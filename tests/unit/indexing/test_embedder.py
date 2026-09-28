from types import SimpleNamespace

import pytest

from rag.core.fakes import FakeEmbedder
from rag.core.interfaces import Embedder
from rag.indexing.embedder import CachingEmbedder, OpenAIEmbedder, make_embedder


class StubOpenAIClient:
    """Mimics client.embeddings.create; returns data out of order, like the API may."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input):
        self.calls.append(list(input))
        data = [SimpleNamespace(index=i, embedding=[float(len(t)), float(i)]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))


def test_openai_embedder_batches_and_keeps_order():
    client = StubOpenAIClient()
    emb = OpenAIEmbedder("m", batch_size=2, client=client)
    vectors = emb.embed(["a", "bb", "ccc"])
    assert client.calls == [["a", "bb"], ["ccc"]]
    assert vectors == [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
    assert emb.embed([]) == []
    assert isinstance(emb, Embedder)


def test_openai_embedder_rejects_bad_batch_size():
    with pytest.raises(ValueError):
        OpenAIEmbedder("m", batch_size=0)


@pytest.fixture
def cache_file(tmp_path):
    return tmp_path / "cache" / "embeddings.sqlite"


def test_cache_hits_skip_the_inner_embedder(cache_file):
    inner = FakeEmbedder(dim=8)
    emb = CachingEmbedder(inner, cache_file, namespace="m")
    first = emb.embed(["a", "b"])
    second = emb.embed(["b", "a", "c"])
    assert inner.calls == [["a", "b"], ["c"]]
    assert second[:2] == [first[1], first[0]]
    assert (emb.hits, emb.misses) == (2, 3)


def test_cache_dedupes_within_one_call(cache_file):
    inner = FakeEmbedder(dim=4)
    vectors = CachingEmbedder(inner, cache_file, namespace="m").embed(["x", "x", "y"])
    assert inner.calls == [["x", "y"]]
    assert vectors[0] == vectors[1]


def test_cache_persists_across_instances(cache_file):
    vec = CachingEmbedder(FakeEmbedder(dim=4), cache_file, namespace="m").embed(["x"])
    inner = FakeEmbedder(dim=4)
    again = CachingEmbedder(inner, cache_file, namespace="m").embed(["x"])
    assert inner.calls == []
    assert again == vec  # identical: fresh vectors are rounded to float32 just like cached ones


def test_cache_namespaces_do_not_mix(cache_file):
    CachingEmbedder(FakeEmbedder(dim=4), cache_file, namespace="model-a").embed(["x"])
    inner = FakeEmbedder(dim=4)
    CachingEmbedder(inner, cache_file, namespace="model-b").embed(["x"])
    assert inner.calls == [["x"]]


def test_make_embedder_uses_settings(settings):
    emb = make_embedder(settings)
    assert emb.namespace == "text-embedding-3-small"
    assert isinstance(emb.inner, OpenAIEmbedder)
    assert (settings.cache_path / "embeddings.sqlite").exists()
    emb.close()
