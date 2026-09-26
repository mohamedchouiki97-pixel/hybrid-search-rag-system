import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from rag.core import interfaces
from rag.core.fakes import (
    FakeEmbedder,
    FakeLLM,
    FakeReranker,
    InMemorySparseIndex,
    InMemoryVectorStore,
    cosine,
    tokenize,
)
from rag.core.models import RetrievedChunk

# ---------- helpers ----------


def test_tokenize():
    assert tokenize("Error ERR-7Q42, retry!") == ["error", "err", "7q42", "retry"]


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine([0, 0], [1, 0]) == 0.0
    with pytest.raises(ValueError):
        cosine([1], [1, 2])


# ---------- protocol conformance ----------


@pytest.mark.parametrize(
    ("fake", "protocol"),
    [
        (FakeEmbedder(), interfaces.Embedder),
        (FakeLLM(), interfaces.LLMClient),
        (FakeReranker(), interfaces.Reranker),
        (InMemoryVectorStore(), interfaces.VectorStore),
        (InMemorySparseIndex(), interfaces.SparseIndex),
    ],
)
def test_fakes_satisfy_protocols(fake, protocol):
    assert isinstance(fake, protocol)


# ---------- FakeEmbedder ----------


def test_embedder_is_deterministic_across_instances():
    a = FakeEmbedder().embed(["hello", "world"])
    b = FakeEmbedder().embed(["hello", "world"])
    assert a == b
    assert a[0] != a[1]


def test_embedder_returns_unit_vectors_of_dim():
    (vec,) = FakeEmbedder(dim=16).embed(["x"])
    assert len(vec) == 16
    assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0)


def test_embedder_overrides_and_records_calls():
    emb = FakeEmbedder(dim=2, overrides={"cats": [1.0, 0.0]})
    assert emb.embed(["cats"]) == [[1.0, 0.0]]
    assert emb.calls == [["cats"]]


def test_embedder_rejects_bad_config():
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)
    with pytest.raises(ValueError):
        FakeEmbedder(dim=3, overrides={"a": [1.0]})


# ---------- FakeLLM ----------


def test_llm_matches_substring_in_system_or_user():
    llm = FakeLLM({"JUDGE": "yes", "capital": "Paris [1]"}, default="fallback")
    assert llm.complete("You are a JUDGE", "claim") == "yes"
    assert llm.complete("sys", "What is the capital?") == "Paris [1]"
    assert llm.complete("sys", "unrelated") == "fallback"
    assert llm.call_count == 3
    assert llm.calls[0] == ("You are a JUDGE", "claim")


def test_llm_first_matching_key_wins():
    llm = FakeLLM({"a": "first", "ab": "second"})
    assert llm.complete("", "ab") == "first"


def test_llm_records_concurrent_calls():
    llm = FakeLLM()
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: llm.complete("s", str(i)), range(100)))
    assert llm.call_count == 100


# ---------- FakeReranker ----------


def test_reranker_sorts_by_overlap_and_truncates(make_chunk):
    chunks = [
        RetrievedChunk(chunk=make_chunk("nothing relevant", chunk_index=0), score=0.9),
        RetrievedChunk(chunk=make_chunk("the lease timeout default", chunk_index=1), score=0.1),
        RetrievedChunk(chunk=make_chunk("lease", chunk_index=2), score=0.5),
    ]
    out = FakeReranker().rerank("what is the lease timeout", chunks, top_n=2)
    assert [rc.chunk.chunk_index for rc in out] == [1, 2]
    assert [rc.rerank_score for rc in out] == [3.0, 1.0]
    assert all(rc.score == rc.rerank_score for rc in out)
    assert chunks[1].rerank_score is None  # input not mutated


# ---------- InMemoryVectorStore ----------


def test_vector_store_query_orders_by_cosine(make_chunk):
    store = InMemoryVectorStore()
    a, b, c = (make_chunk(t, chunk_index=i) for i, t in enumerate("abc"))
    store.add([a, b, c], [[1, 0], [0.7, 0.7], [0, 1]])
    results = store.query([1, 0], k=2)
    assert [ch.chunk_id for ch, _ in results] == [a.chunk_id, b.chunk_id]
    assert results[0][1] == pytest.approx(1.0)


def test_vector_store_upsert_count_ids_delete(make_chunk):
    store = InMemoryVectorStore()
    a1 = make_chunk("a", doc_id="d1", chunk_index=0)
    a2 = make_chunk("b", doc_id="d1", chunk_index=1)
    b1 = make_chunk("c", doc_id="d2", chunk_index=0)
    store.add([a1, a2, b1], [[1, 0]] * 3)
    store.add([a1], [[0, 1]])  # upsert, not duplicate
    assert store.count() == 3
    assert store.ids() == {a1.chunk_id, a2.chunk_id, b1.chunk_id}

    store.delete([a2.chunk_id, "unknown"])
    assert store.ids() == {a1.chunk_id, b1.chunk_id}
    store.delete_doc("d2")
    assert store.ids() == {a1.chunk_id}


def test_vector_store_list_docs(make_chunk):
    store = InMemoryVectorStore()
    store.add(
        [make_chunk("a", doc_id="d2"), make_chunk("b", doc_id="d1", chunk_index=0), make_chunk("c", doc_id="d1", chunk_index=1)],
        [[1.0]] * 3,
    )
    docs = store.list_docs()
    assert [(d.doc_id, d.source_path, d.chunk_count) for d in docs] == [("d1", "d1.md", 2), ("d2", "d2.md", 1)]


def test_vector_store_rejects_length_mismatch(make_chunk):
    with pytest.raises(ValueError):
        InMemoryVectorStore().add([make_chunk("a")], [])


# ---------- InMemorySparseIndex ----------


def test_sparse_index_finds_rare_token_and_deletes(make_chunk):
    idx = InMemorySparseIndex()
    hit = make_chunk("Error ERR-7Q42 means the lease expired", doc_id="errors")
    miss = make_chunk("Backups copy closed segments", doc_id="ops")
    idx.add([hit, miss])
    assert [c.chunk_id for c, _ in idx.query("what is 7Q42", k=5)] == [hit.chunk_id]
    assert idx.query("kafka", k=5) == []

    idx.delete_doc("errors")
    assert idx.ids() == {miss.chunk_id}
    idx.delete([miss.chunk_id])
    assert idx.ids() == set()
