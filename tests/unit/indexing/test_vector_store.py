import pytest

from rag.core.interfaces import VectorStore
from rag.indexing.vector_store import ChromaVectorStore


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(tmp_path / "chroma", collection="test")


def test_protocol(store):
    assert isinstance(store, VectorStore)


def test_add_query_scores_are_cosine(store, make_chunk):
    a, b, c = (make_chunk(t, chunk_index=i) for i, t in enumerate(["a", "b", "c"]))
    store.add([a, b, c], [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    results = store.query([1.0, 0.0], k=2)
    assert [ch.chunk_id for ch, _ in results] == [a.chunk_id, b.chunk_id]
    assert [s for _, s in results] == pytest.approx([1.0, 0.6], abs=1e-4)


def test_query_empty_store_and_k_larger_than_count(store, make_chunk):
    assert store.query([1.0, 0.0], k=5) == []
    store.add([make_chunk("a")], [[1.0, 0.0]])
    assert len(store.query([1.0, 0.0], k=5)) == 1


def test_round_trips_all_chunk_fields(store, make_chunk):
    full = make_chunk("with extras", chunk_index=0, section_heading="Guide > Setup", page_number=3)
    bare = make_chunk("no extras", chunk_index=1)
    store.add([full, bare], [[1.0, 0.0], [0.0, 1.0]])
    assert store.query([1.0, 0.0], k=1)[0][0] == full
    assert store.query([0.0, 1.0], k=1)[0][0] == bare


def test_upsert_count_ids_delete(store, make_chunk):
    a = make_chunk("a", doc_id="d1", chunk_index=0)
    b = make_chunk("b", doc_id="d1", chunk_index=1)
    c = make_chunk("c", doc_id="d2", chunk_index=0)
    store.add([a, b, c], [[1.0, 0.0]] * 3)
    store.add([a], [[0.0, 1.0]])
    assert store.count() == 3
    assert store.ids() == {a.chunk_id, b.chunk_id, c.chunk_id}
    store.delete([b.chunk_id])
    store.delete([])
    assert store.ids() == {a.chunk_id, c.chunk_id}
    store.delete_doc("d2")
    assert store.ids() == {a.chunk_id}


def test_list_docs(store, make_chunk):
    store.add(
        [
            make_chunk("x", doc_id="d2"),
            make_chunk("y", doc_id="d1", chunk_index=0),
            make_chunk("z", doc_id="d1", chunk_index=1),
        ],
        [[1.0, 0.0]] * 3,
    )
    assert [(d.doc_id, d.chunk_count) for d in store.list_docs()] == [("d1", 2), ("d2", 1)]


def test_length_mismatch_rejected(store, make_chunk):
    with pytest.raises(ValueError):
        store.add([make_chunk("a")], [])


def test_persists_across_instances(tmp_path, make_chunk):
    path = tmp_path / "chroma"
    ChromaVectorStore(path, collection="persisted").add([make_chunk("a")], [[1.0, 0.0]])
    assert ChromaVectorStore(path, collection="persisted").count() == 1
    assert ChromaVectorStore(path, collection="other").count() == 0
