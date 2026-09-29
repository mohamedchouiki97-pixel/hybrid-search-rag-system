"""End-to-end flows over real ChromaDB and BM25, with fake embeddings and LLM."""

import shutil

import pytest
from fastapi.testclient import TestClient

from rag.api.main import create_app
from rag.core.fakes import FakeEmbedder, FakeLLM
from rag.core.models import Answer, ChunkStrategy, RetrievalMode
from rag.evaluation.golden import GoldenItem
from rag.evaluation.runner import EvalRunner
from rag.generation.prompts import ANSWER_SYSTEM
from rag.indexing.vector_store import ChromaVectorStore

DOCS = ["configuration", "errors", "installation", "operations", "overview"]


# ---------- ingest -> index ----------


def test_ingest_fixture_corpus_into_both_indexes(make_service, corpus_dir):
    service = make_service(corpus=corpus_dir)
    assert isinstance(service.indexer.vector_store, ChromaVectorStore)
    docs = service.list_documents()
    assert [d.doc_id for d in docs] == DOCS
    assert all(d.chunk_count >= 2 for d in docs)
    assert service.indexer.vector_store.ids() == service.indexer.sparse_index.ids()
    assert sum(d.chunk_count for d in docs) == service.indexer.vector_store.count()


def test_reingest_changes_nothing_and_copies_are_deduplicated(make_service, corpus_dir, tmp_path):
    service = make_service(corpus=corpus_dir)
    ids_before = service.indexer.vector_store.ids()

    again = service.ingest_path(corpus_dir / "errors.md", root=corpus_dir)
    assert again.chunks_skipped_duplicate == 0  # same doc replaces itself
    assert service.indexer.vector_store.ids() == ids_before

    copy = tmp_path / "errors-copy.md"
    shutil.copyfile(corpus_dir / "errors.md", copy)
    result = service.ingest_path(copy, root=tmp_path)
    assert (result.chunks_added, result.chunks_skipped_duplicate) == (0, again.chunks_added)
    assert service.indexer.in_sync()


def test_indexes_survive_a_restart(make_service, corpus_dir):
    count = make_service(corpus=corpus_dir).indexer.vector_store.count()
    reopened = make_service(ingest=False)
    assert reopened.indexer.vector_store.count() == count
    assert reopened.indexer.in_sync()


# ---------- index -> retrieve ----------


def test_known_fact_is_in_top_5(make_service, corpus_dir, known_facts):
    service = make_service(corpus=corpus_dir)
    for fact in known_facts["lookup"]:
        top = service.retriever.retrieve(fact["question"], RetrievalMode.HYBRID, 5)
        assert any(fact["answer"] in rc.chunk.text for rc in top), fact["question"]


def test_hybrid_finds_exact_token_that_dense_misses(make_service, corpus_dir, known_facts):
    rare = known_facts["rare_token"][0]
    # Rig dense search: the question embeds exactly like the Networking section, so
    # dense ranks that first and the error-code chunk is nowhere near the top.
    probe = make_service(corpus=corpus_dir)
    networking = next(rc.chunk for rc in probe.retriever.retrieve("port 7420 TLS", RetrievalMode.HYBRID, 1))
    (vector,) = FakeEmbedder().embed([networking.text])
    service = make_service(corpus=corpus_dir, embedder=FakeEmbedder(overrides={rare["question"]: vector}))

    dense = service.retriever.retrieve(rare["question"], RetrievalMode.DENSE, 1)
    assert rare["token"] not in dense[0].chunk.text
    hybrid = service.retriever.retrieve(rare["question"], RetrievalMode.HYBRID, 1)
    assert rare["token"] in hybrid[0].chunk.text and hybrid[0].sparse_rank == 1


# ---------- retrieve -> generate -> verify ----------


def test_citations_map_to_real_retrieved_chunks(make_service, corpus_dir, llm):
    llm.default = "The broker listens on port 7420 [1]. Traffic between nodes uses TLS 1.3 [2]."
    answer = make_service(corpus=corpus_dir).ask("What port and encryption does the broker use?")
    retrieved_ids = [rc.chunk.chunk_id for rc in answer.retrieved]
    assert [c.chunk_id for c in answer.citations] == retrieved_ids[:2]
    assert all(c.verified for c in answer.citations)


def test_planted_bad_citation_is_flagged(make_service, corpus_dir, llm):
    llm.default = "The broker listens on port 7420 [1]. The broker is written in COBOL [2]. It has 9 lives [7]."
    llm.responses = {
        "CLAIM:\nThe broker is written in COBOL.": '{"supported": false, "reason": "not in the passage"}',
        "CLAIM:": '{"supported": true, "reason": "stated"}',
        "ANSWER TO RATE": '{"score": 1}',
    }
    answer = make_service(corpus=corpus_dir).ask("Tell me about the broker port.")
    verdicts = {c.claim_text: c.verified for c in answer.citations}
    assert verdicts == {
        "The broker listens on port 7420.": True,
        "The broker is written in COBOL.": False,
        "It has 9 lives.": False,  # [7] matches no retrieved chunk
    }
    assert answer.confidence.citation_coverage == pytest.approx(1 / 3)


# ---------- abstain ----------


def test_abstains_without_reaching_the_generator(make_service, corpus_dir, known_facts, llm, cross_encoder):
    service = make_service(corpus=corpus_dir, reranker=cross_encoder)
    for fact in known_facts["no_answer"]:
        answer = service.ask(fact["question"])
        assert answer.abstained, fact["question"]
        assert answer.suggested_docs
    assert all(system != ANSWER_SYSTEM for system, _ in llm.calls)

    answered = service.ask(known_facts["lookup"][0]["question"])
    assert not answered.abstained and answered.confidence.retrieval > 0.9


# ---------- API ----------


def test_api_upload_then_ask(make_service, llm):
    llm.default = "Kelp is harvested every 90 days [1]."
    service = make_service(ingest=False)
    with TestClient(create_app(service)) as client:
        text = b"# Kelp Farm\n\n## Harvest\n\nKelp is harvested every 90 days through the tide gate."
        assert client.post("/v1/ingest", files={"file": ("kelp.md", text)}).status_code == 201
        r = client.post("/v1/ask", json={"question": "How often is kelp harvested through the tide gate?"})
    assert r.status_code == 200
    answer = Answer.model_validate(r.json())
    assert answer.retrieved[0].chunk.doc_id == "kelp"
    assert [(c.marker, c.verified) for c in answer.citations] == [(1, True)]


# ---------- evaluation smoke ----------


def test_eval_runner_smoke(make_service, corpus_dir, known_facts, tmp_path):
    lookup = known_facts["lookup"][:3]
    items = [
        GoldenItem(id=f"s{i}", type="lookup", question=f["question"], gold_answer=f["answer"],
                   gold_chunk_sections=[{"doc_id": f["doc"].removesuffix(".md"),
                                         "section": _heading(corpus_dir / f["doc"], f["section"])}])
        for i, f in enumerate(lookup)
    ] + [GoldenItem(id=f"n{i}", type="no_answer", question=f["question"], gold_answer="Not covered.")
         for i, f in enumerate(known_facts["no_answer"])]  # fmt: skip
    summary = EvalRunner(FakeLLM(default='{"score": 1, "supported_claims": 1, "total_claims": 1}')).run(
        make_service(corpus=corpus_dir), items, tmp_path, label="smoke"
    )
    assert (tmp_path / "smoke.results.json").exists() and (tmp_path / "smoke.summary.md").exists()
    assert summary["overall"]["n"] == 5 and summary["errors"] == 0
    assert summary["by_type"]["lookup"]["recall_at_k"] == 1.0


def _heading(path, section):
    title = path.read_text(encoding="utf-8").splitlines()[0].removeprefix("# ")
    return f"{title} > {section}"


# ---------- chunking switch ----------


def test_all_strategies_share_one_chroma_path_without_colliding(make_service, corpus_dir):
    counts = {}
    for strategy in ChunkStrategy:
        service = make_service(strategy=strategy, corpus=corpus_dir)
        chunks = [rc.chunk for rc in service.retriever.retrieve("broker port", RetrievalMode.HYBRID, 5)]
        assert chunks and all(c.strategy is strategy for c in chunks)
        assert service.indexer.in_sync()
        counts[strategy] = service.indexer.vector_store.count()
    for strategy, count in counts.items():  # each strategy's index untouched by the others
        assert make_service(strategy=strategy, ingest=False).indexer.vector_store.count() == count
