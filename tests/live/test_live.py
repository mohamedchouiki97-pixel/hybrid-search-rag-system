"""Live checks against the real OpenAI APIs. Run by hand: `uv run pytest -m live`.

Cost: a handful of small calls (well under a cent with a mini model).
Needs OPENAI_API_KEY and LLM_MODEL in .env.
"""

import pytest

from rag.api.service import build_service
from rag.core.config import Settings
from rag.core.models import Answer, Citation, RetrievedChunk
from rag.generation.citation_verifier import LLMCitationVerifier
from rag.generation.confidence import ConfidenceScorer
from rag.generation.llm_client import make_llm_client
from rag.indexing.embedder import OpenAIEmbedder
from rag.ingestion.loaders import iter_corpus_files


@pytest.fixture(scope="module")
def live_settings(tmp_path_factory):
    try:
        settings = Settings()  # type: ignore[call-arg]
    except Exception as exc:  # missing LLM_MODEL
        pytest.skip(f"settings incomplete: {exc}")
    if settings.openai_api_key is None:
        pytest.skip("OPENAI_API_KEY not set")
    tmp = tmp_path_factory.mktemp("live")
    return settings.model_copy(
        update={"chroma_path": tmp / "chroma", "documents_path": tmp / "docs", "cache_path": tmp / "cache"}
    )


def test_real_embedder_dimensions(live_settings):
    key = live_settings.openai_api_key.get_secret_value()
    vectors = OpenAIEmbedder(live_settings.embedding_model, api_key=key).embed(["hello", "world"])
    expected = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}.get(live_settings.embedding_model)
    assert len(vectors) == 2
    assert expected is None or len(vectors[0]) == expected


def test_real_judge_returns_parseable_verdicts(live_settings, make_chunk):
    judge = make_llm_client(live_settings)
    passage = RetrievedChunk(chunk=make_chunk("The broker listens on TCP port 7420 by default."), score=1.0)
    answer = Answer(
        question="What port?",
        answer_text="The default port is 7420 [1]. The default port is 9000 [1].",
        citations=[
            Citation(marker=1, chunk_id=passage.chunk.chunk_id, claim_text="The default port is 7420."),
            Citation(marker=1, chunk_id=passage.chunk.chunk_id, claim_text="The default port is 9000."),
        ],
        retrieved=[passage],
    )
    checked = LLMCitationVerifier(judge).verify(answer, [passage])
    assert [c.verified for c in checked.citations] == [True, False]
    assert all(not c.judge_reason.startswith("unparseable") for c in checked.citations)
    assert 0.5 <= ConfidenceScorer(judge).completeness("What port?", "The default port is 7420.") <= 1.0


def test_real_pipeline_answers_and_abstains(live_settings, corpus_dir):
    service = build_service(live_settings, in_memory=True)
    for path in iter_corpus_files(corpus_dir):
        service.ingest_path(path, root=corpus_dir)

    answer = service.ask("What TCP port does the Nimbus broker listen on by default?")
    assert not answer.abstained and "7420" in answer.answer_text
    assert answer.citations and any(c.verified for c in answer.citations)

    assert service.ask("Does Nimbus Queue support the Kafka wire protocol?").abstained
