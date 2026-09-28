from types import SimpleNamespace

import pytest

from rag.core.config import ConfidenceWeights, LLMProvider
from rag.core.fakes import FakeLLM
from rag.core.models import RetrievedChunk
from rag.generation.abstain import ABSTAIN_TEXT, build_abstain_answer, should_abstain
from rag.generation.answering import AnswerFlow, build_answer_flow
from rag.generation.llm_client import OpenAILLMClient, make_llm_client
from rag.generation.prompts import ANSWER_SYSTEM


def rc(make_chunk, text, doc_id, score, heading=None, index=0):
    chunk = make_chunk(text, doc_id=doc_id, chunk_index=index, section_heading=heading)
    return RetrievedChunk(chunk=chunk, score=score, rerank_score=score)


# ---------- abstain ----------


def test_should_abstain():
    assert should_abstain(0.29, 0.3) and not should_abstain(0.3, 0.3)


def test_build_abstain_answer(make_chunk):
    chunks = [
        rc(make_chunk, "a", "errors", 0.2, "Error Codes > ERR-3B10"),
        rc(make_chunk, "b", "errors", 0.15, index=1),
        rc(make_chunk, "c", "operations", 0.1),
        rc(make_chunk, "d", "overview", 0.05),
        rc(make_chunk, "e", "installation", 0.01),
    ]
    answer = build_abstain_answer("Kafka support?", chunks, retrieval=0.2, threshold=0.3)
    assert answer.abstained and answer.answer_text == ABSTAIN_TEXT
    assert answer.suggested_docs == ["errors", "operations", "overview"]  # unique, top 3
    assert answer.found == "Closest matches: Error Codes > ERR-3B10 (errors); operations; overview."
    assert "0.20" in answer.missing and "0.30" in answer.missing
    assert answer.citations == [] and answer.retrieved == chunks
    assert answer.confidence.retrieval == 0.2
    assert answer.confidence.composite == pytest.approx(0.4 * 0.2)


def test_abstain_with_no_chunks():
    answer = build_abstain_answer("q", [], retrieval=0.0, threshold=0.3)
    assert answer.found == "No related passages were found." and answer.suggested_docs == []


# ---------- full flow ----------


def make_flow(llm, threshold=0.3):
    s = SimpleNamespace(abstain_threshold=threshold, confidence_weights=ConfidenceWeights())
    return build_answer_flow(s, llm)


def test_abstain_path_never_calls_the_llm(make_chunk):
    llm = FakeLLM(default="should not be used")
    answer = make_flow(llm).answer("Kafka?", [rc(make_chunk, "unrelated", "d", 0.1)])
    assert answer.abstained
    assert llm.call_count == 0


def test_answer_path_generates_verifies_and_scores(make_chunk):
    chunks = [rc(make_chunk, "The broker listens on port 7420.", "overview", 0.9, "Networking")]
    llm = FakeLLM(
        {
            "Question: What port": "The broker listens on port 7420 [1]. It also sings [2].",
            "CLAIM:\nThe broker listens on port 7420.": '{"supported": true, "reason": "stated"}',
            "ANSWER TO RATE": '{"score": 1}',
        }
    )
    answer = make_flow(llm).answer("What port?", chunks)
    assert not answer.abstained
    assert [(c.marker, c.verified) for c in answer.citations] == [(1, True), (2, False)]  # [2] has no chunk
    assert answer.confidence.retrieval == 0.9
    assert answer.confidence.citation_coverage == 0.5
    assert answer.confidence.completeness == 1.0
    systems = [system for system, _ in llm.calls]
    assert systems.count(ANSWER_SYSTEM) == 1
    assert llm.call_count == 3  # answer + one citation check ([2] skips the judge) + completeness


def test_empty_retrieval_abstains():
    assert AnswerFlow(None, None, None).answer("q", []).abstained  # type: ignore[arg-type]


# ---------- LLM client ----------


class StubChat:
    def __init__(self, reject_temperature=False):
        self.reject_temperature = reject_temperature
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_temperature and "temperature" in kwargs:
            raise BadRequestError("Unsupported value: 'temperature' does not support 0")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])


class BadRequestError(Exception):
    """Same class name as openai.BadRequestError, which is what the client checks."""


def test_openai_client_sends_messages_and_temperature():
    stub = StubChat()
    client = OpenAILLMClient("gpt-test", client=stub)
    assert client.complete("sys", "usr") == "hi"
    [call] = stub.calls
    assert call["model"] == "gpt-test" and call["temperature"] == 0.0
    assert call["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]


def test_openai_client_drops_temperature_once_rejected():
    stub = StubChat(reject_temperature=True)
    client = OpenAILLMClient("reasoning-model", client=stub)
    assert client.complete("s", "u") == "hi"
    assert client.complete("s", "u") == "hi"
    assert ["temperature" in c for c in stub.calls] == [True, False, False]


def test_openai_client_reraises_other_errors():
    class Boom:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(ValueError("x"))))

    with pytest.raises(ValueError):
        OpenAILLMClient("m", client=Boom()).complete("s", "u")


def test_make_llm_client(settings):
    client = make_llm_client(settings)
    assert isinstance(client, OpenAILLMClient) and client.model == "fake-model"
    with pytest.raises(NotImplementedError, match="LLM_PROVIDER=openai"):
        make_llm_client(settings.model_copy(update={"llm_provider": LLMProvider.ANTHROPIC}))
