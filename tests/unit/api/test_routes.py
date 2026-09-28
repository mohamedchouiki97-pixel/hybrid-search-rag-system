import pytest
from fastapi.testclient import TestClient

from rag.api import main, routes
from rag.core.models import Answer, DocumentSummary, IngestResult
from rag.generation.prompts import ANSWER_SYSTEM


@pytest.fixture
def client(service):
    with TestClient(main.create_app(service)) as c:
        yield c


# ---------- health and documents ----------


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["indexes_in_sync"] is True
    assert body["strategy"] == "recursive" and body["chunks"] > 0


def test_documents(client):
    r = client.get("/v1/documents")
    assert r.status_code == 200
    docs = [DocumentSummary.model_validate(d) for d in r.json()]
    assert [d.doc_id for d in docs] == ["configuration", "errors", "installation", "operations", "overview"]
    assert all(d.chunk_count > 0 for d in docs)


# ---------- ask ----------


def test_ask_returns_full_answer(client):
    r = client.post("/v1/ask", json={"question": "What port does the broker listen on?"})
    assert r.status_code == 200
    answer = Answer.model_validate(r.json())
    assert answer.answer_text == "The broker listens on port 7420 [1]."
    assert not answer.abstained
    assert [(c.marker, c.verified) for c in answer.citations] == [(1, True)]
    assert answer.citations[0].chunk_id == answer.retrieved[0].chunk.chunk_id
    assert 1 <= len(answer.retrieved) <= 5
    assert 0 <= answer.confidence.composite <= 1


def test_ask_dense_mode_and_top_k(client):
    r = client.post("/v1/ask", json={"question": "What port does the broker listen on?", "mode": "dense", "top_k": 2})
    assert r.status_code == 200
    retrieved = r.json()["retrieved"]
    assert len(retrieved) == 2 and all(rc["sparse_rank"] is None for rc in retrieved)


def test_ask_abstains_without_calling_the_generator(client, fake_llm):
    r = client.post("/v1/ask", json={"question": "Xylophone zebra quasar?"})
    assert r.status_code == 200
    body = r.json()
    assert body["abstained"] is True and body["suggested_docs"]
    assert all(system != ANSWER_SYSTEM for system, _ in fake_llm.calls)


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "   "}, {}, {"question": "q", "mode": "sparse"},
     {"question": "q", "top_k": 0}, {"question": "q", "top_k": 51}, {"question": 42}],
)  # fmt: skip
def test_ask_rejects_bad_input(client, payload):
    assert client.post("/v1/ask", json=payload).status_code == 422


def test_ask_rejects_non_json(client):
    assert client.post("/v1/ask", content=b"not json", headers={"content-type": "application/json"}).status_code == 422


# ---------- ingest ----------


def test_ingest_then_ask(client):
    text = b"# Kelp Farming\n\n## Harvest\n\nKelp is harvested every 90 days using the tide gate."
    r = client.post("/v1/ingest", files={"file": ("kelp.md", text, "text/markdown")})
    assert r.status_code == 201
    result = IngestResult.model_validate(r.json())
    assert result.doc_id == "kelp" and result.chunks_added >= 1
    assert "kelp" in [d["doc_id"] for d in client.get("/v1/documents").json()]
    answer = client.post("/v1/ask", json={"question": "When is kelp harvested with the tide gate?"}).json()
    assert answer["retrieved"][0]["chunk"]["doc_id"] == "kelp"


def test_ingest_drops_client_path(client):
    r = client.post("/v1/ingest", files={"file": ("../../evil.md", b"# Evil\n\nText.", "text/markdown")})
    assert r.status_code == 201 and r.json()["doc_id"] == "evil"


@pytest.mark.parametrize("name", ["report.docx", "noext", "image.png"])
def test_ingest_unsupported_type_is_415(client, name):
    r = client.post("/v1/ingest", files={"file": (name, b"data")})
    assert r.status_code == 415
    assert "unsupported file type" in r.json()["detail"]


def test_ingest_empty_file_is_422(client):
    r = client.post("/v1/ingest", files={"file": ("empty.md", b"   \n")})
    assert r.status_code == 422 and "empty file" in r.json()["detail"]


def test_ingest_too_large_is_413(client, monkeypatch):
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/v1/ingest", files={"file": ("big.md", b"# Big\n\n" + b"x" * 100)})
    assert r.status_code == 413


def test_ingest_requires_a_file(client):
    assert client.post("/v1/ingest").status_code == 422


# ---------- OpenAPI ----------


def test_openapi_has_examples(client):
    spec = client.get("/openapi.json").json()
    ask = spec["paths"]["/v1/ask"]["post"]
    assert set(ask["requestBody"]["content"]["application/json"]["examples"]) == {"hybrid", "dense"}
    assert "example" in ask["responses"]["200"]["content"]["application/json"]
    assert {"413", "415", "422"} <= set(spec["paths"]["/v1/ingest"]["post"]["responses"])
    assert {"/healthz", "/v1/ask", "/v1/documents", "/v1/ingest"} <= set(spec["paths"])


# ---------- startup ----------


def test_app_builds_real_service_on_startup(monkeypatch, service):
    calls = []
    monkeypatch.setattr(main, "get_settings", lambda: "settings")
    monkeypatch.setattr(main, "build_service", lambda s: calls.append(s) or service)
    with TestClient(main.create_app()) as c:
        assert c.get("/healthz").status_code == 200
    assert calls == ["settings"]
