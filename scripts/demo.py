"""Five questions, one per behaviour, against the running API.

    uv run uvicorn rag.api.main:app          # in one terminal
    uv run python scripts/demo.py            # in another

Set API_URL if the API is not on http://localhost:8000. Costs a few cents at most.
"""

from __future__ import annotations

import os
import textwrap
import time

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

DEMOS = [
    ("1. Plain lookup",
     "My app has one route for /users/me and another for /users/{user_id}, and requests to /users/me keep "
     "landing in the second one. What's going wrong?", ["hybrid"]),
    ("2. Hybrid vs dense: an exact identifier that keyword search catches",
     "What does OAuth2PasswordBearer do when the Authorization header is missing, "
     "and what status code and header should such an error carry?", ["hybrid", "dense"]),
    ("3. Multi-hop: the answer spans two sections",
     "For the login endpoint in the password flow, what exact field names must the client send, "
     "and what keys must my JSON reply contain?", ["hybrid"]),
    ("4. Refusal by the model: retrieval looked relevant, the docs don't answer",
     "How do I issue and rotate refresh tokens alongside the JWT access tokens from the security tutorial?",
     ["hybrid"]),
    ("5. Refusal by the retrieval gate: nothing relevant, no LLM call",
     "How can I expose Prometheus metrics from my FastAPI app?", ["hybrid"]),
]  # fmt: skip


def show(answer: dict, mode: str, seconds: float) -> None:
    c = answer["confidence"]
    print(f"  [{mode}] {seconds:.1f}s | retrieval {c['retrieval']:.2f} | coverage {c['citation_coverage']:.2f} "
          f"| completeness {c['completeness']:.2f} | composite {c['composite']:.2f}")  # fmt: skip
    if answer["abstained"]:
        print(f"  ABSTAINED: {answer['missing']}")
        print(f"  {answer['found']}")
    else:
        print(textwrap.indent(textwrap.fill(answer["answer_text"], 100), "  > "))
        ok = sum(1 for x in answer["citations"] if x["verified"])
        print(f"  citations verified: {ok}/{len(answer['citations'])}")
    for i, rc in enumerate(answer["retrieved"][:3], start=1):
        ch = rc["chunk"]
        ranks = " ".join(
            f"{name}#{rc[key]}" for name, key in (("dense", "dense_rank"), ("bm25", "sparse_rank")) if rc[key]
        )
        print(f"    {i}. {ch['doc_id']} | {ch['section_heading']}  ({ranks})")


def main() -> None:
    with httpx.Client(timeout=180) as client:
        health = client.get(f"{API_URL}/healthz").json()
        print(f"API {API_URL}: {health['status']}, strategy {health['strategy']}, {health['chunks']} chunks\n")
        for title, question, modes in DEMOS:
            print(title)
            print(textwrap.indent(textwrap.fill(question, 100), "  Q: "))
            for mode in modes:
                start = time.perf_counter()
                answer = client.post(f"{API_URL}/v1/ask", json={"question": question, "mode": mode}).json()
                show(answer, mode, time.perf_counter() - start)
            print()


if __name__ == "__main__":
    main()
