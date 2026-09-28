"""Streamlit dashboard for the RAG API.

    uv run streamlit run src/rag/dashboard/app.py

Set API_URL if the API is not on http://localhost:8000.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

from rag.dashboard.helpers import (
    chunk_table,
    citation_status,
    cited_markers,
    confidence_rows,
    format_score,
    linkify_citations,
    source_title,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = httpx.Timeout(180.0, connect=5.0)


def api_get(path: str) -> Any:
    response = httpx.get(f"{API_URL}{path}", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def ask(question: str, mode: str, top_k: int) -> dict[str, Any]:
    response = httpx.post(f"{API_URL}/v1/ask", json={"question": question, "mode": mode, "top_k": top_k}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def render_answer(answer: dict[str, Any], key: str) -> None:
    if answer["abstained"]:
        st.warning(answer["answer_text"])
        st.markdown(f"**Found:** {answer['found']}  \n**Missing:** {answer['missing']}")
        if answer["suggested_docs"]:
            st.markdown("**Try:** " + ", ".join(f"`{d}`" for d in answer["suggested_docs"]))
    else:
        st.markdown(linkify_citations(answer["answer_text"], anchor_prefix=key))

    st.subheader("Confidence")
    for label, value in confidence_rows(answer["confidence"]):
        st.progress(value, text=f"{label}: {value:.2f}")

    if answer["citations"]:
        st.subheader("Citations")
        for c in answer["citations"]:
            st.markdown(f"**[{c['marker']}]** {citation_status(c['verified'])}: {c['claim_text']}")
            if c["judge_reason"]:
                st.caption(c["judge_reason"])

    st.subheader("Retrieved chunks")
    cited = cited_markers(answer)
    for i, rc in enumerate(answer["retrieved"], start=1):
        st.markdown(f'<a id="{key}-{i}"></a>', unsafe_allow_html=True)
        with st.expander(f"[{i}] {source_title(rc['chunk'])} · score {format_score(rc['score'])}", expanded=i in cited):
            st.markdown(rc["chunk"]["text"])
    st.dataframe(chunk_table(answer["retrieved"]), hide_index=True, use_container_width=True)


def sidebar() -> None:
    st.sidebar.header("Index")
    try:
        health = api_get("/healthz")
        docs = api_get("/v1/documents")
    except httpx.HTTPError as exc:
        st.sidebar.error(f"API not reachable at {API_URL}: {exc}")
        return
    icon = "🟢" if health["indexes_in_sync"] else "🟠"
    st.sidebar.markdown(
        f"{icon} **{health['status']}** · strategy `{health['strategy']}`  \n"
        f"{len(docs)} documents · {health['chunks']} chunks"
    )
    upload = st.sidebar.file_uploader("Add a document", type=["md", "txt", "html", "htm", "pdf"])
    if upload is not None and st.sidebar.button("Ingest"):
        response = httpx.post(f"{API_URL}/v1/ingest", files={"file": (upload.name, upload.getvalue())}, timeout=TIMEOUT)
        if response.status_code == 201:
            result = response.json()
            st.sidebar.success(f"{result['doc_id']}: {result['chunks_added']} chunks added")
        else:
            st.sidebar.error(response.json().get("detail", response.text))


def main() -> None:
    st.set_page_config(page_title="Hybrid Search RAG", layout="wide")
    st.title("Hybrid Search RAG")
    sidebar()

    with st.form("ask"):
        question = st.text_area("Question", placeholder="How do I declare an optional query parameter?")
        c1, c2, c3 = st.columns(3)
        mode = c1.radio("Retrieval", ["hybrid", "dense"], horizontal=True)
        top_k = c2.slider("Chunks", 1, 10, 5)
        side_by_side = c3.checkbox("Compare hybrid vs dense side by side")
        submitted = st.form_submit_button("Ask")

    if not submitted or not question.strip():
        return
    try:
        if side_by_side:
            columns = st.columns(2)
            for column, m in zip(columns, ["hybrid", "dense"], strict=True):
                with column:
                    st.header(m)
                    with st.spinner(f"Asking ({m})..."):
                        render_answer(ask(question, m, top_k), key=m)
        else:
            with st.spinner("Asking..."):
                render_answer(ask(question, mode, top_k), key="source")
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    except httpx.HTTPError as exc:
        st.error(f"API not reachable at {API_URL}: {exc}")


main()
