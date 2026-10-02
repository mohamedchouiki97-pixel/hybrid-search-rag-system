# One image for the api, the dashboard and the seed job (compose picks the command).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    PYTHONUNBUFFERED=1

# Unprivileged user, created first so later files are owned by it from the start
# (a `chown -R` after the fact would copy those files into a second layer).
RUN useradd --create-home app && mkdir -p /app/data /opt/hf && chown app /app/data /opt/hf

WORKDIR /app

# Dependencies first, so code changes don't reinstall them. The cache mount keeps
# uv's download cache out of the image (it was 1.7 GB) and reuses it across builds.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY corpus ./corpus
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

USER app

# Bake the cross-encoder into the image so containers never download it at startup.
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${RERANKER_MODEL}')"
ENV HF_HUB_OFFLINE=1

EXPOSE 8000 8501
CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
