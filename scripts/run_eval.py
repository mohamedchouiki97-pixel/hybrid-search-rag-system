"""Run the golden-set evaluation against the real pipeline.

    uv run python scripts/run_eval.py                    # configured strategy, hybrid
    uv run python scripts/run_eval.py --what chunking    # fixed vs recursive vs semantic
    uv run python scripts/run_eval.py --what retrieval   # hybrid vs dense-only
    uv run python scripts/run_eval.py --what all --limit 5   # quick smoke run

Needs OPENAI_API_KEY and LLM_MODEL, and an index built by scripts/seed.py.
Judge calls are cached in <cache_path>/judge.sqlite, so re-runs only pay for
generation. Reports go to reports/eval/ by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.api.service import build_service
from rag.core.config import get_settings
from rag.core.models import Answer, RetrievalMode
from rag.evaluation.chunking_report import compare, run_chunking_comparison
from rag.evaluation.golden import load_catalog, load_golden, validate_golden
from rag.evaluation.judge_cache import CachingLLMClient
from rag.evaluation.runner import EvalRunner, render_summary
from rag.generation.llm_client import make_llm_client


class ModePipeline:
    """The same service, pinned to one retrieval mode."""

    def __init__(self, service, mode: RetrievalMode) -> None:
        self.service, self.mode = service, mode

    def ask(self, question: str) -> Answer:
        return self.service.ask(question, self.mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--what", choices=["single", "chunking", "retrieval", "all"], default="single")
    parser.add_argument("--limit", type=int, default=None, help="only the first N golden questions")
    parser.add_argument("--out", type=Path, default=Path("reports/eval"))
    parser.add_argument("--resume", action="store_true", help="reuse variants that already finished without errors")
    args = parser.parse_args()

    settings = get_settings()
    items = load_golden()
    problems = validate_golden(items, load_catalog())
    if problems:
        print("Golden set is invalid:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    items = items[: args.limit]

    judge = CachingLLMClient(make_llm_client(settings), settings.cache_path / "judge.sqlite", namespace=settings.llm_model)
    runner = EvalRunner(judge, k=settings.rerank_top_n, resume=args.resume, progress=lambda msg: print(f"\r{msg}   ", end="", flush=True))

    def service_for(strategy=None):
        service = build_service(settings, strategy)
        if service.indexer.vector_store.count() == 0:
            raise SystemExit(f"\nThe {service.strategy} index is empty. Run: uv run python scripts/seed.py")
        return service

    if args.what in ("single", "all"):
        service = service_for()
        label = f"{service.strategy}-hybrid"
        summary = runner.run(service, items, args.out, label=label)
        print("\n" + render_summary(label, summary, runner.k))
    if args.what in ("chunking", "all"):
        run_chunking_comparison(runner, service_for, items, args.out)
        print("\n" + (args.out / "chunking_comparison.md").read_text(encoding="utf-8"))
    if args.what in ("retrieval", "all"):
        service = service_for()
        compare(runner, ["hybrid", "dense"], lambda mode: ModePipeline(service, RetrievalMode(mode)), items,
                args.out, title=f"Hybrid vs dense-only ({service.strategy} chunks)", filename="retrieval_comparison.md")  # fmt: skip
        print("\n" + (args.out / "retrieval_comparison.md").read_text(encoding="utf-8"))
    print(f"Judge cache: {judge.hits} hits, {judge.misses} new calls. Reports in {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
