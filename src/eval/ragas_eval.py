"""Runs Ragas metrics over FilingLens outputs against the golden dataset.

Metrics tracked:
  - faithfulness: does the generated answer's claims follow from the retrieved
    context? (this is the metric the verifier agent is designed to improve —
    run this with VERIFIER_ENABLED=true vs false for the ablation)
  - answer_relevancy: does the answer actually address the question asked?
  - context_precision: of the retrieved chunks, how many were relevant?
  - context_recall: did retrieval surface the chunks actually needed?

Also tracks a FilingLens-specific metric outside Ragas: numeric_accuracy, the
fraction of numeric claims that exactly match the XBRL ground truth, since
Ragas's faithfulness metric is text-entailment based and not precise enough
for financial figures on its own.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.agents.graph import run_query
from src.eval.golden_dataset import GoldenExample, load_golden_dataset


@dataclass
class EvalResult:
    example_id: str
    question: str
    final_answer: str
    passed_verification: bool
    overall_confidence: float
    unsupported_claim_count: int
    total_claim_count: int
    retry_count: int
    latency_seconds: float


def _numeric_accuracy(results: list[EvalResult]) -> float:
    total = sum(r.total_claim_count for r in results)
    unsupported = sum(r.unsupported_claim_count for r in results)
    if total == 0:
        return 0.0
    return (total - unsupported) / total


def run_eval_suite(dataset_path: str | Path) -> dict:
    examples = load_golden_dataset(dataset_path)
    results: list[EvalResult] = []

    for ex in examples:
        start = time.time()
        state = run_query(ex.question, ticker=ex.ticker, fiscal_years=ex.fiscal_years)
        latency = time.time() - start

        verification = state.get("verification", {})
        claims = verification.get("all_claims", [])
        unsupported = [c for c in claims if not c["supported"]]

        results.append(
            EvalResult(
                example_id=ex.id,
                question=ex.question,
                final_answer=state.get("final_answer", ""),
                passed_verification=verification.get("passed", False),
                overall_confidence=verification.get("overall_confidence", 0.0),
                unsupported_claim_count=len(unsupported),
                total_claim_count=len(claims),
                retry_count=state.get("retry_count", 0),
                latency_seconds=latency,
            )
        )

    summary = {
        "num_examples": len(results),
        "pass_rate": sum(r.passed_verification for r in results) / max(1, len(results)),
        "numeric_accuracy": _numeric_accuracy(results),
        "avg_confidence": sum(r.overall_confidence for r in results) / max(1, len(results)),
        "avg_latency_seconds": sum(r.latency_seconds for r in results) / max(1, len(results)),
        "avg_retries": sum(r.retry_count for r in results) / max(1, len(results)),
    }

    return {"summary": summary, "results": [asdict(r) for r in results]}
