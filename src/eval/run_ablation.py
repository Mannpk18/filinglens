"""Runs the golden-dataset eval suite twice — once with the Verifier Agent
enabled, once with it disabled (bypassed straight to the writer) — and
produces a before/after comparison. This is the single most important
artifact for the portfolio writeup: it's the difference between "I built a
verifier agent" and "I measured a hallucination-rate reduction of X%."

    python -m src.eval.run_ablation --dataset data/golden_qa/golden_qa.json

Writes:
  eval_runs/ablation_<timestamp>.json   (raw results, both conditions)
  eval_runs/ablation_<timestamp>.md     (human-readable comparison table)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings
from src.eval.ragas_eval import run_eval_suite

RUNS_DIR = Path("eval_runs")


def _numeric_hallucination_rate(summary: dict) -> float:
    """Inverse of numeric_accuracy — the framing that lands in a resume bullet:
    'reduced numeric hallucination rate from X% to Y%.'"""
    return round(100 * (1 - summary["numeric_accuracy"]), 2)


def _run_condition(dataset_path: str, verifier_enabled: bool) -> dict:
    # settings is a module-level singleton read by verifier_agent at call time,
    # so flipping it here before invoking the graph is sufficient — no need to
    # rebuild the graph object itself.
    original = settings.verifier_enabled
    settings.verifier_enabled = verifier_enabled
    try:
        return run_eval_suite(dataset_path)
    finally:
        settings.verifier_enabled = original


def _format_report(with_verifier: dict, without_verifier: dict) -> str:
    w = with_verifier["summary"]
    wo = without_verifier["summary"]

    def delta(key: str, higher_is_better: bool = True) -> str:
        d = w[key] - wo[key]
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        good = (d > 0) == higher_is_better or d == 0
        marker = "better" if good else "worse"
        return f"{arrow} {abs(d):.3f} ({marker})"

    hall_with = _numeric_hallucination_rate(w)
    hall_without = _numeric_hallucination_rate(wo)
    hall_delta = round(hall_without - hall_with, 2)

    lines = [
        "# FilingLens Verifier Agent Ablation",
        "",
        f"Run at {datetime.now(timezone.utc).isoformat()} against "
        f"`{with_verifier['summary']['num_examples']}` golden examples.",
        "",
        "## Headline result",
        "",
        f"**Numeric hallucination rate: {hall_without}% (verifier off) → "
        f"{hall_with}% (verifier on)**, a reduction of {hall_delta} percentage points.",
        "",
        "## Full comparison",
        "",
        "| Metric | Verifier OFF | Verifier ON | Delta |",
        "|---|---|---|---|",
        f"| Pass rate | {wo['pass_rate']:.3f} | {w['pass_rate']:.3f} | {delta('pass_rate')} |",
        f"| Numeric accuracy | {wo['numeric_accuracy']:.3f} | {w['numeric_accuracy']:.3f} | {delta('numeric_accuracy')} |",
        f"| Numeric hallucination rate | {hall_without}% | {hall_with}% | {hall_delta} pp lower |",
        f"| Avg confidence | {wo['avg_confidence']:.3f} | {w['avg_confidence']:.3f} | {delta('avg_confidence')} |",
        f"| Avg latency (s) | {wo['avg_latency_seconds']:.2f} | {w['avg_latency_seconds']:.2f} | {delta('avg_latency_seconds', higher_is_better=False)} |",
        f"| Avg retries | {wo['avg_retries']:.2f} | {w['avg_retries']:.2f} | n/a (0 by definition when off) |",
        "",
        "## Reading this table",
        "",
        "- **Pass rate** and **numeric accuracy** should go up with the verifier on — "
        "that's the correctness win.",
        "- **Latency** should go up too — that's the cost of the retry loop. Report both "
        "numbers together; a verifier that's 100% accurate but 10x slower is a real "
        "tradeoff, not a free win, and saying so in an interview is a stronger signal "
        "than only reporting the accuracy number.",
        "- If pass rate does NOT improve with the verifier on, check `eval_runs/*.json` "
        "for per-example `unsupported_claim_count` — it usually means the reasoning "
        "agent's `evidence_id` tagging is inconsistent (see reasoning_agent.py prompt) "
        "rather than a verifier bug.",
        "",
        "## Resume framing",
        "",
        f'> "Designed a verifier-agent architecture that reduced numeric hallucination '
        f'rate from {hall_without}% to {hall_with}% ({hall_delta}pp) on a hand-curated '
        f'{w["num_examples"]}-question golden dataset, measured via an automated eval '
        f'harness gating CI."',
        "",
        "Do not publish this exact sentence with these numbers until you've run this "
        "script against your own ingested filings and golden dataset — these are "
        "computed from whatever `data/golden_qa/golden_qa.json` contains when you run it.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/golden_qa/golden_qa.json")
    args = parser.parse_args()

    print("Running eval WITH verifier enabled...")
    with_verifier = _run_condition(args.dataset, verifier_enabled=True)
    print(json.dumps(with_verifier["summary"], indent=2))

    print("\nRunning eval WITH verifier disabled...")
    without_verifier = _run_condition(args.dataset, verifier_enabled=False)
    print(json.dumps(without_verifier["summary"], indent=2))

    report_md = _format_report(with_verifier, without_verifier)
    print("\n" + report_md)

    RUNS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RUNS_DIR / f"ablation_{stamp}.json"
    md_path = RUNS_DIR / f"ablation_{stamp}.md"

    json_path.write_text(json.dumps(
        {"with_verifier": with_verifier, "without_verifier": without_verifier}, indent=2
    ))
    md_path.write_text(report_md)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}  <- put this in your portfolio README")


if __name__ == "__main__":
    main()
