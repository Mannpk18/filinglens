"""CLI: python -m src.eval.run_eval --dataset data/golden_qa/golden_qa.json

Writes results to eval_runs/<timestamp>.json so results are comparable across
runs (this is what your CI regression gate in .github/workflows/eval.yml reads
to fail a PR that drops pass_rate or numeric_accuracy below the prior run).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.eval.ragas_eval import run_eval_suite

RUNS_DIR = Path("eval_runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/golden_qa/golden_qa.json")
    parser.add_argument("--min-pass-rate", type=float, default=None,
                         help="Exit nonzero if pass_rate falls below this (for CI)")
    parser.add_argument("--min-numeric-accuracy", type=float, default=None)
    args = parser.parse_args()

    output = run_eval_suite(args.dataset)
    summary = output["summary"]

    print(json.dumps(summary, indent=2))

    RUNS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RUNS_DIR / f"{stamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote full results to {out_path}")

    failed = False
    if args.min_pass_rate is not None and summary["pass_rate"] < args.min_pass_rate:
        print(f"FAIL: pass_rate {summary['pass_rate']:.3f} < threshold {args.min_pass_rate}")
        failed = True
    if args.min_numeric_accuracy is not None and summary["numeric_accuracy"] < args.min_numeric_accuracy:
        print(f"FAIL: numeric_accuracy {summary['numeric_accuracy']:.3f} < threshold {args.min_numeric_accuracy}")
        failed = True

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
