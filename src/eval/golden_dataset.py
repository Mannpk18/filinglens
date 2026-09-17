"""Loads the hand-built golden QA dataset used for evaluation.

Each entry should be a question you can answer yourself by reading the actual
filing, with the ground-truth answer and the specific evidence (XBRL concept
or filing section) that supports it. This is deliberately hand-curated, not
LLM-generated — an LLM-generated eval set risks the same model grading its own
homework, which undermines the credibility of the numbers you'd put on a resume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldenExample:
    id: str
    question: str
    ticker: str
    fiscal_years: list[int]
    ground_truth_answer: str
    ground_truth_contexts: list[str]  # expected supporting excerpts/facts
    category: str  # "numeric" | "narrative" | "comparative"


def load_golden_dataset(path: str | Path) -> list[GoldenExample]:
    data = json.loads(Path(path).read_text())
    return [
        GoldenExample(
            id=row["id"],
            question=row["question"],
            ticker=row["ticker"],
            fiscal_years=row.get("fiscal_years", []),
            ground_truth_answer=row["ground_truth_answer"],
            ground_truth_contexts=row.get("ground_truth_contexts", []),
            category=row.get("category", "comparative"),
        )
        for row in data
    ]
