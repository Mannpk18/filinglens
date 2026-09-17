"""Shared state passed between every node in the LangGraph. Keeping this as a
single typed dict (rather than each agent inventing its own return shape) is
what makes the verify -> retry conditional edge possible: the verifier only
needs to inspect `verification` and can send the graph back to `retrieval_agent`
or `table_agent` with `retry_count` incremented and `retry_reason` set.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class NumericFact(TypedDict):
    concept: str
    fiscal_year: int
    value: float
    unit: str
    source: str  # "xbrl"


class RetrievedChunk(TypedDict):
    chunk_id: int
    text: str
    section: str
    fiscal_year: int
    score: float


class Claim(TypedDict):
    text: str
    claim_type: Literal["numeric", "narrative"]
    supported: bool
    evidence_id: str | None  # chunk_id or xbrl concept key
    confidence: float


class VerificationResult(TypedDict):
    all_claims: list[Claim]
    overall_confidence: float
    passed: bool
    failure_reason: str | None


class FilingLensState(TypedDict, total=False):
    # input
    question: str
    ticker: str | None
    fiscal_years: list[int] | None

    # router output
    query_type: Literal["numeric", "narrative", "comparative", "multi_company"]

    # retrieval outputs
    retrieved_chunks: list[RetrievedChunk]
    numeric_facts: list[NumericFact]

    # reasoning output
    draft_answer: str
    draft_claims: list[Claim]

    # verification
    verification: VerificationResult
    retry_count: int
    retry_reason: str | None

    # final output
    final_answer: str
    citations: list[str]
