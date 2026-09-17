"""Verifier Agent — the core differentiator of this system.

For each claim the Reasoning Agent produced:
  - numeric claims are checked deterministically (no LLM call): the claimed
    number must match a value actually present in state["numeric_facts"]
    within a small tolerance, catching both hallucinated figures and
    arithmetic slips.
  - narrative claims are checked with an LLM entailment call against the
    specific retrieved chunk the claim cited: does the chunk actually support
    this statement, or did the reasoning agent overreach?

If overall confidence is below threshold, the graph routes back to retrieval
or table agent with a `retry_reason` describing what's missing, up to
settings.max_verify_retries times, then returns the best available answer
with unsupported claims flagged rather than silently dropped.
"""

from __future__ import annotations

import re

from src.agents.llm import call_llm
from src.agents.state import Claim, FilingLensState
from src.config import settings

NARRATIVE_ENTAILMENT_SYSTEM_PROMPT = """You are a fact-checker. Given a source \
excerpt and a claim, decide if the excerpt supports the claim.

Respond with only one word: SUPPORTED, CONTRADICTED, or UNSUPPORTED (meaning \
the excerpt is unrelated or insufficient to judge)."""

NUMERIC_TOLERANCE_PCT = 0.5  # allow half a percent rounding slack


def _verify_numeric_claim(claim: Claim, numeric_facts: list[dict]) -> tuple[bool, float]:
    facts_by_id = {f"{f['concept']}:{f['fiscal_year']}": f for f in numeric_facts}
    fact = facts_by_id.get(claim.get("evidence_id"))
    if fact is None:
        return False, 0.0

    numbers_in_claim = re.findall(r"[-+]?\d[\d,]*\.?\d*", claim["text"])
    if not numbers_in_claim:
        return False, 0.2

    fact_value = fact["value"]
    for raw_num in numbers_in_claim:
        try:
            claimed = float(raw_num.replace(",", ""))
        except ValueError:
            continue
        # claim might state the value in millions/billions or as a raw figure —
        # check both the raw value and common scaled representations
        for scale in (1, 1e3, 1e6, 1e9):
            if fact_value == 0:
                continue
            scaled = fact_value / scale
            if abs(claimed - scaled) / abs(scaled) < (NUMERIC_TOLERANCE_PCT / 100):
                return True, 1.0
    return False, 0.3


def _verify_narrative_claim(claim: Claim, retrieved_chunks: list[dict]) -> tuple[bool, float]:
    chunks_by_id = {str(c["chunk_id"]): c for c in retrieved_chunks}
    chunk = chunks_by_id.get(str(claim.get("evidence_id")))
    if chunk is None:
        return False, 0.0

    verdict = call_llm(
        system=NARRATIVE_ENTAILMENT_SYSTEM_PROMPT,
        user=f"Excerpt:\n{chunk['text']}\n\nClaim:\n{claim['text']}",
        max_tokens=10,
    ).strip().upper()

    if "SUPPORTED" in verdict and "UN" not in verdict and "CONTRADICTED" not in verdict:
        return True, 0.9
    return False, 0.1


def verifier_agent(state: FilingLensState) -> FilingLensState:
    claims = state.get("draft_claims") or []
    numeric_facts = state.get("numeric_facts") or []
    retrieved_chunks = state.get("retrieved_chunks") or []

    verified_claims: list[Claim] = []
    for claim in claims:
        if claim["claim_type"] == "numeric":
            supported, confidence = _verify_numeric_claim(claim, numeric_facts)
        else:
            supported, confidence = _verify_narrative_claim(claim, retrieved_chunks)
        verified_claims.append({**claim, "supported": supported, "confidence": confidence})

    if not verified_claims:
        overall_confidence = 0.5  # no checkable claims extracted; neutral, not a pass
        passed = False
        failure_reason = "No verifiable claims were extracted from the draft answer."
    else:
        overall_confidence = sum(c["confidence"] for c in verified_claims) / len(verified_claims)
        unsupported = [c for c in verified_claims if not c["supported"]]
        passed = len(unsupported) == 0
        failure_reason = (
            None if passed
            else "Unsupported claims: " + "; ".join(c["text"][:100] for c in unsupported[:3])
        )

    verification = {
        "all_claims": verified_claims,
        "overall_confidence": overall_confidence,
        "passed": passed,
        "failure_reason": failure_reason,
    }

    retry_count = state.get("retry_count", 0)
    should_retry = (
        not passed
        and settings.verifier_enabled
        and retry_count < settings.max_verify_retries
    )

    return {
        **state,
        "verification": verification,
        "retry_count": retry_count + 1 if should_retry else retry_count,
        "retry_reason": failure_reason if should_retry else None,
    }


def route_after_verification(state: FilingLensState) -> str:
    """Conditional edge function for the LangGraph: decides whether to loop
    back for more evidence or proceed to the writer."""
    verification = state.get("verification", {})
    if verification.get("passed"):
        return "writer_agent"
    if state.get("retry_reason"):
        # route back to whichever agent is more likely to fill the gap
        unsupported_types = {
            c["claim_type"] for c in verification.get("all_claims", []) if not c["supported"]
        }
        if "numeric" in unsupported_types and "narrative" not in unsupported_types:
            return "table_agent"
        return "retrieval_agent"
    return "writer_agent"  # retries exhausted, ship best-effort answer with flags
