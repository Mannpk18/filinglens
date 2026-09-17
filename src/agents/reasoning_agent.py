"""Reasoning Agent: drafts an answer and, critically, emits a structured list
of discrete claims (numeric and narrative) rather than just prose. This claim
list is what the Verifier Agent checks — verifying "the answer" as one blob is
much weaker than verifying each factual assertion independently.
"""

from __future__ import annotations

import json

from src.agents.llm import call_llm
from src.agents.state import FilingLensState

REASONING_SYSTEM_PROMPT = """You are a financial analyst assistant. Using ONLY \
the numeric facts and retrieved filing excerpts provided, draft an answer to \
the user's question.

Then break your answer down into a list of discrete claims. Each claim must be \
either:
  - "numeric": a specific number/figure/ratio you stated (must come directly \
    from the provided numeric facts, no calculations beyond simple arithmetic \
    on those facts)
  - "narrative": a qualitative statement about what the filing says (must be \
    traceable to a specific retrieved excerpt)

Do not include any claim you cannot trace to the provided evidence. If the \
evidence is insufficient to answer part of the question, say so explicitly \
rather than guessing.

Respond with ONLY valid JSON in this exact shape:
{
  "draft_answer": "...",
  "claims": [
    {"text": "...", "claim_type": "numeric", "evidence_id": "<concept:fiscal_year or chunk_id>"},
    {"text": "...", "claim_type": "narrative", "evidence_id": "<chunk_id>"}
  ]
}"""


def _format_evidence(state: FilingLensState) -> str:
    parts = []
    facts = state.get("numeric_facts") or []
    if facts:
        parts.append("NUMERIC FACTS (from SEC XBRL, ground truth):")
        for f in facts:
            parts.append(
                f"- {f['concept']} FY{f['fiscal_year']}: {f['value']:,.0f} {f['unit']} [id: {f['concept']}:{f['fiscal_year']}]"
            )
    chunks = state.get("retrieved_chunks") or []
    if chunks:
        parts.append("\nRETRIEVED FILING EXCERPTS:")
        for c in chunks:
            parts.append(f"- [chunk_id: {c['chunk_id']}, section: {c['section']}] {c['text'][:600]}")
    return "\n".join(parts) if parts else "No evidence retrieved."


def reasoning_agent(state: FilingLensState) -> FilingLensState:
    user_prompt = (
        f"Question: {state['question']}\n\nEvidence:\n{_format_evidence(state)}"
    )
    raw = call_llm(system=REASONING_SYSTEM_PROMPT, user=user_prompt, max_tokens=1500)

    try:
        parsed = json.loads(raw.strip().strip("`").removeprefix("json").strip())
        draft_answer = parsed["draft_answer"]
        claims = [
            {
                "text": c["text"],
                "claim_type": c["claim_type"],
                "evidence_id": c.get("evidence_id"),
                "supported": False,  # verifier fills this in
                "confidence": 0.0,
            }
            for c in parsed.get("claims", [])
        ]
    except (json.JSONDecodeError, KeyError):
        draft_answer = raw
        claims = []

    return {"draft_answer": draft_answer, "draft_claims": claims}
