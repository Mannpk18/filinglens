"""Writer Agent: formats the final answer, appending inline citations back to
either an XBRL concept/fiscal-year or a retrieved chunk's filing section, and
flags any claim that failed verification instead of silently presenting it as
fact — this is what keeps the system honest when retries are exhausted.
"""

from __future__ import annotations

from src.agents.state import FilingLensState


def writer_agent(state: FilingLensState) -> FilingLensState:
    verification = state.get("verification", {})
    claims = verification.get("all_claims", [])
    answer = state.get("draft_answer", "")

    citations = []
    flagged_lines = []
    for c in claims:
        if c["claim_type"] == "numeric":
            label = f"XBRL:{c.get('evidence_id')}"
        else:
            label = f"chunk:{c.get('evidence_id')}"
        citations.append(label)
        if not c["supported"]:
            flagged_lines.append(f"⚠ Unverified claim: \"{c['text']}\" (source: {label})")

    final_answer = answer
    if flagged_lines:
        final_answer += "\n\n---\n" + "\n".join(flagged_lines)
        final_answer += (
            "\n\nNote: the claims above could not be automatically verified "
            "against source data and should be independently checked."
        )

    return {"final_answer": final_answer, "citations": citations}
