"""Table Agent: pulls structured numeric facts from SEC XBRL data. This is
the agent that makes FilingLens trustworthy on numbers — it never asks the LLM
to read a number out of retrieved text; it fetches the tagged, structured
value directly from SEC's companyfacts API.

The LLM is used here only for one narrow purpose: mapping the user's natural-
language ask ("R&D spend", "top-line revenue") onto the small set of GAAP
concept keys in xbrl_client.GAAP_CONCEPTS, since users won't say "us-gaap:
ResearchAndDevelopmentExpense" verbatim.
"""

from __future__ import annotations

import json

from src.agents.llm import call_llm
from src.agents.state import FilingLensState
from src.data.xbrl_client import GAAP_CONCEPTS, XbrlClient

CONCEPT_MAPPING_SYSTEM_PROMPT = f"""Map the financial concepts mentioned in the \
user's question to keys from this list: {list(GAAP_CONCEPTS.keys())}.

Respond with ONLY a JSON array of matching keys, e.g. ["revenue", "rd_expense"]. \
If none match, respond with []. Do not include any other text."""


def _extract_concept_keys(question: str) -> list[str]:
    raw = call_llm(system=CONCEPT_MAPPING_SYSTEM_PROMPT, user=question, max_tokens=100)
    try:
        keys = json.loads(raw.strip())
        return [k for k in keys if k in GAAP_CONCEPTS]
    except (json.JSONDecodeError, TypeError):
        return []


def table_agent(state: FilingLensState, xbrl: XbrlClient | None = None) -> FilingLensState:
    xbrl = xbrl or XbrlClient()
    ticker = state.get("ticker")
    if not ticker:
        return {**state, "numeric_facts": []}

    concept_keys = _extract_concept_keys(state["question"])
    if not concept_keys:
        return {**state, "numeric_facts": []}

    facts = []
    for key in concept_keys:
        try:
            series = xbrl.get_concept_series(ticker, key, state.get("fiscal_years"))
        except ValueError:
            continue
        for point in series:
            facts.append(
                {
                    "concept": key,
                    "fiscal_year": point.fiscal_year,
                    "value": point.value,
                    "unit": point.unit,
                    "source": "xbrl",
                }
            )
    return {**state, "numeric_facts": facts}
