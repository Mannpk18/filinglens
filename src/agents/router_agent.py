"""Router Agent: classifies the incoming question so the graph can decide
which downstream agents to invoke. A pure narrative question ("what does
management say about supply chain risk?") never needs to touch XBRL; a pure
numeric question ("what was R&D expense in FY2023?") often doesn't need
narrative retrieval at all. Skipping unnecessary agent hops is a real latency/
cost optimization worth calling out in a system design writeup.
"""

from __future__ import annotations

from src.agents.llm import call_llm
from src.agents.state import FilingLensState

ROUTER_SYSTEM_PROMPT = """You are a query router for a financial filings analysis \
system. Classify the user's question into exactly one category:

- numeric: purely about a financial figure or ratio (e.g. "what was revenue in 2023")
- narrative: purely about qualitative disclosures (e.g. "what risks did they mention")
- comparative: requires both numeric data AND narrative context, or numbers across \
multiple periods analyzed together (e.g. "did R&D spend growth match what they said \
about R&D priorities")
- multi_company: question spans more than one company

Respond with only the single category word, nothing else."""


def router_agent(state: FilingLensState) -> FilingLensState:
    category = call_llm(
        system=ROUTER_SYSTEM_PROMPT,
        user=state["question"],
        max_tokens=10,
    ).strip().lower()

    valid = {"numeric", "narrative", "comparative", "multi_company"}
    if category not in valid:
        category = "comparative"  # safest default: run both retrieval paths

    return {**state, "query_type": category}
