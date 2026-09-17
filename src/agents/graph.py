"""Wires the six agents into a LangGraph StateGraph.

Graph shape:

    router_agent
        │
        ├── numeric ──────────► table_agent ──────────────┐
        ├── narrative ────────► retrieval_agent ───────────┤
        └── comparative/multi ► retrieval_agent+table_agent┤
                                                            ▼
                                                    reasoning_agent
                                                            │
                                                            ▼
                                                    verifier_agent
                                                       │       │
                                        (failed, retry)│       │(passed, or retries exhausted)
                                                        ▼       ▼
                                   retrieval_agent/table_agent  writer_agent
                                              │
                                              └──► reasoning_agent (loop)

This is a real conditional loop, not a fixed pipeline — that's the part worth
walking an interviewer through.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.reasoning_agent import reasoning_agent
from src.agents.retrieval_agent import retrieval_agent
from src.agents.router_agent import router_agent
from src.agents.state import FilingLensState
from src.agents.table_agent import table_agent
from src.agents.verifier_agent import route_after_verification, verifier_agent
from src.agents.writer_agent import writer_agent


def _route_after_router(state: FilingLensState) -> list[str]:
    query_type = state.get("query_type", "comparative")
    if query_type == "numeric":
        return ["table_agent"]
    if query_type == "narrative":
        return ["retrieval_agent"]
    # comparative / multi_company: gather both structured and narrative evidence
    return ["retrieval_agent", "table_agent"]


def build_graph():
    graph = StateGraph(FilingLensState)

    graph.add_node("router_agent", router_agent)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("table_agent", table_agent)
    graph.add_node("reasoning_agent", reasoning_agent)
    graph.add_node("verifier_agent", verifier_agent)
    graph.add_node("writer_agent", writer_agent)

    graph.set_entry_point("router_agent")

    graph.add_conditional_edges(
        "router_agent",
        _route_after_router,
        {"retrieval_agent": "retrieval_agent", "table_agent": "table_agent"},
    )

    # both retrieval and table agent feed into reasoning; LangGraph fans-in
    # automatically when a query type triggers both
    graph.add_edge("retrieval_agent", "reasoning_agent")
    graph.add_edge("table_agent", "reasoning_agent")

    graph.add_edge("reasoning_agent", "verifier_agent")

    graph.add_conditional_edges(
        "verifier_agent",
        route_after_verification,
        {
            "retrieval_agent": "retrieval_agent",
            "table_agent": "table_agent",
            "writer_agent": "writer_agent",
        },
    )

    graph.add_edge("writer_agent", END)

    return graph.compile()


def run_query(question: str, ticker: str | None = None, fiscal_years: list[int] | None = None) -> FilingLensState:
    app = build_graph()
    initial_state: FilingLensState = {
        "question": question,
        "ticker": ticker,
        "fiscal_years": fiscal_years,
        "retry_count": 0,
    }
    return app.invoke(initial_state)
