"""Retrieval Agent: pulls narrative context via hybrid search. Runs for
'narrative' and 'comparative' query types. On a verifier-triggered retry, it
re-queries with a reformulated question that incorporates the verifier's
`retry_reason`, rather than blindly repeating the same search."""

from __future__ import annotations

from src.agents.llm import call_llm
from src.agents.state import FilingLensState
from src.config import settings
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.vector_store import VectorStore

REFORMULATE_SYSTEM_PROMPT = """The retrieval step below did not find enough \
evidence to support an answer. Rewrite the search query to be more likely to \
find the missing evidence. Respond with only the rewritten query."""


def retrieval_agent(state: FilingLensState, store: VectorStore | None = None) -> FilingLensState:
    store = store or VectorStore()
    query = state["question"]

    if state.get("retry_reason"):
        query = call_llm(
            system=REFORMULATE_SYSTEM_PROMPT,
            user=f"Original question: {state['question']}\nMissing evidence: {state['retry_reason']}",
            max_tokens=200,
        ).strip()

    results = hybrid_search(
        query=query,
        store=store,
        ticker=state.get("ticker"),
        fiscal_years=state.get("fiscal_years"),
        top_k=settings.retrieval_top_k,
    )

    retrieved_chunks = [
        {
            "chunk_id": r.chunk_id,
            "text": r.text,
            "section": r.section,
            "fiscal_year": r.fiscal_year,
            "score": r.fused_score,
        }
        for r in results
    ]
    return {"retrieved_chunks": retrieved_chunks}
