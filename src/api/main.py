"""FastAPI app exposing FilingLens as an HTTP service.

    uvicorn src.api.main:app --reload
    curl -X POST localhost:8000/query -H 'content-type: application/json' \\
      -d '{"question": "How has Apple R&D spend as % of revenue changed over 3 years?", "ticker": "AAPL"}'
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.agents.graph import run_query

app = FastAPI(title="FilingLens", version="0.1.0")


class QueryRequest(BaseModel):
    question: str
    ticker: str | None = None
    fiscal_years: list[int] | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    verification_passed: bool
    overall_confidence: float
    retry_count: int


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    state = run_query(req.question, ticker=req.ticker, fiscal_years=req.fiscal_years)
    verification = state.get("verification", {})
    return QueryResponse(
        answer=state.get("final_answer", ""),
        citations=state.get("citations", []),
        verification_passed=verification.get("passed", False),
        overall_confidence=verification.get("overall_confidence", 0.0),
        retry_count=state.get("retry_count", 0),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
