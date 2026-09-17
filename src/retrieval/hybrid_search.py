"""Hybrid retrieval: fuses dense (pgvector cosine) and sparse (BM25) rankings
with Reciprocal Rank Fusion. Dense retrieval alone misses exact-term matches
that matter a lot in financial filings (specific dollar figures, defined terms
like "Class A Common Stock", ticker-specific product names) — BM25 catches
those. RRF is used instead of a learned re-ranker to keep the system simple
and explainable, with a documented upgrade path to a cross-encoder re-ranker.
"""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.retrieval.embeddings import embed_query
from src.retrieval.vector_store import SearchResult, VectorStore


@dataclass
class FusedResult:
    chunk_id: int
    text: str
    ticker: str
    fiscal_year: int
    form: str
    section: str
    fused_score: float


def _bm25_rank(
    query: str, corpus: list[tuple[int, str]], top_k: int
) -> list[tuple[int, int]]:
    """Returns [(chunk_id, rank)] for the top_k BM25 matches, rank 1-indexed."""
    if not corpus:
        return []
    ids = [c[0] for c in corpus]
    tokenized_corpus = [c[1].lower().split() for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunk_id, rank + 1) for rank, (chunk_id, _score) in enumerate(ranked)]


def hybrid_search(
    query: str,
    store: VectorStore,
    ticker: str | None = None,
    fiscal_years: list[int] | None = None,
    top_k: int = 8,
    rrf_k: int = 60,
) -> list[FusedResult]:
    query_emb = embed_query(query)
    dense_results: list[SearchResult] = store.dense_search(
        query_emb, ticker=ticker, fiscal_years=fiscal_years, top_k=top_k * 3
    )
    dense_by_id = {r.chunk_id: r for r in dense_results}
    dense_ranks = {r.chunk_id: i + 1 for i, r in enumerate(dense_results)}

    corpus = store.all_texts_for_bm25(ticker=ticker, fiscal_years=fiscal_years)
    bm25_ranks = dict(_bm25_rank(query, corpus, top_k=top_k * 3))

    # Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across retrieval methods
    all_ids = set(dense_ranks) | set(bm25_ranks)
    fused_scores: dict[int, float] = {}
    for cid in all_ids:
        score = 0.0
        if cid in dense_ranks:
            score += 1 / (rrf_k + dense_ranks[cid])
        if cid in bm25_ranks:
            score += 1 / (rrf_k + bm25_ranks[cid])
        fused_scores[cid] = score

    top_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

    corpus_text_by_id = dict(corpus)
    results: list[FusedResult] = []
    for cid in top_ids:
        if cid in dense_by_id:
            r = dense_by_id[cid]
            results.append(
                FusedResult(
                    chunk_id=cid, text=r.text, ticker=r.ticker,
                    fiscal_year=r.fiscal_year, form=r.form, section=r.section,
                    fused_score=fused_scores[cid],
                )
            )
        else:
            results.append(
                FusedResult(
                    chunk_id=cid, text=corpus_text_by_id.get(cid, ""),
                    ticker=ticker or "", fiscal_year=0, form="", section="",
                    fused_score=fused_scores[cid],
                )
            )
    return results
