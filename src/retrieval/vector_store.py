"""pgvector-backed store for filing chunks.

Chose Postgres + pgvector over a managed vector DB (Pinecone/Qdrant) deliberately:
it's free to self-host, keeps metadata filtering (ticker, fiscal_year, form,
section) in the same SQL query as the vector search instead of a second round
trip, and is a more common production pattern to defend in an interview than
"I used a managed service." The README documents the Pinecone/Qdrant swap path
for when this needs to scale past a few thousand filings.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from src.config import settings


@dataclass
class SearchResult:
    chunk_id: int
    text: str
    ticker: str
    fiscal_year: int
    form: str
    section: str
    score: float


class VectorStore:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or settings.database_url

    def _connect(self):
        conn = psycopg.connect(self.dsn, autocommit=True)
        register_vector(conn)
        return conn

    def upsert_chunks(
        self,
        ticker: str,
        fiscal_year: int,
        form: str,
        accession_number: str,
        chunks: list,  # list[Chunk] from src.data.chunking
        embeddings: list[list[float]],
    ) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                for chunk, emb in zip(chunks, embeddings):
                    cur.execute(
                        """
                        INSERT INTO filing_chunks
                            (ticker, fiscal_year, form, accession_number,
                             section, chunk_index, text, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (accession_number, chunk_index) DO UPDATE
                            SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                        """,
                        (
                            ticker, fiscal_year, form, accession_number,
                            chunk.section, chunk.chunk_index, chunk.text, emb,
                        ),
                    )
                return len(chunks)

    def dense_search(
        self,
        query_embedding: list[float],
        ticker: str | None = None,
        fiscal_years: list[int] | None = None,
        top_k: int = 8,
    ) -> list[SearchResult]:
        filters = []
        params: list = [query_embedding]
        if ticker:
            filters.append("ticker = %s")
            params.append(ticker)
        if fiscal_years:
            filters.append("fiscal_year = ANY(%s)")
            params.append(fiscal_years)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        sql = f"""
            SELECT id, text, ticker, fiscal_year, form, section,
                   1 - (embedding <=> %s) AS score
            FROM filing_chunks
            {where_clause}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        params = [query_embedding] + params[1:] + [query_embedding, top_k]

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            SearchResult(
                chunk_id=r[0], text=r[1], ticker=r[2], fiscal_year=r[3],
                form=r[4], section=r[5], score=float(r[6]),
            )
            for r in rows
        ]

    def all_texts_for_bm25(
        self, ticker: str | None = None, fiscal_years: list[int] | None = None
    ) -> list[tuple[int, str]]:
        """Pulls (id, text) pairs for building/refreshing the in-memory BM25
        index used by hybrid_search.py. For a portfolio-scale corpus (a few
        thousand chunks per company) this is fine in memory; at real scale
        you'd move to a persistent inverted index (e.g. OpenSearch)."""
        filters = []
        params: list = []
        if ticker:
            filters.append("ticker = %s")
            params.append(ticker)
        if fiscal_years:
            filters.append("fiscal_year = ANY(%s)")
            params.append(fiscal_years)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT id, text FROM filing_chunks {where_clause}", params)
                return cur.fetchall()
