"""Ingests a company's recent 10-K/10-Q filings: fetch from EDGAR, chunk,
embed, and upsert into pgvector, plus pre-fetches XBRL facts to warm the
table_agent's data.

    python scripts/run_ingest.py --ticker AAPL --forms 10-K,10-Q --years 3

Requires network access to data.sec.gov / www.sec.gov (SEC EDGAR) — run this
locally or from a deployment with outbound internet access, not inside a
network-restricted sandbox.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

sys.path.insert(0, ".")

from src.config import settings
from src.data.chunking import chunk_filing
from src.data.edgar_client import EdgarClient
from src.retrieval.embeddings import embed_texts
from src.retrieval.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--forms", default="10-K,10-Q")
    parser.add_argument("--years", type=int, default=3)
    args = parser.parse_args()

    forms = args.forms.split(",")
    cutoff_year = datetime.now().year - args.years

    edgar = EdgarClient()
    store = VectorStore()

    filings = edgar.get_filings(args.ticker, forms=forms, limit=args.years * len(forms) + 2)
    filings = [f for f in filings if int(f.filing_date[:4]) >= cutoff_year]

    print(f"Found {len(filings)} filings for {args.ticker} since {cutoff_year}")

    for filing in filings:
        print(f"  Fetching {filing.form} filed {filing.filing_date} ({filing.accession_number})")
        text = edgar.fetch_document_text(filing)
        chunks = chunk_filing(
            text,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        print(f"    -> {len(chunks)} chunks, embedding...")
        embeddings = embed_texts([c.text for c in chunks])

        fiscal_year = int(filing.filing_date[:4])
        n = store.upsert_chunks(
            ticker=args.ticker,
            fiscal_year=fiscal_year,
            form=filing.form,
            accession_number=filing.accession_number,
            chunks=chunks,
            embeddings=embeddings,
        )
        print(f"    -> upserted {n} chunks")

    print("Done. XBRL facts are fetched on-demand by the table agent, no separate step needed.")


if __name__ == "__main__":
    main()
