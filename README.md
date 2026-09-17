# FilingLens

A multi-agent system that answers questions about public companies' SEC filings, using LangGraph to orchestrate six agents that route, retrieve, reason, and — critically — **verify every claim against source data before it reaches the user**.

```
"How has Apple's R&D spend as a % of revenue changed over the last 3 fiscal years?"
```

FilingLens answers this by pulling XBRL figures for R&D expense and revenue directly from SEC's structured data (never asking an LLM to read a number out of prose), cross-checking every generated claim against that data or the retrieved filing text, and flagging anything it can't verify instead of presenting it as fact.

## Why this exists

Most RAG demos stop at "retrieve chunks, generate an answer." For financial data, that's not good enough — a plausible-sounding hallucinated number is worse than no answer. FilingLens is built around one core idea: **separate generation from verification**, and never let a numeric claim ship without being checked against ground truth.

## Architecture

Six agents wired into a LangGraph `StateGraph` with a real conditional retry loop, not a fixed pipeline:

```
                router_agent
                    │
    ┌───────────────┼────────────────────┐
    │ numeric        │ narrative          │ comparative / multi_company
    ▼                ▼                    ▼
table_agent    retrieval_agent    retrieval_agent + table_agent
    │                │                    │
    └────────────────┴────────────────────┘
                     ▼
              reasoning_agent
                     │
                     ▼
              verifier_agent
               │           │
   (failed, retry)      (passed, or retries exhausted)
               │           │
   retrieval_agent /       ▼
   table_agent        writer_agent
        │
        └──► reasoning_agent (loop)
```

| Agent | Role |
|---|---|
| **Router** | Classifies the question (`numeric` / `narrative` / `comparative` / `multi_company`) so the graph only invokes the agents it actually needs |
| **Retrieval** | Hybrid search (dense + BM25, fused with Reciprocal Rank Fusion) over chunked 10-K/10-Q text for qualitative context |
| **Table** | Pulls structured financial figures straight from SEC's XBRL `companyfacts` API — the LLM only maps natural language ("R&D spend") to a GAAP concept key, it never reads numbers out of text |
| **Reasoning** | Drafts an answer and, critically, decomposes it into a list of discrete, independently-traceable claims (numeric or narrative) instead of one prose blob |
| **Verifier** | Checks every claim: numeric claims deterministically against `numeric_facts` (no LLM call, tolerance-based match), narrative claims via an LLM entailment check against the specific chunk cited. Routes back to retrieval/table with a `retry_reason` if confidence is too low, up to `MAX_VERIFY_RETRIES` |
| **Writer** | Formats the final answer with inline citations (XBRL concept or filing chunk), and explicitly flags any claim that failed verification rather than hiding it |

## Key design decisions

- **Postgres + pgvector over a managed vector DB.** Free to self-host, and keeps metadata filtering (ticker, fiscal year, form, section) in the same SQL query as the vector search instead of a second round trip.
- **Hybrid search, not dense-only.** Dense retrieval alone misses exact-term matches that matter in filings (specific dollar figures, defined terms like "Class A Common Stock"); BM25 catches those. Fused with RRF rather than a learned re-ranker, to keep the system simple and explainable.
- **Section-aware chunking.** Splits on standard 10-K/10-Q "Item N." headings before sub-chunking by token count, so a chunk never loses its heading context.
- **Numeric ground truth from XBRL, always.** Any dollar figure or percentage in an answer is computed from SEC's tagged structured data, not extracted by an LLM reading prose.
- **Claims, not answers, get verified.** Verifying one discrete assertion at a time is a much stronger check than asking an LLM to grade its own entire response.

## Project structure

```
src/
├── agents/          router, retrieval, table, reasoning, verifier, writer agents + graph.py (LangGraph wiring)
├── api/              FastAPI app (main.py)
├── data/              EDGAR client, XBRL client, filing chunking
├── retrieval/       embeddings, hybrid search, pgvector store
├── eval/              golden dataset loader, Ragas eval suite, verifier on/off ablation runner
├── observability/  Langfuse tracing
└── config.py         env-driven settings (pydantic-settings)
scripts/
├── run_ingest.py    fetch -> chunk -> embed -> upsert a ticker's filings
└── setup_db.sql     pgvector schema
data/golden_qa/     hand-curated eval questions
web/index.html      static frontend
tests/                  pytest unit tests
.github/workflows/  CI eval gate on PRs touching agents/retrieval/data
```

## Setup

**Requirements:** Python 3.11+, a Postgres instance with the `pgvector` extension available, an Anthropic API key.

```bash
git clone https://github.com/Mannpk18/filinglens.git
cd filinglens
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in ANTHROPIC_API_KEY, DATABASE_URL, and SEC_USER_AGENT (SEC requires a
# real contact email in the User-Agent header -- see .env.example)

psql "$DATABASE_URL" -f scripts/setup_db.sql
```

## Ingest a company's filings

```bash
python scripts/run_ingest.py --ticker AAPL --forms 10-K,10-Q --years 3
```

This fetches recent filings from EDGAR, chunks and embeds the text into pgvector, and warms nothing else — XBRL numeric facts are fetched on demand by the table agent, so there's no separate ingestion step for numbers.

## Run

**As an API:**

```bash
uvicorn src.api.main:app --reload
```

```bash
curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "How has Apple R&D spend as % of revenue changed over 3 years?", "ticker": "AAPL"}'
```

**Programmatically:**

```python
from src.agents.graph import run_query

result = run_query("What supply chain risks does Apple's most recent 10-K disclose?", ticker="AAPL")
print(result["final_answer"])
```

## Evaluation

The eval suite runs Ragas metrics (faithfulness, answer relevancy, context precision, context recall) plus a FilingLens-specific `numeric_accuracy` metric (fraction of numeric claims matching XBRL ground truth exactly) against a hand-curated golden dataset in `data/golden_qa/golden_qa.json`.

```bash
python -m src.eval.run_eval --dataset data/golden_qa/golden_qa.json
```

**Verifier ablation** — the artifact that turns "I built a verifier agent" into a measured claim — runs the suite once with the verifier enabled and once with it bypassed, and writes a before/after comparison:

```bash
python -m src.eval.run_ablation --dataset data/golden_qa/golden_qa.json
```

> **Note:** `data/golden_qa/golden_qa.json` currently ships with placeholder `"REPLACE ME"` ground-truth answers. Fill these in against the actual filings before the eval numbers mean anything — the dataset is deliberately hand-curated rather than LLM-generated so the eval results aren't the same model grading its own homework.

A CI workflow (`.github/workflows/eval.yml`) runs this suite on every PR touching agent, retrieval, or data code, and fails the build if `pass_rate` or `numeric_accuracy` regresses.

## Testing

```bash
pytest tests/ -v
```

## Roadmap / known limitations

- `data/golden_qa/golden_qa.json` ground truths need to be filled in from real filings before eval numbers are meaningful.
- BM25/dense fusion uses RRF rather than a learned cross-encoder re-ranker — documented as a future upgrade path once retrieval quality needs to scale past a portfolio-sized corpus.
- `edgar_client.py` and `run_ingest.py` make live calls to `data.sec.gov` / `www.sec.gov` and need normal outbound internet access — they won't run inside network-restricted sandboxes.
