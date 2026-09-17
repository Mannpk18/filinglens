"""Client for SEC EDGAR: resolves a ticker to a CIK, lists a company's recent
filings, and downloads filing documents.

SEC requires a descriptive User-Agent on every request (see
https://www.sec.gov/os/accessing-edgar-data) — set SEC_USER_AGENT in .env to
your own contact info or SEC will start rate-limiting / blocking you.

Note: this module makes live HTTP calls to data.sec.gov / www.sec.gov and will
not run inside network-restricted sandboxes. Run it from an environment with
normal internet access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

TICKER_TO_CIK_URL = "https://www.sec.gov/files/company_tickers.json"


@dataclass
class FilingRef:
    cik: str
    accession_number: str
    form: str
    filing_date: str
    primary_document: str
    company_name: str

    @property
    def document_url(self) -> str:
        acc_nodash = self.accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{acc_nodash}/{self.primary_document}"
        )


class EdgarClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.sec_user_agent})
        self._ticker_map: dict[str, str] | None = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _get(self, url: str) -> requests.Response:
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        # SEC asks for <=10 req/sec; be a polite citizen for a solo-engineer project
        time.sleep(0.15)
        return resp

    def ticker_to_cik(self, ticker: str) -> str:
        """Resolves a ticker like 'AAPL' to a zero-padded 10-digit CIK string."""
        if self._ticker_map is None:
            data = self._get(TICKER_TO_CIK_URL).json()
            self._ticker_map = {
                row["ticker"].upper(): str(row["cik_str"]).zfill(10)
                for row in data.values()
            }
        cik = self._ticker_map.get(ticker.upper())
        if cik is None:
            raise ValueError(f"Ticker '{ticker}' not found in SEC ticker map")
        return cik

    def get_filings(
        self, ticker: str, forms: list[str], limit: int = 12
    ) -> list[FilingRef]:
        """Returns recent filings of the given form types (e.g. ['10-K', '10-Q'])
        for a ticker, most recent first."""
        cik = self.ticker_to_cik(ticker)
        data = self._get(f"{settings.edgar_submissions_url}/CIK{cik}.json").json()
        recent = data["filings"]["recent"]
        company_name = data.get("name", ticker)

        refs: list[FilingRef] = []
        for i, form in enumerate(recent["form"]):
            if form not in forms:
                continue
            refs.append(
                FilingRef(
                    cik=cik,
                    accession_number=recent["accessionNumber"][i],
                    form=form,
                    filing_date=recent["filingDate"][i],
                    primary_document=recent["primaryDocument"][i],
                    company_name=company_name,
                )
            )
            if len(refs) >= limit:
                break
        return refs

    def fetch_document_text(self, filing: FilingRef) -> str:
        """Downloads a filing's primary document (HTML) and returns raw text.
        Kept dependency-light: strips tags with a simple parser rather than
        pulling in a heavy HTML stack, since filing structure is used later
        for section detection in chunking.py, not here."""
        from html.parser import HTMLParser

        class _TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.chunks: list[str] = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip and data.strip():
                    self.chunks.append(data.strip())

        html = self._get(filing.document_url).text
        parser = _TextExtractor()
        parser.feed(html)
        return "\n".join(parser.chunks)
