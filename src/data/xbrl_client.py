"""Client for SEC's XBRL 'companyfacts' API — structured, machine-readable
financial statement data (revenue, R&D expense, net income, etc.) tagged by
GAAP concept and fiscal period.

This is the ground-truth numeric source for FilingLens. Any answer involving a
dollar figure or a percentage is computed from this data, not extracted by an
LLM reading prose, which is what keeps numeric answers verifiable.
"""

from __future__ import annotations

from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from src.data.edgar_client import EdgarClient

# Common US-GAAP concept tags. Extend this map as you hit new questions —
# the XBRL taxonomy has thousands of concepts, this covers the frequent ones.
GAAP_CONCEPTS = {
    "revenue": "Revenues",
    "revenue_alt": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "rd_expense": "ResearchAndDevelopmentExpense",
    "net_income": "NetIncomeLoss",
    "operating_income": "OperatingIncomeLoss",
    "total_assets": "Assets",
    "total_liabilities": "Liabilities",
    "gross_profit": "GrossProfit",
    "sga_expense": "SellingGeneralAndAdministrativeExpense",
}


@dataclass
class FactPoint:
    concept: str
    value: float
    unit: str
    fiscal_year: int
    fiscal_period: str  # e.g. "FY", "Q1", "Q2"
    form: str
    filed: str
    frame: str  # e.g. "CY2023"


class XbrlClient:
    def __init__(self, edgar: EdgarClient | None = None):
        self.edgar = edgar or EdgarClient()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _company_facts(self, cik: str) -> dict:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        return self.edgar._get(url).json()

    def get_concept_series(
        self, ticker: str, concept_key: str, fiscal_years: list[int] | None = None
    ) -> list[FactPoint]:
        """Returns annual (form=10-K, fiscal_period=FY) values for a GAAP
        concept, e.g. get_concept_series('AAPL', 'rd_expense', [2022,2023,2024])."""
        cik = self.edgar.ticker_to_cik(ticker)
        facts = self._company_facts(cik)
        concept_name = GAAP_CONCEPTS.get(concept_key, concept_key)

        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        concept_data = us_gaap.get(concept_name)
        if concept_data is None:
            raise ValueError(
                f"Concept '{concept_name}' not found for {ticker}. "
                f"Available concepts: {len(us_gaap)} total — check GAAP_CONCEPTS mapping."
            )

        points: list[FactPoint] = []
        for unit, entries in concept_data.get("units", {}).items():
            for entry in entries:
                if entry.get("form") != "10-K" or entry.get("fp") != "FY":
                    continue
                fy = entry.get("fy")
                if fiscal_years and fy not in fiscal_years:
                    continue
                points.append(
                    FactPoint(
                        concept=concept_name,
                        value=float(entry["val"]),
                        unit=unit,
                        fiscal_year=fy,
                        fiscal_period=entry.get("fp", ""),
                        form=entry.get("form", ""),
                        filed=entry.get("filed", ""),
                        frame=entry.get("frame", ""),
                    )
                )
        # dedupe by fiscal year, keep most recently filed value (handles restatements)
        by_year: dict[int, FactPoint] = {}
        for p in points:
            existing = by_year.get(p.fiscal_year)
            if existing is None or p.filed > existing.filed:
                by_year[p.fiscal_year] = p
        return sorted(by_year.values(), key=lambda p: p.fiscal_year)

    def compute_ratio_series(
        self, ticker: str, numerator_key: str, denominator_key: str,
        fiscal_years: list[int] | None = None,
    ) -> list[dict]:
        """E.g. R&D as % of revenue across years. This is real arithmetic on
        real structured data — no LLM involved in producing the number."""
        num = {p.fiscal_year: p.value for p in self.get_concept_series(ticker, numerator_key, fiscal_years)}
        den = {p.fiscal_year: p.value for p in self.get_concept_series(ticker, denominator_key, fiscal_years)}
        years = sorted(set(num) & set(den))
        return [
            {
                "fiscal_year": y,
                "numerator": num[y],
                "denominator": den[y],
                "ratio_pct": round(100 * num[y] / den[y], 2),
            }
            for y in years
        ]
