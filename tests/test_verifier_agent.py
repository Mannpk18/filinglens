import sys
sys.path.insert(0, ".")

from src.agents.verifier_agent import _verify_numeric_claim

NUMERIC_FACTS = [
    {"concept": "rd_expense", "fiscal_year": 2024, "value": 31_370_000_000, "unit": "USD", "source": "xbrl"},
    {"concept": "revenue", "fiscal_year": 2024, "value": 391_035_000_000, "unit": "USD", "source": "xbrl"},
]


def test_verify_numeric_claim_matches_raw_value():
    claim = {
        "text": "R&D expense was $31,370,000,000 in FY2024.",
        "claim_type": "numeric",
        "evidence_id": "rd_expense:2024",
    }
    supported, confidence = _verify_numeric_claim(claim, NUMERIC_FACTS)
    assert supported
    assert confidence == 1.0


def test_verify_numeric_claim_matches_billions_scale():
    claim = {
        "text": "R&D expense was approximately $31.37 billion in FY2024.",
        "claim_type": "numeric",
        "evidence_id": "rd_expense:2024",
    }
    supported, confidence = _verify_numeric_claim(claim, NUMERIC_FACTS)
    assert supported


def test_verify_numeric_claim_rejects_hallucinated_value():
    claim = {
        "text": "R&D expense was $50 billion in FY2024.",
        "claim_type": "numeric",
        "evidence_id": "rd_expense:2024",
    }
    supported, confidence = _verify_numeric_claim(claim, NUMERIC_FACTS)
    assert not supported


def test_verify_numeric_claim_missing_evidence_id_fails():
    claim = {
        "text": "R&D expense was $31.37 billion in FY2024.",
        "claim_type": "numeric",
        "evidence_id": "nonexistent:2024",
    }
    supported, confidence = _verify_numeric_claim(claim, NUMERIC_FACTS)
    assert not supported
    assert confidence == 0.0


def test_verify_numeric_claim_computes_ratio_correctly():
    # sanity check: 31.37B / 391.035B = ~8.02%
    ratio = 100 * NUMERIC_FACTS[0]["value"] / NUMERIC_FACTS[1]["value"]
    assert 7.5 < ratio < 8.5
