import sys
sys.path.insert(0, ".")

from src.eval.run_ablation import _format_report, _numeric_hallucination_rate


def test_numeric_hallucination_rate_is_inverse_of_accuracy():
    summary = {"numeric_accuracy": 0.95}
    assert _numeric_hallucination_rate(summary) == 5.0


def test_numeric_hallucination_rate_handles_perfect_accuracy():
    summary = {"numeric_accuracy": 1.0}
    assert _numeric_hallucination_rate(summary) == 0.0


def test_format_report_shows_improvement_when_verifier_helps():
    with_verifier = {"summary": {
        "num_examples": 10, "pass_rate": 0.9, "numeric_accuracy": 0.95,
        "avg_confidence": 0.85, "avg_latency_seconds": 3.0, "avg_retries": 0.5,
    }}
    without_verifier = {"summary": {
        "num_examples": 10, "pass_rate": 0.6, "numeric_accuracy": 0.7,
        "avg_confidence": 0.5, "avg_latency_seconds": 1.5, "avg_retries": 0.0,
    }}
    report = _format_report(with_verifier, without_verifier)
    assert "better" in report
    assert "worse" in report  # latency should be flagged as worse (cost of retries)
    assert "Resume framing" in report
    assert "5.0%" in report  # 1 - 0.95 numeric accuracy -> 5.0% hallucination rate


def test_format_report_includes_headline_percentages():
    with_verifier = {"summary": {
        "num_examples": 5, "pass_rate": 1.0, "numeric_accuracy": 1.0,
        "avg_confidence": 1.0, "avg_latency_seconds": 2.0, "avg_retries": 0.0,
    }}
    without_verifier = {"summary": {
        "num_examples": 5, "pass_rate": 0.4, "numeric_accuracy": 0.4,
        "avg_confidence": 0.4, "avg_latency_seconds": 1.0, "avg_retries": 0.0,
    }}
    report = _format_report(with_verifier, without_verifier)
    assert "0.0% (verifier on)" in report
    assert "60.0% (verifier off)" in report
