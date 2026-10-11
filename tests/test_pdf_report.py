"""PDF report generation checks."""

import base64
import re
import zlib

from src.reports.pdf_generator import generate_build_report_pdf


def _extracted_content_streams(pdf_path) -> str:
    """Extract ReportLab's compressed page-content streams using stdlib only."""
    raw = pdf_path.read_bytes()
    streams = re.findall(rb"stream\r?\n(.*?)endstream", raw, re.DOTALL)
    content = []
    for stream in streams:
        try:
            # ReportLab uses ASCII85 followed by Flate compression by default.
            encoded = stream.strip().removesuffix(b"~>")
            content.append(
                zlib.decompress(base64.a85decode(encoded)).decode("latin-1", errors="ignore")
            )
        except (ValueError, zlib.error):
            continue
    return "\n".join(content)


def test_generate_build_report_pdf_creates_valid_pdf_with_suggestion(tmp_path) -> None:
    output = tmp_path / "build_report.pdf"
    generate_build_report_pdf(
        {
            "pipeline_id": "demo-repo",
            "build_id": 42,
            "timestamp": "2026-08-31T12:00:00Z",
            "prediction": {"predicted_status": "failed", "failure_probability": 0.82, "confidence": 0.82},
            "anomaly": {"is_anomaly": True, "anomaly_score": -0.21},
            "root_cause": {"likely_cause": "network, timeout"},
            "top_terms": ["network", "timeout"],
            "optimization_suggestion": {"category": "Flaky/Timing", "title": "Flaky network-dependent tests detected", "suggestion": "Add retry-on-failure logic for transient tests."},
            "explanation": [
                {"feature": "total_files_changed", "value": 177.0, "shap_value": 0.42, "direction": "increases_failure_risk"},
                {"feature": "gh_is_pr", "value": 0.0, "shap_value": -0.13, "direction": "decreases_failure_risk"},
            ],
        },
        str(output),
    )
    assert output.exists()
    assert output.stat().st_size > 0
    assert output.read_bytes().startswith(b"%PDF")
    content = _extracted_content_streams(output)
    assert "Prediction Explanation" in content
    assert "total_files_changed" in content
    assert "0.42" in content
    assert "increases failure risk" in content
    assert "Flaky network-dependent tests detected" in content


def test_generate_build_report_pdf_handles_empty_or_missing_explanation(tmp_path) -> None:
    for name, explanation in (("empty", []), ("missing", None)):
        output = tmp_path / f"{name}.pdf"
        build_data = {
            "pipeline_id": "demo-repo",
            "build_id": 43,
            "timestamp": "2026-08-31T12:00:00Z",
            "prediction": {"predicted_status": "passed", "failure_probability": 0.1, "confidence": 0.9},
            "anomaly": {"is_anomaly": False, "anomaly_score": 0.1},
        }
        if explanation is not None:
            build_data["explanation"] = explanation
        generate_build_report_pdf(build_data, str(output))
        assert output.read_bytes().startswith(b"%PDF")
        assert "Prediction Explanation" not in _extracted_content_streams(output)
