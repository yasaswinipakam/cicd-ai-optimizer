"""Tests for structured, post-failure CI log diagnostics."""

from src.diagnostics.log_analysis import analyze_failed_log, sanitize_log


def test_log_analysis_classifies_and_redacts_common_secret_shapes() -> None:
    result = analyze_failed_log("ModuleNotFoundError: No module named 'requests'\nghp_abcdefghijklmnopqrstuvwxyz")
    assert result.category == "dependency_build_config"
    assert result.matched_signals
    assert "ghp_abcdefghijklmnopqrstuvwxyz" not in result.sanitized_log


def test_log_analysis_uses_safe_unknown_fallback() -> None:
    result = analyze_failed_log("runner returned unexpected diagnostic code 999")
    assert result.category == "unknown"
    assert result.matched_signals == []
    assert sanitize_log("password: abc123") == "password=[REDACTED]"
