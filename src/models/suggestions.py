"""Actionable, rule-based CI/CD optimization suggestions."""

from __future__ import annotations


SUGGESTION_RULES = [
    {
        "keywords": ["timeout", "network", "connection", "socket", "retry"],
        "category": "Flaky/Timing",
        "title": "Flaky network-dependent tests detected",
        "suggestion": "These failures cluster around network or timeout-related test names. Add retry-on-failure for transient tests and mock external calls where practical.",
    },
    {
        "keywords": ["dependency", "require", "import", "module", "version", "package"],
        "category": "Dependency/Build Config",
        "title": "Dependency or build configuration issue detected",
        "suggestion": "Review dependency versions, lockfiles, and module resolution in the affected build. Pin compatible versions and make the build environment reproducible.",
    },
    {
        "keywords": ["assert", "expect", "should", "test", "spec"],
        "category": "Test Reliability",
        "title": "Test assertion failures detected",
        "suggestion": "Review the affected assertions and their test data. Separate deterministic product defects from brittle expectations, then reduce redundant tests where coverage overlaps.",
    },
    {
        "keywords": ["compile", "syntax", "build", "non-test failure"],
        "category": "Non-Test Failure",
        "title": "Non-test build failure detected",
        "suggestion": "Inspect compiler and build-setup output before running the test suite. Add an early validation stage for syntax, compilation, and required build configuration.",
    },
]


def get_suggestion(
    top_terms: list[str], is_anomaly: bool = False, anomaly_score: float | None = None
) -> dict:
    """Map root-cause terms to a practical remediation recommendation."""
    normalized_terms = " ".join(str(term).lower() for term in top_terms)
    result = None
    for rule in SUGGESTION_RULES:
        if any(keyword in normalized_terms for keyword in rule["keywords"]):
            result = {key: rule[key] for key in ("category", "title", "suggestion")}
            break
    if result is None:
        result = {
            "category": "Resource/Performance" if is_anomaly else "Unclassified",
            "title": "Manual build-log review recommended",
            "suggestion": "Review build logs manually — no clear pattern detected.",
        }
    if is_anomaly and result["category"] != "Resource/Performance":
        score_text = f"{anomaly_score:.2f}" if anomaly_score is not None else "unavailable"
        result["suggestion"] += (
            f" This build's pre-build input profile is also unusual relative to historical builds "
            f"(score: {score_text}) — consider checking for resource contention or infrastructure issues alongside the above."
        )
    return result
