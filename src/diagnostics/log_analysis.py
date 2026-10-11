"""Privacy-aware, rule-based analysis of real failed CI job logs.

This module is post-failure diagnostics only.  Its output must never be passed
to the pre-build prediction feature matrix.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


MAX_LOG_CHARS = 200_000


@dataclass(frozen=True)
class FailureAnalysis:
    """Serializable structured diagnostic result for one failed CI job."""

    category: str
    title: str
    summary: str
    signature: str
    matched_signals: list[str]
    sanitized_log: str


# Ordered from more specific signals to broad fallback.  These are operational
# diagnostic labels, not pre-build ML features or claims of causal certainty.
RULES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "resource_performance",
        "Resource or runner exhaustion",
        "The log indicates resource pressure or a terminated runner.",
        (r"out of memory", r"oom[- ]?killed", r"no space left on device", r"disk quota", r"exit code 137"),
    ),
    (
        "flaky_timing_network",
        "Timeout or network instability",
        "The log contains a timeout or network/connectivity failure signal.",
        (r"timed? ?out", r"econnreset", r"econnrefused", r"connection (?:reset|refused|failed)", r"temporary failure in name resolution"),
    ),
    (
        "dependency_build_config",
        "Dependency or build configuration failure",
        "The log indicates a missing, incompatible, or unresolved dependency/configuration.",
        (r"modulenotfounderror", r"cannot find module", r"no module named", r"could not resolve", r"dependency .* not found", r"version conflict"),
    ),
    (
        "compilation_syntax",
        "Compilation or syntax failure",
        "The log contains a compiler, parser, or build-time syntax failure.",
        (r"syntaxerror", r"compilation failed", r"compile error", r"error: expected", r"build failed"),
    ),
    (
        "test_assertion",
        "Test assertion failure",
        "The log contains an assertion or expectation mismatch.",
        (r"assertionerror", r"assert .* failed", r"expected:.*received:", r"expected .* to (?:be|equal)", r"\bfailures?:\s*[1-9]"),
    ),
)


def sanitize_log(log_text: str, limit: int = MAX_LOG_CHARS) -> str:
    """Bound retained log size and redact common secrets before persistence."""
    text = str(log_text or "")[:limit]
    replacements = (
        (r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{10,}\b", "[REDACTED_GITHUB_TOKEN]"),
        (r"(?i)(authorization:\s*bearer\s+)[^\s]+", r"\1[REDACTED]"),
        (r"(?i)\b(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*[^\s'\"]+", r"\1=[REDACTED]"),
        (r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def _signal_lines(sanitized_log: str, patterns: tuple[str, ...]) -> list[str]:
    matched: list[str] = []
    for line in sanitized_log.splitlines():
        compact = " ".join(line.strip().split())
        if compact and any(re.search(pattern, compact, flags=re.IGNORECASE) for pattern in patterns):
            matched.append(compact[:300])
        if len(matched) == 3:
            break
    return matched


def analyze_failed_log(log_text: str) -> FailureAnalysis:
    """Return a deterministic, explainable post-failure log classification."""
    sanitized = sanitize_log(log_text)
    for category, title, summary, patterns in RULES:
        signals = _signal_lines(sanitized, patterns)
        if signals:
            canonical = re.sub(r"\d+", "#", "\n".join(signals).lower())
            signature = hashlib.sha256(f"{category}:{canonical}".encode()).hexdigest()[:20]
            return FailureAnalysis(category, title, summary, signature, signals, sanitized)
    signature = hashlib.sha256(re.sub(r"\d+", "#", sanitized.lower()).encode()).hexdigest()[:20]
    return FailureAnalysis(
        "unknown",
        "Unclassified failure",
        "No supported failure signature was found; review the sanitized log excerpt.",
        signature,
        [],
        sanitized,
    )
