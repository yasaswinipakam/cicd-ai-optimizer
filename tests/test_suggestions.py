"""Tests for root-cause optimization suggestion mapping."""

import pytest

from src.models.suggestions import get_suggestion


@pytest.mark.parametrize(
    ("terms", "category"),
    [
        (["network", "timeout"], "Flaky/Timing"),
        (["module", "version"], "Dependency/Build Config"),
        (["assert", "expect"], "Test Reliability"),
        (["compile", "syntax"], "Non-Test Failure"),
    ],
)
def test_suggestions_match_failure_categories(terms, category):
    assert get_suggestion(terms)["category"] == category


def test_suggestions_fall_back_for_unknown_terms():
    suggestion = get_suggestion(["zargle", "flimflam"])
    assert suggestion["title"] == "Manual build-log review recommended"
    assert "Review build logs manually" in suggestion["suggestion"]
