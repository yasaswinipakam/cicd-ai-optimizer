"""Static contract checks for the active manual-build submission form."""

from pathlib import Path


SUBMIT_BUILD_PAGE = (
    Path(__file__).resolve().parents[1] / "dashboard" / "src" / "pages" / "SubmitBuildPage.jsx"
)


def test_manual_submit_sends_raw_github_actions_inputs() -> None:
    """The form sends the run's raw commit counts; the predictor derives totals and flags itself."""
    source = SUBMIT_BUILD_PAGE.read_text(encoding="utf-8")

    for raw in ("additions:", "deletions:", "files_modified:", "num_parents:", "event:", "workflow_id:"):
        assert raw in source
    for derived in ("total_churn", "is_merge", "is_pr:", "is_default_branch", "msg_len"):
        assert derived not in source
