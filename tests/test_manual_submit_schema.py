"""Static contract checks for the active manual-build submission form."""

from pathlib import Path


SUBMIT_BUILD_PAGE = (
    Path(__file__).resolve().parents[1] / "dashboard" / "src" / "pages" / "SubmitBuildPage.jsx"
)


def test_manual_submit_uses_canonical_file_count_inputs() -> None:
    """The active form sends canonical raw counts; it never controls a derived total."""
    source = SUBMIT_BUILD_PAGE.read_text(encoding="utf-8")

    assert "gh_diff_files_added" in source
    assert "gh_diff_files_deleted" in source
    assert "gh_diff_files_modified" in source
    assert "git_diff_src_files" not in source
    assert '"total_files_changed"' not in source
