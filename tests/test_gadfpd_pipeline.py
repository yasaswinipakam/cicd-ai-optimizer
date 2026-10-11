"""Tests for the separate, non-production GADFPD research ingestion."""

import pandas as pd

from src.data.gadfpd_pipeline import build_gadfpd_research_dataset


def test_gadfpd_ingestion_keeps_only_labelable_github_actions_rows(tmp_path, monkeypatch) -> None:
    raw = pd.DataFrame(
        [
            {"conclusion": "success", "workflow_id": 1, "repo": "org/a", "created_at": "2026-01-01T10:00:00Z", "event": "push", "head_branch": "main", "run_number": 1, "msg_len": 2, "is_merge": False, "num_parents": 1, "additions": 3, "deletions": 1, "total_churn": 4, "files_modified": 2, "time_since_last_commit": 12},
            {"conclusion": "failure", "workflow_id": 2, "repo": "org/a", "created_at": "2026-01-02T10:00:00Z", "event": "pull_request", "head_branch": "feature", "run_number": 2, "msg_len": 5, "is_merge": False, "num_parents": 1, "additions": 30, "deletions": 10, "total_churn": 40, "files_modified": 8, "time_since_last_commit": 30},
            {"conclusion": "cancelled", "workflow_id": 3, "repo": "org/a", "created_at": "2026-01-03T10:00:00Z", "event": "push", "head_branch": "main", "run_number": 3, "msg_len": 2, "is_merge": False, "num_parents": 1, "additions": 1, "deletions": 1, "total_churn": 2, "files_modified": 1, "time_since_last_commit": 4},
            {"conclusion": "passed", "workflow_id": None, "repo": "legacy/a", "created_at": "2016-01-01T10:00:00Z", "event": "push", "head_branch": "master", "run_number": 1, "msg_len": 2, "is_merge": False, "num_parents": 1, "additions": 1, "deletions": 1, "total_churn": 2, "files_modified": 1, "time_since_last_commit": 4},
        ]
    )
    input_path = tmp_path / "source.csv"
    output_path = tmp_path / "research.parquet"
    raw.to_csv(input_path, index=False)
    monkeypatch.setattr("src.data.gadfpd_pipeline.REPORTS_DIR", tmp_path)
    monkeypatch.setattr("src.data.gadfpd_pipeline.SUMMARY_JSON", tmp_path / "summary.json")
    monkeypatch.setattr("src.data.gadfpd_pipeline.SUMMARY_MD", tmp_path / "summary.md")
    monkeypatch.setattr("src.data.gadfpd_pipeline.REPOSITORY_MANIFEST", tmp_path / "manifest.csv")

    summary = build_gadfpd_research_dataset(input_path, output_path, chunksize=2)
    processed = pd.read_parquet(output_path)
    assert summary["research_rows"] == 2
    assert processed["target_failed"].tolist() == [0, 1]
    assert set(processed["global_temporal_split"]) == {"train", "test"}
    assert "duration" not in processed.columns
    assert (tmp_path / "manifest.csv").exists()
