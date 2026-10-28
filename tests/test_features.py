"""Tests for the leak-free feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.features.availability_guard import FeatureAvailabilityError, assert_features_available_for_prediction
from src.features.engineer import add_rolling_history, engineer_features, training_failure_prior
from src.features.schema import FEATURE_COLS


def _history_frame() -> pd.DataFrame:
    # Two builds in repo a (first one cold) and one cold build in repo b.
    return pd.DataFrame(
        {
            "gh_project_name": ["a", "a", "a", "b"],
            "tr_build_id": [1, 1, 2, 3],
            "tr_status": ["failed", "passed", "passed", "passed"],
            "tr_duration": [60, 60, 60, 60],
            "gh_build_started_at": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T05:00:00Z", "2024-01-01T06:00:00Z"]
            ),
            "git_prev_commit_resolution_status": ["build_found"] * 4,
            "gh_lang": ["ruby"] * 4,
        }
    )


def test_training_failure_prior_is_build_level() -> None:
    # Build 1 failed (one of its two jobs), builds 2 and 3 passed: 1/3, not 1/4.
    assert training_failure_prior(_history_frame()) == pytest.approx(1 / 3)


def test_cold_start_history_uses_training_prior_not_zero(tmp_path) -> None:
    history = add_rolling_history(_history_frame())
    assert history.loc[[0, 1, 3], "repo_fail_rate_last5"].isna().all()
    assert history.loc[2, "repo_fail_rate_last5"] == 1.0
    with pytest.raises(ValueError, match="cold_start_prior"):
        engineer_features(history, encoder_path=tmp_path / "e.joblib")
    result, _ = engineer_features(history, encoder_path=tmp_path / "e.joblib", cold_start_prior=0.25)
    assert result.loc[[0, 1, 3], "repo_prior_fail_rate"].tolist() == [0.25, 0.25, 0.25]
    assert result.loc[2, "repo_prior_fail_rate"] == 1.0
    assert not np.isnan(result[FEATURE_COLS].to_numpy(float)).any()


@pytest.mark.parametrize("column", ["gh_num_commits_in_push", "gh_num_pr_comments", "gh_num_issue_comments", "gh_num_commit_comments"])
def test_excluded_features_are_blocked_by_registry(column) -> None:
    assert column not in FEATURE_COLS
    with pytest.raises(FeatureAvailabilityError):
        assert_features_available_for_prediction([column])


def test_engineer_features_adds_expected_columns_and_encodes_missing_values(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "git_prev_commit_resolution_status": ["passed", None, "failed"],
            "gh_lang": ["Python", "Java", None],
            "git_branch": ["master", "feature/test", None],
            "gh_by_core_team_member": [1, 0, 1],
            "gh_is_pr": [0, 1, 1],
            "git_diff_src_churn": [10, 0, 5],
            "git_diff_test_churn": [5, 3, 0],
            "git_diff_src_files": [2, 1, 4],
            "git_diff_test_files": [1, 2, 0],
            "gh_build_started_at": [
                "2024-01-01T02:00:00Z",
                "2024-01-06T12:00:00Z",
                "2024-01-07T22:00:00Z",
            ],
            "tr_log_status": ["ignored", "ignored", "ignored"],
        }
    )

    encoder_path = tmp_path / "encoders.joblib"
    result, feature_columns = engineer_features(source, encoder_path=encoder_path)

    assert set(FEATURE_COLS).issubset(result.columns)
    assert feature_columns == FEATURE_COLS
    assert result["prev_status_enc"].notna().all()
    assert result["lang_enc"].notna().all()
    assert result["is_master"].tolist() == [1, 0, 0]
    assert result["test_to_src_ratio"].tolist() == [0.5, 0.0, 0.0]
    assert result["total_files_changed"].tolist() == [3.0, 3.0, 4.0]
    assert encoder_path.exists()


def test_feature_schema_excludes_post_build_log_columns() -> None:
    assert not any(column.startswith("tr_log_") for column in FEATURE_COLS)


def test_engineer_features_uses_canonical_travistorrent_file_count_columns(tmp_path) -> None:
    """File totals use added + deleted + modified, not test-case/source subsets."""
    source = pd.DataFrame(
        {
            "git_prev_commit_resolution_status": ["passed"],
            "gh_lang": ["Ruby"],
            "gh_diff_files_added": [2],
            "gh_diff_files_deleted": [1],
            "gh_diff_files_modified": [3],
            "gh_diff_src_files": [99],
            "gh_diff_tests_added": [88],
            "gh_diff_tests_deleted": [77],
            "gh_build_started_at": ["2024-01-01T02:00:00Z"],
        }
    )

    result, _ = engineer_features(source, encoder_path=tmp_path / "encoders.joblib")

    assert result["total_files_changed"].tolist() == [6.0]


def test_engineer_features_preserves_valid_notebook_prebuild_numeric_fields(tmp_path) -> None:
    """Additional notebook fields remain numeric, safe, and part of the canonical schema."""
    source = pd.DataFrame(
        {
            "git_prev_commit_resolution_status": ["passed"],
            "gh_lang": ["Python"],
            "gh_build_started_at": ["2024-01-01T02:00:00Z"],
            "git_diff_src_churn": [40],
            "gh_diff_tests_added": [3],
            "gh_diff_tests_deleted": [2],
            "gh_diff_doc_files": [4],
            "gh_num_commits_on_files_touched": [9],
            "git_num_all_built_commits": [12],
            "gh_sloc": [5000],
            "gh_team_size": [7],
            "gh_repo_age": [1200],
            "gh_repo_num_commits": [400],
            "tr_build_number": [88],
        }
    )

    result, feature_columns = engineer_features(source, encoder_path=tmp_path / "encoders.joblib")

    assert feature_columns == FEATURE_COLS
    assert result.loc[0, "gh_diff_tests_added"] == 3.0
    assert result.loc[0, "gh_sloc"] == 5000.0
    assert result.loc[0, "tr_build_number"] == 88.0
    assert result.loc[0, "gh_test_cases_per_kloc"] == 0.0


def test_file_count_invalid_values_fall_back_to_zero_and_transform_does_not_refit_encoders(tmp_path) -> None:
    training = pd.DataFrame(
        {
            "git_prev_commit_resolution_status": ["passed", "failed"],
            "gh_lang": ["Ruby", "Python"],
            "gh_build_started_at": ["2024-01-01T02:00:00Z", "2024-01-01T03:00:00Z"],
        }
    )
    _, _ = engineer_features(training, encoder_path=tmp_path / "encoders.joblib")

    import joblib

    encoders = joblib.load(tmp_path / "encoders.joblib")
    original_classes = {name: encoder.classes_.copy() for name, encoder in encoders.items()}
    incoming = pd.DataFrame(
        {
            "git_prev_commit_resolution_status": ["unseen-status"],
            "gh_lang": ["unseen-language"],
            "gh_diff_files_added": ["bad"],
            "gh_diff_files_deleted": [None],
            "gh_diff_files_modified": [float("nan")],
            "gh_build_started_at": ["2024-01-01T04:00:00Z"],
        }
    )

    result, feature_columns = engineer_features(
        incoming, encoders=encoders, fit_encoders=False, save_encoders=False
    )

    assert result["total_files_changed"].tolist() == [0.0]
    assert feature_columns == FEATURE_COLS
    for name, encoder in encoders.items():
        assert (encoder.classes_ == original_classes[name]).all()
