"""Leak-free feature engineering for CI/CD build failure prediction."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.features.availability_guard import assert_features_available_for_prediction
from src.features.schema import DIRECT_PREBUILD_FEATURE_COLS, FEATURE_COLS


ENCODER_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "encoders.joblib"


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric source column, treating absent and invalid values as zero."""
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _total_files_changed(df: pd.DataFrame) -> pd.Series:
    """Return changed-file count from the canonical TravisTorrent file fields.

    TravisTorrent defines ``gh_diff_files_added``, ``gh_diff_files_deleted``,
    and ``gh_diff_files_modified`` as counts of files changed by the commits
    that triggered the build.  ``gh_diff_tests_*`` are *test-case* counts and
    ``gh_diff_src_files`` is only a source-file subset, so neither can stand in
    for this total.  The legacy ``git_diff_*_files`` pair remains a fallback for
    older API clients that cannot send the canonical fields.
    """
    canonical = (
        "gh_diff_files_added",
        "gh_diff_files_deleted",
        "gh_diff_files_modified",
    )
    if any(column in df.columns for column in canonical):
        return sum((_numeric_column(df, column) for column in canonical), start=pd.Series(0.0, index=df.index))
    return (
        _numeric_column(df, "git_diff_src_files")
        + _numeric_column(df, "git_diff_test_files")
    )


def _add_time_features(df: pd.DataFrame) -> None:
    """Backfill Task 2 time columns when this function receives raw timestamps."""
    time_columns = {"hour", "day_of_week", "month", "is_weekend", "is_night"}
    if time_columns.issubset(df.columns):
        return
    if "gh_build_started_at" not in df.columns:
        missing = ", ".join(sorted(time_columns.difference(df.columns)))
        raise ValueError(f"Missing time feature columns and gh_build_started_at: {missing}")

    timestamp = pd.to_datetime(df["gh_build_started_at"], errors="coerce", utc=True)
    df["hour"] = timestamp.dt.hour
    df["day_of_week"] = timestamp.dt.dayofweek
    df["month"] = timestamp.dt.month
    df["is_weekend"] = timestamp.dt.dayofweek.isin([5, 6])
    df["is_night"] = timestamp.dt.hour.isin(range(0, 6))


def engineer_features(
    df: pd.DataFrame,
    *,
    encoders: dict[str, LabelEncoder] | None = None,
    fit_encoders: bool = True,
    save_encoders: bool = True,
    encoder_path: Path = ENCODER_PATH,
) -> tuple[pd.DataFrame, list[str]]:
    """Add pre-build features using either newly fitted or supplied encoders.

    The function does not use ``tr_log_*`` data. Those fields are emitted after a
    build runs and would leak the label into a build-failure prediction model.
    Only the training workflow may fit and persist encoders. Evaluation and other
    downstream workflows must pass the training encoders with ``fit_encoders=False``
    to avoid changing category mappings or overwriting inference artifacts.
    """
    engineered = df.copy()
    _add_time_features(engineered)

    if not fit_encoders and encoders is None:
        raise ValueError("Transform-only feature engineering requires fitted encoders.")
    fitted_encoders: dict[str, LabelEncoder] = {} if encoders is None else encoders
    categorical_sources = {
        "prev_status_enc": "git_prev_commit_resolution_status",
        "lang_enc": "gh_lang",
    }
    for feature, source in categorical_sources.items():
        values = (
            engineered[source].fillna("__missing__").astype(str)
            if source in engineered.columns
            else pd.Series("__missing__", index=engineered.index)
        )
        if fit_encoders:
            encoder = LabelEncoder()
            engineered[feature] = encoder.fit_transform(values)
            fitted_encoders[feature] = encoder
        else:
            encoder = fitted_encoders[feature]
            known_values = values.where(values.isin(encoder.classes_), "__missing__")
            fallback = "__missing__" if "__missing__" in encoder.classes_ else encoder.classes_[0]
            known_values = known_values.where(known_values.isin(encoder.classes_), fallback)
            engineered[feature] = encoder.transform(known_values)

    branch = engineered.get("git_branch", pd.Series("", index=engineered.index))
    engineered["is_master"] = branch.fillna("").astype(str).eq("master").astype(int)
    engineered["by_core"] = _numeric_column(engineered, "gh_by_core_team_member").astype(int)
    engineered["is_pr"] = _numeric_column(engineered, "gh_is_pr").astype(int)

    source_churn = _numeric_column(engineered, "git_diff_src_churn")
    test_churn = _numeric_column(engineered, "git_diff_test_churn")
    engineered["test_to_src_ratio"] = (test_churn / source_churn.mask(source_churn == 0)).fillna(0.0)
    engineered["total_files_changed"] = _total_files_changed(engineered)
    engineered["total_test_changes"] = test_churn

    # Materialize every direct schema field only after aggregate calculations.
    # This preserves the legacy file-count fallback when canonical gh_diff_* keys
    # were genuinely absent from a sparse live payload.
    for column in DIRECT_PREBUILD_FEATURE_COLS:
        engineered[column] = _numeric_column(engineered, column)

    # Keep boolean time features numeric and guarantee the schema is model-ready.
    for column in ("is_weekend", "is_night"):
        engineered[column] = engineered[column].fillna(False).astype(int)
    for column in ("hour", "day_of_week", "month"):
        engineered[column] = _numeric_column(engineered, column).astype(int)

    if save_encoders:
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted_encoders, encoder_path)
    # This does not alter feature values; it prevents future schema regressions.
    assert_features_available_for_prediction(FEATURE_COLS)
    return engineered, FEATURE_COLS.copy()
