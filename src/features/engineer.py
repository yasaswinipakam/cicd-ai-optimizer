"""Leak-free feature engineering for CI/CD build failure prediction."""

import heapq
from pathlib import Path

import joblib
import numpy as np
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


HISTORY_COLS = ("repo_fail_rate_last5", "repo_fail_rate_last20", "repo_prior_fail_rate")
STREAK_COLS = ("repo_streak_fails", "repo_builds_since_success", "repo_hours_since_last_build")

# Auxiliary columns used by add_rolling_history to compute finish-time-safe windows.
# tr_duration is a post-build field of PRIOR builds used to determine when those
# builds completed — it is never a feature of the current build being scored.
HISTORY_AUX_COLS = ("tr_build_id", "tr_duration")

def training_failure_prior(train: pd.DataFrame, label_col: str = "tr_status", build_id_col: str = "tr_build_id") -> float:
    """Build-level failure rate of the training rows: the cold-start history fill.

    Rate features are NaN until a repository has a completed build; they are
    filled with this value, derived from the training split only, never 0.0
    (which would signal "this repo never fails") and never a global constant.
    """
    failed = train[label_col].eq("failed")
    if build_id_col in train.columns:
        failed = failed.groupby(train[build_id_col]).any()
    return float(failed.mean())


def _safe_history_for_repo(
    start_ns_arr: np.ndarray,
    finish_ns_arr: np.ndarray,
    failed_arr: np.ndarray,
    build_id_arr: np.ndarray,
    buffer_ns: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Heap-based finish-time-safe history for one repository's build sequence.

    A build contributes to history only after its last job has finished strictly
    before ``current_start - buffer_ns``.  buffer_ns=0 matches the Phase-1
    rigor protocol's 0h sensitivity; pass 3_600_000_000_000 for the 1h buffer
    or 21_600_000_000_000 for the 6h buffer used in the Phase-1 sensitivity study.

    Returns six arrays: rate5, rate20, rate_prior, streak_fails,
    builds_since_success, hours_since_last_build.
    """
    n = len(start_ns_arr)
    rate5 = np.full(n, np.nan)
    rate20 = np.full(n, np.nan)
    rate_prior = np.full(n, np.nan)
    streak_fails = np.zeros(n)
    builds_since_success = np.zeros(n)
    hours_since_last = np.zeros(n)

    pending: list[tuple[int, int, int]] = []  # (finish_ns, build_id, failed)
    completed: list[int] = []
    total_failed = 0
    current_streak = 0
    last_success_idx = -1   # index into completed; -1 = no success yet
    last_finish_ns = 0      # max finish_ns of all drained builds

    for i in range(n):
        t_start = int(start_ns_arr[i])
        cutoff = t_start - buffer_ns

        # Drain all builds that finished strictly before the cutoff.
        while pending and pending[0][0] < cutoff:
            fin_ns, _, label = heapq.heappop(pending)
            completed.append(label)
            total_failed += label
            if label == 0:
                current_streak = 0
                last_success_idx = len(completed) - 1
            else:
                current_streak += 1
            if fin_ns > last_finish_ns:
                last_finish_ns = fin_ns

        n_done = len(completed)
        if n_done > 0:
            rate5[i] = float(np.mean(completed[-5:]))
            rate20[i] = float(np.mean(completed[-20:]))
            rate_prior[i] = total_failed / n_done
            streak_fails[i] = float(current_streak)
            builds_since_success[i] = float(
                n_done - 1 - last_success_idx if last_success_idx >= 0 else n_done
            )
            hours_since_last[i] = max(0.0, (t_start - last_finish_ns) / 3_600_000_000_000.0)

        heapq.heappush(pending, (int(finish_ns_arr[i]), int(build_id_arr[i]), int(failed_arr[i])))

    return rate5, rate20, rate_prior, streak_fails, builds_since_success, hours_since_last


def add_rolling_history(
    df: pd.DataFrame,
    label_col: str = "tr_status",
    duration_col: str = "tr_duration",
    build_id_col: str = "tr_build_id",
    finish_buffer_s: float = 0.0,
) -> pd.DataFrame:
    """Add per-repo rolling failure-rate features with finish-time safety.

    When *duration_col* and *build_id_col* are present the function uses a
    heap keyed by each build's finish time (start + duration), so only builds
    that completed before ``current_start - finish_buffer_s`` contribute to the
    window.  finish_buffer_s=0 matches the Phase-1 0h sensitivity; pass 3600
    or 21600 for the 1h / 6h sensitivity buffers from the Phase-1 study.

    Missing or non-positive durations fall back to the per-repo p95 duration,
    then the global p95, rather than zero.  Using zero would make those builds
    appear to finish instantly, leaking their outcomes into concurrent builds.

    Rate features are NaN for cold-start builds; ``engineer_features`` fills
    them with a training-split prior (see ``training_failure_prior``).

    Must be called on the full chronologically-sorted dataset BEFORE any split.
    """
    out = df.copy()
    for col in HISTORY_COLS:
        out[col] = np.nan
    for col in STREAK_COLS:
        out[col] = 0.0

    has_aux = (
        duration_col in out.columns
        and build_id_col in out.columns
        and "gh_build_started_at" in out.columns
    )

    if not has_aux:
        # Fallback when auxiliary columns are absent (live inference path).
        # shift(1) on start-time order is sufficient for the cold-start / single-
        # row inference case; concurrent-build leakage only affects batch training.
        _FAIL_TMP = "__failed_tmp__"
        out[_FAIL_TMP] = out[label_col].eq("failed").astype(float)
        out["repo_fail_rate_last5"] = (
            out.groupby("gh_project_name")[_FAIL_TMP]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
        out["repo_fail_rate_last20"] = (
            out.groupby("gh_project_name")[_FAIL_TMP]
            .transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
        )
        out["repo_prior_fail_rate"] = (
            out.groupby("gh_project_name")[_FAIL_TMP]
            .transform(lambda x: x.expanding().mean().shift(1))
        )
        # Streak / gap features default to 0 in the no-aux fallback path.
        out.drop(columns=[_FAIL_TMP], inplace=True)
        return out

    # Compute per-build finish time: start_ns + duration_ns.
    # Missing/zero/negative durations fall back to the per-repo p95, then the
    # global p95.  Using 0 would make those builds appear to finish instantly,
    # allowing their outcomes to leak into concurrent builds' history windows.
    start_ts = pd.to_datetime(out["gh_build_started_at"], utc=True, errors="coerce")
    # pandas 3 may store microseconds; durations below are added in nanoseconds.
    start_ns = start_ts.dt.as_unit("ns").astype("int64")
    duration_raw = pd.to_numeric(out[duration_col], errors="coerce")
    pos_mask = duration_raw > 0

    # Per-repo p95 from strictly-positive durations in the supplied dataset.
    # This is an approximation when the dataset spans train+test; for a fallback
    # that only affects data-quality edge cases the bias is negligible.
    repo_p95: pd.Series = (
        duration_raw[pos_mask]
        .groupby(out.loc[pos_mask, "gh_project_name"])
        .quantile(0.95)
        if pos_mask.any()
        else pd.Series(dtype=float)
    )
    global_p95_s = float(duration_raw[pos_mask].quantile(0.95)) if pos_mask.any() else 3600.0
    duration_fallback = out["gh_project_name"].map(repo_p95).fillna(global_p95_s)
    duration_s = duration_raw.where(pos_mask, duration_fallback)
    finished_ns = start_ns + (duration_s * 1_000_000_000).astype("int64")

    out["_start_ns_tmp"] = start_ns
    out["_finished_ns_tmp"] = finished_ns

    # Build-level deduplication: one entry per build using max finish time and
    # OR-of-failures (a build fails if any job fails), matching Phase-1 collapse.
    out["_build_finished_ns_tmp"] = out.groupby(build_id_col)["_finished_ns_tmp"].transform("max")

    builds = (
        out.groupby(build_id_col, sort=False)
        .agg(
            _repo=("gh_project_name", "first"),
            _start_ns=("_start_ns_tmp", "first"),
            _finished_ns=("_build_finished_ns_tmp", "first"),
            _failed=(label_col, lambda x: int(x.eq("failed").any())),
        )
        .reset_index()
    )

    # Allocate result arrays indexed by build_id.
    idx = builds[build_id_col]
    build_rate5 = pd.Series(np.nan, index=idx)
    build_rate20 = pd.Series(np.nan, index=idx)
    build_rate_prior = pd.Series(np.nan, index=idx)
    build_streak = pd.Series(0.0, index=idx)
    build_since_success = pd.Series(0.0, index=idx)
    build_hours_since = pd.Series(0.0, index=idx)

    buffer_ns = int(finish_buffer_s * 1_000_000_000)
    for _, repo_builds in builds.groupby("_repo"):
        sorted_b = repo_builds.sort_values(["_start_ns", build_id_col], kind="mergesort")
        r5, r20, rp, streak, since_suc, hrs = _safe_history_for_repo(
            sorted_b["_start_ns"].to_numpy(),
            sorted_b["_finished_ns"].to_numpy(),
            sorted_b["_failed"].to_numpy(),
            sorted_b[build_id_col].to_numpy(),
            buffer_ns=buffer_ns,
        )
        build_rate5.loc[sorted_b[build_id_col]] = r5
        build_rate20.loc[sorted_b[build_id_col]] = r20
        build_rate_prior.loc[sorted_b[build_id_col]] = rp
        build_streak.loc[sorted_b[build_id_col]] = streak
        build_since_success.loc[sorted_b[build_id_col]] = since_suc
        build_hours_since.loc[sorted_b[build_id_col]] = hrs

    # Map build-level history back to job-level rows.
    out["repo_fail_rate_last5"] = out[build_id_col].map(build_rate5)
    out["repo_fail_rate_last20"] = out[build_id_col].map(build_rate20)
    out["repo_prior_fail_rate"] = out[build_id_col].map(build_rate_prior)
    out["repo_streak_fails"] = out[build_id_col].map(build_streak).fillna(0.0)
    out["repo_builds_since_success"] = out[build_id_col].map(build_since_success).fillna(0.0)
    out["repo_hours_since_last_build"] = out[build_id_col].map(build_hours_since).fillna(0.0)

    out.drop(
        columns=["_start_ns_tmp", "_finished_ns_tmp", "_build_finished_ns_tmp"],
        inplace=True,
    )
    return out


def engineer_features(
    df: pd.DataFrame,
    *,
    encoders: dict[str, LabelEncoder] | None = None,
    fit_encoders: bool = True,
    save_encoders: bool = True,
    encoder_path: Path = ENCODER_PATH,
    cold_start_prior: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add pre-build features using either newly fitted or supplied encoders.

    ``cold_start_prior`` (from ``training_failure_prior`` on the training split)
    fills rate-history NaNs.  It is required whenever computed history has
    cold-start rows; frames without any history columns get a constant 0.0.

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

    for column in HISTORY_COLS:
        if column not in engineered.columns:
            engineered[column] = 0.0
            continue
        values = pd.to_numeric(engineered[column], errors="coerce")
        if values.isna().any():
            if cold_start_prior is None:
                raise ValueError(f"{column} has cold-start rows; pass cold_start_prior from training_failure_prior(train).")
            values = values.fillna(cold_start_prior)
        engineered[column] = values
    for column in STREAK_COLS:
        engineered[column] = _numeric_column(engineered, column)

    if save_encoders:
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted_encoders, encoder_path)
    # This does not alter feature values; it prevents future schema regressions.
    assert_features_available_for_prediction(FEATURE_COLS)
    return engineered, FEATURE_COLS.copy()
