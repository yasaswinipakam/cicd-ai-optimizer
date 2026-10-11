"""Create a leakage-safe, GitHub-Actions-only research dataset from GADFPD.

This is intentionally independent from the TravisTorrent processing/training
pipeline and does not modify FEATURE_COLS, production artifacts, or the API.
Its output is for modern-platform research only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "final_research_dataset_MASTER.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "gadfpd_github_actions_research.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUMMARY_JSON = REPORTS_DIR / "gadfpd_ingestion_summary.json"
SUMMARY_MD = REPORTS_DIR / "gadfpd_ingestion_summary.md"
REPOSITORY_MANIFEST = REPORTS_DIR / "gadfpd_repository_manifest.csv"

TARGET_COLUMN = "target_failed"
TIMESTAMP_COLUMN = "created_at"
# These are data retained for future feature engineering, not production FEATURE_COLS.
PRE_RUN_RAW_FIELDS = [
    "event", "head_branch", "run_number", "msg_len", "is_merge", "num_parents",
    "additions", "deletions", "total_churn", "files_modified", "time_since_last_commit",
]
DERIVED_TIME_FIELDS = ["hour", "day_of_week", "month", "is_weekend", "is_night"]
METADATA_FIELDS = ["repo", "workflow_id", TIMESTAMP_COLUMN]
INPUT_COLUMNS = [
    "conclusion", "workflow_id", *METADATA_FIELDS, *PRE_RUN_RAW_FIELDS,
]
# Only terminal outcomes with an unambiguous binary interpretation are labels.
PASS_CONCLUSIONS = {"success"}
FAIL_CONCLUSIONS = {"failure", "startup_failure"}


def _prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Keep GitHub Actions rows and form a binary label without using it as input."""
    gha = chunk.loc[chunk["workflow_id"].notna()].copy()
    normalized = gha["conclusion"].astype("string").str.lower().str.strip()
    labelable = normalized.isin(PASS_CONCLUSIONS | FAIL_CONCLUSIONS)
    gha = gha.loc[labelable].copy()
    normalized = normalized.loc[labelable]
    gha[TARGET_COLUMN] = normalized.isin(FAIL_CONCLUSIONS).astype("int8")
    gha[TIMESTAMP_COLUMN] = pd.to_datetime(gha[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    gha = gha.dropna(subset=[TIMESTAMP_COLUMN])
    for column in [
        "run_number", "msg_len", "num_parents", "additions", "deletions", "total_churn",
        "files_modified", "time_since_last_commit",
    ]:
        gha[column] = pd.to_numeric(gha[column], errors="coerce")
    gha["is_merge"] = gha["is_merge"].astype("boolean")
    gha["event"] = gha["event"].fillna("__missing__").astype(str)
    gha["head_branch"] = gha["head_branch"].fillna("__missing__").astype(str)
    gha["hour"] = gha[TIMESTAMP_COLUMN].dt.hour.astype("int8")
    gha["day_of_week"] = gha[TIMESTAMP_COLUMN].dt.dayofweek.astype("int8")
    gha["month"] = gha[TIMESTAMP_COLUMN].dt.month.astype("int8")
    gha["is_weekend"] = gha["day_of_week"].isin([5, 6]).astype("int8")
    gha["is_night"] = gha["hour"].between(0, 5).astype("int8")
    # conclusion is retained solely for auditability of target creation; it is
    # never listed in PRE_RUN_RAW_FIELDS or a future feature matrix.
    return gha[[*METADATA_FIELDS, "conclusion", *PRE_RUN_RAW_FIELDS, *DERIVED_TIME_FIELDS, TARGET_COLUMN]]


def build_gadfpd_research_dataset(
    input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT, chunksize: int = 50_000
) -> dict:
    """Build modern GHA research data and chronological/repository manifests."""
    if not input_path.exists():
        raise FileNotFoundError(f"GADFPD input not found: {input_path}")
    total_input_rows = 0
    github_actions_rows = 0
    parts = []
    for chunk in pd.read_csv(input_path, usecols=INPUT_COLUMNS, chunksize=chunksize, low_memory=False):
        total_input_rows += len(chunk)
        github_actions_rows += int(chunk["workflow_id"].notna().sum())
        parts.append(_prepare_chunk(chunk))
    clean = pd.concat(parts, ignore_index=True).sort_values(TIMESTAMP_COLUMN, kind="mergesort")
    if clean.empty:
        raise ValueError("No labelable GitHub Actions success/failure records found.")
    split_index = int(len(clean) * 0.8)
    clean["global_temporal_split"] = "test"
    clean.iloc[:split_index, clean.columns.get_loc("global_temporal_split")] = "train"
    if clean.loc[clean["global_temporal_split"].eq("test"), TIMESTAMP_COLUMN].min() < clean.loc[
        clean["global_temporal_split"].eq("train"), TIMESTAMP_COLUMN
    ].max():
        raise RuntimeError("Global temporal split overlaps chronologically.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(output_path, index=False, engine="pyarrow")

    manifest = (
        clean.groupby("repo", dropna=False)
        .agg(
            builds=(TARGET_COLUMN, "size"),
            failures=(TARGET_COLUMN, "sum"),
            failure_rate=(TARGET_COLUMN, "mean"),
            first_build=(TIMESTAMP_COLUMN, "min"),
            last_build=(TIMESTAMP_COLUMN, "max"),
        )
        .reset_index()
        .sort_values("builds", ascending=False)
    )
    # A held-out-repository study can use every sufficiently large repository;
    # this flag is a transparent eligibility rule, not cherry-picking.
    manifest["eligible_for_repo_holdout"] = (manifest["builds"] >= 500) & (manifest["failures"] >= 30)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(REPOSITORY_MANIFEST, index=False)
    summary = {
        "source": str(input_path),
        "output": str(output_path),
        "scope": "GitHub-Actions-only research data; separate from TravisTorrent and production artifacts.",
        "input_rows": int(total_input_rows),
        "github_actions_rows_before_label_filter": int(github_actions_rows),
        "research_rows": int(len(clean)),
        "repositories": int(clean["repo"].nunique()),
        "failures": int(clean[TARGET_COLUMN].sum()),
        "failure_rate": float(clean[TARGET_COLUMN].mean()),
        "date_start": str(clean[TIMESTAMP_COLUMN].min()),
        "date_end": str(clean[TIMESTAMP_COLUMN].max()),
        "train_rows": int(split_index),
        "test_rows": int(len(clean) - split_index),
        "pre_run_fields": PRE_RUN_RAW_FIELDS + DERIVED_TIME_FIELDS,
        "excluded_leakage_fields": ["conclusion", "status", "duration", "run_started_at", "updated_at"],
        "label_rule": {"passed": sorted(PASS_CONCLUSIONS), "failed": sorted(FAIL_CONCLUSIONS)},
        "excluded_terminal_conclusions": ["cancelled", "skipped", "action_required"],
        "repository_holdout_eligible": int(manifest["eligible_for_repo_holdout"].sum()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(
        "# GADFPD GitHub Actions Research Ingestion\n\n"
        "This dataset is separate from TravisTorrent and from the deployed model. It contains only rows with a GitHub Actions `workflow_id` and an unambiguous success/failure conclusion.\n\n"
        f"- Labelled GitHub Actions runs: {summary['research_rows']:,} across {summary['repositories']:,} repositories\n"
        f"- Failures: {summary['failures']:,} ({summary['failure_rate']:.2%})\n"
        f"- Chronological range: {summary['date_start']} to {summary['date_end']}\n"
        f"- Global chronological split: {summary['train_rows']:,} train / {summary['test_rows']:,} final test\n"
        f"- Repository holdout eligible: {summary['repository_holdout_eligible']:,}\n\n"
        "`conclusion` is retained only as an audit label. `status`, `duration`, `run_started_at`, and `updated_at` are excluded from prediction research because they are available during or after execution.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the separate GADFPD GitHub Actions research dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_gadfpd_research_dataset(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
