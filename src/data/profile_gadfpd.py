"""Profile GADFPD separately before any GitHub Actions model research.

This script deliberately does not alter the TravisTorrent pipeline, production
artifacts, or live predictor. It inventories the modern dataset and documents a
strict pre-run feature boundary for a future, platform-specific study.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "final_research_dataset_MASTER.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
JSON_PATH = REPORTS_DIR / "gadfpd_schema_audit.json"
MARKDOWN_PATH = REPORTS_DIR / "gadfpd_schema_audit.md"

# These are a research schema proposal, not changes to FEATURE_COLS or the live
# model.  The two groups are intentionally explicit to make leakage review easy.
PROPOSED_PRE_RUN_FIELDS = {
    "repo": "Repository identity; use for group-aware evaluation and optional local calibration, not as a global-model feature.",
    "event": "Workflow trigger is known when the run is created.",
    "head_branch": "Branch is known at workflow dispatch.",
    "run_number": "Workflow run sequence is assigned at run creation.",
    "created_at": "Run creation time; derive calendar features only.",
    "workflow_id": "Workflow configuration identity is known at dispatch; assess generalization carefully.",
    "msg_len": "Commit-message length is known before the workflow starts.",
    "is_merge": "Merge-commit status is known from commit metadata before execution.",
    "num_parents": "Commit graph structure is known before execution.",
    "additions": "Commit-level additions are available from the commit before the run starts.",
    "deletions": "Commit-level deletions are available from the commit before the run starts.",
    "total_churn": "Commit-level churn is available before the run starts.",
    "files_modified": "Changed-file count is available before the run starts.",
    "time_since_last_commit": "Historical commit interval can be computed before dispatch.",
}
EXCLUDED_OR_DIAGNOSTIC_FIELDS = {
    "conclusion": "Post-run outcome; this is the research target, never a predictive feature.",
    "status": "Execution state becomes known during/after workflow execution; exclude from pre-run prediction.",
    "duration": "Observed only after or during execution; post-build leakage.",
    "run_started_at": "Execution-start timestamp; retain for chronology only, not a pre-dispatch feature.",
    "updated_at": "Updated after workflow progress/completion; post-build leakage.",
    "run_id": "Opaque execution identifier with no stable predictive semantics.",
    "repo_name": "Duplicate/mostly missing repository field; use canonical repo only after validation.",
    "commit_sha": "Opaque commit identifier; use only for joins and deduplication.",
    "head_sha": "Opaque commit identifier; use only for joins and deduplication.",
    "node_id": "Opaque platform identifier; never a model feature.",
    "run_attempt": "May encode retries following an earlier failure; exclude absent a proven pre-run definition.",
    "author_name": "Personally identifying and brittle categorical field; do not use as a global feature.",
    "author_email": "Personally identifying field; do not use.",
    "committer_name": "Personally identifying and brittle categorical field; do not use.",
    "committer_email": "Personally identifying field; do not use.",
    "authored_date": "Raw commit timestamp; derive calendar features from a documented timestamp instead.",
    "committed_date": "Raw commit timestamp; derive calendar features from a documented timestamp instead.",
    "message": "Raw text can be studied separately, but is excluded from the first tabular baseline to avoid fragile lexical leakage.",
    "timestamp_raw": "Duplicate raw timestamp field; use only to validate chronology.",
    "commit_date": "Duplicate/mostly missing timestamp field; exclude pending data-quality validation.",
    "author": "Duplicate/mostly missing author field; exclude for privacy and quality reasons.",
    "actor_login": "Platform actor identifier; high-cardinality and repository-specific; exclude from the global baseline.",
    "triggering_actor_login": "Platform actor identifier; high-cardinality and repository-specific; exclude from the global baseline.",
    "timestamp": "Duplicate/mostly missing timestamp field; exclude pending data-quality validation.",
}


def profile_gadfpd(path: Path = DATA_PATH, chunksize: int = 50_000) -> dict:
    """Read the CSV in chunks and write a reproducible schema/data-quality audit."""
    if not path.exists():
        raise FileNotFoundError(f"GADFPD file not found: {path}")
    row_count = 0
    missing: dict[str, int] = {}
    dtypes: dict[str, str] = {}
    outcomes: Counter[str] = Counter()
    repositories: set[str] = set()
    timestamp_min = None
    timestamp_max = None
    columns: list[str] | None = None
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if columns is None:
            columns = list(chunk.columns)
            missing = {column: 0 for column in columns}
            dtypes = {column: str(chunk[column].dtype) for column in columns}
        row_count += len(chunk)
        for column in columns:
            missing[column] += int(chunk[column].isna().sum())
        outcomes.update(chunk["conclusion"].fillna("<missing>").value_counts().astype(int).to_dict())
        repositories.update(chunk["repo"].dropna().astype(str).unique())
        timestamps = pd.to_datetime(chunk["created_at"], utc=True, errors="coerce").dropna()
        if not timestamps.empty:
            timestamp_min = timestamps.min() if timestamp_min is None else min(timestamp_min, timestamps.min())
            timestamp_max = timestamps.max() if timestamp_max is None else max(timestamp_max, timestamps.max())
    assert columns is not None
    classified = []
    for column in columns:
        if column in PROPOSED_PRE_RUN_FIELDS:
            disposition, reason = "candidate_pre_run", PROPOSED_PRE_RUN_FIELDS[column]
        else:
            disposition, reason = "excluded_or_diagnostic", EXCLUDED_OR_DIAGNOSTIC_FIELDS.get(
                column, "Unreviewed field: exclude until its timing and semantics are documented."
            )
        classified.append(
            {
                "column": column,
                "dtype_sample": dtypes[column],
                "missing_count": missing[column],
                "missing_rate": missing[column] / row_count,
                "disposition": disposition,
                "reason": reason,
            }
        )
    report = {
        "dataset": "GADFPD / final_research_dataset_MASTER.csv",
        "path": str(path),
        "row_count": row_count,
        "column_count": len(columns),
        "repository_count": len(repositories),
        "created_at_min": str(timestamp_min),
        "created_at_max": str(timestamp_max),
        "conclusion_counts": dict(outcomes),
        "proposed_pre_run_fields": list(PROPOSED_PRE_RUN_FIELDS),
        "fields": classified,
        "scope_note": "Separate GitHub Actions research dataset. Do not merge with TravisTorrent or deploy to the live API without separate temporal and repository-aware evaluation.",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# GADFPD Schema and Availability Audit",
        "",
        f"- File: `{path.name}`",
        f"- Rows: {row_count:,}; columns: {len(columns)}; repositories: {len(repositories):,}",
        f"- `created_at` range: {timestamp_min} to {timestamp_max}",
        f"- Conclusion counts: {dict(outcomes)}",
        "- Scope: separate GitHub Actions research data; no merge with TravisTorrent and no live-model change.",
        "",
        "| Field | Sample dtype | Missing | Disposition | Reason |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for field in classified:
        lines.append(
            f"| `{field['column']}` | {field['dtype_sample']} | {field['missing_rate']:.2%} | "
            f"{field['disposition']} | {field['reason']} |"
        )
    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GADFPD without changing any ML artifacts.")
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    report = profile_gadfpd(args.input)
    print(json.dumps({key: report[key] for key in ("row_count", "column_count", "repository_count", "conclusion_counts")}, indent=2))
    print(f"Wrote {JSON_PATH} and {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
