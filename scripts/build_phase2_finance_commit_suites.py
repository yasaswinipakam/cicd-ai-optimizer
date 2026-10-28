"""Build and audit leakage-safe commit/check-suite units for Phase 2.

The source API exposes workflow runs.  Multiple runs can belong to the same
commit SHA, so this builder creates one target per repository/SHA rather than
treating matrix/workflow fan-out as independent code changes.  It performs no
feature engineering, split, calibration, or model fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "phase2_finance_v1" / "workflow_runs_raw.parquet"
OUT = ROOT / "data" / "processed" / "phase2_finance_v1_commit_suites.parquet"
REPORTS = ROOT / "reports"


def main() -> None:
    runs = pd.read_parquet(RAW)
    runs["trigger_time"] = pd.to_datetime(runs.created_at, utc=True, errors="coerce")
    runs["finish_time"] = pd.to_datetime(runs.updated_at, utc=True, errors="coerce")
    included = (
        runs.status.eq("completed")
        & runs.run_attempt.eq(1)
        & runs.event.isin(["push", "pull_request"])
        & runs.conclusion.isin(["success", "failure", "startup_failure"])
    )
    filtered = runs.loc[included].copy()
    missing_sha = filtered.head_sha.isna() | filtered.head_sha.eq("")
    excluded_missing_sha = filtered.loc[missing_sha]
    filtered = filtered.loc[~missing_sha].copy()
    if filtered.trigger_time.isna().any() or filtered.finish_time.isna().any():
        raise ValueError("eligible workflow run missing trigger or completion timestamp")
    if (filtered.finish_time < filtered.trigger_time).any():
        raise ValueError("eligible workflow run completion precedes trigger")
    filtered["ordinary_failure"] = filtered.conclusion.eq("failure")
    filtered["startup_failure"] = filtered.conclusion.eq("startup_failure")
    grouped = filtered.groupby(["repository", "head_sha"], sort=True, as_index=False)
    suites = grouped.agg(
        trigger_time=("trigger_time", "min"),
        finish_time=("finish_time", "max"),
        target_failure=("ordinary_failure", "max"),
        has_startup_failure=("startup_failure", "max"),
        constituent_workflow_runs=("id", "size"),
        workflow_ids=("workflow_id", "nunique"),
        workflow_names=("name", lambda s: " | ".join(sorted({str(v) for v in s.dropna()}))),
        events=("event", lambda s: " | ".join(sorted({str(v) for v in s.dropna()}))),
        first_run_id=("id", "min"),
    )
    suites["duration_seconds"] = (suites.finish_time - suites.trigger_time).dt.total_seconds()
    if suites.duplicated(["repository", "head_sha"]).any():
        raise AssertionError("duplicate commit/check-suite identifier")
    if (suites.finish_time < suites.trigger_time).any():
        raise AssertionError("suite completion precedes suite trigger")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    suites.to_parquet(OUT, index=False)

    rows = []
    for repo, d in suites.sort_values(["repository", "trigger_time", "head_sha"], kind="mergesort").groupby("repository", sort=False):
        cut = int(.8 * len(d))
        span = (d.trigger_time.max() - d.trigger_time.min()).days
        final = d.iloc[cut:]
        rows.append({
            "repository": repo,
            "commit_check_suites": len(d),
            "ordinary_failure_suites": int(d.target_failure.sum()),
            "startup_failure_suites": int(d.has_startup_failure.sum()),
            "both_failure_types": int((d.target_failure & d.has_startup_failure).sum()),
            "earliest_trigger": d.trigger_time.min(),
            "latest_trigger": d.trigger_time.max(),
            "span_days": span,
            "final_20_suites": len(final),
            "final_20_ordinary_failures": int(final.target_failure.sum()),
            "median_workflow_runs_per_suite": float(d.constituent_workflow_runs.median()),
            "max_workflow_runs_per_suite": int(d.constituent_workflow_runs.max()),
        })
    audit = pd.DataFrame(rows)
    audit["eligible"] = (
        audit.span_days.ge(180)
        & audit.commit_check_suites.ge(500)
        & audit.ordinary_failure_suites.ge(30)
        & audit.final_20_ordinary_failures.ge(10)
    )
    audit.to_csv(REPORTS / "phase2_finance_commit_suite_eligibility.csv", index=False)
    manifest = {
        "study": "phase2_finance_pilot_v1_1",
        "source": str(RAW.relative_to(ROOT)),
        "unit": "repository + head_sha commit/check-suite",
        "source_filter": "completed, run_attempt==1, event in {push,pull_request}, conclusion in {success,failure,startup_failure}",
        "ordinary_failure_label": "any constituent conclusion == failure",
        "startup_failure": "reported separately; not merged into ordinary failure label",
        "trigger_time": "minimum constituent created_at",
        "finish_time": "maximum constituent updated_at",
        "excluded_missing_head_sha_runs": int(len(excluded_missing_sha)),
        "output": str(OUT.relative_to(ROOT)),
    }
    (REPORTS / "phase2_finance_commit_suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(audit.to_string(index=False))
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
