"""Protocol v1.2 Fineract commit/check-suites: drop zero-job failed runs, add stream key.

Reads the v1.1 raw runs, the fetched job lists for failed runs, and the v1.1
annotated suites (for commit size and descriptive badges). Writes
data/processed/phase2_finance_v1_2_suites.parquet and a small audit json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "phase2_finance_v1" / "workflow_runs_raw.parquet"
JOBS = ROOT / "data" / "raw" / "phase2_finance_v1_jobs" / "fineract_failure_run_jobs.jsonl"
V1_SUITES = ROOT / "data" / "processed" / "phase2_finance_v1_annotated_suites.parquet"
OUT = ROOT / "data" / "processed" / "phase2_finance_v1_2_suites.parquet"
AUDIT = ROOT / "reports" / "phase2_finance_v1_2_suite_audit.json"
FAILED = ("failure", "startup_failure")


def main() -> None:
    r = pd.read_parquet(RAW)
    r = r[r.repository.eq("apache/fineract") & r.run_attempt.eq(1) & r.status.eq("completed")
          & r.event.isin(["push", "pull_request"]) & r.conclusion.isin(["success", *FAILED])].copy()
    records = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    # A transient fetch error is superseded by a later successful retry of the same run.
    job_count = {x["run_id"]: x["job_count"] for x in records if "error" not in x}
    failed = r.conclusion.isin(FAILED)
    missing = set(r.loc[failed, "id"].astype(int)) - set(job_count)
    if missing:
        raise SystemExit(f"{len(missing)} failed runs have no fetched job list")
    r["job_count"] = r.id.astype(int).map(job_count)
    zero_job_failed = failed & r.job_count.eq(0)
    kept = r[~zero_job_failed].copy()
    for c in ("created_at", "updated_at"):
        kept[c] = pd.to_datetime(kept[c], utc=True)
    kept = kept.sort_values(["created_at", "id"], kind="mergesort")
    suites = kept.groupby("head_sha", sort=False).agg(
        trigger_time=("created_at", "min"), finish_time=("updated_at", "max"),
        target_failure=("conclusion", lambda s: int(s.eq("failure").any())),
        has_startup_failure=("conclusion", lambda s: bool(s.eq("startup_failure").any())),
        constituent_workflow_runs=("id", "size"), stream=("head_branch", "first"),
    ).reset_index()
    v1 = pd.read_parquet(V1_SUITES)
    v1 = v1[v1.repository.eq("apache/fineract")]
    keep_cols = ["head_sha", "additions", "deletions", "total_changes", "changed_file_count",
                 *[c for c in v1.columns if c.startswith("badge_")]]
    suites = suites.merge(v1[keep_cols], on="head_sha", how="inner", validate="one_to_one")
    suites.insert(0, "repository", "apache/fineract")
    suites.to_parquet(OUT, index=False)
    v1_fail = int(v1.target_failure.sum())
    dur_h = (suites.finish_time - suites.trigger_time).dt.total_seconds() / 3600
    audit = {
        "failed_runs_checked": int(failed.sum()), "zero_job_failed_runs_dropped": int(zero_job_failed.sum()),
        "zero_job_failed_runs_by_workflow": r.loc[zero_job_failed, "name"].value_counts().to_dict(),
        "v1_1_suites": len(v1), "v1_2_suites": len(suites), "suites_dropped": len(v1) - len(suites),
        "v1_1_failures": v1_fail, "v1_2_failures": int(suites.target_failure.sum()),
        "v1_1_prevalence": round(v1_fail / len(v1), 4), "v1_2_prevalence": round(float(suites.target_failure.mean()), 4),
        "v1_2_suite_hours_median": round(float(dur_h.median()), 3), "v1_2_suite_hours_p90": round(float(dur_h.quantile(.9)), 2),
        "v1_2_suite_hours_max": round(float(dur_h.max()), 1), "streams": int(suites.stream.nunique()),
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
