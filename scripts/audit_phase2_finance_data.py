"""Create the pre-modelling Phase 2 GitHub Actions data audit.

This script intentionally stops before feature engineering, splitting, or
fitting.  It reports both raw workflow-run counts and the same-commit
duplication that must be resolved before choosing the prediction unit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "phase2_finance_v1"
REPORTS = ROOT / "reports"


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    manifest = json.loads((RAW / "phase2_finance_collection_manifest.json").read_text())
    raw = pd.read_parquet(RAW / "workflow_runs_raw.parquet")
    raw["trigger_time"] = pd.to_datetime(raw.created_at, utc=True, errors="coerce")
    # This is a transparent *provisional workflow-run filter*, not a final
    # analytic data set. It makes the impact of the unit decision visible.
    retained = raw.loc[
        raw.status.eq("completed")
        & raw.run_attempt.eq(1)
        & raw.event.isin(["push", "pull_request"])
        & raw.conclusion.isin(["success", "failure", "startup_failure"])
    ].copy()
    retained["primary_failure"] = retained.conclusion.eq("failure")

    raw_summary = raw.groupby("repository", as_index=False).agg(
        raw_workflow_runs=("id", "size"), unique_run_ids=("id", "nunique"),
        earliest_raw_trigger=("trigger_time", "min"), latest_raw_trigger=("trigger_time", "max"),
        duplicate_run_ids=("id", lambda s: int(s.duplicated().sum())),
    )
    provisional = retained.groupby("repository", as_index=False).agg(
        provisional_first_attempt_runs=("id", "size"),
        primary_failures=("primary_failure", "sum"),
        startup_failures=("conclusion", lambda s: int(s.eq("startup_failure").sum())),
        unique_head_shas=("head_sha", "nunique"),
        earliest_retained_trigger=("trigger_time", "min"), latest_retained_trigger=("trigger_time", "max"),
    )
    provisional["retained_span_days"] = (provisional.latest_retained_trigger - provisional.earliest_retained_trigger).dt.days
    provisional["additional_runs_sharing_sha"] = provisional.provisional_first_attempt_runs - provisional.unique_head_shas
    provisional["sha_reuse_share"] = provisional.additional_runs_sharing_sha / provisional.provisional_first_attempt_runs
    eligibility = raw_summary.merge(provisional, on="repository", how="left")
    eligibility["provisional_workflow_run_eligible"] = (
        eligibility.retained_span_days.ge(180)
        & eligibility.provisional_first_attempt_runs.ge(500)
        & eligibility.primary_failures.ge(30)
    )
    # Final-20% failures are only a reporting/eligibility diagnostic here. No
    # model, feature choice, or threshold has been read or selected.
    final_rows = []
    for repo, d in retained.sort_values(["repository", "trigger_time", "id"]).groupby("repository", sort=False):
        cut = int(.8 * len(d))
        final_rows.append({"repository": repo, "provisional_final_20_runs": len(d) - cut, "provisional_final_20_failures": int(d.iloc[cut:].primary_failure.sum())})
    eligibility = eligibility.merge(pd.DataFrame(final_rows), on="repository", how="left")
    eligibility["provisional_workflow_run_eligible"] &= eligibility.provisional_final_20_failures.ge(10)
    eligibility.to_csv(REPORTS / "phase2_finance_candidate_eligibility.csv", index=False)

    outcomes = pd.crosstab(raw.repository, raw.conclusion.fillna("missing"), dropna=False).reset_index()
    events = pd.crosstab(raw.repository, raw.event.fillna("missing"), dropna=False).reset_index()
    outcomes.to_csv(REPORTS / "phase2_finance_raw_outcomes.csv", index=False)
    events.to_csv(REPORTS / "phase2_finance_raw_events.csv", index=False)

    terms = manifest.get("repository_terminations", {})
    truncated = ", ".join(sorted(terms)) if terms else "none"
    lines = [
        "# Phase 2 finance pilot: pre-modelling data audit",
        "",
        "## Collection",
        "",
        f"Retrieved: {manifest['retrieved_at_utc']}. Raw workflow-run records: {len(raw):,}. Repositories with an explicit GitHub pagination ceiling: {truncated}.",
        "",
        "The raw pages and their SHA-256 hashes are retained in `data/raw/phase2_finance_v1/`. A pagination ceiling means the extract is the newest accessible window, not a claim of complete repository history.",
        "",
        "## Raw integrity and provisional first-attempt filter",
        "",
        markdown_table(eligibility),
        "",
        "The provisional filter is completed first-attempt `push`/`pull_request` workflow runs with conclusion success, failure, or startup_failure. `failure` is the primary label; `startup_failure` remains separate. Cancelled, skipped, action-required, queued, scheduled, manual, and non-CI event records are retained in raw data but excluded from the provisional analytic filter.",
        "",
        "## Blocking unit-of-analysis decision",
        "",
        "Workflow-run IDs are unique, but many retained workflow runs share a commit SHA. Therefore workflow runs cannot yet be described as independent code changes. Splitting or averaging them as independent build-like units could overweight matrix/workflow-rich commits and allow the same change to occur on both sides of a temporal boundary.",
        "",
        "Before modelling, choose one pre-specified unit:",
        "",
        "1. **Commit/check-suite unit (recommended):** group first-attempt eligible runs by repository and head SHA; label failed if any constituent workflow run failed; start at the earliest constituent trigger and finish at the latest constituent completion. This aligns the prediction unit with a code change and prevents same-SHA cross-split duplication.",
        "2. **Workflow-run unit:** retain individual runs, but purge every training run sharing a test head SHA and prohibit same-SHA outcome history. This answers a narrower question about ranking CI checks, not code changes.",
        "",
        "No model fitting, calibration, feature selection, or result table is authorized until this decision is recorded. The provisional workflow-run eligibility column is not a final eligibility result under the recommended commit/check-suite unit.",
        "",
        "## Files produced",
        "",
        "- `reports/phase2_finance_candidate_eligibility.csv`",
        "- `reports/phase2_finance_raw_outcomes.csv`",
        "- `reports/phase2_finance_raw_events.csv`",
    ]
    (REPORTS / "phase2_finance_data_audit.md").write_text("\n".join(lines) + "\n")
    print(eligibility.to_string(index=False))
    print("STOP: choose commit/check-suite or workflow-run unit before modelling.")


if __name__ == "__main__":
    main()
