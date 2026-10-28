"""Fetch job lists for Fineract first-attempt runs labelled failure/startup_failure.

Protocol v1.2 needs to know whether a "failed" workflow run executed any job:
runs awaiting maintainer approval are later expired as failures with zero jobs.
Uses the authenticated `gh` CLI; public data only. Resumable via the output file.
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "phase2_finance_v1" / "workflow_runs_raw.parquet"
OUT_DIR = ROOT / "data" / "raw" / "phase2_finance_v1_jobs"
OUT = OUT_DIR / "fineract_failure_run_jobs.jsonl"


def _targets() -> pd.DataFrame:
    r = pd.read_parquet(RAW)
    keep = (r.repository.eq("apache/fineract") & r.run_attempt.eq(1) & r.status.eq("completed")
            & r.event.isin(["push", "pull_request"]) & r.conclusion.isin(["failure", "startup_failure"]))
    return r.loc[keep, ["id", "name", "conclusion"]]


def _jobs(run_id: int) -> dict:
    out = subprocess.run(
        ["gh", "api", "--paginate", f"/repos/apache/fineract/actions/runs/{run_id}/jobs?per_page=100",
         "--jq", ".jobs[] | {name, status, conclusion, started_at, completed_at}"],
        capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return {"run_id": run_id, "error": out.stderr.strip()[:300]}
    jobs = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    return {"run_id": run_id, "job_count": len(jobs), "jobs": jobs,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["run_id"] for l in OUT.read_text().splitlines() if l.strip() and "error" not in json.loads(l)}
    todo = [int(i) for i in _targets().id if int(i) not in done]
    print(f"{len(done)} already fetched, {len(todo)} to fetch", flush=True)
    # One API call per run; 8 in flight keeps well inside the hourly rate limit.
    with OUT.open("a") as fh, ThreadPoolExecutor(max_workers=8) as pool:
        for n, record in enumerate(pool.map(_jobs, todo), 1):
            fh.write(json.dumps(record) + "\n"); fh.flush()
            if n % 250 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
