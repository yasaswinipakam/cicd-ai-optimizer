"""Collect the GitHub Actions confirmation dataset for the frozen Tier E recipe.

Implements reports/phase1_v2_gha_confirmation_protocol.md. Collection only: no
labels are modelled and nothing is scored here. Every step is resumable.

    python scripts/collect_gha_confirmation.py select    # pick 30 repositories
    python scripts/collect_gha_confirmation.py runs      # list their workflow runs
    python scripts/collect_gha_confirmation.py commits   # fetch commit details
    python scripts/collect_gha_confirmation.py build     # write the dataset CSV

Uses the token of the authenticated `gh` CLI; only public data is read.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "phase2_gha_confirm"
CANDIDATES = ROOT / "reports" / "phase1_v2_gha_confirmation_candidates.csv"
DATASET = OUT / "gha_confirmation_dataset.csv"
FIRST_DATASET = ROOT / "data" / "raw" / "final_research_dataset_MASTER.csv"
API = "https://api.github.com"
QUERY = "stars:>10000 archived:false fork:false pushed:>2026-10-01"
WINDOW = (datetime(2026, 8, 15, tzinfo=timezone.utc), datetime(2026, 10, 1, tzinfo=timezone.utc))
EVENTS = ("push", "pull_request")
FINANCE_PILOT = {"apache/fineract", "actualbudget/actual", "killbill/killbill", "firefly-iii/firefly-iii"}
NON_SOFTWARE = ("awesome", "interview", "roadmap", "tutorial", "course", "list")
MIN_RUNS, MIN_FAILURES, REPOSITORIES, RUN_CAP = 500, 25, 30, 4000


class GitHub:
    """Small client that waits out primary and secondary rate limits."""

    def __init__(self) -> None:
        token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
        self.client = httpx.Client(base_url=API, timeout=60, headers={
            "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "cicd-ai-optimizer-gha-confirmation"})
        self.lock = threading.Lock()

    def get(self, path: str, **params) -> httpx.Response:
        for attempt in range(8):
            try:
                response = self.client.get(path, params=params)
            except httpx.TransportError:
                time.sleep(2 ** attempt); continue
            if response.status_code in (403, 429) and ("rate limit" in response.text.lower() or "retry-after" in response.headers):
                with self.lock:
                    reset = int(response.headers.get("x-ratelimit-reset", time.time() + 60))
                    wait = int(response.headers.get("retry-after", max(reset - time.time(), 1))) + 2
                    print(f"rate limited; waiting {wait}s", flush=True); time.sleep(wait)
                continue
            if response.status_code >= 500:
                time.sleep(2 ** attempt); continue
            return response
        return response


def _slug(repo: str) -> str:
    return repo.replace("/", "__")


def _window_query() -> str:
    start, end = WINDOW
    return f"{start:%Y-%m-%dT%H:%M:%SZ}..{end - timedelta(seconds=1):%Y-%m-%dT%H:%M:%SZ}"


def first_dataset_repositories() -> set[str]:
    raw = pd.read_csv(FIRST_DATASET, usecols=["repo", "conclusion", "event", "run_attempt"], low_memory=False)
    used = raw[raw.event.isin(EVENTS) & raw.run_attempt.eq(1) & raw.conclusion.isin(["success", "failure", "startup_failure"])]
    return set(used.repo.unique())


def select(gh: GitHub) -> None:
    excluded_first = first_dataset_repositories()
    candidates = []
    for page in (1, 2, 3):
        body = gh.get("/search/repositories", q=QUERY, sort="stars", order="desc", per_page=100, page=page).json()
        candidates += [(item["full_name"], item["stargazers_count"]) for item in body["items"]]
        time.sleep(2)  # search API: 30 requests per minute
    rows, selected = [], 0
    for rank, (repo, stars) in enumerate(candidates, 1):
        row = {"rank": rank, "repository": repo, "stars": stars, "excluded": "", "runs": None, "failures": None,
               "eligible": False, "selected": False}
        if repo in excluded_first:
            row["excluded"] = "first GitHub Actions dataset"
        elif repo in FINANCE_PILOT:
            row["excluded"] = "finance pilot"
        elif any(word in repo.lower() for word in NON_SOFTWARE):
            row["excluded"] = "list or teaching material"
        elif selected < REPOSITORIES:
            counts = {}
            for conclusion in ("success", "failure"):
                counts[conclusion] = sum(
                    gh.get(f"/repos/{repo}/actions/runs", event=event, status=conclusion, created=_window_query(), per_page=1)
                    .json().get("total_count", 0) for event in EVENTS)
            row["runs"], row["failures"] = counts["success"] + counts["failure"], counts["failure"]
            row["eligible"] = row["runs"] >= MIN_RUNS and row["failures"] >= MIN_FAILURES
            if row["eligible"]:
                selected += 1; row["selected"] = True
            print(f"{rank:3d} {repo}: {row['runs']} runs, {row['failures']} failures -> {'selected' if row['selected'] else 'not eligible'}", flush=True)
        rows.append(row)
    pd.DataFrame(rows).to_csv(CANDIDATES, index=False)
    print(f"selected {selected} repositories", flush=True)


def _chunks(start: datetime, end: datetime, days: float):
    while start < end:
        stop = min(start + timedelta(days=days), end)
        yield start, stop
        start = stop


def _list_runs(gh: GitHub, repo: str, event: str, start: datetime, end: datetime) -> list[dict]:
    """All runs in [start, end); the API returns at most 1,000 per query, so split long ranges."""
    created = f"{start:%Y-%m-%dT%H:%M:%SZ}..{end - timedelta(seconds=1):%Y-%m-%dT%H:%M:%SZ}"
    first = gh.get(f"/repos/{repo}/actions/runs", event=event, created=created, per_page=100, page=1).json()
    if first.get("total_count", 0) > 1000 and end - start > timedelta(hours=2):
        middle = start + (end - start) / 2
        return _list_runs(gh, repo, event, start, middle) + _list_runs(gh, repo, event, middle, end)
    runs, page, body = [], 1, first
    while True:
        runs += [{
            "run_id": r["id"], "repo": repo, "workflow_id": r["workflow_id"], "event": r["event"],
            "head_branch": r["head_branch"], "head_sha": r["head_sha"], "run_attempt": r["run_attempt"],
            "status": r["status"], "conclusion": r["conclusion"], "created_at": r["created_at"],
            "updated_at": r["updated_at"], "run_started_at": r.get("run_started_at"),
            "actor_login": (r.get("actor") or {}).get("login")} for r in body.get("workflow_runs", [])]
        if len(body.get("workflow_runs", [])) < 100:
            return runs
        page += 1
        body = gh.get(f"/repos/{repo}/actions/runs", event=event, created=created, per_page=100, page=page).json()


def _qualifies(run: dict) -> bool:
    return run["run_attempt"] == 1 and run["conclusion"] in ("success", "failure", "startup_failure")


def collect_runs(gh: GitHub) -> None:
    """List runs day by day from the window start; stop once a repository has RUN_CAP
    qualifying runs, since only its earliest RUN_CAP are kept (see capped_runs)."""
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    for repo in pd.read_csv(CANDIDATES).query("selected").repository:
        path = OUT / "runs" / f"{_slug(repo)}.json"
        if path.exists():
            continue
        runs: list[dict] = []
        for start, stop in _chunks(*WINDOW, days=1):
            runs += [run for event in EVENTS for run in _list_runs(gh, repo, event, start, stop)]
            if sum(map(_qualifies, runs)) >= RUN_CAP:
                break
        path.write_text(json.dumps(runs))
        print(f"{repo}: {len(runs)} runs listed, {sum(map(_qualifies, runs))} qualifying, through {stop:%Y-%m-%d}", flush=True)


def capped_runs() -> pd.DataFrame:
    """Qualifying first-attempt runs, earliest RUN_CAP per repository."""
    frames = []
    for path in sorted((OUT / "runs").glob("*.json")):
        runs = pd.DataFrame(json.loads(path.read_text()))
        if runs.empty:
            continue
        runs["created_at"] = pd.to_datetime(runs.created_at, utc=True)
        keep = runs.run_attempt.eq(1) & runs.conclusion.isin(["success", "failure", "startup_failure"])
        frames.append(runs[keep].drop_duplicates("run_id").sort_values(["created_at", "run_id"]).head(RUN_CAP))
    return pd.concat(frames, ignore_index=True)


def _commit(gh: GitHub, repo: str, sha: str) -> dict:
    response = gh.get(f"/repos/{repo}/commits/{sha}")
    if response.status_code != 200:
        return {"repo": repo, "sha": sha, "error": response.status_code}
    body = response.json(); files = len(body.get("files", []))
    page = 1
    while len(body.get("files", [])) == 300 and page < 10:  # the API pages changed files 300 at a time
        page += 1
        body = gh.get(f"/repos/{repo}/commits/{sha}", page=page).json(); files += len(body.get("files", []))
    first = response.json()
    return {"repo": repo, "sha": sha, "additions": first["stats"]["additions"], "deletions": first["stats"]["deletions"],
            "files_modified": files, "msg_len": len(first["commit"]["message"]), "num_parents": len(first["parents"]),
            "committed_date": first["commit"]["committer"]["date"]}


def collect_commits(gh: GitHub) -> None:
    path = OUT / "commits.jsonl"
    done = set()
    if path.exists():
        done = {(r["repo"], r["sha"]) for r in map(json.loads, path.read_text().splitlines()) if "error" not in r}
    todo = sorted({(r.repo, r.head_sha) for r in capped_runs().itertuples()} - done)
    print(f"{len(done)} commits fetched, {len(todo)} to fetch", flush=True)
    with path.open("a") as handle, ThreadPoolExecutor(max_workers=6) as pool:
        for n, record in enumerate(pool.map(lambda key: _commit(gh, *key), todo), 1):
            handle.write(json.dumps(record) + "\n"); handle.flush()
            if n % 500 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def build() -> None:
    runs = capped_runs()
    commits = pd.DataFrame([r for r in map(json.loads, (OUT / "commits.jsonl").read_text().splitlines()) if "error" not in r])
    commits = commits.drop_duplicates(["repo", "sha"])
    commits["committed_date"] = pd.to_datetime(commits.committed_date, utc=True)
    commits = commits.sort_values(["repo", "committed_date", "sha"])
    commits["time_since_last_commit"] = commits.groupby("repo").committed_date.diff().dt.total_seconds()
    data = runs.merge(commits.rename(columns={"sha": "head_sha"}), on=["repo", "head_sha"], how="inner")
    data["commit_sha"] = data.head_sha
    data["total_churn"] = data.additions + data.deletions
    data["is_merge"] = data.num_parents > 1
    started = pd.to_datetime(data.run_started_at, utc=True)
    data["duration"] = (pd.to_datetime(data.updated_at, utc=True) - started).dt.total_seconds()
    data.to_csv(DATASET, index=False)
    print(json.dumps({"qualifying_runs": len(runs), "runs_with_commit": len(data), "dropped_no_commit": len(runs) - len(data),
                      "repositories": int(data.repo.nunique()), "failure_rate": float(data.conclusion.ne("success").mean())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["select", "runs", "commits", "build"])
    step = parser.parse_args().step
    if step == "build":
        build()
    else:
        {"select": select, "runs": collect_runs, "commits": collect_commits}[step](GitHub())
