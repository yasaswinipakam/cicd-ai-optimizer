"""Collect auditable public GitHub Actions workflow-run metadata for Phase 2.

This is collection only: it never creates labels, features, train/test splits,
or model results.  Every downloaded API page is kept verbatim and SHA-256
hashed so the later data audit can trace the normalized table to its source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "reports" / "phase2_finance_candidate_repositories.csv"
DEFAULT_OUT = ROOT / "data" / "raw" / "phase2_finance_v1"
API = "https://api.github.com"
RUN_COLUMNS = [
    "repository", "id", "workflow_id", "name", "event", "status", "conclusion",
    "run_attempt", "head_branch", "head_sha", "created_at", "updated_at",
    "run_started_at", "html_url", "api_retrieved_at_utc",
]


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--token-env", default="GITHUB_TOKEN", help="Optional PAT environment-variable name.")
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--max-pages-per-repo", type=int, default=0, help="0 means fetch every available page.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries for transient network disconnects.")
    parser.add_argument("--pause-seconds", type=float, default=0.2, help="Pause between API pages; set higher if GitHub disconnects.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned collection without network or file writes.")
    args = parser.parse_args()
    if not 1 <= args.per_page <= 100 or args.max_pages_per_repo < 0 or args.max_retries < 0 or args.pause_seconds < 0:
        raise SystemExit("invalid collection arguments")

    cohort = pd.read_csv(CANDIDATES)
    repos = cohort.loc[cohort.selection_status.eq("predeclared_candidate"), "repository"].tolist()
    if not repos:
        raise SystemExit("No predeclared candidates found.")
    if args.dry_run:
        print("Phase 2 collection dry run; no network or files written.")
        for repo in repos:
            print(f"GET {API}/repos/{repo}/actions/runs?per_page={args.per_page}&page=1")
        return

    token = os.environ.get(args.token_env)
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "cicd-ai-optimizer-phase2-pilot"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    out = args.out_dir.resolve()
    raw = out / "raw_pages"
    raw.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    pages: list[dict] = []
    terminations: dict[str, dict] = {}
    with httpx.Client(timeout=60, follow_redirects=True, headers=headers) as client:
        for repo in repos:
            for page in range(1, args.max_pages_per_repo + 1 if args.max_pages_per_repo else sys.maxsize):
                url = f"{API}/repos/{repo}/actions/runs"
                raw_path = raw / f"{repo.replace('/', '--')}__page_{page:04d}.json"
                if raw_path.exists():
                    body = raw_path.read_bytes()
                    payload = json.loads(body)
                    page_url = f"{url}?per_page={args.per_page}&page={page}"
                    http_status = "cached"
                    fetched = False
                    print(f"RESUME {repo} page {page}: using saved raw response")
                else:
                    fetched = True
                    response = None
                    for attempt in range(args.max_retries + 1):
                        try:
                            response = client.get(url, params={"per_page": args.per_page, "page": page})
                            break
                        except httpx.RequestError as exc:
                            if attempt >= args.max_retries:
                                raise RuntimeError(f"network error after {args.max_retries + 1} attempts on {repo} page {page}") from exc
                            delay = min(60.0, 2.0 ** attempt)
                            print(f"RETRY {repo} page {page} after {type(exc).__name__}; sleeping {delay:.0f}s")
                            time.sleep(delay)
                    assert response is not None
                    if response.status_code == 403 and "rate limit" in response.text.lower():
                        raise RuntimeError(f"GitHub API rate limit reached while collecting {repo}; rerun later with {args.token_env} set.")
                    # GitHub's workflow-runs endpoint can reject a page beyond
                    # its result window (observed at page 401).  Preserve that
                    # response and mark the repository as truncated rather
                    # than silently calling the newest 40,000 records "all".
                    if response.status_code == 422 and page > 1:
                        terminal = raw / f"{repo.replace('/', '--')}__terminal_page_{page:04d}.json"
                        terminal.write_bytes(response.content)
                        terminations[repo] = {
                            "reason": "github_workflow_runs_pagination_limit",
                            "attempted_page": page,
                            "http_status": response.status_code,
                            "response_sha256": digest(response.content),
                            "terminal_response_file": str(terminal.relative_to(out)),
                        }
                        print(f"STOP {repo} at page {page}: GitHub pagination limit recorded")
                        break
                    response.raise_for_status()
                    body = response.content
                    raw_path.write_bytes(body)
                    payload = response.json()
                    page_url = str(response.url)
                    http_status = response.status_code
                runs = payload.get("workflow_runs", [])
                pages.append({"repository": repo, "page": page, "url": page_url, "http_status": http_status, "records": len(runs), "sha256": digest(body), "raw_file": str(raw_path.relative_to(out)), "retrieved_at_utc": retrieved_at})
                for item in runs:
                    row = {column: item.get(column) for column in RUN_COLUMNS if column not in {"repository", "api_retrieved_at_utc"}}
                    row.update({"repository": repo, "api_retrieved_at_utc": retrieved_at})
                    records.append(row)
                if not raw_path.exists() or http_status != "cached":
                    print(f"COLLECTED {repo} page {page}: {len(runs)} workflow runs")
                if len(runs) < args.per_page:
                    break
                if fetched and args.pause_seconds:
                    time.sleep(args.pause_seconds)

    runs = pd.DataFrame(records, columns=RUN_COLUMNS)
    runs.to_parquet(out / "workflow_runs_raw.parquet", index=False)
    pd.DataFrame(pages).to_csv(out / "collection_pages.csv", index=False)
    manifest = {
        "study": "phase2_finance_pilot_v1",
        "retrieved_at_utc": retrieved_at,
        "github_api": API,
        "candidate_file": str(CANDIDATES.relative_to(ROOT)),
        "repositories": repos,
        "token_used": bool(token),
        "per_page": args.per_page,
        "max_pages_per_repo": args.max_pages_per_repo,
        "max_retries": args.max_retries,
        "pause_seconds": args.pause_seconds,
        "raw_page_count": len(pages),
        "workflow_run_records": len(runs),
        "repository_terminations": terminations,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (out / "phase2_finance_collection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"DONE: {len(runs)} raw workflow runs saved under {out}")


if __name__ == "__main__":
    main()
