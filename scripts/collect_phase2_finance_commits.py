"""Resumably collect public commit metadata for already eligible Phase 2 suites.

Commit metadata supplies pre-trigger change size and source-path information.
It is deliberately separate from outcome collection, preserves every raw API
response, and does not fit or score a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "data" / "processed" / "phase2_finance_v1_commit_suites.parquet"
ELIGIBILITY = ROOT / "reports" / "phase2_finance_commit_suite_eligibility.csv"
OUT = ROOT / "data" / "raw" / "phase2_finance_v1_commits"
API = "https://api.github.com"


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="For smoke collection only; 0 collects all eligible suites.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pause_seconds < 0 or args.max_retries < 0 or args.limit < 0:
        raise SystemExit("invalid collection arguments")
    eligible = pd.read_csv(ELIGIBILITY).query("eligible").repository.tolist()
    suites = pd.read_parquet(SUITES).loc[lambda d: d.repository.isin(eligible), ["repository", "head_sha"]].drop_duplicates()
    if args.limit:
        suites = suites.iloc[: args.limit]
    if args.dry_run:
        print(f"eligible repositories: {eligible}")
        print(f"commit metadata requests planned: {len(suites)}")
        print(suites.head(5).to_string(index=False))
        return
    token = os.environ.get(args.token_env)
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "cicd-ai-optimizer-phase2-pilot"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    out = args.out_dir.resolve(); raw_dir = out / "raw_commits"; raw_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat(); records=[]
    with httpx.Client(timeout=60, follow_redirects=True, headers=headers) as client:
        for position, row in enumerate(suites.itertuples(index=False), start=1):
            safe_repo = row.repository.replace("/", "--")
            raw_path = raw_dir / f"{safe_repo}__{row.head_sha}.json"
            if raw_path.exists():
                payload = json.loads(raw_path.read_bytes()); source = "cached"
            else:
                url = f"{API}/repos/{row.repository}/commits/{row.head_sha}"
                response = None
                for attempt in range(args.max_retries + 1):
                    try:
                        response = client.get(url); break
                    except httpx.RequestError as exc:
                        if attempt >= args.max_retries:
                            raise RuntimeError(f"network error for {row.repository}@{row.head_sha}") from exc
                        time.sleep(min(60.0, 2.0**attempt))
                assert response is not None
                if response.status_code == 403 and "rate limit" in response.text.lower():
                    raise RuntimeError(f"GitHub API rate limit reached; rerun later with {args.token_env} set.")
                response.raise_for_status(); raw_path.write_bytes(response.content)
                payload = response.json(); source = "api"
                if args.pause_seconds: time.sleep(args.pause_seconds)
            files = payload.get("files", [])
            stats = payload.get("stats", {})
            records.append({
                "repository": row.repository, "head_sha": row.head_sha,
                "commit_author_time": payload.get("commit", {}).get("author", {}).get("date"),
                "commit_committer_time": payload.get("commit", {}).get("committer", {}).get("date"),
                "additions": stats.get("additions"), "deletions": stats.get("deletions"), "total_changes": stats.get("total"),
                "changed_file_count": len(files), "changed_paths": json.dumps(sorted(str(f.get("filename")) for f in files if f.get("filename"))),
                "raw_file": str(raw_path.relative_to(out)), "raw_sha256": sha256(raw_path.read_bytes()),
                "api_retrieved_at_utc": retrieved_at,
            })
            if source == "api" or position % 100 == 0:
                print(f"{source.upper()} {position}/{len(suites)} {row.repository}@{row.head_sha[:12]}")
    table = pd.DataFrame(records)
    table.to_parquet(out / "commit_metadata.parquet", index=False)
    manifest = {"study":"phase2_finance_pilot_v1_1", "retrieved_at_utc":retrieved_at, "eligible_repositories":eligible, "requested_commits":len(suites), "records":len(table), "token_used":bool(token), "source_suites":str(SUITES.relative_to(ROOT))}
    (out / "commit_collection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"DONE: {len(table)} commit metadata records under {out}")


if __name__ == "__main__":
    main()
