"""Stream sampled processed builds to a local CI webhook endpoint for demos."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean.parquet"


def github_payload(row: pd.Series, index: int) -> dict:
    """Convert one processed dataset row into a workflow_run-shaped payload."""
    status = "success" if row.get("tr_status") == "passed" else "failure"
    return {
        "action": "completed",
        "repository": {"name": str(row.get("gh_repository_name") or "simulated-repo")},
        "workflow_run": {
            "id": index,
            "name": "simulated-ci",
            "status": "completed",
            "conclusion": status,
            "run_started_at": str(row.get("gh_build_started_at")),
            "head_branch": str(row.get("git_branch") or "main"),
            "head_repository": {"name": str(row.get("gh_repository_name") or "simulated-repo")},
            "pull_requests": [],
        },
    }


def jenkins_payload(row: pd.Series, index: int) -> dict:
    return {
        "job_name": str(row.get("gh_repository_name") or "simulated-jenkins-job"),
        "build_number": index,
        "result": "SUCCESS" if row.get("tr_status") == "passed" else "FAILURE",
        "duration": 0,
        "timestamp": str(row.get("gh_build_started_at")),
        "branch": str(row.get("git_branch") or "main"),
        "language": row.get("gh_lang"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Send sampled build events to the local CI webhook API.")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--endpoint", choices=["github-actions", "jenkins"], default="github-actions")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if not secret:
        raise SystemExit("Set GITHUB_WEBHOOK_SECRET before simulating CI traffic.")
    if not DATA_PATH.exists():
        raise SystemExit(f"Processed dataset not found: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    sample = df.sample(n=min(args.count, len(df)), random_state=42)
    mapper = github_payload if args.endpoint == "github-actions" else jenkins_payload
    with httpx.Client(base_url="http://localhost:8000", timeout=10) as client:
        for index, (_, row) in enumerate(sample.iterrows(), start=1):
            payload = mapper(row, index)
            if args.endpoint == "github-actions":
                # GitHub signs the exact raw body, so serialize once and submit those
                # same bytes with the standard HMAC-SHA256 header.
                body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
                response = client.post(
                    "/webhooks/github-actions",
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Hub-Signature-256": f"sha256={signature}",
                    },
                )
            else:
                # Jenkins has no provider-standard HMAC header in this project.
                response = client.post(
                    "/webhooks/jenkins",
                    json=payload,
                    headers={"X-Webhook-Secret": secret},
                )
            response.raise_for_status()
            print(f"Sent {index}/{len(sample)}: {response.json()['pipeline_name']}")
            if index < len(sample):
                time.sleep(args.delay)


if __name__ == "__main__":
    main()
