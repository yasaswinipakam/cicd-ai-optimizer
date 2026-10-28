"""Fill an empty database with the public demo's runs, through the real webhook.

Replays ``data/demo/seed_runs.json`` (built by ``scripts/build_demo_seed.py``)
as signed GitHub ``workflow_run`` events, in time order, against the app
in-process: each run is scored when requested and gets its outcome when it
completes, exactly as a connected repository would.  Commit sizes come from
the seed file instead of the GitHub API.

    DATABASE_URL=sqlite:///./demo.db python scripts/seed_demo.py
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = ROOT / "data" / "demo" / "seed_runs.json"


def main() -> None:
    os.environ["GITHUB_WEBHOOK_SECRET"] = secret = secrets.token_hex(16)
    os.environ.pop("DEMO_READ_ONLY", None)
    from fastapi.testclient import TestClient

    from api.db.database import SessionLocal
    from api.db.models import BuildLog
    from api.main import app
    from api.routers import webhooks

    with SessionLocal() as db:
        if db.query(BuildLog).count():
            print("database already has builds; not seeding")
            return
    events = json.loads(SEED.read_text())
    commits = {event["run"]["head_sha"]: event["commit"] for event in events if event["action"] == "requested"}

    async def seeded_commit(_repository: str, sha: str) -> dict:
        return commits.get(sha, {})

    webhooks._fetch_commit = seeded_commit
    alerts = 0
    with TestClient(app) as client:
        for event in events:
            run = {**event["run"], "status": "completed" if event["action"] == "completed" else "queued"}
            if event["action"] == "completed":
                run["conclusion"] = event["conclusion"]
            body = json.dumps({"action": event["action"], "repository": {"full_name": event["repository"]},
                               "workflow_run": run}).encode()
            signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            response = client.post("/webhooks/github-actions", content=body,
                                   headers={"X-Hub-Signature-256": signature, "X-GitHub-Event": "workflow_run"})
            response.raise_for_status()
            alerts += bool(response.json().get("is_alert"))
    print(f"replayed {len(events)} events for {sum(e['action'] == 'requested' for e in events)} runs; {alerts} alerts")


if __name__ == "__main__":
    main()
