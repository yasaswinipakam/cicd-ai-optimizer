"""Build the public demo's seed data from real GitHub Actions runs.

Takes the first 150 first-attempt push/pull_request runs of three public
repositories from the deployed model's holdout window (runs it was never
trained on), and writes them as a time-ordered list of webhook events to
``data/demo/seed_runs.json``.  ``scripts/seed_demo.py`` replays them.

PR authors and non-default branch names are replaced with stable pseudonyms;
repository names, workflows, commit sizes and outcomes are kept.

    python scripts/build_demo_seed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the model artifact unpickles src.models.tier_e
SOURCE = ROOT / "data" / "raw" / "phase2_gha_confirm" / "gha_confirmation_dataset.csv"
MODEL = ROOT / "artifacts" / "tier_e_gha_blend.joblib"
OUT = ROOT / "data" / "demo" / "seed_runs.json"
REPOSITORIES = ("twbs/bootstrap", "freeCodeCamp/freeCodeCamp", "yt-dlp/yt-dlp")
RUNS_PER_REPOSITORY = 150
DEFAULT_BRANCHES = ("main", "master")


def main() -> None:
    holdout_start = pd.Timestamp(joblib.load(MODEL).metadata["holdout_window"][0])
    d = pd.read_csv(SOURCE, low_memory=False)
    for column in ("created_at", "updated_at", "committed_date"):
        d[column] = pd.to_datetime(d[column], utc=True, format="ISO8601")
    d = d[d.repo.isin(REPOSITORIES) & d.run_attempt.eq(1) & d.event.isin(["push", "pull_request"])
          & d.conclusion.isin(["success", "failure", "startup_failure"]) & (d.created_at >= holdout_start)]
    d = d.sort_values(["created_at", "run_id"]).groupby("repo").head(RUNS_PER_REPOSITORY)

    aliases: dict[tuple[str, str], str] = {}
    def alias(repo: str, kind: str, value) -> str:
        key = (repo, f"{kind}:{value}")
        return aliases.setdefault(key, f"{kind}-{sum(k[0] == repo and k[1].startswith(kind) for k in aliases) + 1}")

    events = []
    for run in d.itertuples():
        branch = run.head_branch if run.head_branch in DEFAULT_BRANCHES else alias(run.repo, "branch", run.head_branch)
        payload = {
            "id": int(run.run_id), "workflow_id": int(run.workflow_id), "name": "CI", "event": run.event,
            "head_branch": branch, "head_sha": run.head_sha, "run_attempt": 1,
            "actor": {"login": alias(run.repo, "dev", run.actor_login)},
        }
        # What the API would fetch from the commits API (see api/routers/webhooks._fetch_commit).
        stats = {"additions": int(run.additions), "deletions": int(run.deletions),
                 "files_modified": int(run.files_modified), "num_parents": int(run.num_parents),
                 "msg_len": int(run.msg_len), "committed_at": run.committed_date.isoformat()}
        events.append((run.created_at, 0, {"repository": run.repo, "action": "requested", "run": payload, "commit": stats}))
        events.append((run.updated_at, 1, {"repository": run.repo, "action": "completed", "run": payload,
                                           "conclusion": run.conclusion}))
    events.sort(key=lambda item: (item[0], item[1], item[2]["run"]["id"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps([event for _, _, event in events], separators=(",", ":")) + "\n")
    print(f"{len(d)} runs, {len(events)} events -> {OUT.relative_to(ROOT)}")
    print(d.groupby("repo").conclusion.apply(lambda c: round(c.ne("success").mean(), 3)).to_string())


if __name__ == "__main__":
    main()
