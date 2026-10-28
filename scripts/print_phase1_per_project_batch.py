"""Print a deterministic, resumable batch of per-project online units.

The evaluator itself owns checkpointing, so rerunning a printed batch is safe:
finished project/fold parts are skipped and only missing parts are computed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ELIGIBILITY = ROOT / "reports" / "phase1_v2_travis_per_project_eligibility.csv"


def slug(repo: str) -> str:
    return repo.replace("/", "--")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-index", type=int, required=True, help="Zero-based deterministic batch index.")
    parser.add_argument("--batch-size", type=int, default=5, help="Projects per batch; default keeps long runs manageable.")
    args = parser.parse_args()
    if args.batch_index < 0 or args.batch_size < 1:
        raise SystemExit("batch-index must be >=0 and batch-size must be >=1")
    eligible = pd.read_csv(ELIGIBILITY).sort_values(["builds", "gh_project_name"], ascending=[False, True], kind="stable")
    lo = args.batch_index * args.batch_size
    selected = eligible.iloc[lo : lo + args.batch_size]
    if selected.empty:
        raise SystemExit(f"No projects in batch {args.batch_index}; eligible projects: {len(eligible)}")
    units = [f"travis_project_online_{slug(repo)}_C_rf_ros" for repo in selected.gh_project_name]
    print(",".join(units))


if __name__ == "__main__":
    main()
