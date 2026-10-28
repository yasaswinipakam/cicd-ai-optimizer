"""Live Tier E context must equal the training-time history pass, build by build."""

import numpy as np
import pandas as pd

from src.features.stream_context import CONTEXT_COLS, PriorBuild, stream_context, stream_key
from src.models.phase1_rigor_evaluation import NORM_SOURCE, add_safe_history, add_stream_history


def _synthetic_builds(seed: int = 7, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2015-01-01", tz="UTC") + pd.to_timedelta(np.sort(rng.integers(0, 20 * 24 * 60, n)), unit="m")
    # Long, overlapping runs so many builds start while others are in flight.
    finish = start + pd.to_timedelta(rng.integers(1, 6 * 60, n), unit="m")
    is_pr = rng.random(n) < 0.4
    d = pd.DataFrame({
        "tr_build_id": np.arange(1000, 1000 + n),
        "gh_project_name": rng.choice(["acme/api", "acme/web"], n),
        "gh_build_started_at": start,
        "_finished_at": finish,
        "target_failed": (rng.random(n) < 0.3).astype(int),
        "gh_is_pr": is_pr,
        "gh_pull_req_num": np.where(is_pr, rng.integers(1, 15, n), np.nan),
        "git_branch": rng.choice(["master", "dev", "feature-x"], n),
        "git_trigger_commit": [f"sha{i}" for i in range(n)],
        "job_count": rng.integers(1, 5, n),
    })
    d[NORM_SOURCE] = 0.0
    # Ties on start time exercise the build-id tie order.
    d.loc[10, "gh_build_started_at"] = d.loc[9, "gh_build_started_at"]
    d = d.sort_values(["gh_build_started_at", "tr_build_id"], kind="mergesort").reset_index(drop=True)
    d["stream"] = [stream_key(r.gh_is_pr, r.gh_pull_req_num, r.git_branch) for r in d.itertuples()]
    d["tr_prev_build"] = d.groupby(["gh_project_name", "git_branch"]).tr_build_id.shift()
    return d


def test_live_context_matches_training_history_pass() -> None:
    d = _synthetic_builds()
    batch = add_stream_history(add_safe_history(d))
    checked = 0
    for repo, rows in d.groupby("gh_project_name", sort=False):
        prior: list[PriorBuild] = []
        for i in rows.index:
            row = d.loc[i]
            live = stream_context(
                prior, started_at=row.gh_build_started_at, stream=row.stream,
                previous_build_id=None if pd.isna(row.tr_prev_build) else int(row.tr_prev_build),
            )
            for column in CONTEXT_COLS:
                expected = batch.at[i, column]
                assert (pd.isna(expected) and pd.isna(live[column])) or np.isclose(live[column], expected), (repo, i, column)
            prior.append(PriorBuild(int(row.tr_build_id), row.gh_build_started_at, row.stream,
                                    row._finished_at, bool(row.target_failed)))
            checked += 1
    assert checked == len(d)
    # The synthetic history must actually exercise the in-flight paths.
    assert batch.ctx_stream_pending.eq(1).any() and batch.ctx_prevbuild_pending.eq(1).any()


def test_running_build_counts_as_in_flight_not_history() -> None:
    t = pd.Timestamp("2026-01-01 12:00", tz="UTC")
    prior = [
        PriorBuild(1, t - pd.Timedelta(hours=3), "brmain", t - pd.Timedelta(hours=2), True),
        PriorBuild(2, t - pd.Timedelta(minutes=30), "brmain"),  # still running
    ]
    ctx = stream_context(prior, started_at=t, stream="brmain", previous_build_id=2)
    assert ctx["hist_stream_completed"] == 1 and ctx["hist_stream_prev_failed"] == 1
    assert ctx["ctx_repo_pending"] == 1 and ctx["ctx_stream_pending"] == 1 and ctx["ctx_prevbuild_pending"] == 1
    assert np.isnan(ctx["hist_prevbuild_failed"]) and ctx["ctx_repo_started_1h"] == 1
    assert ctx["ctx_hours_since_stream_start"] == 0.5


def test_stream_key_separates_pull_requests_from_branches() -> None:
    assert stream_key(True, 42, "master") == "pr42.0"
    assert stream_key(True, None, "master") == "brmaster"
    assert stream_key(False, 42, "dev") == "brdev"
