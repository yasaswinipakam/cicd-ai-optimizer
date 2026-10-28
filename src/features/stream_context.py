"""Live Tier E history and in-flight context for one incoming build.

Training builds these columns in one chronological pass per repository
(``add_safe_history`` and ``add_stream_history`` in
``phase1_rigor_evaluation``).  This module computes the same values for one new
build from the builds a pipeline has already seen, with the same rules:

* an outcome counts only if that build finished strictly before the new build
  started, and completed outcomes are ordered by (finish, start, build id);
* a stream is a pull request (``pr<number>``) or, for push builds, a branch
  (``br<name>``);
* in-flight fields use only the existence of earlier starts, never outcomes.

``tests/test_stream_context.py`` checks it against the training pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from src.models.phase1_rigor_evaluation import EWMA_ALPHA

HOUR_NS = 3_600_000_000_000
CONTEXT_COLS = [
    "hist_prev_failed", "hist_repo_fail_rate_5", "hist_repo_fail_rate_all",
    "hist_stream_prev_failed", "hist_stream_fail_rate_5", "hist_stream_fail_rate_all", "hist_stream_completed",
    "hist_stream_streak", "hist_prevbuild_failed", "hist_repo_ewma",
    "ctx_prevbuild_pending", "ctx_stream_pending", "ctx_repo_pending", "ctx_repo_started_1h",
    "ctx_hours_since_stream_start",
]


@dataclass(frozen=True)
class PriorBuild:
    """A build the pipeline saw before the one being scored."""

    build_id: int
    started_at: datetime
    stream: str
    finished_at: datetime | None = None  # when its outcome became known; None while running
    failed: bool | None = None


def stream_key(is_pr, pr_number, branch) -> str:
    """Training's stream key: the PR for PR builds with a number, else the branch."""
    has_number = pr_number is not None and not pd.isna(pr_number) and str(pr_number).strip() != ""
    if bool(is_pr) and has_number:
        number = pd.to_numeric(pr_number, errors="coerce")
        return f"pr{float(number)}" if pd.notna(number) else f"pr{pr_number}"
    return f"br{'' if branch is None or pd.isna(branch) else branch}"


def gha_stream_key(event, actor, branch) -> str:
    """GitHub Actions stream key, as in ``scripts/evaluate_gha_tier_e.py``.

    Runs carry no PR number, so a pull request is identified by its author and
    head branch; a push run's stream is its branch.
    """
    branch = "" if branch is None or pd.isna(branch) else str(branch)
    if str(event) == "pull_request":
        return f"pr{'' if actor is None or pd.isna(actor) else actor}:{branch}"
    return f"br{branch}"


def _ns(value) -> int:
    return pd.Timestamp(value).tz_localize("UTC").value if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).value


def stream_context(prior: Sequence[PriorBuild], *, started_at, stream: str,
                   previous_build_id: int | None = None) -> dict[str, float]:
    """Tier E history/in-flight features for a build starting at ``started_at``.

    ``prior`` holds every earlier build of the same pipeline (earlier start, or
    the same start but created first).  ``previous_build_id`` is Travis's
    ``tr_prev_build``; live callers pass the latest earlier build on the same
    branch, which is how Travis defines it.
    """
    t0 = _ns(started_at)
    out = {c: np.nan for c in CONTEXT_COLS}
    done = sorted(
        (b for b in prior if b.finished_at is not None and b.failed is not None and _ns(b.finished_at) < t0),
        key=lambda b: (_ns(b.finished_at), _ns(b.started_at), b.build_id),
    )
    labels = [int(b.failed) for b in done]
    if labels:
        out["hist_prev_failed"] = labels[-1]
        out["hist_repo_fail_rate_5"] = float(np.mean(labels[-5:]))
        out["hist_repo_fail_rate_all"] = sum(labels) / len(labels)
        ewma = float(labels[0])
        for label in labels[1:]:
            ewma = EWMA_ALPHA * label + (1 - EWMA_ALPHA) * ewma
        out["hist_repo_ewma"] = ewma
    history = [int(b.failed) for b in done if b.stream == stream]
    out["hist_stream_completed"] = len(history)
    if history:
        out["hist_stream_prev_failed"] = history[-1]
        out["hist_stream_fail_rate_5"] = float(np.mean(history[-5:]))
        out["hist_stream_fail_rate_all"] = float(np.mean(history))
        streak = 0
        for label in history:
            streak = streak + 1 if label else 0
        out["hist_stream_streak"] = streak
    done_ids = {b.build_id: int(b.failed) for b in done}
    if previous_build_id is not None:
        if previous_build_id in done_ids:
            out["hist_prevbuild_failed"] = done_ids[previous_build_id]; out["ctx_prevbuild_pending"] = 0
        elif any(b.build_id == previous_build_id for b in prior):
            out["ctx_prevbuild_pending"] = 1
    in_stream = [b for b in prior if b.stream == stream]
    if in_stream:
        last = max(in_stream, key=lambda b: (_ns(b.started_at), b.build_id))
        if _ns(last.started_at) < t0:
            out["ctx_hours_since_stream_start"] = (t0 - _ns(last.started_at)) / 3.6e12
            out["ctx_stream_pending"] = 0 if last.build_id in done_ids else 1
    out["ctx_repo_pending"] = len(prior) - len(done)
    out["ctx_repo_started_1h"] = sum(_ns(b.started_at) >= t0 - HOUR_NS for b in prior)
    return out
