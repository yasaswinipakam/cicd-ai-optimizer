"""Per-repository review budget: when a scored build becomes an alert.

The model was evaluated by within-repository recall at a 15% review budget:
each repository's riskiest 15% of builds.  A single global threshold does not
implement that, because repositories differ in how risky their builds look.
Live, a build is an alert when its risk reaches the 85th percentile of its
repository's most recent scored builds.

Until a repository has ``MIN_HISTORY`` scored builds, it gets the *typical
repository* threshold saved with the model: the median, over holdout
repositories, of each one's own 85th-percentile risk.  Neither threshold uses
outcome labels.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REVIEW_BUDGET = 0.15
WINDOW = 100  # most recent scored builds of the repository
MIN_HISTORY = 20


def repository_threshold(recent: Iterable[float], fallback: float) -> tuple[float, str]:
    """Alert threshold for the next build, from the repository's earlier risk scores (oldest first).

    Returns the threshold and its basis: ``"repository_budget"``, or
    ``"typical_repository"`` while the repository has too few scored builds.
    """
    scores = [float(p) for p in recent if p is not None and np.isfinite(p)][-WINDOW:]
    if len(scores) < MIN_HISTORY:
        return float(fallback), "typical_repository"
    return float(np.quantile(scores, 1 - REVIEW_BUDGET)), "repository_budget"


def typical_repository_threshold(predictions: pd.DataFrame) -> float:
    """Median over repositories of each one's (1 - budget) risk quantile; needs ``repo`` and ``probability``."""
    return float(predictions.groupby("repo").probability.quantile(1 - REVIEW_BUDGET).median())


def simulate(predictions: pd.DataFrame, fallback: float) -> pd.Series:
    """Alerts the policy would raise on time-ordered ``repo``/``probability`` rows, as live."""
    alerts = pd.Series(False, index=predictions.index)
    for _, rows in predictions.groupby("repo", sort=False):
        earlier: list[float] = []
        for index, probability in rows.probability.items():
            threshold, _ = repository_threshold(earlier, fallback)
            alerts.at[index] = probability >= threshold
            earlier.append(probability)
    return alerts
