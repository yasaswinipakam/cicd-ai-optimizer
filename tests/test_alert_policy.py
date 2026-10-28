"""Per-repository review budget."""

import numpy as np
import pandas as pd

from src.inference.alert_policy import MIN_HISTORY, WINDOW, repository_threshold, simulate, typical_repository_threshold


def test_threshold_uses_recent_scores_once_there_are_enough() -> None:
    assert repository_threshold([0.9] * (MIN_HISTORY - 1), fallback=0.5) == (0.5, "typical_repository")
    scores = list(np.linspace(0, 1, MIN_HISTORY + 1))
    threshold, basis = repository_threshold(scores, fallback=0.5)
    assert basis == "repository_budget" and threshold == np.quantile(scores, 0.85)
    # Only the most recent WINDOW scores count; missing scores are skipped.
    old_then_new = [1.0] * 50 + [0.1] * WINDOW + [float("nan")]
    assert repository_threshold(old_then_new, fallback=0.5)[0] == 0.1


def test_simulation_gives_every_repository_its_budget() -> None:
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"repo": np.repeat(["low", "high"], 500),
                         "probability": np.concatenate([rng.uniform(0, 0.2, 500), rng.uniform(0.6, 1, 500)])})
    fallback = typical_repository_threshold(rows)
    alerts = simulate(rows, fallback)
    rates = alerts.groupby(rows.repo).mean()
    # A global threshold would flag nearly all of "high" and none of "low".
    assert rates.between(0.10, 0.22).all()
