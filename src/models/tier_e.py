"""Deployable Tier E blend: the frozen Phase-1 research model as a fitted object.

``fit_predict_lgbm("lgbmblend", ...)`` in ``phase1_rigor_evaluation`` fits and
scores in one call, which suits walk-forward evaluation but cannot be saved.
This class runs the same steps with the same helpers and parameters, keeping
the fitted pieces so live inference can score one build at a time:

1. binary LightGBM and LightGBM ``rank_xendcg`` ranker (repo x month queries);
2. each score standardised with *training*-score moments, averaged 50/50;
3. one global Platt map from the blended training score to a probability.

``train_tier_e`` checks that this reproduces the saved walk-forward fold
predictions before anything is deployed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.phase1_rigor_evaluation import _lgbm_frame, _lgbm_params, _logit, _rank_groups


class TierEBlend:
    """Binary + ranking LightGBM blend with a global probability map."""

    def __init__(self, numeric: list[str], categorical: list[str]):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.metadata: dict = {}
        # Training medians for inputs a live payload may omit (set by train_tier_e).
        self.input_fill: dict[str, float] = {}

    @property
    def feature_cols(self) -> list[str]:
        return self.numeric + self.categorical

    def frame(self, d: pd.DataFrame) -> pd.DataFrame:
        """Model matrix with training categories; unseen categories become missing."""
        return _lgbm_frame(d, self.numeric, self.categorical, self.categories_)

    def fit(self, train: pd.DataFrame) -> "TierEBlend":
        """Fit on build rows with target_failed, gh_project_name and gh_build_started_at."""
        import lightgbm as lgb

        x = _lgbm_frame(train, self.numeric, self.categorical)
        self.categories_ = {c: x[c].cat.categories for c in self.categorical}
        y = train.target_failed
        self.binary_ = lgb.LGBMClassifier(**_lgbm_params(), objective="binary").fit(x, y)
        order, sizes = _rank_groups(train)
        self.ranker_ = lgb.LGBMRanker(**_lgbm_params(), objective="rank_xendcg").fit(
            x.iloc[order], y.to_numpy()[order], group=sizes
        )
        binary, rank = self._raw_scores(x)
        self.binary_moments_ = (float(binary.mean()), float(binary.std()))
        self.rank_moments_ = (float(rank.mean()), float(rank.std()))
        self.platt_ = LogisticRegression(max_iter=500).fit(self._blend(binary, rank).reshape(-1, 1), y)
        assert self.platt_.coef_[0, 0] > 0, "score-to-probability map must be increasing"
        return self

    def _raw_scores(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return _logit(self.binary_.predict_proba(x)[:, 1]), self.ranker_.predict(x)

    def _blend(self, binary: np.ndarray, rank: np.ndarray) -> np.ndarray:
        (bm, bs), (rm, rs) = self.binary_moments_, self.rank_moments_
        # Same operation order as the research blend, so scores match bit for bit.
        return 0.5 * ((rank - rm) / rs) + 0.5 * ((binary - bm) / bs)

    def predict_proba(self, d: pd.DataFrame) -> np.ndarray:
        """Two-column probabilities, like a scikit-learn classifier."""
        p = self.platt_.predict_proba(self._blend(*self._raw_scores(self.frame(d))).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])

    def contributions(self, d: pd.DataFrame) -> pd.DataFrame:
        """Per-feature SHAP contributions to the binary half of the blend (log-odds).

        The ranker has no probability scale, so explanations describe the
        binary model; its direction agrees with the blend for nearly all rows.
        """
        values = self.binary_.booster_.predict(self.frame(d), pred_contrib=True)
        return pd.DataFrame(np.asarray(values)[:, :-1], columns=self.feature_cols, index=d.index)
