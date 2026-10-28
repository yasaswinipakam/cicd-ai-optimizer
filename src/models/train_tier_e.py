"""Train and deploy the Tier E blend (no job-count features) for live inference.

The deployed model is exactly the walk-forward fold-10 model: trained on
chronological folds 1-9 and evaluated on fold 10, the most recent tenth of
builds.  Keeping the held-out fold means the deployed artifact carries its own
out-of-sample numbers and a threshold chosen without labels:

* its fold-10 predictions must reproduce the saved research predictions
  (``travis_walk_10_Enojobs_lgbmblend``), so the deployed code is the
  evaluated code;
* the alert threshold is the 85th percentile of those fold-10 probabilities,
  i.e. a 15% review budget, matching the evaluated recall@15% operating point.

Job-count features are left out because most CI systems do not report a
build-matrix size before the build starts; the evaluated cost is 0.003
within-repo AUC (0.695 vs 0.698, folds 7-10).

Usage: ``python -m src.models.train_tier_e``.  Writes
``artifacts/tier_e_blend.joblib`` and points ``artifacts/best_model.txt`` at it.
The random-forest artifacts stay in place: the anomaly detector and the
what-if derived features still use the 39-feature matrix.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features.stream_context import CONTEXT_COLS
from src.models.phase1_rigor_evaluation import _walk_data, _within_repo_auc, tier_features
from src.models.tier_e import TierEBlend

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
SAVED = ROOT / "reports" / "phase1_v2_parts" / "travis_walk_10_Enojobs_lgbmblend__finish_buffer_hours_0.parquet"
MODEL_NAME = "tier_e_blend"
ALERT_BUDGET = 0.15


def main() -> None:
    data, meta = _walk_data(0)
    folds = np.array_split(np.arange(len(data)), 10)
    train, test = data.iloc[np.concatenate(folds[:9])], data.iloc[folds[9]]
    numeric, categorical = tier_features("Enojobs")
    model = TierEBlend(numeric, categorical).fit(train)
    p = model.predict_proba(test)[:, 1]
    # Change and repository inputs are never missing in training, so a live
    # payload that omits one is filled with its training median (as the
    # random-forest path imputes).  History/in-flight columns stay missing at
    # cold start, which training saw and the model handles natively.
    direct = [c for c in numeric if c not in CONTEXT_COLS]
    if train[direct].isna().any().any():
        raise AssertionError("direct Tier E inputs unexpectedly missing in training")
    model.input_fill = {c: float(v) for c, v in train[direct].median().items()}

    saved = pd.read_parquet(SAVED)
    if not np.array_equal(saved.build_id.to_numpy(), test.tr_build_id.to_numpy()):
        raise AssertionError("fold-10 rows differ from the saved research predictions")
    max_diff = float(np.max(np.abs(saved.probability.to_numpy() - p)))
    if max_diff > 1e-6:
        raise AssertionError(f"deployed model does not reproduce research predictions (max diff {max_diff:.3g})")

    y = test.target_failed.to_numpy()
    threshold = float(np.quantile(p, 1 - ALERT_BUDGET))
    alert = p >= threshold
    pred = pd.DataFrame({"repo": test.gh_project_name.to_numpy(), "label": y, "probability": p})
    macro, _, repos = _within_repo_auc(pred)
    model.metadata = {
        "model_name": MODEL_NAME,
        "variant": "Enojobs_lgbmblend",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_builds": len(train),
        "train_through": train.gh_build_started_at.max().isoformat(),
        "holdout_builds": len(test),
        "holdout_window": [test.gh_build_started_at.min().isoformat(), test.gh_build_started_at.max().isoformat()],
        "reproduces_research_fold10_max_abs_diff": max_diff,
        "decision_threshold": threshold,
        "threshold_policy": f"{ALERT_BUDGET:.0%} alert budget: {1 - ALERT_BUDGET:.0%} quantile of fold-10 probabilities (no labels used)",
        "holdout_metrics": {
            "macro_within_repo_auc": macro,
            "within_repo_auc_repositories": repos,
            "pooled_auc": float(roc_auc_score(y, p)),
            "prevalence": float(y.mean()),
            "alert_rate": float(alert.mean()),
            "precision_at_threshold": float(y[alert].mean()),
            "recall_at_threshold": float(y[alert].sum() / y.sum()),
        },
        "finish_buffer_hours": 0,
        "stream_definition": "PR builds: (pipeline, PR number); push builds: (pipeline, branch)",
        "job_rows": meta["job_rows"],
    }
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / f"{MODEL_NAME}.joblib")
    (ARTIFACTS / "best_model.txt").write_text(f"{MODEL_NAME}\n", encoding="utf-8")
    print(pd.Series(model.metadata["holdout_metrics"]).round(4).to_string())
    print(f"threshold {threshold:.4f}; reproduces research fold 10 (max diff {max_diff:.2g})")


if __name__ == "__main__":
    main()
