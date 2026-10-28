"""Train and deploy the Tier E blend on GitHub Actions runs for live inference.

The app scores GitHub Actions workflow runs, so the deployed model is trained
on the two GitHub Actions datasets (60 repositories) where the recipe was
tested and confirmed (``reports/phase1_v2_gha_confirmation_results.md``).

It is exactly the combined analysis's walk-forward fold-10 model: trained on
chronological folds 1-9 and evaluated on fold 10, the most recent tenth of
runs.  As in ``train_tier_e`` (the Travis deployment):

* its fold-10 predictions must reproduce the saved research predictions
  (``phase1_v2_gha_combined_parts/primary_0h_fold10_E_lgbmblend``), so the
  deployed code is the evaluated code;
* the alert threshold is the 85th percentile of those fold-10 probabilities,
  a 15% review budget chosen without labels.

The holdout also reports a *new-repository* score: fold 10 rescored with the
workflow category removed.  Every workflow in a repository the model never saw
is an unseen category, so this is what a newly connected repository gets.

Usage: ``python -m src.models.train_tier_e_gha`` (about 20 minutes).  Needs
both raw GHA CSVs.  Writes ``artifacts/tier_e_gha_blend.joblib`` and points
``artifacts/best_model.txt`` at it; ``train_tier_e`` points it back at Travis.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_gha_tier_e as gha  # noqa: E402
from compare_phase1_walkforward_tiers import per_repository_metrics  # noqa: E402
from src.features.stream_context import CONTEXT_COLS  # noqa: E402
from src.models.tier_e import TierEBlend  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
RAW = [ROOT / "data" / "raw" / "final_research_dataset_MASTER.csv",
       ROOT / "data" / "raw" / "phase2_gha_confirm" / "gha_confirmation_dataset.csv"]
SAVED = ROOT / "reports" / "phase1_v2_gha_combined_parts" / "primary_0h_fold10_E_lgbmblend.parquet"
MODEL_NAME = "tier_e_gha_blend"
ALERT_BUDGET = 0.15


def _holdout_metrics(test: pd.DataFrame, p: np.ndarray, threshold: float) -> dict:
    y = test.target_failed.to_numpy()
    per = per_repository_metrics(pd.DataFrame({"repo": test.gh_project_name.to_numpy(), "label": y, "probability": p}))
    alert = p >= threshold
    return {
        "macro_within_repo_auc": float(per.within_repo_auc.mean()),
        "within_repo_auc_repositories": int(per.within_repo_auc.notna().sum()),
        "within_repo_recall_at_15": float(per.within_repo_recall_at_15.mean()),
        "pooled_auc": float(roc_auc_score(y, p)),
        "prevalence": float(y.mean()),
        "alert_rate": float(alert.mean()),
        "precision_at_threshold": float(y[alert].mean()),
        "recall_at_threshold": float(y[alert].sum() / y.sum()),
    }


def main() -> None:
    gha.configure("gha_combined", RAW, gha.REPORTS / "phase1_v2_gha_confirmation_protocol.md")
    runs, counts = gha.load_runs("primary")
    data = gha.with_history(runs, 0)
    folds = gha.chronological_folds(data.gh_build_started_at)
    train, test = data.iloc[np.concatenate(folds[:9])], data.iloc[folds[9]]
    gha._assert_split(train, test)
    numeric, categorical = gha.FEATURES["E"]
    model = TierEBlend(numeric, categorical).fit(train)
    p = model.predict_proba(test)[:, 1]
    # Inputs a live payload may omit get their training median; history and
    # in-flight columns stay missing at cold start, as in training.
    model.input_fill = {c: float(v) for c, v in train[[c for c in numeric if c not in CONTEXT_COLS]].median().items()}

    saved = pd.read_parquet(SAVED)
    if not np.array_equal(saved.build_id.to_numpy(), test.tr_build_id.to_numpy()):
        raise AssertionError("fold-10 rows differ from the saved research predictions")
    max_diff = float(np.max(np.abs(saved.probability.to_numpy() - p)))
    if max_diff > 1e-6:
        raise AssertionError(f"deployed model does not reproduce research predictions (max diff {max_diff:.3g})")

    threshold = float(np.quantile(p, 1 - ALERT_BUDGET))
    unseen = model.predict_proba(test.assign(**{c: None for c in categorical}))[:, 1]
    model.metadata = {
        "model_name": MODEL_NAME,
        "variant": "E_lgbmblend",
        "input_schema": "github_actions",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data": [str(path.relative_to(ROOT)) for path in RAW],
        "repositories": int(data.gh_project_name.nunique()),
        "train_builds": len(train),
        "train_through": train.gh_build_started_at.max().isoformat(),
        "holdout_builds": len(test),
        "holdout_window": [test.gh_build_started_at.min().isoformat(), test.gh_build_started_at.max().isoformat()],
        "reproduces_research_fold10_max_abs_diff": max_diff,
        "decision_threshold": threshold,
        "threshold_policy": f"{ALERT_BUDGET:.0%} alert budget: {1 - ALERT_BUDGET:.0%} quantile of fold-10 probabilities (no labels used)",
        "holdout_metrics": _holdout_metrics(test, p, threshold),
        "new_repository_holdout_metrics": _holdout_metrics(test, unseen, threshold),
        "finish_buffer_hours": 0,
        "stream_definition": "pull_request runs: (repository, actor:head_branch); push runs: (repository, branch)",
        "previous_build_definition": "previous run of the same workflow on the same stream",
        "data_counts": counts,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / f"{MODEL_NAME}.joblib")
    (ARTIFACTS / "best_model.txt").write_text(f"{MODEL_NAME}\n", encoding="utf-8")
    for key in ("holdout_metrics", "new_repository_holdout_metrics"):
        print(key, pd.Series(model.metadata[key]).round(4).to_dict())
    print(f"threshold {threshold:.4f}; reproduces research fold 10 (max diff {max_diff:.2g})")


if __name__ == "__main__":
    main()
