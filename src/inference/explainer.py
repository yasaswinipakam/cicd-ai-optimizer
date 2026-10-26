"""Fast, tree-model SHAP explanations for individual build predictions."""

from __future__ import annotations

import logging

import pandas as pd
try:
    import shap
except ImportError:  # Allows the rest of the API to run until optional SHAP is installed.
    shap = None


logger = logging.getLogger(__name__)


class PredictionExplainer:
    """Produce JSON-safe local feature attributions for tree classifiers."""

    def __init__(self, model, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self.explainer = None
        if shap is None:
            logger.warning("SHAP is not installed; returning an empty explanation.")
        elif model.__class__.__name__ == "SVC":
            # KernelExplainer is too slow for synchronous live API predictions.
            logger.warning("SHAP explanations are unavailable for SVM models; returning an empty explanation.")
        elif model.__class__.__name__ == "StackingClassifier":
            # Extract the XGBoost base estimator for fast, exact SHAP values.
            # StackingClassifier.estimators_ is the list of fitted base estimators
            # in the same order as .estimators (the (name, est) pairs).
            xgb_est = None
            for (name, _), fitted in zip(model.estimators, model.estimators_):
                if "xgb" in name or fitted.__class__.__name__ == "XGBClassifier":
                    xgb_est = fitted
                    break
            if xgb_est is not None:
                self.explainer = shap.TreeExplainer(xgb_est)
                logger.info("SHAP using XGBoost base estimator from StackingClassifier.")
            else:
                logger.warning("No XGBClassifier found in StackingClassifier; SHAP unavailable.")
        else:
            self.explainer = shap.TreeExplainer(model)

    def explain(self, build_features_row: pd.DataFrame, top_n: int = 8) -> list[dict]:
        """Return top local SHAP contributions, sorted by absolute magnitude."""
        if self.explainer is None:
            return []
        values = self.explainer.shap_values(build_features_row)
        # SHAP versions/models vary: binary outputs can be a list or a 3-D array.
        if isinstance(values, list):
            row_values = values[1][0] if len(values) > 1 else values[0][0]
        elif getattr(values, "ndim", 0) == 3:
            row_values = values[0, :, 1] if values.shape[-1] > 1 else values[0, :, 0]
        else:
            row_values = values[0]
        row = build_features_row.iloc[0]
        explanation = [
            {
                "feature": str(feature),
                "value": float(row[feature]),
                "shap_value": float(shap_value),
                "direction": "increases_failure_risk" if float(shap_value) >= 0 else "decreases_failure_risk",
            }
            for feature, shap_value in zip(self.feature_cols, row_values)
        ]
        return sorted(explanation, key=lambda item: abs(item["shap_value"]), reverse=True)[:top_n]
