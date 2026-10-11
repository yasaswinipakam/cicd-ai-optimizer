"""JSON-safe SHAP explanation tests."""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.inference.explainer import PredictionExplainer

pytest.importorskip("shap")


def test_tree_explainer_returns_plain_float_attributions():
    features = pd.DataFrame({"churn": [0, 1, 2, 100, 120, 150], "files": [0, 1, 1, 10, 12, 15]})
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(features, [0, 0, 0, 1, 1, 1])
    explanation = PredictionExplainer(model, list(features.columns)).explain(features.iloc[[5]], top_n=1)
    assert len(explanation) <= 1
    assert isinstance(explanation[0]["shap_value"], float)
    assert isinstance(explanation[0]["value"], float)
