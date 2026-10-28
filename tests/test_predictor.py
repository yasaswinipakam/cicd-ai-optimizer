"""Tests for artifact-backed, sparse-payload CI/CD inference."""

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.features.schema import FEATURE_COLS
from src.inference.predictor import CICDPredictor
from src.models.anomaly import AnomalyDetector
from src.models.clustering import RootCauseClusterer


def test_full_analysis_handles_minimal_and_unseen_live_payload(tmp_path) -> None:
    matrix = pd.DataFrame(
        [[0.0] * len(FEATURE_COLS), [1.0] * len(FEATURE_COLS)], columns=FEATURE_COLS
    )
    imputer = SimpleImputer(strategy="median").fit(matrix)
    scaler = StandardScaler().fit(matrix)
    model = DummyClassifier(strategy="constant", constant=1).fit(matrix, [0, 1])
    joblib.dump(model, tmp_path / "random_forest.joblib")
    joblib.dump(
        {"imputer": imputer, "scaler": scaler, "feature_cols": FEATURE_COLS.copy(), "cold_start_prior": 0.2},
        tmp_path / "preprocessing.joblib",
    )
    joblib.dump(
        {
            "prev_status_enc": LabelEncoder().fit(["__missing__", "passed"]),
            "lang_enc": LabelEncoder().fit(["__missing__", "python"]),
        },
        tmp_path / "encoders.joblib",
    )
    AnomalyDetector(contamination=0.5).fit(imputer.transform(matrix)).save(
        str(tmp_path / "anomaly_detector.joblib")
    )
    RootCauseClusterer(n_clusters=2).fit(
        pd.Series(["database connection timeout", "database connection refused"])
    ).save(str(tmp_path / "root_cause_clusterer.joblib"))
    (tmp_path / "best_model.txt").write_text("random_forest\n", encoding="utf-8")

    predictor = CICDPredictor(str(tmp_path))
    canonical_counts = predictor._engineer_single_row(
        {
            "gh_diff_files_added": 34,
            "gh_diff_files_deleted": 6,
            "gh_diff_files_modified": 137,
            "gh_diff_src_files": 99,
            "gh_diff_tests_added": 88,
        }
    )
    assert canonical_counts.loc[0, "total_files_changed"] == 177.0
    for column in ("repo_fail_rate_last5", "repo_fail_rate_last20", "repo_prior_fail_rate"):
        assert canonical_counts.loc[0, column] == 0.2
    supplied = predictor._engineer_single_row({"repo_fail_rate_last5": 0.6})
    assert supplied.loc[0, "repo_fail_rate_last5"] == 0.6
    result = predictor.full_analysis(
        {"git_diff_src_churn": 50, "gh_lang": "brainfuck", "gh_is_pr": False},
        "database connection timeout",
    )

    assert set(result) == {"prediction", "anomaly", "root_cause", "optimization_suggestion", "explanation"}
    assert set(result["prediction"]) == {"failure_probability", "predicted_status", "confidence"}
    assert 0 <= result["prediction"]["failure_probability"] <= 1
    assert isinstance(result["anomaly"]["is_anomaly"], bool)
    assert isinstance(result["root_cause"]["cluster_id"], int)
    assert result["optimization_suggestion"]["category"] == "Flaky/Timing"

    for missing_log in (None, "", "   "):
        unavailable = predictor.full_analysis({"gh_lang": "python"}, missing_log)
        assert unavailable["root_cause"] is None
        assert unavailable["optimization_suggestion"] is None

    predictor.model = DummyClassifier(strategy="constant", constant=0).fit(matrix, [0, 1])
    passed_result = predictor.full_analysis({"gh_lang": "python"}, "database connection timeout")
    assert passed_result["prediction"]["predicted_status"] == "passed"
    assert isinstance(passed_result["root_cause"]["cluster_id"], int)
    assert passed_result["optimization_suggestion"] is None
