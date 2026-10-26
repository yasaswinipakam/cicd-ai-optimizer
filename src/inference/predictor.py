"""Artifact-backed inference service core for incoming CI/CD builds."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import pandas as pd

from src.features.availability_guard import assert_features_available_for_prediction
from src.models.anomaly import AnomalyDetector
from src.models.clustering import RootCauseClusterer
from src.models.suggestions import get_suggestion
from src.inference.explainer import PredictionExplainer
from src.inference.calibration import ProbabilityCalibrator
from src.features.engineer import COLD_START_RATE, HISTORY_COLS, STREAK_COLS
from src.features.schema import DIRECT_PREBUILD_FEATURE_COLS


class CICDPredictor:
    """Score one incoming build without refitting any training preprocessing."""

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        best_model_path = self.artifacts_dir / "best_model.txt"
        if not best_model_path.exists():
            raise FileNotFoundError(f"Best-model marker not found: {best_model_path}")
        self.best_model_name = best_model_path.read_text(encoding="utf-8").strip()
        self.model_path = self.artifacts_dir / f"{self.best_model_name}.joblib"
        self.model = joblib.load(self.model_path)
        self.model_artifact_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self.preprocessing = joblib.load(self.artifacts_dir / "preprocessing.joblib")
        calibration_metadata = self.preprocessing.get("calibration", {})
        calibration_path = self.artifacts_dir / calibration_metadata.get(
            "artifact", "probability_calibrator.joblib"
        )
        self.calibration_method = calibration_metadata.get("method", "raw")
        self.calibrator: ProbabilityCalibrator | None = (
            joblib.load(calibration_path) if self.calibration_method != "raw" and calibration_path.exists() else None
        )
        self.calibrator_artifact_sha256 = (
            hashlib.sha256(calibration_path.read_bytes()).hexdigest() if self.calibrator is not None else None
        )
        self.encoders = joblib.load(self.artifacts_dir / "encoders.joblib")
        self.anomaly_detector = AnomalyDetector.load(str(self.artifacts_dir / "anomaly_detector.joblib"))
        self.root_cause_clusterer = RootCauseClusterer.load(
            str(self.artifacts_dir / "root_cause_clusterer.joblib")
        )
        self.feature_cols: list[str] = self.preprocessing["feature_cols"]
        assert_features_available_for_prediction(self.feature_cols)
        self._explainer: PredictionExplainer | None = None

    @property
    def decision_threshold(self) -> float:
        """Return the persisted operating threshold, defaulting for legacy artifacts."""
        return float(self.preprocessing.get("decision_threshold", 0.5))

    @staticmethod
    def _number(value, default: float = 0.0) -> float:
        """Coerce API values to numbers, using a sensible zero default."""
        converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return default if pd.isna(converted) else float(converted)

    def _first_number(self, raw: dict, *keys: str) -> float:
        """Read the first provided numeric alias from a sparse API payload."""
        for key in keys:
            if key in raw and raw[key] is not None:
                return self._number(raw[key])
        return 0.0

    def _total_files_changed(self, raw: dict) -> float:
        """Mirror the canonical TravisTorrent changed-file calculation."""
        canonical = (
            "gh_diff_files_added",
            "gh_diff_files_deleted",
            "gh_diff_files_modified",
        )
        if any(key in raw for key in canonical):
            return sum(self._number(raw.get(key)) for key in canonical)
        # Compatibility only for older API clients; training data uses gh_diff_*.
        return self._first_number(raw, "git_diff_src_files") + self._first_number(
            raw, "git_diff_test_files"
        )

    @staticmethod
    def _safe_encode(value, encoder) -> int:
        """Encode known classes and use a stable default for unseen categories."""
        normalized = "__missing__" if value is None or pd.isna(value) else str(value)
        try:
            return int(encoder.transform([normalized])[0])
        except ValueError:
            fallback = "__missing__" if "__missing__" in encoder.classes_ else encoder.classes_[0]
            return int(encoder.transform([fallback])[0])

    def _engineer_single_row(self, build_features: dict) -> pd.DataFrame:
        """Apply Task 3's feature formulas safely to a sparse live API payload."""
        raw = dict(build_features)
        timestamp = pd.to_datetime(raw.get("gh_build_started_at"), errors="coerce", utc=True)
        hour = self._number(raw.get("hour"), timestamp.hour if not pd.isna(timestamp) else 0)
        day_of_week = self._number(
            raw.get("day_of_week"), timestamp.dayofweek if not pd.isna(timestamp) else 0
        )
        month = self._number(raw.get("month"), timestamp.month if not pd.isna(timestamp) else 0)
        is_weekend = self._number(raw.get("is_weekend"), int(day_of_week in (5, 6)))
        is_night = self._number(raw.get("is_night"), int(0 <= hour < 6))

        source_churn = self._number(raw.get("git_diff_src_churn"))
        test_churn = self._number(raw.get("git_diff_test_churn"))
        row = {
            **{column: self._number(raw.get(column)) for column in DIRECT_PREBUILD_FEATURE_COLS},
            "prev_status_enc": self._safe_encode(
                raw.get("git_prev_commit_resolution_status"), self.encoders["prev_status_enc"]
            ),
            "lang_enc": self._safe_encode(raw.get("gh_lang"), self.encoders["lang_enc"]),
            "is_master": int(str(raw.get("git_branch", "")).lower() == "master"),
            "by_core": int(self._number(raw.get("gh_by_core_team_member"))),
            "is_pr": int(self._number(raw.get("gh_is_pr"))),
            "test_to_src_ratio": test_churn / source_churn if source_churn else 0.0,
            "total_files_changed": self._total_files_changed(raw),
            "total_test_changes": test_churn,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "is_night": is_night,
        }
        # Rolling history features — caller supplies these from the build-history DB.
        # Rate features default to COLD_START_RATE (global failure prior) when absent.
        # Streak/gap features default to 0.0 (no prior streak known).
        for col in HISTORY_COLS:
            row[col] = self._number(raw.get(col), COLD_START_RATE)
        for col in STREAK_COLS:
            row[col] = self._number(raw.get(col), 0.0)

        # Reindexing enforces the training schema and supplies defaults if it expands later.
        return pd.DataFrame([row]).reindex(columns=self.feature_cols, fill_value=0.0)

    def _prepared_features(self, build_features: dict):
        """Transform a live build into the same imputed matrix used in training."""
        engineered = self._engineer_single_row(build_features)
        values = self.preprocessing["imputer"].transform(engineered)
        return pd.DataFrame(values, columns=self.feature_cols)

    def predict_failure(self, build_features: dict) -> dict:
        """Return failure probability, binary status prediction, and confidence."""
        matrix = self._prepared_features(build_features)
        model_input = self.preprocessing["scaler"].transform(matrix) if self.best_model_name == "svm" else matrix
        raw_probability = float(self.model.predict_proba(model_input)[0, 1])
        probability = float(self.calibrator.transform([raw_probability])[0]) if self.calibrator else raw_probability
        predicted_status = "failed" if probability >= self.decision_threshold else "passed"
        return {
            "failure_probability": probability,
            "predicted_status": predicted_status,
            "confidence": float(max(probability, 1 - probability)),
        }

    def detect_anomaly(self, build_features: dict) -> dict:
        """Return unusual pre-build input-profile status, not failure risk."""
        matrix = self._prepared_features(build_features)
        return {
            "is_anomaly": bool(self.anomaly_detector.predict(matrix)[0] == -1),
            "anomaly_score": float(self.anomaly_detector.anomaly_scores(matrix)[0]),
        }

    def analyze_root_cause(self, failed_test_log: str | None) -> dict | None:
        """Assign a non-empty failed-test log to a human-readable cluster.

        Root-cause analysis is post-failure diagnostics, not a fallback label for
        every prediction. A missing or whitespace-only failed-test log therefore
        has no available root cause rather than being placed in a synthetic
        "unknown" bucket.
        """
        if failed_test_log is None or not str(failed_test_log).strip():
            return None
        cluster_id = int(self.root_cause_clusterer.predict(pd.Series([failed_test_log]))[0])
        terms = self.root_cause_clusterer.top_terms_per_cluster().get(cluster_id, ["unknown failure"])
        return {"cluster_id": cluster_id, "likely_cause": ", ".join(terms[:3])}

    def full_analysis(
        self, build_features: dict, failed_test_log: str | None = None, include_explanation: bool = False
    ) -> dict:
        """Combine prediction, anomaly detection, and optional post-failure diagnosis."""
        prediction = self.predict_failure(build_features)
        anomaly = self.detect_anomaly(build_features)
        root_cause = self.analyze_root_cause(failed_test_log)
        actual_failed = str(build_features.get("tr_status", "")).lower() == "failed"
        if root_cause is not None and (prediction["predicted_status"] == "failed" or actual_failed):
            top_terms = self.root_cause_clusterer.top_terms_per_cluster().get(
                root_cause["cluster_id"], ["unknown failure"]
            )
            optimization_suggestion = get_suggestion(
                top_terms,
                is_anomaly=anomaly["is_anomaly"],
                anomaly_score=anomaly["anomaly_score"],
            )
        else:
            optimization_suggestion = None
        explanation = []
        if include_explanation:
            if self._explainer is None:
                self._explainer = PredictionExplainer(self.model, self.feature_cols)
            explanation = self._explainer.explain(self._prepared_features(build_features))
        return {
            "prediction": prediction,
            "anomaly": anomaly,
            "root_cause": root_cause,
            "optimization_suggestion": optimization_suggestion,
            "explanation": explanation,
        }
