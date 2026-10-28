"""Artifact-backed inference service core for incoming CI/CD builds."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.availability_guard import assert_features_available_for_prediction
from src.models.anomaly import AnomalyDetector
from src.models.clustering import RootCauseClusterer
from src.models.suggestions import get_suggestion
from src.inference.calibration import ProbabilityCalibrator
from src.features.engineer import HISTORY_COLS, STREAK_COLS
from src.features.schema import DIRECT_PREBUILD_FEATURE_COLS
from src.models.tier_e import TierEBlend

TIME_COLS = ("hour", "day_of_week", "month", "is_weekend", "is_night")
# Branches the GitHub Actions training data treats as the default branch.
GHA_DEFAULT_BRANCHES = ("main", "master")


class CICDPredictor:
    """Score one incoming build without refitting any training preprocessing.

    The failure model is either a 39-feature classifier (random forest etc.)
    or the Tier E blend, which also reads stream history and in-flight
    context.  The anomaly detector always uses the 39-feature matrix.
    """

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
        self.anomaly_feature_cols: list[str] = self.preprocessing["feature_cols"]
        assert_features_available_for_prediction(self.anomaly_feature_cols)
        self.is_tier_e = isinstance(self.model, TierEBlend)
        # Which CI provider's inputs the failure model reads: "travis" or "github_actions".
        self.input_schema = self.model.metadata.get("input_schema", "travis") if self.is_tier_e else "travis"
        if self.is_tier_e:
            # The blend carries its own global Platt map and threshold; the
            # random forest's calibrator and threshold do not apply to it.
            self.calibration_method, self.calibrator, self.calibrator_artifact_sha256 = "platt_global", None, None
            prefix = "tier_e_gha" if self.input_schema == "github_actions" else "tier_e"
            self.preprocessing = {**self.preprocessing, "training_mode": f"{prefix}:{self.model.metadata['variant']}"}
            self.feature_cols: list[str] = self.model.feature_cols
            assert_features_available_for_prediction(self.feature_cols)
        else:
            self.feature_cols = self.anomaly_feature_cols
        self._explainer = None  # PredictionExplainer, created on first random-forest explanation

    @property
    def uses_stream_context(self) -> bool:
        """Whether callers should supply Tier E history (see src.features.stream_context)."""
        return self.is_tier_e

    @property
    def decision_threshold(self) -> float:
        """Return the persisted operating threshold, defaulting for legacy artifacts."""
        if self.is_tier_e:
            return float(self.model.metadata["decision_threshold"])
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
        # Absent rates get the training split's failure prior saved with the model.
        # Streak/gap features default to 0.0 (no prior streak known).
        for col in HISTORY_COLS:
            row[col] = self._number(raw.get(col), self.preprocessing["cold_start_prior"])
        for col in STREAK_COLS:
            row[col] = self._number(raw.get(col), 0.0)

        # Reindexing enforces the training schema and supplies defaults if it expands later.
        return pd.DataFrame([row]).reindex(columns=self.anomaly_feature_cols, fill_value=0.0)

    def _prepared_features(self, build_features: dict):
        """Transform a live build into the same imputed matrix used in training."""
        engineered = self._engineer_single_row(build_features)
        values = self.preprocessing["imputer"].transform(engineered)
        return pd.DataFrame(values, columns=self.anomaly_feature_cols)

    @staticmethod
    def _number_or_missing(value) -> float:
        converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float("nan") if pd.isna(converted) else float(converted)

    def _tier_e_frame(self, build_features: dict) -> pd.DataFrame:
        """One Tier E row: absent history stays missing (cold start); absent change inputs get training medians.

        History and in-flight columns come from ``stream_context`` (the API
        derives them from the pipeline's stored builds).  Time features follow
        preprocessing: UTC hour, weekday, month, weekend, night = 00-05h; a
        payload without a start time is scored as starting now.
        """
        raw = dict(build_features)
        timestamp = pd.to_datetime(raw.get("gh_build_started_at"), errors="coerce", utc=True)
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.now(tz="UTC")
        derived = {
            "hour": timestamp.hour, "day_of_week": timestamp.day_of_week, "month": timestamp.month,
            "is_weekend": int(timestamp.day_of_week in (5, 6)), "is_night": int(0 <= timestamp.hour < 6),
        }
        if self.input_schema == "github_actions":
            derived.update(self._gha_derived(raw))
        else:
            derived.update({
                "is_master": int(str(raw.get("git_branch", "")) == "master"),
                "gh_is_pr": int(str(raw.get("gh_is_pr", "")).strip().lower() in ("true", "1", "1.0")),
            })
        fill = getattr(self.model, "input_fill", {})
        row = {}
        for column in self.model.numeric:
            if column in derived and (column not in TIME_COLS or raw.get(column) is None):
                row[column] = float(derived[column])
            else:
                value = self._number_or_missing(raw.get(column))
                # Change/repository inputs were never missing in training: use the training median.
                row[column] = fill[column] if np.isnan(value) and column in fill else value
        for column in self.model.categorical:
            value = derived.get(column, raw.get(column))
            row[column] = None if value is None or pd.isna(value) else str(value)
        return pd.DataFrame([row], columns=self.feature_cols)

    def _gha_derived(self, raw: dict) -> dict:
        """GitHub Actions inputs defined as in ``scripts/evaluate_gha_tier_e.py``.

        A run is ``push`` or ``pull_request``; its commit's size comes from the
        commits API (additions, deletions, files_modified, num_parents).  Only
        values that can be derived are returned; the rest fall back to the
        payload, then to training medians.
        """
        out: dict = {"is_pr": int(str(raw.get("event", "")) == "pull_request"),
                     "is_default_branch": int(str(raw.get("head_branch") or "") in GHA_DEFAULT_BRANCHES)}
        additions, deletions = (self._number_or_missing(raw.get(key)) for key in ("additions", "deletions"))
        if not (np.isnan(additions) or np.isnan(deletions)) and raw.get("total_churn") is None:
            out["total_churn"] = additions + deletions
        parents = self._number_or_missing(raw.get("num_parents"))
        if not np.isnan(parents) and raw.get("is_merge") is None:
            out["is_merge"] = int(parents > 1)
        if raw.get("msg_len") is None and isinstance(raw.get("commit_message"), str):
            out["msg_len"] = len(raw["commit_message"])
        if raw.get("workflow") is None and raw.get("workflow_id") is not None:
            out["workflow"] = str(raw["workflow_id"])
        return out

    def _tier_e_explanation(self, build_features: dict, top_n: int = 8) -> list[dict]:
        """Top SHAP contributions of the blend's binary half, for observed numeric inputs."""
        frame = self._tier_e_frame(build_features)
        contributions = self.model.contributions(frame).iloc[0]
        row = frame.iloc[0]
        items = [
            {
                "feature": feature,
                "value": float(row[feature]),
                "shap_value": float(contributions[feature]),
                "direction": "increases_failure_risk" if contributions[feature] >= 0 else "decreases_failure_risk",
            }
            # Missing history and categorical inputs have no numeric value to show.
            for feature in self.model.numeric if pd.notna(row[feature])
        ]
        return sorted(items, key=lambda item: abs(item["shap_value"]), reverse=True)[:top_n]

    def predict_failure(self, build_features: dict) -> dict:
        """Return failure probability, binary status prediction, and confidence."""
        if self.is_tier_e:
            raw_probability = float(self.model.predict_proba(self._tier_e_frame(build_features))[0, 1])
        else:
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
        if include_explanation and self.is_tier_e:
            explanation = self._tier_e_explanation(build_features)
        elif include_explanation:
            if self._explainer is None:
                # Imported here: SHAP costs ~100 MB and only the random-forest path uses it.
                from src.inference.explainer import PredictionExplainer
                self._explainer = PredictionExplainer(self.model, self.feature_cols)
            explanation = self._explainer.explain(self._prepared_features(build_features))
        return {
            "prediction": prediction,
            "anomaly": anomaly,
            "root_cause": root_cause,
            "optimization_suggestion": optimization_suggestion,
            "explanation": explanation,
        }
