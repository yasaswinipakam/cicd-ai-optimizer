"""Authenticated build and dashboard endpoint tests."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.db.database import Base, get_db
from api.main import app
from api.routers.pipelines import get_predictor


class StubPredictor:
    best_model_name = "xgboost"
    decision_threshold = 0.4
    model_artifact_sha256 = "test-model-sha256"
    preprocessing = {"training_mode": "full"}
    feature_cols = ["git_diff_src_churn", "lang_enc"]
    calibration_method = "raw"
    calibrator_artifact_sha256 = None

    def full_analysis(
        self, build_features: dict, failed_test_log: str | None = None, include_explanation: bool = False
    ) -> dict:
        return {
            "prediction": {
                "failure_probability": 0.73,
                "predicted_status": "failed",
                "confidence": 0.73,
            },
            "anomaly": {"is_anomaly": True, "anomaly_score": -0.18},
            "root_cause": {"cluster_id": 1, "likely_cause": "database, timeout"},
            "explanation": [
                {
                    "feature": "total_files_changed",
                    "value": 177.0,
                    "shap_value": 0.42,
                    "direction": "increases_failure_risk",
                }
            ] if include_explanation else [],
        }

    def analyze_root_cause(self, failed_test_log: str | None) -> dict | None:
        return {"cluster_id": 1, "likely_cause": "database, timeout"} if failed_test_log else None


@pytest.fixture
def client() -> TestClient:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = session_factory()
    session.commit()

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_predictor] = lambda: StubPredictor()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    session.close()
    Base.metadata.drop_all(engine)



def test_build_persistence_pipeline_listing_and_dashboard_summary(client: TestClient) -> None:
    first = client.post(
        "/pipelines/demo-repo/builds",
        json={
            "git_diff_src_churn": 50,
            "gh_diff_files_added": 34,
            "gh_diff_files_deleted": 6,
            "gh_diff_files_modified": 137,
            "gh_lang": "ruby",
            "gh_is_pr": False,
        },
    )
    assert first.status_code == 200
    assert first.json()["failure_probability"] == 0.73
    assert first.json()["is_anomaly"] is True
    assert first.json()["pipeline_name"] == "demo-repo"

    second = client.post(
        "/pipelines/demo-repo/builds",
        json={"build_features": {"gh_lang": "ruby"}, "build_status": "passed"},
    )
    assert second.status_code == 200
    pipelines = client.get("/pipelines")
    assert pipelines.status_code == 200
    assert pipelines.json()[0]["name"] == "demo-repo"
    history = client.get("/pipelines/demo-repo/builds")
    assert history.status_code == 200
    assert len(history.json()) == 2
    summary = client.get("/dashboard/summary")
    assert summary.status_code == 200
    assert summary.json()["total_builds_analyzed"] == 2
    assert summary.json()["anomalies_detected"] == 2
    assert summary.json()["top_root_causes"][0]["label"] == "database, timeout"


def test_pdf_report_receives_persisted_explanation(client: TestClient) -> None:
    response = client.post(
        "/pipelines/demo-repo/builds?explain=true",
        json={"gh_lang": "ruby"},
    )
    assert response.status_code == 200
    build_id = response.json()["build_log_id"]
    history = client.get("/pipelines/demo-repo/builds")
    assert history.json()[0]["explanation"][0]["feature"] == "total_files_changed"
    report = client.get(f"/pipelines/demo-repo/builds/{build_id}/report.pdf")
    assert report.status_code == 200
    assert report.content.startswith(b"%PDF")


def test_shadow_prediction_records_later_actual_outcome_without_blocking(client: TestClient) -> None:
    shadow = client.post(
        "/pipelines/shadow-repo/shadow-builds?explain=true",
        json={"git_diff_src_churn": 500, "gh_lang": "python"},
    )
    assert shadow.status_code == 200
    record = shadow.json()
    assert record["decision_threshold"] == 0.4
    assert record["is_alert"] is True
    assert record["training_mode"] == "full"
    assert record["model_version"] == "test-model-sha256"
    assert record["calibration_method"] == "raw"
    assert record["calibration_version"] is None

    pending_history = client.get("/pipelines/shadow-repo/builds")
    assert pending_history.json()[0]["build_status"] == "pending"
    outcome = client.patch(
        f"/pipelines/shadow-repo/shadow-builds/{record['build_log_id']}/outcome",
        json={"build_status": "failed", "failed_test_log": "database connection timeout"},
    )
    assert outcome.status_code == 200
    assert outcome.json()["build_status"] == "failed"
    assert outcome.json()["root_cause_label"] == "database, timeout"
    assert client.patch(
        f"/pipelines/shadow-repo/shadow-builds/{record['build_log_id']}/outcome",
        json={"build_status": "passed"},
    ).status_code == 409


def test_builds_without_a_reported_outcome_never_count_as_failures(client: TestClient) -> None:
    # The stub predicts "failed"; with no reported status the outcome is unknown.
    unknown = client.post("/pipelines/rate-repo/builds", json={"gh_lang": "ruby"})
    client.post("/pipelines/rate-repo/builds",
                json={"build_features": {"gh_lang": "ruby"}, "build_status": "passed"})
    history = client.get("/pipelines/rate-repo/builds").json()
    assert {build["id"]: build["build_status"] for build in history}[unknown.json()["build_log_id"]] == "unknown"
    summary = client.get("/dashboard/summary").json()
    assert summary["builds_with_outcome"] == 1 and summary["overall_failure_rate"] == 0.0
    trend = client.get("/dashboard/failure-trends").json()
    assert [(day["total_builds"], day["failed_builds"], day["scored_builds"]) for day in trend] == [(1, 0, 2)]
    assert trend[0]["failure_rate"] == 0.0 and trend[0]["average_predicted_risk"] == pytest.approx(0.73)


def test_chart_endpoints_report_model_distribution_and_recent_risk(client: TestClient) -> None:
    for _ in range(3):
        client.post("/pipelines/chart-repo/builds", json={"gh_lang": "ruby"})
    pipeline = next(p for p in client.get("/pipelines").json() if p["name"] == "chart-repo")
    assert pipeline["recent_failure_probabilities"] == [0.73, 0.73, 0.73]
    distribution = client.get("/dashboard/risk-distribution?threshold=0.75").json()
    assert len(distribution["bins"]) == 10 and distribution["bins"][7] == {"start": 0.7, "end": 0.8, "count": 3}
    assert distribution["total"] == 3 and distribution["flagged"] == 0
    assert client.get("/dashboard/risk-distribution").json()["flagged"] is None
    model = client.get("/model").json()
    assert model == {"model_name": "xgboost", "training_mode": "full", "decision_threshold": 0.4, "holdout_metrics": None}


def test_build_granularity_trend_gives_one_point_per_scored_build(client: TestClient) -> None:
    client.post("/pipelines/trend-repo/builds", json={"build_features": {"gh_lang": "ruby"}, "build_status": "failed"})
    client.post("/pipelines/trend-repo/builds", json={"gh_lang": "ruby"})
    client.post("/pipelines/trend-repo/builds", json={"build_features": {"gh_lang": "ruby"}, "build_status": "passed"})
    points = client.get("/dashboard/failure-trends?granularity=build").json()
    assert [(p["total_builds"], p["failed_builds"], p["failure_rate"]) for p in points] == [(1, 1, 1.0), (1, 1, 1.0), (2, 1, 0.5)]
    assert all(p["average_predicted_risk"] == pytest.approx(0.73) for p in points)
