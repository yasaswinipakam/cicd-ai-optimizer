"""End-to-end authenticated build submission workflow test."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.auth.security import get_password_hash
from api.db.database import Base, get_db
from api.db.models import BuildLog, User, UserRole
from api.main import app
from api.routers.pipelines import get_predictor


class EndToEndPredictor:
    """Deterministic artifact substitute, keeping this test independent of training data."""

    best_model_name = "xgboost"

    def full_analysis(self, build_features, failed_test_log=None):
        churn = float(build_features.get("git_diff_src_churn", 0))
        probability = min(0.95, 0.35 + churn / 20_000)
        return {
            "prediction": {
                "failure_probability": probability,
                "predicted_status": "failed" if probability >= 0.5 else "passed",
                "confidence": max(probability, 1 - probability),
            },
            "anomaly": {"is_anomaly": churn > 1_000, "anomaly_score": -0.2},
            "root_cause": {"cluster_id": 1, "likely_cause": "timeout, integration"},
        }


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = session_factory()
    session.add(User(username="e2e-dev", hashed_password=get_password_hash("correct-password"), role=UserRole.DEVELOPER))
    session.commit()

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_predictor] = lambda: EndToEndPredictor()
    with TestClient(app) as test_client:
        yield test_client, session_factory
    app.dependency_overrides.clear()
    session.close()
    Base.metadata.drop_all(engine)


def test_authenticated_build_submission_persists_and_updates_dashboard(client) -> None:
    test_client, session_factory = client
    login = test_client.post("/auth/login", data={"username": "e2e-dev", "password": "correct-password"})
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    submitted = test_client.post(
        "/pipelines/e2e-repo/builds",
        headers=headers,
        json={
            "git_diff_src_churn": 5_000,
            "gh_diff_files_added": 34,
            "gh_diff_files_deleted": 6,
            "gh_diff_files_modified": 137,
            "git_diff_test_churn": 900,
            "git_prev_commit_resolution_status": "failed",
            "gh_lang": "python",
            "gh_is_pr": True,
            "git_branch": "feature/large-change",
        },
    )
    assert submitted.status_code == 200
    response = submitted.json()
    assert isinstance(response["failure_probability"], float)
    assert 0 <= response["failure_probability"] <= 1
    assert response["predicted_status"] in {"passed", "failed"}
    assert session_factory().query(BuildLog).count() == 1

    summary = test_client.get("/dashboard/summary", headers=headers)
    assert summary.status_code == 200
    assert summary.json()["total_builds_analyzed"] == 1
    assert summary.json()["pipelines_monitored"] == 1
