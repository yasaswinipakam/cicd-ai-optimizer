"""Tests for the database-free, single-intervention optimization simulator."""

from copy import deepcopy

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.auth.security import get_password_hash
from api.db.database import Base, get_db
from api.db.models import BuildLog, Prediction, User, UserRole
from api.main import app
from api.routers.pipelines import get_predictor


class RecordingPredictor:
    """Deterministic predictor stub proving the endpoint reuses one predictor path."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @staticmethod
    def _total(features: dict) -> float:
        return float(sum(float(features.get(key, 0) or 0) for key in (
            "gh_diff_files_added", "gh_diff_files_deleted", "gh_diff_files_modified"
        )))

    def _prepared_features(self, features: dict) -> pd.DataFrame:
        return pd.DataFrame({"total_files_changed": [self._total(features)]})

    def full_analysis(self, features: dict, failed_test_log=None, include_explanation: bool = False) -> dict:
        self.calls.append(deepcopy(features))
        risk = min(1.0, (self._total(features) + float(features.get("git_diff_src_churn", 0) or 0)) / 1000)
        return {
            "prediction": {
                "failure_probability": risk,
                "predicted_status": "failed" if risk >= 0.5 else "passed",
            },
            "explanation": [
                {
                    "feature": "total_files_changed",
                    "value": self._total(features),
                    "shap_value": 0.2,
                    "direction": "increases_failure_risk",
                }
            ] if include_explanation else [],
        }


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    session.add(
        User(
            username="optimization-user",
            hashed_password=get_password_hash("password"),
            role=UserRole.DEVELOPER,
        )
    )
    session.commit()
    predictor = RecordingPredictor()

    def override_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_predictor] = lambda: predictor
    with TestClient(app) as test_client:
        yield test_client, factory, predictor
    app.dependency_overrides.clear()
    session.close()
    Base.metadata.drop_all(engine)


def _headers(test_client: TestClient) -> dict[str, str]:
    token = test_client.post(
        "/auth/login", data={"username": "optimization-user", "password": "password"}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _payload() -> dict:
    return {
        "build_features": {
            "git_diff_src_churn": 50,
            "gh_diff_files_added": 34,
            "gh_diff_files_deleted": 6,
            "gh_diff_files_modified": 137,
            "gh_lang": "ruby",
        },
        "intervention": {
            "field": "gh_diff_files_modified",
            "value": 100,
            "reason": "Hypothetically split a wide change.",
        },
        "include_explanation": True,
    }


def test_optimization_simulation_is_single_intervention_deterministic_and_database_free(client) -> None:
    test_client, factory, predictor = client
    headers = _headers(test_client)
    payload = _payload()
    original_payload = deepcopy(payload)
    before_logs = factory().query(BuildLog).count()
    before_predictions = factory().query(Prediction).count()

    first = test_client.post("/predict/whatif/optimization", headers=headers, json=payload)
    second = test_client.post("/predict/whatif/optimization", headers=headers, json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    body = first.json()
    assert body == second.json()
    assert payload == original_payload
    assert body["simulation_type"] == "model_based_hypothetical"
    assert body["baseline"]["derived_features"]["total_files_changed"] == 177.0
    assert body["simulation"]["derived_features"]["total_files_changed"] == 140.0
    assert body["intervention"]["original_value"] == 137
    assert body["intervention"]["simulated_value"] == 100.0
    assert body["risk_delta"] == pytest.approx(
        body["simulation"]["failure_probability"] - body["baseline"]["failure_probability"]
    )
    assert body["percentage_point_change"] == pytest.approx(body["risk_delta"] * 100)
    assert body["baseline"]["explanation"][0]["feature"] == "total_files_changed"
    assert "Model-based simulation only" in body["disclaimer"]
    assert "not guarantee" in body["disclaimer"]
    assert factory().query(BuildLog).count() == before_logs
    assert factory().query(Prediction).count() == before_predictions
    assert len(predictor.calls) == 4  # baseline + simulation for each identical request
    assert predictor.calls[0]["gh_diff_files_modified"] == 137
    assert predictor.calls[1]["gh_diff_files_modified"] == 100.0


@pytest.mark.parametrize(
    "intervention",
    [
        {"field": "total_files_changed", "value": 10},
        {"field": "tr_log_buildduration", "value": 10},
        {"field": "gh_team_size", "value": 10},
        {"field": "gh_diff_files_modified", "value": -1},
        {"field": "gh_diff_files_modified", "value": 501},
        {"field": "gh_diff_files_modified", "value": "not-a-number"},
    ],
)
def test_optimization_simulation_rejects_invalid_interventions(client, intervention) -> None:
    test_client, factory, _predictor = client
    headers = _headers(test_client)
    before_logs = factory().query(BuildLog).count()
    response = test_client.post(
        "/predict/whatif/optimization",
        headers=headers,
        json={"build_features": {"gh_lang": "ruby"}, "intervention": intervention},
    )
    assert response.status_code == 422
    assert factory().query(BuildLog).count() == before_logs


def test_optimization_simulation_rejects_multiple_and_post_build_inputs(client) -> None:
    test_client, factory, _predictor = client
    headers = _headers(test_client)
    before_logs = factory().query(BuildLog).count()
    multiple = test_client.post(
        "/predict/whatif/optimization",
        headers=headers,
        json={
            "build_features": {},
            "interventions": [
                {"field": "git_diff_src_churn", "value": 1},
                {"field": "git_diff_test_churn", "value": 1},
            ],
        },
    )
    leakage = test_client.post(
        "/predict/whatif/optimization",
        headers=headers,
        json={
            "build_features": {"tr_log_buildduration": 20},
            "intervention": {"field": "git_diff_src_churn", "value": 1},
        },
    )
    assert multiple.status_code == 422
    assert leakage.status_code == 422
    assert factory().query(BuildLog).count() == before_logs
