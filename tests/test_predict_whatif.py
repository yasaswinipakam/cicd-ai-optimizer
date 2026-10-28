"""Database-free What-If prediction endpoint test."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.db.database import Base, get_db
from api.db.models import BuildLog
from api.main import app
from api.routers.pipelines import get_predictor


class StubPredictor:
    def predict_failure(self, _features): return {"failure_probability": 0.42}
    def detect_anomaly(self, _features): return {"is_anomaly": False}
    def full_analysis(self, _features, include_explanation=False):
        return {"explanation": [{"feature": "git_diff_src_churn", "value": 2000.0, "shap_value": 0.3,
                                 "direction": "increases_failure_risk"}] if include_explanation else []}


@pytest.fixture
def client():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine); factory = sessionmaker(bind=engine)
    session = factory()
    def override_db():
        db = factory()
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db] = override_db; app.dependency_overrides[get_predictor] = lambda: StubPredictor()
    with TestClient(app) as test_client: yield test_client, factory
    app.dependency_overrides.clear(); session.close(); Base.metadata.drop_all(engine)


def test_what_if_prediction_never_creates_build_records(client):
    test_client, factory = client
    before = factory().query(BuildLog).count()
    response = test_client.post("/predict/whatif", json={"git_diff_src_churn": 2000, "gh_lang": "python"})
    assert response.status_code == 200
    assert 0 <= response.json()["failure_probability"] <= 1
    assert factory().query(BuildLog).count() == before


def test_what_if_explanation_is_opt_in(client):
    test_client, _factory = client
    plain = test_client.post("/predict/whatif", json={"git_diff_src_churn": 2000}).json()
    explained = test_client.post("/predict/whatif?explain=true", json={"git_diff_src_churn": 2000}).json()
    assert "explanation" not in plain
    assert explained["explanation"][0]["feature"] == "git_diff_src_churn"
