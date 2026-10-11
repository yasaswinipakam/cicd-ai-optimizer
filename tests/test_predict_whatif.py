"""Database-free What-If prediction endpoint test."""

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


class StubPredictor:
    def predict_failure(self, _features): return {"failure_probability": 0.42}
    def detect_anomaly(self, _features): return {"is_anomaly": False}


@pytest.fixture
def client():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine); factory = sessionmaker(bind=engine)
    session = factory(); session.add(User(username="whatif", hashed_password=get_password_hash("password"), role=UserRole.DEVELOPER)); session.commit()
    def override_db():
        db = factory()
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db] = override_db; app.dependency_overrides[get_predictor] = lambda: StubPredictor()
    with TestClient(app) as test_client: yield test_client, factory
    app.dependency_overrides.clear(); session.close(); Base.metadata.drop_all(engine)


def test_what_if_prediction_never_creates_build_records(client):
    test_client, factory = client
    token = test_client.post("/auth/login", data={"username": "whatif", "password": "password"}).json()["access_token"]
    before = factory().query(BuildLog).count()
    response = test_client.post("/predict/whatif", headers={"Authorization": f"Bearer {token}"}, json={"git_diff_src_churn": 2000, "gh_lang": "python"})
    assert response.status_code == 200
    assert 0 <= response.json()["failure_probability"] <= 1
    assert factory().query(BuildLog).count() == before
