"""Authentication endpoint tests with an isolated in-memory database."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.auth.security import get_password_hash
from api.db.database import Base, get_db
from api.db.models import User, UserRole
from api.main import app


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = testing_session()
    session.add(
        User(
            username="api-dev",
            hashed_password=get_password_hash("correct-password"),
            role=UserRole.DEVELOPER,
        )
    )
    session.commit()

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    session.close()
    Base.metadata.drop_all(engine)


def test_login_and_current_user(client: TestClient) -> None:
    failed_login = client.post(
        "/auth/login", data={"username": "api-dev", "password": "wrong-password"}
    )
    assert failed_login.status_code == 401

    login = client.post(
        "/auth/login", data={"username": "api-dev", "password": "correct-password"}
    )
    assert login.status_code == 200
    payload = login.json()
    assert payload["token_type"] == "bearer"
    assert payload["role"] == "developer"

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {payload['access_token']}"})
    assert me.status_code == 200
    assert me.json() == {"username": "api-dev", "role": "developer"}
