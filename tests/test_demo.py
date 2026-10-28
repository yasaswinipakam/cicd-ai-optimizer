"""Public demo: read-only API mode and the single-service app serving the dashboard."""

import importlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.db.database import Base, get_db
from api.db.models import BuildLog
from api.main import app
from api.routers.pipelines import get_predictor
from tests.test_pipelines import StubPredictor


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_predictor] = lambda: StubPredictor()
    yield factory
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)


def test_read_only_mode_scores_without_saving_and_blocks_other_writes(session_factory, monkeypatch) -> None:
    monkeypatch.setenv("DEMO_READ_ONLY", "1")
    client = TestClient(app)
    assert client.get("/status").json() == {"read_only": True}
    scored = client.post("/pipelines/demo/builds", json={"gh_lang": "ruby"})
    assert scored.status_code == 200 and scored.json()["failure_probability"] == 0.73
    with session_factory() as db:
        assert db.query(BuildLog).count() == 0
    for path in ("/pipelines/demo/shadow-builds", "/webhooks/github-actions",
                 "/pipelines/a/predict/b/shadow-builds", "/pipelines/demo/shadow-builds/1/outcome"):
        assert client.post(path, json={}).status_code == 403, path
    assert client.get("/pipelines").status_code == 200
    monkeypatch.delenv("DEMO_READ_ONLY")
    client.post("/pipelines/demo/builds", json={"gh_lang": "ruby"})
    with session_factory() as db:
        assert db.query(BuildLog).count() == 1


def test_demo_app_serves_api_under_prefix_and_dashboard_routes(session_factory, monkeypatch, tmp_path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html>dashboard")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    monkeypatch.setenv("DASHBOARD_DIST", str(tmp_path))
    monkeypatch.setenv("DEMO_READ_ONLY", "1")
    import api.demo
    demo = TestClient(importlib.reload(api.demo).app)
    assert demo.get("/api/status").json() == {"read_only": True}
    assert demo.post("/api/pipelines/demo/builds", json={"gh_lang": "ruby"}).status_code == 200
    assert demo.post("/api/pipelines/demo/shadow-builds", json={}).status_code == 403
    assert demo.get("/assets/app.js").text == "console.log(1)"
    for route in ("/", "/pipelines/twbs%2Fbootstrap", "/what-if", "/..%2f..%2fREADME.md"):
        assert demo.get(route).text == "<!doctype html>dashboard", route
