"""Database setup creates no accounts, and migrates sign-in era databases without losing data."""

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

import api.db.init_db as init_module
import api.db.migrations as migrations_module


@pytest.fixture()
def engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    for module in (init_module, migrations_module):
        monkeypatch.setattr(module, "engine", engine)
    return engine


def test_fresh_database_has_no_accounts(engine) -> None:
    init_module.init_db()
    inspector = inspect(engine)
    assert "users" not in inspector.get_table_names()
    assert "owner_id" not in {c["name"] for c in inspector.get_columns("pipelines")}


def test_sign_in_era_database_keeps_pipelines_and_builds(engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR(100) NOT NULL, "
                                "hashed_password VARCHAR(255) NOT NULL)"))
        connection.execute(text("CREATE TABLE pipelines (id INTEGER PRIMARY KEY, name VARCHAR(255) NOT NULL, "
                                "status VARCHAR(50) NOT NULL, created_at DATETIME NOT NULL, "
                                "owner_id INTEGER NOT NULL REFERENCES users(id))"))
        connection.execute(text("INSERT INTO users VALUES (1, 'admin', 'hash')"))
        connection.execute(text("INSERT INTO pipelines VALUES (7, 'acme', 'active', '2026-01-01', 1)"))
    init_module.init_db()
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO build_logs (pipeline_id, timestamp, build_status, stage_durations, raw_features) "
                                "VALUES (7, '2026-01-02', 'passed', '{}', '{}')"))
    init_module.init_db()  # idempotent
    inspector = inspect(engine)
    assert "users" not in inspector.get_table_names()
    assert "owner_id" not in {c["name"] for c in inspector.get_columns("pipelines")}
    with engine.connect() as connection:
        assert connection.execute(text("SELECT id, name FROM pipelines")).all() == [(7, "acme")]
        assert connection.execute(text("SELECT pipeline_id FROM build_logs")).scalar() == 7
        assert connection.execute(text("PRAGMA foreign_key_check")).all() == []
