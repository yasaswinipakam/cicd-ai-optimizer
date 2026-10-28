"""Seeding never uses a shared default password."""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.db.init_db as init_module
import api.db.migrations as migrations_module
from api.db.models import User


@pytest.fixture()
def database(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    sessions = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    for module in (init_module, migrations_module):
        monkeypatch.setattr(module, "engine", engine)
    monkeypatch.setattr(init_module, "SessionLocal", sessions)
    for name in ("ADMIN_PASSWORD", "DEV_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    return sessions


def _users(sessions) -> dict[str, User]:
    with sessions() as session:
        return {user.username: user for user in session.scalars(select(User))}


def test_admin_password_comes_from_environment_and_dev_is_opt_in(database, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_PASSWORD", "a-real-admin-passphrase")
    init_module.init_db()
    users = _users(database)
    assert set(users) == {"admin"}
    assert init_module.password_context.verify("a-real-admin-passphrase", users["admin"].hashed_password)


def test_unset_admin_password_is_random_and_printed_once(database, capsys) -> None:
    init_module.init_db()
    printed = capsys.readouterr().out
    generated = printed.split("(shown once): ")[1].split()[0]
    admin = _users(database)["admin"]
    assert len(generated) >= 20 and init_module.password_context.verify(generated, admin.hashed_password)
    assert not init_module.password_context.verify("changeme123", admin.hashed_password)


@pytest.mark.parametrize("bad", ["changeme123", "replace-with-a-strong-admin-password", "short"])
def test_placeholder_default_and_short_passwords_are_refused(database, monkeypatch, bad) -> None:
    monkeypatch.setenv("ADMIN_PASSWORD", bad)
    with pytest.raises(SystemExit):
        init_module.init_db()
    assert _users(database) == {}


def test_set_password_replaces_an_existing_password(database, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_PASSWORD", "a-real-admin-passphrase")
    monkeypatch.setenv("DEV_PASSWORD", "a-real-dev-passphrase")
    init_module.init_db()
    init_module.set_password("dev", "a-brand-new-passphrase")
    dev = _users(database)["dev"]
    assert init_module.password_context.verify("a-brand-new-passphrase", dev.hashed_password)
    with pytest.raises(SystemExit):
        init_module.set_password("nobody", "a-brand-new-passphrase")
