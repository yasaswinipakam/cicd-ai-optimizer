"""Initialize local tables and seed accounts when the database is empty.

No password is hard-coded.  The administrator password comes from
``ADMIN_PASSWORD``; if it is unset, a random one is generated and printed
once.  The optional developer account is created only when ``DEV_PASSWORD``
is set.  Use ``--set-password USERNAME`` to change a password later, for
example on a database seeded by an older version with a shared default.
"""

import argparse
import getpass
import os
import secrets

from passlib.context import CryptContext
from sqlalchemy import select

from api.db.database import Base, SessionLocal, engine
from api.db.models import User, UserRole
from api.db.migrations import apply_migrations


password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
MIN_PASSWORD_LENGTH = 12
# Values that must never become a real password: the .env.example placeholders
# and the shared default earlier versions seeded.
REJECTED_PASSWORDS = {"replace-with-a-strong-admin-password", "changeme123"}


def _checked(password: str, source: str) -> str:
    if password in REJECTED_PASSWORDS or password.startswith("replace-with-"):
        raise SystemExit(f"{source} is still a placeholder or known default; set a real password.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise SystemExit(f"{source} must be at least {MIN_PASSWORD_LENGTH} characters.")
    return password


def init_db() -> None:
    """Create tables and add the initial accounts exactly once."""
    Base.metadata.create_all(bind=engine)
    applied = apply_migrations()
    session = SessionLocal()
    try:
        if session.scalar(select(User).limit(1)) is None:
            admin_password = os.getenv("ADMIN_PASSWORD")
            if admin_password:
                admin_password = _checked(admin_password, "ADMIN_PASSWORD")
            else:
                admin_password = secrets.token_urlsafe(18)
                print(f"ADMIN_PASSWORD is unset; generated admin password (shown once): {admin_password}")
            users = [User(username="admin", hashed_password=password_context.hash(admin_password),
                          role=UserRole.ADMINISTRATOR)]
            dev_password = os.getenv("DEV_PASSWORD")
            if dev_password:
                users.append(User(username="dev", hashed_password=password_context.hash(_checked(dev_password, "DEV_PASSWORD")),
                                  role=UserRole.DEVELOPER))
            session.add_all(users)
            session.commit()
            print(f"Created tables and seeded: {', '.join(user.username for user in users)}.")
        else:
            print("Database tables already exist; seed users were not changed.")
        if applied:
            print(f"Applied schema migrations: {', '.join(applied)}")
    finally:
        session.close()


def set_password(username: str, password: str) -> None:
    """Replace one existing account's password."""
    session = SessionLocal()
    try:
        user = session.scalar(select(User).where(User.username == username))
        if user is None:
            raise SystemExit(f"No user named {username!r}.")
        user.hashed_password = password_context.hash(_checked(password, "The new password"))
        session.commit()
        print(f"Password updated for {username}.")
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set-password", metavar="USERNAME", help="prompt for and set a new password for USERNAME")
    args = parser.parse_args()
    if args.set_password:
        first = getpass.getpass(f"New password for {args.set_password}: ")
        if first != getpass.getpass("Repeat it: "):
            raise SystemExit("Passwords did not match.")
        set_password(args.set_password, first)
    else:
        init_db()
