"""Initialize local tables and seed development users when the database is empty."""

import os

from passlib.context import CryptContext
from sqlalchemy import select

from api.db.database import Base, SessionLocal, engine
from api.db.models import User, UserRole
from api.db.migrations import apply_migrations


password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def init_db() -> None:
    """Create tables and add initial admin/developer accounts exactly once."""
    Base.metadata.create_all(bind=engine)
    applied = apply_migrations()
    session = SessionLocal()
    try:
        if session.scalar(select(User).limit(1)) is None:
            admin_password = os.getenv("ADMIN_PASSWORD", "changeme123")
            if "ADMIN_PASSWORD" not in os.environ:
                print("WARNING: ADMIN_PASSWORD is unset; change the default admin password immediately.")
            session.add_all(
                [
                    User(
                        username="admin",
                        hashed_password=password_context.hash(admin_password),
                        role=UserRole.ADMINISTRATOR,
                    ),
                    User(
                        username="dev",
                        hashed_password=password_context.hash("changeme123"),
                        role=UserRole.DEVELOPER,
                    ),
                ]
            )
            session.commit()
            print("Created tables and seeded admin and dev users.")
        else:
            print("Database tables already exist; seed users were not changed.")
        if applied:
            print(f"Applied schema migrations: {', '.join(applied)}")
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
