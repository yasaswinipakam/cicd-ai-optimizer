"""Create the database tables and apply schema migrations.

The application has no sign-in, so there are no accounts to seed.
"""

from api.db.database import Base, engine
from api.db.migrations import apply_migrations


def init_db() -> None:
    """Create missing tables, then bring an existing database up to date."""
    Base.metadata.create_all(bind=engine)
    applied = apply_migrations()
    print("Database tables are ready.")
    if applied:
        print(f"Applied schema migrations: {', '.join(applied)}")


if __name__ == "__main__":
    init_db()
