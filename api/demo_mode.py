"""Public demo mode (``DEMO_READ_ONLY=1``): nothing a visitor sends is stored."""

import os


def read_only() -> bool:
    return os.getenv("DEMO_READ_ONLY", "").lower() in ("1", "true", "yes")
