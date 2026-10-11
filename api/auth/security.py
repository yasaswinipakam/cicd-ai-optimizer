"""Password hashing and JWT creation/validation helpers."""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext


ALGORITHM = "HS256"
DEV_JWT_SECRET = "dev-only-change-before-production-cicd-ai-optimizer"
password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_warned_about_dev_secret = False


def _jwt_secret() -> str:
    """Return configured signing key, refusing an insecure fallback in production."""
    global _warned_about_dev_secret
    secret = os.getenv("JWT_SECRET")
    if secret:
        return secret
    environment = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development")).lower()
    if environment in {"production", "prod"}:
        raise RuntimeError("JWT_SECRET must be set when APP_ENV is production.")
    if not _warned_about_dev_secret:
        print("WARNING: JWT_SECRET is unset; using a development-only signing key.")
        _warned_about_dev_secret = True
    return DEV_JWT_SECRET


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return password_context.verify(plain, hashed)


def get_password_hash(plain: str) -> str:
    """Create a bcrypt password hash."""
    return password_context.hash(plain)


def create_access_token(data: dict, expires_minutes: int = 60) -> str:
    """Create a signed access token with an expiry claim."""
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    return jwt.encode(payload, _jwt_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode a valid JWT or raise the standard unauthenticated response."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[ALGORITHM])
        if not payload.get("sub"):
            raise credentials_error
        return payload
    except JWTError as error:
        raise credentials_error from error


def verify_github_signature(
    payload_bytes: bytes, signature_header: str | None, secret: str
) -> bool:
    """Verify GitHub's HMAC-SHA256 webhook signature in constant time."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.removeprefix("sha256="))
