"""Authentication endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from api.auth.dependencies import get_current_user
from api.auth.security import create_access_token, verify_password
from api.db.database import get_db
from api.db.models import User


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> dict:
    """Exchange a username/password pair for a bearer token."""
    user = db.query(User).filter(User.username == form_data.username).first()
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    role = user.role.value if hasattr(user.role, "value") else user.role
    return {
        "access_token": create_access_token({"sub": user.username, "role": role}),
        "token_type": "bearer",
        "role": role,
    }


@router.get("/me")
def me(current_user: User = Depends(get_current_user)) -> dict:
    """Return the authenticated caller's identity and role."""
    role = current_user.role.value if hasattr(current_user.role, "value") else current_user.role
    return {"username": current_user.username, "role": role}
