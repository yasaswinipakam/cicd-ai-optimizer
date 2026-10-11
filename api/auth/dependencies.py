"""FastAPI authentication and role-based authorization dependencies."""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from api.auth.security import decode_token
from api.db.database import get_db
from api.db.models import User


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """Resolve the current user from the JWT subject."""
    username = decode_token(token)["sub"]
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(role: str):
    """Return a dependency that permits only one named application role."""
    def role_dependency(current_user: User = Depends(get_current_user)) -> User:
        current_role = current_user.role.value if hasattr(current_user.role, "value") else current_user.role
        if current_role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return role_dependency
