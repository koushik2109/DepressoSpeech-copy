"""FastAPI dependencies for authentication and role-based access control."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from src.models import User
from src.utils.auth import decode_token

security = HTTPBearer(auto_error=False)

# In-process token→user_id cache to skip redundant JWT decode on repeated requests.
# Max 512 entries; evict the oldest key when full.
_TOKEN_CACHE: dict[str, str] = {}
_TOKEN_CACHE_MAX = 512


def _cache_get(token: str) -> str | None:
    return _TOKEN_CACHE.get(token)


def _cache_set(token: str, user_id: str) -> None:
    if len(_TOKEN_CACHE) >= _TOKEN_CACHE_MAX:
        oldest = next(iter(_TOKEN_CACHE))
        del _TOKEN_CACHE[oldest]
    _TOKEN_CACHE[token] = user_id


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate the current user from the Bearer token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )

    token = credentials.credentials

    cached_user_id = _cache_get(token)
    if cached_user_id is None:
        payload = decode_token(token)
        if not payload or payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )
        cached_user_id = payload.get("sub")
        if not cached_user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        _cache_set(token, cached_user_id)

    result = await db.execute(select(User).where(User.id == cached_user_id))
    user = result.scalar_one_or_none()
    if not user or user.status != "active":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user


async def require_patient(user: User = Depends(get_current_user)) -> User:
    if user.role != "patient":
        raise HTTPException(status_code=403, detail="Patient access required")
    return user


async def require_doctor(user: User = Depends(get_current_user)) -> User:
    if user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required")
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
