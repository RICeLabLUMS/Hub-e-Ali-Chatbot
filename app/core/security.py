import secrets

from fastapi import Header, HTTPException, status
from app.core.config import settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> str:
    # Constant-time compare so an attacker can't extract the key one byte at a
    # time by measuring response timing. compare_digest requires equal-length
    # bytes inputs; the conversion handles that uniformly.
    expected = (settings.API_KEY or "").encode("utf-8")
    presented = (x_api_key or "").encode("utf-8")
    if not presented or not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return x_api_key  # type: ignore[return-value]
