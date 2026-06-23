"""Discord OAuth2 + password login + HMAC-signed session cookie."""

import hashlib
import hmac
import json
import os
import time
import logging

import aiohttp
from aiohttp import web

log = logging.getLogger("huacheng.dashboard.auth")

_DISCORD_API = "https://discord.com/api/v10"
_COOKIE_NAME = "oc_session"
_COOKIE_MAX_AGE = 7 * 24 * 3600  # 7 days


def _get_secret() -> bytes:
    return os.environ["DASHBOARD_SECRET"].encode()


def _sign(payload: str) -> str:
    return hmac.new(_get_secret(), payload.encode(), hashlib.sha256).hexdigest()


def create_session_cookie(user_id: str, username: str) -> str:
    """Create an HMAC-signed session cookie value."""
    data = json.dumps({"uid": user_id, "name": username, "exp": int(time.time()) + _COOKIE_MAX_AGE})
    sig = _sign(data)
    return f"{data}|{sig}"


def parse_session_cookie(raw: str) -> dict | None:
    """Verify and parse a session cookie. Returns user dict or None."""
    if "|" not in raw:
        return None
    data, sig = raw.rsplit("|", 1)
    if not hmac.compare_digest(sig, _sign(data)):
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload


def get_session_user(request: web.Request) -> dict | None:
    """Extract the authenticated user from request cookies."""
    raw = request.cookies.get(_COOKIE_NAME)
    if not raw:
        return None
    return parse_session_cookie(raw)


def set_session_cookie(response: web.Response, user_id: str, username: str) -> None:
    """Set the session cookie on a response."""
    value = create_session_cookie(user_id, username)
    response.set_cookie(
        _COOKIE_NAME,
        value,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="Lax",
        path="/",
    )


def clear_session_cookie(response: web.Response) -> None:
    response.del_cookie(_COOKIE_NAME, path="/")


# ---------------------------------------------------------------------------
# Auth mode detection
# ---------------------------------------------------------------------------

def has_oauth_config() -> bool:
    """Check if Discord OAuth2 is fully configured."""
    return bool(
        os.getenv("DISCORD_CLIENT_ID")
        and os.getenv("DISCORD_CLIENT_SECRET")
        and os.getenv("DASHBOARD_REDIRECT_URI")
    )


def has_password_config() -> bool:
    """Check if password login is configured."""
    return bool(os.getenv("DASHBOARD_PASSWORD"))


def verify_password(password: str) -> bool:
    """Verify the dashboard password."""
    expected = os.getenv("DASHBOARD_PASSWORD", "")
    if not expected:
        return False
    return hmac.compare_digest(password, expected)


# ---------------------------------------------------------------------------
# Discord OAuth2
# ---------------------------------------------------------------------------

def get_oauth_url() -> str:
    client_id = os.environ["DISCORD_CLIENT_ID"]
    redirect_uri = os.environ["DASHBOARD_REDIRECT_URI"]
    return (
        f"https://discord.com/oauth2/authorize"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope=identify"
    )


async def exchange_code(code: str) -> dict | None:
    """Exchange OAuth2 code for user info. Returns {'id': ..., 'username': ...} or None."""
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": os.environ["DASHBOARD_REDIRECT_URI"],
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    auth = aiohttp.BasicAuth(
        os.environ["DISCORD_CLIENT_ID"],
        os.environ["DISCORD_CLIENT_SECRET"],
    )
    async with aiohttp.ClientSession() as session:
        # Exchange code for token
        async with session.post(f"{_DISCORD_API}/oauth2/token", data=data, auth=auth, headers=headers) as resp:
            if resp.status != 200:
                log.warning(f"OAuth token exchange failed: {resp.status}")
                return None
            token_data = await resp.json()

        access_token = token_data.get("access_token")
        if not access_token:
            return None

        # Get user info
        async with session.get(
            f"{_DISCORD_API}/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
        ) as resp:
            if resp.status != 200:
                log.warning(f"OAuth user fetch failed: {resp.status}")
                return None
            return await resp.json()


def is_allowed_user(user_id: str) -> bool:
    allowed = os.environ.get("DASHBOARD_ALLOWED_USERS", "")
    if not allowed:
        return True  # no restriction if not set
    return user_id in allowed.split(",")
