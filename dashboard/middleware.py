"""Authentication middleware for the dashboard."""

from aiohttp import web

from .auth import get_session_user

# Paths that don't require authentication
_PUBLIC_PATHS = frozenset({
    "/auth/login", "/auth/callback", "/auth/logout",
    "/auth/password",  # password login POST
    "/api/webhook/github",
})


@web.middleware
async def auth_middleware(request: web.Request, handler):
    # Allow static files and public auth paths
    if request.path.startswith("/static/") or request.path in _PUBLIC_PATHS:
        return await handler(request)

    user = get_session_user(request)
    if not user:
        raise web.HTTPFound("/auth/login")

    request["user"] = user
    return await handler(request)
