"""aiohttp web application factory."""

from pathlib import Path

import jinja2
from aiohttp import web

from .middleware import auth_middleware
from .routes import setup_routes

_HERE = Path(__file__).parent


def create_app(bot_context: dict) -> web.Application:
    app = web.Application(middlewares=[auth_middleware])
    app["bot"] = bot_context

    # Jinja2 template engine (used directly, no aiohttp_jinja2 needed)
    app["jinja"] = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_HERE / "templates")),
        autoescape=True,
    )

    # Static files
    app.router.add_static("/static/", path=str(_HERE / "static"), name="static")

    # Routes
    setup_routes(app)

    return app
