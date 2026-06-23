"""Dashboard page routes + API endpoints."""

import asyncio
import functools
import hashlib
import hmac
import logging
import os
import subprocess

from aiohttp import web

from . import auth as _auth

log = logging.getLogger("huacheng.dashboard.routes")


def _render(request: web.Request, template_name: str, ctx: dict | None = None) -> web.Response:
    """Render a Jinja2 template to an HTML response."""
    jinja = request.app["jinja"]
    tpl = jinja.get_template(template_name)
    ctx = ctx or {}
    ctx.setdefault("user", request.get("user"))
    html = tpl.render(**ctx)
    return web.Response(text=html, content_type="text/html")


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

async def login_page(request: web.Request) -> web.Response:
    error = request.query.get("error")
    oauth_enabled = _auth.has_oauth_config()
    password_enabled = _auth.has_password_config()
    return _render(request, "login.html", {
        "oauth_url": _auth.get_oauth_url() if oauth_enabled else None,
        "oauth_enabled": oauth_enabled,
        "password_enabled": password_enabled,
        "error": error,
        "user": None,
    })


async def password_login(request: web.Request) -> web.Response:
    """Handle password-based login."""
    data = await request.post()
    password = data.get("password", "")
    if _auth.verify_password(password):
        resp = web.HTTPFound("/")
        _auth.set_session_cookie(resp, "admin", "Admin")
        log.info("Dashboard login via password")
        return resp
    raise web.HTTPFound("/auth/login?error=密碼錯誤")


async def oauth_callback(request: web.Request) -> web.Response:
    code = request.query.get("code")
    if not code:
        raise web.HTTPFound("/auth/login?error=缺少授權碼")

    user_info = await _auth.exchange_code(code)
    if not user_info:
        raise web.HTTPFound("/auth/login?error=Discord+驗證失敗")

    user_id = user_info["id"]
    username = user_info.get("global_name") or user_info.get("username", "unknown")

    if not _auth.is_allowed_user(user_id):
        raise web.HTTPFound("/auth/login?error=你沒有權限存取此面板")

    resp = web.HTTPFound("/")
    _auth.set_session_cookie(resp, user_id, username)
    log.info(f"Dashboard login: {username} ({user_id})")
    return resp


async def logout(request: web.Request) -> web.Response:
    resp = web.HTTPFound("/auth/login")
    _auth.clear_session_cookie(resp)
    return resp


# ---------------------------------------------------------------------------
# Dashboard pages
# ---------------------------------------------------------------------------

async def dashboard_page(request: web.Request) -> web.Response:
    bot = request.app["bot"]
    client = bot["client"]
    from watchlist import get_state, get_watchlist, is_tw_market_active, is_us_market_active

    state = get_state()
    return _render(request, "dashboard.html", {
        "bot_ready": client.is_ready(),
        "guild_count": len(client.guilds) if client.is_ready() else 0,
        "tw_active": is_tw_market_active(),
        "us_active": is_us_market_active(),
        "monitor_enabled": state["enabled"],
        "watchlist_count": len(get_watchlist()),
    })


async def watchlist_page(request: web.Request) -> web.Response:
    return _render(request, "watchlist.html")


async def settings_page(request: web.Request) -> web.Response:
    bot = request.app["bot"]
    return _render(request, "settings.html", {
        "prompt": bot["get_prompt"](),
        "greeting_template": bot["get_greeting_template"](),
        "greeting_prompts": bot["get_greeting_prompts"](),
    })


async def settings_save(request: web.Request) -> web.Response:
    bot = request.app["bot"]
    data = await request.post()

    # Save AI prompt
    new_prompt = data.get("prompt", "").strip()
    if new_prompt:
        bot["set_prompt"](new_prompt)

    # Save greeting template
    new_template = data.get("greeting_template", "").strip()
    if new_template:
        bot["set_greeting_template"](new_template)

    # Save individual greeting prompts
    new_greeting_prompts = {}
    for hour in bot["get_greeting_prompts"]():
        key = f"greeting_{hour}"
        val = data.get(key, "").strip()
        if val:
            new_greeting_prompts[hour] = val
    if new_greeting_prompts:
        bot["set_greeting_prompts"](new_greeting_prompts)

    log.info("Dashboard: settings saved")
    return _render(request, "settings.html", {
        "prompt": bot["get_prompt"](),
        "greeting_template": bot["get_greeting_template"](),
        "greeting_prompts": bot["get_greeting_prompts"](),
        "flash_msg": "設定已儲存",
    })


async def chat_history_page(request: web.Request) -> web.Response:
    bot = request.app["bot"]
    sessions = bot["get_chat_sessions"]()
    client = bot["client"]

    # Build channel name map
    channels = {}
    for ch_id in sessions:
        ch = client.get_channel(ch_id)
        channels[ch_id] = getattr(ch, "name", str(ch_id)) if ch else str(ch_id)

    selected = request.query.get("channel")
    if selected:
        try:
            selected = int(selected)
        except ValueError:
            selected = None

    display_sessions = {}
    if selected and selected in sessions:
        display_sessions[selected] = sessions[selected]
    else:
        display_sessions = sessions

    return _render(request, "chat_history.html", {
        "sessions": sessions,
        "channels": channels,
        "selected_channel": selected,
        "display_sessions": display_sessions,
    })


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

async def api_watchlist_add(request: web.Request) -> web.Response:
    """Add a stock. Accepts form data (HTMX) or JSON."""
    content_type = request.content_type
    if "json" in content_type:
        data = await request.json()
    else:
        data = await request.post()

    ticker_raw = data.get("ticker", "").strip()
    if not ticker_raw:
        return await _scores_partial(request, error_msg="請輸入股票代碼")

    from prediction import _resolve_ticker
    import yfinance as yf

    resolved = _resolve_ticker(ticker_raw)
    loop = asyncio.get_running_loop()

    try:
        info = await loop.run_in_executor(
            None, lambda: yf.Ticker(resolved).info
        )
        name = info.get("shortName") or info.get("longName") or resolved
    except Exception:
        return await _scores_partial(request, error_msg=f"找不到股票 {ticker_raw}")

    from watchlist import add_stock
    input_code = ticker_raw.strip().upper()
    ok = add_stock(input_code, resolved, name)
    if ok:
        msg = f"已新增 {name} ({resolved})"
    else:
        msg = f"{name} ({resolved}) 已在清單中"

    return await _scores_partial(request, success_msg=msg)


async def api_watchlist_remove(request: web.Request) -> web.Response:
    content_type = request.content_type
    if "json" in content_type:
        data = await request.json()
    else:
        data = await request.post()

    ticker = data.get("ticker", "").strip()
    if not ticker:
        return await _scores_partial(request, error_msg="缺少股票代碼")

    from watchlist import remove_stock
    ok = remove_stock(ticker)
    msg = f"已移除 {ticker}" if ok else f"{ticker} 不在監控清單中"
    return await _scores_partial(request, success_msg=msg if ok else None, error_msg=None if ok else msg)


async def api_monitor_toggle(request: web.Request) -> web.Response:
    bot = request.app["bot"]
    from watchlist import get_state, set_monitor

    state = get_state()
    new_enabled = not state["enabled"]
    set_monitor(enabled=new_enabled)

    # Start/stop the monitor task
    monitor = bot.get("monitor_loop")
    if monitor:
        if new_enabled and not monitor.is_running():
            monitor.start()
        elif not new_enabled and monitor.is_running():
            monitor.stop()

    log.info(f"Dashboard: monitor toggled to {'ON' if new_enabled else 'OFF'}")
    raise web.HTTPFound("/")


async def api_market_status(request: web.Request) -> web.Response:
    from watchlist import is_tw_market_active, is_us_market_active
    return web.json_response({
        "tw_active": is_tw_market_active(),
        "us_active": is_us_market_active(),
    })


async def api_watchlist_scores(request: web.Request) -> web.Response:
    """Return all watchlist scores as JSON."""
    from watchlist import get_watchlist
    from prediction import quick_analysis

    loop = asyncio.get_running_loop()
    wl = get_watchlist()
    results = []
    for item in wl:
        try:
            r = await loop.run_in_executor(
                None, functools.partial(quick_analysis, item["ticker"])
            )
            a = r["analysis"]
            results.append({
                "name": item["name"],
                "ticker": item["ticker"],
                "current": a["current"],
                "change_pct": a["change_pct"],
                "total_score": a["total_score"],
                "verdict": a["verdict"],
                "ta_confidence": a.get("ta_confidence", 0),
                "ta_confidence_max": a.get("ta_confidence_max", 40),
            })
        except Exception as e:
            results.append({
                "name": item["name"],
                "ticker": item["ticker"],
                "error": str(e),
            })
    return web.json_response(results)


# ---------------------------------------------------------------------------
# HTMX partials
# ---------------------------------------------------------------------------

async def _scores_partial(request: web.Request,
                          error_msg: str | None = None,
                          success_msg: str | None = None) -> web.Response:
    """Render the stock_table partial with current scores."""
    from watchlist import get_watchlist
    from prediction import quick_analysis

    loop = asyncio.get_running_loop()
    wl = get_watchlist()
    stocks = []

    for item in wl:
        entry = {
            "name": item["name"],
            "ticker": item["ticker"],
        }
        try:
            r = await loop.run_in_executor(
                None, functools.partial(quick_analysis, item["ticker"])
            )
            a = r["analysis"]
            score = a["total_score"]
            entry.update({
                "current": a["current"],
                "change_pct": a["change_pct"],
                "total_score": score,
                "verdict": a["verdict"],
                "ta_confidence": a.get("ta_confidence", 0),
                "ta_confidence_max": a.get("ta_confidence_max", 40),
                "score_class": (
                    "strong-buy" if score >= 75 else
                    "buy" if score >= 60 else
                    "hold" if score >= 45 else
                    "weak"
                ),
                "error": None,
            })
        except Exception as e:
            entry["error"] = str(e)
        stocks.append(entry)

    return _render(request, "_partials/stock_table.html", {
        "stocks": stocks,
        "error_msg": error_msg,
        "success_msg": success_msg,
    })


async def scores_partial_handler(request: web.Request) -> web.Response:
    return await _scores_partial(request)


# ---------------------------------------------------------------------------
# Arena pages + API
# ---------------------------------------------------------------------------

async def arena_page(request: web.Request) -> web.Response:
    bot = request.app["bot"]
    arena = bot.get("arena")
    if not arena:
        return _render(request, "arena.html", {"bots": [], "trades": [], "risk_events": []})

    loop = asyncio.get_running_loop()
    status = await loop.run_in_executor(None, arena.get_status)

    from arena import arena_db as _adb
    trades = _adb.get_recent_trades(limit=20)
    risk_events = _adb.get_risk_events(limit=10)

    return _render(request, "arena.html", {
        "bots": status.get("bots", []),
        "trades": trades,
        "risk_events": risk_events,
    })


async def api_arena_chart(request: web.Request) -> web.Response:
    """回傳 Arena 權益曲線圖片作為 <img> tag。"""
    bot = request.app["bot"]
    arena = bot.get("arena")
    if not arena:
        return web.Response(text="<p>Arena not initialized</p>", content_type="text/html")

    loop = asyncio.get_running_loop()
    buf = await loop.run_in_executor(None, arena.draw_equity_chart)
    if not buf:
        return web.Response(text="<p>Not enough data for chart</p>", content_type="text/html")

    import base64
    img_b64 = base64.b64encode(buf.read()).decode()
    html = f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;border-radius:8px;" alt="Equity Curve">'
    return web.Response(text=html, content_type="text/html")


async def api_arena_status(request: web.Request) -> web.Response:
    bot = request.app["bot"]
    arena = bot.get("arena")
    if not arena:
        return web.json_response({"error": "arena not initialized"}, status=503)
    loop = asyncio.get_running_loop()
    status = await loop.run_in_executor(None, arena.get_status)
    return web.json_response(status)


async def alerts_partial_handler(request: web.Request) -> web.Response:
    from watchlist import get_state
    from datetime import datetime, timezone

    state = get_state()
    cooldowns = state.get("cooldowns", {})

    alerts = {}
    for ticker, info in cooldowns.items():
        try:
            t = datetime.fromisoformat(info["time"])
            time_str = t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            time_str = info.get("time", "?")
        alerts[ticker] = {
            "score": info.get("score", 0),
            "level": info.get("level", "?"),
            "time_str": time_str,
        }

    return _render(request, "_partials/alert_list.html", {"alerts": alerts})


# ---------------------------------------------------------------------------
# GitHub Webhook — 自動部署
# ---------------------------------------------------------------------------

def _verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify X-Hub-Signature-256 from GitHub webhook."""
    if not signature.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


async def api_webhook_github(request: web.Request) -> web.Response:
    """GitHub webhook endpoint for auto-deploy."""
    webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    if not webhook_secret:
        log.warning("GitHub webhook called but GITHUB_WEBHOOK_SECRET not set")
        return web.json_response({"error": "webhook not configured"}, status=503)

    # Verify signature
    body = await request.read()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not _verify_github_signature(body, signature, webhook_secret):
        log.warning("GitHub webhook: invalid signature")
        return web.json_response({"error": "invalid signature"}, status=403)

    event = request.headers.get("X-GitHub-Event", "")
    log.info(f"GitHub webhook received: event={event}")

    if event != "push":
        return web.json_response({"status": "ignored", "event": event})

    # Run deploy.sh in background
    deploy_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deploy.sh")
    if not os.path.exists(deploy_script):
        log.error(f"deploy.sh not found at {deploy_script}")
        return web.json_response({"error": "deploy.sh not found"}, status=500)

    async def _run_deploy():
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", deploy_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                log.info(f"Deploy succeeded:\n{stdout.decode()}")
            else:
                log.error(f"Deploy failed (code {proc.returncode}):\n{stderr.decode()}")

            # Notify on Discord if possible
            bot = request.app.get("bot")
            if bot:
                client = bot["client"]
                from watchlist import get_chat_channel, get_state
                ch_id = get_chat_channel() or get_state().get("channel_id")
                if ch_id and client.is_ready():
                    channel = client.get_channel(ch_id)
                    if channel:
                        status = "success" if proc.returncode == 0 else "FAILED"
                        await channel.send(
                            f"🚀 **自動部署 [{status}]**\n"
                            f"```\n{stdout.decode()[-500:]}\n```"
                        )
        except Exception as e:
            log.error(f"Deploy error: {e}")

    asyncio.create_task(_run_deploy())

    return web.json_response({"status": "deploying"})


# ---------------------------------------------------------------------------
# Route setup
# ---------------------------------------------------------------------------

def setup_routes(app: web.Application) -> None:
    # Auth
    app.router.add_get("/auth/login", login_page)
    app.router.add_post("/auth/password", password_login)
    app.router.add_get("/auth/callback", oauth_callback)
    app.router.add_get("/auth/logout", logout)

    # Pages
    app.router.add_get("/", dashboard_page)
    app.router.add_get("/watchlist", watchlist_page)
    app.router.add_get("/settings", settings_page)
    app.router.add_post("/settings", settings_save)
    app.router.add_get("/chat-history", chat_history_page)

    # API
    app.router.add_post("/api/watchlist/add", api_watchlist_add)
    app.router.add_post("/api/watchlist/remove", api_watchlist_remove)
    app.router.add_get("/api/watchlist/scores", api_watchlist_scores)
    app.router.add_post("/api/monitor/toggle", api_monitor_toggle)
    app.router.add_get("/api/market-status", api_market_status)

    # Arena
    app.router.add_get("/arena", arena_page)
    app.router.add_get("/api/arena/chart", api_arena_chart)
    app.router.add_get("/api/arena/status", api_arena_status)

    # GitHub webhook
    app.router.add_post("/api/webhook/github", api_webhook_github)

    # HTMX partials
    app.router.add_get("/api/partials/scores", scores_partial_handler)
    app.router.add_get("/api/partials/alerts", alerts_partial_handler)
