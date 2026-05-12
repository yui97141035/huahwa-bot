"""Arena SQLite 資料庫 — 交易紀錄、持倉、績效快照。"""

import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

_log = logging.getLogger("arena.db")

_TW = timezone(timedelta(hours=8))
_DB_PATH = Path(__file__).resolve().parent.parent / "arena.db"
_lock = threading.Lock()


def _now_tw() -> str:
    return datetime.now(_TW).strftime("%Y-%m-%d %H:%M:%S")


def _today_tw() -> str:
    return datetime.now(_TW).strftime("%Y-%m-%d")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_DB_PATH), timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def init_db() -> None:
    """建立所有資料表（冪等）。"""
    with _lock:
        c = _conn()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS bots (
            bot_id   TEXT PRIMARY KEY,
            name     TEXT NOT NULL,
            status   TEXT NOT NULL DEFAULT 'active'  -- active / eliminated / champion
        );

        CREATE TABLE IF NOT EXISTS portfolios (
            bot_id  TEXT PRIMARY KEY REFERENCES bots(bot_id),
            cash    REAL NOT NULL DEFAULT 50.0
        );

        CREATE TABLE IF NOT EXISTS positions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id      TEXT NOT NULL REFERENCES bots(bot_id),
            ticker      TEXT NOT NULL,
            market      TEXT NOT NULL DEFAULT 'US',  -- US / TW
            shares      REAL NOT NULL DEFAULT 0,
            entry_price REAL NOT NULL,
            stop_loss   REAL,
            opened_at   TEXT NOT NULL,
            closed_at   TEXT,
            status      TEXT NOT NULL DEFAULT 'open'  -- open / closed
        );
        CREATE INDEX IF NOT EXISTS idx_positions_bot_status
            ON positions(bot_id, status);

        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id      TEXT NOT NULL REFERENCES bots(bot_id),
            ticker      TEXT NOT NULL,
            market      TEXT NOT NULL DEFAULT 'US',
            side        TEXT NOT NULL,  -- buy / sell
            shares      REAL NOT NULL,
            price       REAL NOT NULL,
            cost        REAL NOT NULL DEFAULT 0,  -- 手續費
            pnl         REAL,
            reason      TEXT,
            order_type  TEXT NOT NULL DEFAULT 'market',
            executed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_trades_bot ON trades(bot_id);

        CREATE TABLE IF NOT EXISTS daily_snapshots (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id   TEXT NOT NULL REFERENCES bots(bot_id),
            date     TEXT NOT NULL,
            equity   REAL NOT NULL,   -- cash + 持倉市值
            cash     REAL NOT NULL,
            UNIQUE(bot_id, date)
        );

        CREATE TABLE IF NOT EXISTS metrics (
            bot_id          TEXT PRIMARY KEY REFERENCES bots(bot_id),
            total_return_pct REAL DEFAULT 0,
            sharpe_ratio    REAL DEFAULT 0,
            max_drawdown    REAL DEFAULT 0,
            win_rate        REAL DEFAULT 0,
            profit_factor   REAL DEFAULT 0,
            total_trades    INTEGER DEFAULT 0,
            winning_trades  INTEGER DEFAULT 0,
            losing_trades   INTEGER DEFAULT 0,
            updated_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS strategy_params (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id    TEXT NOT NULL REFERENCES bots(bot_id),
            param_key TEXT NOT NULL,
            old_value REAL,
            new_value REAL,
            changed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sparams_bot ON strategy_params(bot_id);

        CREATE TABLE IF NOT EXISTS risk_events (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id    TEXT NOT NULL REFERENCES bots(bot_id),
            event     TEXT NOT NULL,  -- stop_loss / daily_limit / position_limit
            detail    TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_risk_bot ON risk_events(bot_id);
        """)
        c.commit()
        c.close()
        _log.info("arena_db: schema initialized")


# ---------------------------------------------------------------------------
# Bot helpers
# ---------------------------------------------------------------------------

def ensure_bot(bot_id: str, name: str, initial_cash: float = 50.0) -> None:
    """確保 bot 存在，若不存在則新增。"""
    with _lock:
        c = _conn()
        existing = c.execute("SELECT bot_id FROM bots WHERE bot_id=?", (bot_id,)).fetchone()
        if not existing:
            c.execute("INSERT INTO bots(bot_id, name) VALUES(?,?)", (bot_id, name))
            c.execute("INSERT INTO portfolios(bot_id, cash) VALUES(?,?)", (bot_id, initial_cash))
            c.execute(
                "INSERT INTO metrics(bot_id, updated_at) VALUES(?,?)",
                (bot_id, _now_tw()),
            )
            c.commit()
            _log.info(f"arena_db: created bot {bot_id} ({name}) with ${initial_cash}")
        c.close()


def get_bot(bot_id: str) -> dict | None:
    c = _conn()
    row = c.execute("SELECT * FROM bots WHERE bot_id=?", (bot_id,)).fetchone()
    c.close()
    return dict(row) if row else None


def get_all_bots() -> list[dict]:
    c = _conn()
    rows = c.execute("SELECT * FROM bots").fetchall()
    c.close()
    return [dict(r) for r in rows]


def set_bot_status(bot_id: str, status: str) -> None:
    with _lock:
        c = _conn()
        c.execute("UPDATE bots SET status=? WHERE bot_id=?", (status, bot_id))
        c.commit()
        c.close()


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def get_cash(bot_id: str) -> float:
    c = _conn()
    row = c.execute("SELECT cash FROM portfolios WHERE bot_id=?", (bot_id,)).fetchone()
    c.close()
    return float(row["cash"]) if row else 0.0


def update_cash(bot_id: str, delta: float) -> float:
    """增減現金，回傳新餘額。"""
    with _lock:
        c = _conn()
        c.execute("UPDATE portfolios SET cash = cash + ? WHERE bot_id=?", (delta, bot_id))
        c.commit()
        row = c.execute("SELECT cash FROM portfolios WHERE bot_id=?", (bot_id,)).fetchone()
        c.close()
        return float(row["cash"]) if row else 0.0


def set_cash(bot_id: str, cash: float) -> None:
    with _lock:
        c = _conn()
        c.execute("UPDATE portfolios SET cash=? WHERE bot_id=?", (cash, bot_id))
        c.commit()
        c.close()


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def open_position(bot_id: str, ticker: str, market: str,
                  shares: float, entry_price: float, stop_loss: float | None = None) -> int:
    with _lock:
        c = _conn()
        cur = c.execute(
            """INSERT INTO positions(bot_id, ticker, market, shares, entry_price, stop_loss, opened_at)
               VALUES(?,?,?,?,?,?,?)""",
            (bot_id, ticker, market, shares, entry_price, stop_loss, _now_tw()),
        )
        c.commit()
        row_id = cur.lastrowid
        c.close()
        return row_id


def close_position(position_id: int) -> None:
    with _lock:
        c = _conn()
        c.execute(
            "UPDATE positions SET status='closed', closed_at=? WHERE id=?",
            (_now_tw(), position_id),
        )
        c.commit()
        c.close()


def get_open_positions(bot_id: str) -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT * FROM positions WHERE bot_id=? AND status='open'", (bot_id,)
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_position_by_ticker(bot_id: str, ticker: str) -> dict | None:
    c = _conn()
    row = c.execute(
        "SELECT * FROM positions WHERE bot_id=? AND ticker=? AND status='open'",
        (bot_id, ticker),
    ).fetchone()
    c.close()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def record_trade(bot_id: str, ticker: str, market: str, side: str,
                 shares: float, price: float, cost: float = 0,
                 pnl: float | None = None, reason: str | None = None,
                 order_type: str = "market") -> int:
    with _lock:
        c = _conn()
        cur = c.execute(
            """INSERT INTO trades(bot_id, ticker, market, side, shares, price, cost, pnl, reason, order_type, executed_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (bot_id, ticker, market, side, shares, price, cost, pnl, reason, order_type, _now_tw()),
        )
        c.commit()
        row_id = cur.lastrowid
        c.close()
        return row_id


def get_recent_trades(bot_id: str | None = None, limit: int = 20) -> list[dict]:
    c = _conn()
    if bot_id:
        rows = c.execute(
            "SELECT * FROM trades WHERE bot_id=? ORDER BY id DESC LIMIT ?",
            (bot_id, limit),
        ).fetchall()
    else:
        rows = c.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_closed_trades(bot_id: str) -> list[dict]:
    """取得所有已完成的賣出交易（用於計算勝率等）。"""
    c = _conn()
    rows = c.execute(
        "SELECT * FROM trades WHERE bot_id=? AND side='sell' AND pnl IS NOT NULL ORDER BY id",
        (bot_id,),
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Daily Snapshots
# ---------------------------------------------------------------------------

def save_snapshot(bot_id: str, equity: float, cash: float,
                  date: str | None = None) -> None:
    date = date or _today_tw()
    with _lock:
        c = _conn()
        c.execute(
            """INSERT INTO daily_snapshots(bot_id, date, equity, cash) VALUES(?,?,?,?)
               ON CONFLICT(bot_id, date) DO UPDATE SET equity=excluded.equity, cash=excluded.cash""",
            (bot_id, date, equity, cash),
        )
        c.commit()
        c.close()


def get_snapshots(bot_id: str, limit: int = 90) -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT * FROM daily_snapshots WHERE bot_id=? ORDER BY date DESC LIMIT ?",
        (bot_id, limit),
    ).fetchall()
    c.close()
    return [dict(r) for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def update_metrics(bot_id: str, **kwargs) -> None:
    """更新績效指標，kwargs: total_return_pct, sharpe_ratio, max_drawdown, win_rate, profit_factor, total_trades, winning_trades, losing_trades."""
    if not kwargs:
        return
    kwargs["updated_at"] = _now_tw()
    sets = ", ".join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [bot_id]
    with _lock:
        c = _conn()
        c.execute(f"UPDATE metrics SET {sets} WHERE bot_id=?", vals)
        c.commit()
        c.close()


def get_metrics(bot_id: str) -> dict | None:
    c = _conn()
    row = c.execute("SELECT * FROM metrics WHERE bot_id=?", (bot_id,)).fetchone()
    c.close()
    return dict(row) if row else None


def get_all_metrics() -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT m.*, b.name, b.status FROM metrics m JOIN bots b ON m.bot_id=b.bot_id"
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Strategy Params
# ---------------------------------------------------------------------------

def log_param_change(bot_id: str, param_key: str,
                     old_value: float, new_value: float) -> None:
    with _lock:
        c = _conn()
        c.execute(
            "INSERT INTO strategy_params(bot_id, param_key, old_value, new_value, changed_at) VALUES(?,?,?,?,?)",
            (bot_id, param_key, old_value, new_value, _now_tw()),
        )
        c.commit()
        c.close()


def get_param_history(bot_id: str, limit: int = 20) -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT * FROM strategy_params WHERE bot_id=? ORDER BY id DESC LIMIT ?",
        (bot_id, limit),
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Risk Events
# ---------------------------------------------------------------------------

def log_risk_event(bot_id: str, event: str, detail: str | None = None) -> None:
    with _lock:
        c = _conn()
        c.execute(
            "INSERT INTO risk_events(bot_id, event, detail, created_at) VALUES(?,?,?,?)",
            (bot_id, event, detail, _now_tw()),
        )
        c.commit()
        c.close()


def get_risk_events(bot_id: str | None = None, limit: int = 20) -> list[dict]:
    c = _conn()
    if bot_id:
        rows = c.execute(
            "SELECT * FROM risk_events WHERE bot_id=? ORDER BY id DESC LIMIT ?",
            (bot_id, limit),
        ).fetchall()
    else:
        rows = c.execute(
            "SELECT * FROM risk_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    c.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_trading_days(bot_id: str) -> int:
    """計算該 Bot 有快照的交易天數。"""
    c = _conn()
    row = c.execute(
        "SELECT COUNT(DISTINCT date) AS cnt FROM daily_snapshots WHERE bot_id=?",
        (bot_id,),
    ).fetchone()
    c.close()
    return int(row["cnt"]) if row else 0


def get_daily_pnl(bot_id: str, date: str | None = None) -> float:
    """當日已實現 PnL。"""
    date = date or _today_tw()
    c = _conn()
    row = c.execute(
        "SELECT COALESCE(SUM(pnl), 0) AS total FROM trades WHERE bot_id=? AND executed_at LIKE ?",
        (bot_id, f"{date}%"),
    ).fetchone()
    c.close()
    return float(row["total"]) if row else 0.0
