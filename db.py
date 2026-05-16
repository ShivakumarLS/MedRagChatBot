import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = "ratings.db"

def _utc_now() -> str:
    return datetime.utcnow().isoformat()

def init_db():
    """Initialize SQLite schema (users, chats, ratings)."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            role TEXT NOT NULL,          -- 'user' | 'bot'
            content TEXT NOT NULL,
            latency_ms INTEGER,
            created_at TEXT NOT NULL
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            user_message TEXT,
            bot_reply TEXT,
            helpful INTEGER,          -- 1 = yes, 0 = no
            latency_ms INTEGER,
            created_at TEXT
        )
        """)
        con.commit()

def create_user(username: str, password: str) -> Tuple[bool, str]:
    username = (username or "").strip()
    password = (password or "").strip()

    if not username or not password:
        return False, "Username and password are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    pw_hash = generate_password_hash(password)

    try:
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, pw_hash, _utc_now()),
            )
            con.commit()
        return True, "ok"
    except sqlite3.IntegrityError:
        return False, "Username already exists."

def verify_user(username: str, password: str) -> bool:
    username = (username or "").strip()
    password = (password or "").strip()
    if not username or not password:
        return False

    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if not row:
        return False

    return check_password_hash(row[0], password)

def save_chat(username: str, role: str, content: str, latency_ms: Optional[int] = None) -> None:
    username = (username or "").strip()
    role = (role or "").strip()
    content = (content or "").strip()
    if not username or role not in ("user", "bot") or not content:
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO chats (username, role, content, latency_ms, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, role, content, int(latency_ms) if latency_ms is not None else None, _utc_now()),
        )
        con.commit()

def get_chat_history(username: str, limit: int = 50) -> List[Dict]:
    username = (username or "").strip()
    if not username:
        return []

    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT role, content, latency_ms, created_at FROM chats WHERE username = ? ORDER BY id ASC LIMIT ?",
            (username, int(limit)),
        ).fetchall()

    return [
        {"role": r[0], "content": r[1], "latency_ms": r[2], "created_at": r[3]}
        for r in rows
    ]

def save_rating(username, user_message, bot_reply, helpful, latency_ms):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        INSERT INTO ratings (username, user_message, bot_reply, helpful, latency_ms, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            username,
            user_message,
            bot_reply,
            int(helpful),
            int(latency_ms) if latency_ms is not None else None,
            _utc_now()
        ))
        con.commit()
