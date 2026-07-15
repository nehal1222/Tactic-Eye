import sqlite3
from datetime import datetime, timezone

DB_PATH = "app_data.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            video_name TEXT NOT NULL,
            frames_processed INTEGER NOT NULL,
            output_path TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subscription (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            is_premium INTEGER NOT NULL DEFAULT 0,
            stripe_customer_id TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("INSERT OR IGNORE INTO subscription (id, is_premium) VALUES (1, 0)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS waitlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            name TEXT,
            email TEXT NOT NULL UNIQUE
        )
    """)
    conn.commit()
    conn.close()


def add_history(video_name, frames_processed, output_path):
    conn = get_connection()
    conn.execute(
        "INSERT INTO history (timestamp, video_name, frames_processed, output_path) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), video_name, frames_processed, output_path),
    )
    conn.commit()
    conn.close()


def get_history(limit=20):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows


def is_premium():
    conn = get_connection()
    row = conn.execute("SELECT is_premium FROM subscription WHERE id = 1").fetchone()
    conn.close()
    return bool(row["is_premium"])


def set_premium(is_premium_flag, stripe_customer_id=None):
    conn = get_connection()
    conn.execute(
        "UPDATE subscription SET is_premium = ?, stripe_customer_id = ?, updated_at = ? WHERE id = 1",
        (int(is_premium_flag), stripe_customer_id, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def add_to_waitlist(email, name=None):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO waitlist (timestamp, name, email) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), name, email),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # email already on the waitlist
    finally:
        conn.close()


def get_waitlist_count():
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) AS total FROM waitlist").fetchone()
    conn.close()
    return row["total"]


def get_all_history():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM history ORDER BY id").fetchall()
    conn.close()
    return rows


def get_all_waitlist():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM waitlist ORDER BY id").fetchall()
    conn.close()
    return rows
