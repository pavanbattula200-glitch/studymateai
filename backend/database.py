"""
StudyMate AI — SQLite Database Layer
Handles connection management and schema initialisation.
All SQL uses parameterised queries to prevent injection.
"""

import sqlite3
from config import Config


def get_db() -> sqlite3.Connection:
    """Open and return a new database connection with row_factory set."""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they do not already exist. Safe to call on every start."""
    conn = get_db()
    try:
        conn.executescript("""
            -- Users table
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                password    TEXT    NOT NULL,         -- bcrypt / werkzeug hash
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- Study history table
            CREATE TABLE IF NOT EXISTS study_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                type        TEXT    NOT NULL,  -- summarize | explain | quiz | ask
                topic       TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)
        conn.commit()
    finally:
        conn.close()
