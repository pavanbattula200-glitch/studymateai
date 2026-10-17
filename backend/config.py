"""
StudyMate AI — Backend Configuration
Reads values from .env (via python-dotenv).
Falls back to safe defaults so the app starts without a .env file during development.
"""

import os
import secrets
from dotenv import load_dotenv

# Load .env from the backend/ directory (where this file lives)
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


class Config:
    # Flask session signing key — MUST be changed in production
    SECRET_KEY: str = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

    # SQLite DB path — stored in backend/ by default
    DATABASE_PATH: str = os.path.join(
        os.path.dirname(__file__),
        os.environ.get("DATABASE_PATH", "studymate.db"),
    )

    # Session cookie settings
    SESSION_COOKIE_HTTPONLY: bool = True
    # Lax for local cross-port; None (with Secure=True) for cross-domain production
    SESSION_COOKIE_SAMESITE: str = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax")
    SESSION_COOKIE_SECURE: bool = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() in ("true", "1")

    # Ollama / AI model configuration
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "gemma4:latest")

    # Frontend origins (for CORS) — supports single URL or comma-separated list
    _raw_origins = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")
    FRONTEND_ORIGINS: list[str] = [orig.strip() for orig in _raw_origins.split(",") if orig.strip()]
    FRONTEND_ORIGIN: str = FRONTEND_ORIGINS[0] if FRONTEND_ORIGINS else "http://localhost:3000"
