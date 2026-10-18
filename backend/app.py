"""
StudyMate AI — Flask Application Entry Point
Run with:  python app.py
Or:        gunicorn app:app
"""

import os
import sys
import threading
from datetime import timedelta
from flask import Flask
from flask_cors import CORS

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import Config
from database import init_db
from auth import auth_bp
from routes import study_bp
import ollama_helper


def create_app() -> Flask:
    app = Flask(__name__)

    # ── Configuration ──────────────────────────────────────────────────────
    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.config["SESSION_COOKIE_HTTPONLY"] = Config.SESSION_COOKIE_HTTPONLY
    app.config["SESSION_COOKIE_SAMESITE"] = Config.SESSION_COOKIE_SAMESITE
    app.config["SESSION_COOKIE_SECURE"] = Config.SESSION_COOKIE_SECURE
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

    # ── CORS ──────────────────────────────────────────────────────────────
    # Allow configured frontend origins to send cookies (credentials).
    CORS(
        app,
        origins=Config.FRONTEND_ORIGINS,
        supports_credentials=True,   # required for session cookies
        allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept"],
        expose_headers=["Content-Type"],
        methods=["GET", "POST", "DELETE", "OPTIONS"],
    )

    # ── Blueprints ────────────────────────────────────────────────────────
    app.register_blueprint(auth_bp)
    app.register_blueprint(study_bp)

    # ── Database ──────────────────────────────────────────────────────────
    with app.app_context():
        init_db()

    # Pre-warm model in background thread so offline inferences start without load delay
    def _background_warmup():
        try:
            ollama_helper.warmup_model()
        except Exception:
            pass

    threading.Thread(target=_background_warmup, daemon=True).start()

    return app


app = create_app()


if __name__ == "__main__":
    port = Config.PORT
    debug = Config.FLASK_DEBUG

    print("=" * 55)
    print("  StudyMate AI — Flask Backend (Offline Mode)")
    print(f"  URL : http://localhost:{port}")
    print("  DB  :", Config.DATABASE_PATH)
    print("  AI  :", Config.OLLAMA_BASE_URL, "/", Config.OLLAMA_MODEL)
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
