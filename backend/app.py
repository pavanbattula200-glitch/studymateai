"""
StudyMate AI — Flask Application Entry Point
Run with:  python app.py
Or:        gunicorn app:app
"""

import os
from datetime import timedelta
from flask import Flask
from flask_cors import CORS

from config import Config
from database import init_db
from auth import auth_bp
from routes import study_bp


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
        allow_headers=["Content-Type"],
        methods=["GET", "POST", "DELETE", "OPTIONS"],
    )

    # ── Blueprints ────────────────────────────────────────────────────────
    app.register_blueprint(auth_bp)
    app.register_blueprint(study_bp)

    # ── Database ──────────────────────────────────────────────────────────
    with app.app_context():
        init_db()

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true")

    print("=" * 55)
    print("  StudyMate AI — Flask Backend")
    print(f"  URL : http://localhost:{port}")
    print("  DB  :", Config.DATABASE_PATH)
    print("  AI  :", Config.OLLAMA_BASE_URL, "/", Config.OLLAMA_MODEL)
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=debug)
