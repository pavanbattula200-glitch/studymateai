"""
StudyMate AI — Authentication Routes
POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me

Rules:
- Username and hashed password stored in SQLite (no plaintext, no email).
- Passwords hashed with Werkzeug's generate_password_hash (pbkdf2:sha256).
- Server-side Flask sessions (signed cookies).
- Never return the password hash in any response.
"""

from flask import Blueprint, request, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


# ── helpers ──────────────────────────────────────────────────────────────────

def current_user_id() -> int | None:
    return session.get("user_id")


def require_login():
    """Return a 401 error response if the user is not logged in, else None."""
    if not current_user_id():
        return jsonify({"error": "Authentication required."}), 401
    return None


def _user_row(user_id: int) -> dict | None:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── routes ────────────────────────────────────────────────────────────────────

@auth_bp.post("/register")
def register():
    """Create a new local account. Requires username + password."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    # Validation
    if not username:
        return jsonify({"error": "Username is required."}), 400
    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters."}), 400
    if len(username) > 40:
        return jsonify({"error": "Username must be 40 characters or fewer."}), 400
    if not password:
        return jsonify({"error": "Password is required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    pw_hash = generate_password_hash(password)

    conn = get_db()
    try:
        # Check for duplicate username (COLLATE NOCASE in schema handles case)
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            return jsonify({"error": "That username is already taken."}), 409

        cursor = conn.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, pw_hash),
        )
        conn.commit()
        user_id = cursor.lastrowid
    finally:
        conn.close()

    # Auto-login after registration
    session.clear()
    session["user_id"] = user_id
    session.permanent = True

    return jsonify({
        "message": "Account created successfully.",
        "user": {"id": user_id, "username": username},
    }), 201


@auth_bp.post("/login")
def login():
    """Authenticate a local account."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, username, password FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()

    # Use constant-time comparison via check_password_hash
    if not row or not check_password_hash(row["password"], password):
        return jsonify({"error": "Invalid username or password."}), 401

    session.clear()
    session["user_id"] = row["id"]
    session.permanent = True

    return jsonify({
        "message": "Logged in successfully.",
        "user": {"id": row["id"], "username": row["username"]},
    }), 200


@auth_bp.post("/logout")
def logout():
    """Destroy the current session."""
    session.clear()
    return jsonify({"message": "Logged out."}), 200


@auth_bp.get("/me")
def me():
    """Return the current logged-in user (for session restoration)."""
    uid = current_user_id()
    if not uid:
        return jsonify({"error": "Not authenticated."}), 401

    user = _user_row(uid)
    if not user:
        session.clear()
        return jsonify({"error": "User no longer exists."}), 401

    return jsonify({"user": user}), 200
