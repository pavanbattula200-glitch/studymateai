"""
StudyMate AI — Study Feature Routes
GET    /api/health
POST   /api/summarize
POST   /api/explain
POST   /api/quiz/generate
POST   /api/ask
GET    /api/history
DELETE /api/history/<id>
DELETE /api/history         (clear all for current user)
"""

from flask import Blueprint, request, session, jsonify
from auth import current_user_id, require_login
from database import get_db
import ollama_helper

study_bp = Blueprint("study", __name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _log_history(user_id: int, activity_type: str, topic: str) -> None:
    """Write a study history record — fails silently so AI errors don't hide it."""
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO study_history (user_id, type, topic) VALUES (?, ?, ?)",
            (user_id, activity_type, topic[:200]),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ── health ────────────────────────────────────────────────────────────────────

@study_bp.get("/api/health")
def health():
    """Public health check. Also reports Ollama availability."""
    ollama_status = ollama_helper.check_ollama()
    return jsonify({
        "status": "ok",
        "backend": "StudyMate AI Flask Backend",
        "ollama": ollama_status,
    }), 200


# ── study features ────────────────────────────────────────────────────────────

@study_bp.post("/api/summarize")
def summarize():
    err = require_login()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Notes text is required."}), 400
    if len(text) > 10_000:
        return jsonify({"error": "Notes are too long (max 10 000 characters)."}), 400

    result = ollama_helper.summarize(text)
    if not result["ok"]:
        return jsonify({"error": result["error"]}), 503

    _log_history(current_user_id(), "summarize", text[:80])
    return jsonify({"summary": result["summary"]}), 200


@study_bp.post("/api/explain")
def explain():
    err = require_login()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    context = (data.get("context") or "").strip()
    if not topic:
        return jsonify({"error": "Topic is required."}), 400

    result = ollama_helper.explain(topic, context)
    if not result["ok"]:
        return jsonify({"error": result["error"]}), 503

    _log_history(current_user_id(), "explain", topic)
    return jsonify({
        "explanation": result.get("explanation", ""),
        "key_points":  result.get("key_points", []),
        "example":     result.get("example", ""),
    }), 200


@study_bp.post("/api/quiz/generate")
def quiz_generate():
    err = require_login()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    num_q = int(data.get("num_questions") or 5)
    difficulty = (data.get("difficulty") or "medium").strip()

    if not topic:
        return jsonify({"error": "Topic is required."}), 400
    num_q = max(1, min(num_q, 20))  # clamp 1–20

    result = ollama_helper.generate_quiz(topic, num_q, difficulty)
    if not result["ok"]:
        return jsonify({"error": result["error"]}), 503

    _log_history(current_user_id(), "quiz", topic)
    return jsonify({"questions": result["questions"]}), 200


@study_bp.post("/api/ask")
def ask():
    err = require_login()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    notes    = (data.get("notes") or "").strip()
    question = (data.get("question") or "").strip()
    if not notes:
        return jsonify({"error": "Notes are required."}), 400
    if not question:
        return jsonify({"error": "Question is required."}), 400

    result = ollama_helper.ask_notes(notes, question)
    if not result["ok"]:
        return jsonify({"error": result["error"]}), 503

    _log_history(current_user_id(), "ask", question[:80])
    return jsonify({"answer": result["answer"]}), 200


# ── history ───────────────────────────────────────────────────────────────────

@study_bp.get("/api/history")
def get_history():
    err = require_login()
    if err:
        return err

    uid = current_user_id()
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, type, topic, created_at FROM study_history "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT 200",
            (uid,),
        ).fetchall()
    finally:
        conn.close()

    return jsonify({"history": [dict(r) for r in rows]}), 200


@study_bp.delete("/api/history/<int:record_id>")
def delete_history_item(record_id: int):
    err = require_login()
    if err:
        return err

    uid = current_user_id()
    conn = get_db()
    try:
        # Only delete if it belongs to the current user
        row = conn.execute(
            "SELECT id FROM study_history WHERE id = ? AND user_id = ?",
            (record_id, uid),
        ).fetchone()
        if not row:
            return jsonify({"error": "Record not found."}), 404
        conn.execute("DELETE FROM study_history WHERE id = ?", (record_id,))
        conn.commit()
    finally:
        conn.close()

    return jsonify({"message": "Deleted."}), 200


@study_bp.delete("/api/history")
def clear_history():
    err = require_login()
    if err:
        return err

    uid = current_user_id()
    conn = get_db()
    try:
        conn.execute("DELETE FROM study_history WHERE user_id = ?", (uid,))
        conn.commit()
    finally:
        conn.close()

    return jsonify({"message": "History cleared."}), 200
