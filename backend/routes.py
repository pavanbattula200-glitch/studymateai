"""
StudyMate AI — Study Feature Routes
GET    /api/health
GET    /api/model/status
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

def _log_history(user_id: int | None, activity_type: str, topic: str) -> None:
    """Write a study history record — fails silently so AI errors don't hide it."""
    if not user_id:
        return
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


# ── health & model status ─────────────────────────────────────────────────────

@study_bp.get("/api/health")
def health():
    """Public health check. Also reports Ollama availability and memory status."""
    ollama_status = ollama_helper.check_ollama()
    uid = current_user_id()
    user_info = None
    if uid:
        try:
            from auth import _user_row
            user_info = _user_row(uid)
        except Exception:
            pass

    return jsonify({
        "status": "ok",
        "backend": "StudyMate AI Flask Backend",
        "ollama": ollama_status,
        "user": user_info,
    }), 200


@study_bp.get("/api/model/status")
def model_status():
    """Check whether the model is installed, loaded in memory, and ready for inference."""
    details = ollama_helper.get_model_details()
    return jsonify(details), 200


# ── study features ────────────────────────────────────────────────────────────

@study_bp.post("/api/summarize")
def summarize():
    err = require_login()
    if err:
        return err

    # Support JSON payload or form data, checking both 'text' and 'notes'
    data = request.get_json(silent=True) or {}
    text = (
        data.get("text")
        or data.get("notes")
        or request.form.get("text")
        or request.form.get("notes")
        or ""
    ).strip()

    if not text:
        return jsonify({"error": "Notes text is required."}), 400
    if len(text) > 15_000:
        return jsonify({"error": "Notes are too long (max 15 000 characters)."}), 400

    try:
        result = ollama_helper.summarize(text)
    except Exception as exc:
        return jsonify({"error": f"Summarization error: {exc}"}), 500

    if not result.get("ok"):
        return jsonify({"error": result.get("error", "Summarization failed.")}), 503

    summary = (result.get("summary") or "").strip()
    if not summary:
        return jsonify({"error": "The local Gemma4 model produced an empty summary. Please try again."}), 500

    _log_history(current_user_id(), "summarize", text[:80])
    return jsonify({"summary": summary}), 200


@study_bp.post("/api/explain")
def explain():
    err = require_login()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or request.form.get("topic") or "").strip()
    context = (data.get("context") or request.form.get("context") or "").strip()
    if not topic:
        return jsonify({"error": "Topic is required."}), 400

    try:
        result = ollama_helper.explain(topic, context)
    except Exception as exc:
        return jsonify({"error": f"Explanation error: {exc}"}), 500

    if not result.get("ok"):
        return jsonify({"error": result.get("error", "Explanation failed.")}), 503

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
    topic = (data.get("topic") or request.form.get("topic") or "").strip()
    num_q_val = data.get("num_questions") or request.form.get("num_questions") or 5
    try:
        num_q = int(num_q_val)
    except (ValueError, TypeError):
        num_q = 5
    difficulty = (data.get("difficulty") or request.form.get("difficulty") or "medium").strip()

    if not topic:
        return jsonify({"error": "Topic is required."}), 400
    num_q = max(1, min(num_q, 20))  # clamp 1–20

    try:
        result = ollama_helper.generate_quiz(topic, num_q, difficulty)
    except Exception as exc:
        return jsonify({"error": f"Quiz generation error: {exc}"}), 500

    if not result.get("ok"):
        return jsonify({"error": result.get("error", "Quiz generation failed.")}), 503

    _log_history(current_user_id(), "quiz", topic)
    return jsonify({"questions": result.get("questions", [])}), 200


@study_bp.post("/api/ask")
def ask():
    err = require_login()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    notes = (data.get("notes") or request.form.get("notes") or "").strip()
    question = (data.get("question") or request.form.get("question") or "").strip()
    if not notes:
        return jsonify({"error": "Notes are required."}), 400
    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        result = ollama_helper.ask_notes(notes, question)
    except Exception as exc:
        return jsonify({"error": f"Ask notes error: {exc}"}), 500

    if not result.get("ok"):
        return jsonify({"error": result.get("error", "Q&A failed.")}), 503

    _log_history(current_user_id(), "ask", question[:80])
    return jsonify({"answer": result.get("answer", "")}), 200


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
