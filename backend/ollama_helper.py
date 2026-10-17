"""
StudyMate AI — Ollama / Local AI Helper
Connects to locally running Ollama via the /api/chat endpoint.
Uses the model specified in .env (e.g. gemma4:latest).
All operations run 100% locally and offline without external dependencies.
"""

import json
import logging
import requests
from config import Config

logger = logging.getLogger(__name__)

# Timeout in seconds for AI inference calls (local models can take 10-60s depending on hardware)
INFERENCE_TIMEOUT = 180


def get_base_url() -> str:
    return Config.OLLAMA_BASE_URL.rstrip("/")


def get_available_models() -> list[str]:
    """Fetch list of model names currently installed in local Ollama."""
    try:
        resp = requests.get(f"{get_base_url()}/api/tags", timeout=5)
        if resp.status_code == 200:
            return [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
    except Exception:
        pass
    return []


def resolve_model() -> str:
    """
    Resolve which model to use:
    1. Read Config.OLLAMA_MODEL (from .env).
    2. If matched against available models in Ollama, use it.
    3. If tag is omitted (e.g. 'gemma4' vs 'gemma4:latest'), match appropriately.
    4. Fall back to first installed model if configured model is missing.
    """
    configured = (Config.OLLAMA_MODEL or "").strip()
    available = get_available_models()
    if not available:
        return configured or "gemma4:latest"

    # Exact match
    if configured in available:
        return configured

    # Match ignoring tag (e.g. 'gemma4' matches 'gemma4:latest')
    for m in available:
        base_name = m.split(":")[0]
        if configured == base_name or (":" in configured and configured.split(":")[0] == base_name):
            return m

    # Fallback to first available installed model
    return available[0]


def _ollama_chat(messages: list[dict], format_type: str | None = None) -> dict:
    """
    Call the Ollama /api/chat endpoint with stream: false.
    Returns {"ok": True, "text": "...", "model": "..."} or {"ok": False, "error": "..."}.
    """
    base_url = get_base_url()
    model_to_use = resolve_model()
    url = f"{base_url}/api/chat"

    payload = {
        "model": model_to_use,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3,
        },
    }
    if format_type:
        payload["format"] = format_type

    try:
        resp = requests.post(url, json=payload, timeout=INFERENCE_TIMEOUT)

        if resp.status_code == 200:
            data = resp.json()
            message_obj = data.get("message", {})
            content = (message_obj.get("content") or "").strip()
            return {"ok": True, "text": content, "model": model_to_use}

        # Handle non-200 responses with clear messages
        error_msg = f"HTTP {resp.status_code}"
        try:
            err_json = resp.json()
            if "error" in err_json:
                error_msg = err_json["error"]
        except Exception:
            if resp.text:
                error_msg = resp.text[:200]

        available = get_available_models()
        avail_str = ", ".join(available) if available else "none detected"
        return {
            "ok": False,
            "error": f"Ollama error: {error_msg}. Configured: '{Config.OLLAMA_MODEL}', Available in Ollama: [{avail_str}]",
        }

    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "error": (
                "Ollama is not reachable on http://localhost:11434. "
                "Make sure Ollama is running (`ollama serve`)."
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "error": "Ollama response timed out. The local model took too long to complete.",
        }
    except Exception as exc:
        return {"ok": False, "error": f"Ollama request error: {exc}"}


def check_ollama() -> dict:
    """Return status dict describing whether Ollama and the configured model are available."""
    try:
        available = get_available_models()
        active = resolve_model()
        model_ready = bool(available and (active in available or any(active.split(":")[0] in m for m in available)))
        return {
            "ollama_running": True,
            "model_ready": model_ready,
            "configured_model": Config.OLLAMA_MODEL,
            "active_model": active,
            "available_models": available,
        }
    except Exception:
        return {
            "ollama_running": False,
            "model_ready": False,
            "configured_model": Config.OLLAMA_MODEL,
            "active_model": Config.OLLAMA_MODEL,
            "available_models": [],
        }


def summarize(text: str) -> dict:
    """Feature 1: Summarize Notes using local Gemma via /api/chat."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert study assistant. Summarize the user's study notes clearly, "
                "concisely, and accurately in well-structured paragraphs or bullet points. "
                "Focus on the core ideas, definitions, and key takeaways. Do not invent information."
            ),
        },
        {
            "role": "user",
            "content": f"Please summarize the following study notes:\n\n{text}",
        },
    ]

    result = _ollama_chat(messages)
    if result["ok"]:
        return {"ok": True, "summary": result["text"]}
    return result


def explain(topic: str, context: str = "") -> dict:
    """Feature 2: Explain Concepts using local Gemma via /api/chat."""
    ctx_instruction = f"\n\nContext to incorporate:\n{context}" if context.strip() else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert tutor. Explain academic and technical concepts in simple, "
                "intuitive language suitable for students. Always output valid JSON matching this schema:\n"
                '{"explanation": "...", "key_points": ["point 1", "point 2", ...], "example": "..."}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Explain the concept '{topic}'.{ctx_instruction}\n\n"
                "Provide a clear explanation, 3-5 bullet key points, and a memorable real-world example."
            ),
        },
    ]

    result = _ollama_chat(messages, format_type="json")
    if not result["ok"]:
        return result

    try:
        raw = result["text"]
        # Strip potential markdown formatting if returned
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
        parsed = json.loads(raw)
        return {
            "ok": True,
            "explanation": parsed.get("explanation", ""),
            "key_points": parsed.get("key_points", []),
            "example": parsed.get("example", ""),
        }
    except Exception:
        # Graceful fallback if JSON parsing fails
        return {
            "ok": True,
            "explanation": result["text"],
            "key_points": [],
            "example": "",
        }


def generate_quiz(topic: str, num_questions: int = 5, difficulty: str = "medium") -> dict:
    """Feature 3: Generate Quiz questions using local Gemma via /api/chat."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic test designer. Create multiple-choice quizzes that test understanding. "
                "Each question must have exactly 4 plausible options (labeled in text or an array) "
                "and an integer correct_index from 0 to 3 pointing to the correct option. "
                "Always output valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Create a {difficulty} difficulty quiz about '{topic}' with exactly {num_questions} questions.\n"
                "Respond in this JSON format:\n"
                '{"questions": [{"question": "...", "options": ["Option 1", "Option 2", "Option 3", "Option 4"], "correct_index": 0}]}'
            ),
        },
    ]

    result = _ollama_chat(messages, format_type="json")
    if not result["ok"]:
        return result

    try:
        raw = result["text"]
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
        parsed = json.loads(raw)
        questions = parsed.get("questions", [])
        return {"ok": True, "questions": questions}
    except Exception:
        return {"ok": False, "error": "The AI model returned an invalid quiz structure. Please try again."}


def ask_notes(notes: str, question: str) -> dict:
    """Feature 4: Ask Your Notes (grounded Q&A) using local Gemma via /api/chat."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict, helpful study assistant. Answer the student's question using "
                "ONLY the notes provided. If the answer cannot be found or deduced from the notes, "
                "state clearly: 'This information is not covered in your provided notes.'"
            ),
        },
        {
            "role": "user",
            "content": (
                f"NOTES:\n{notes}\n\n"
                f"QUESTION:\n{question}\n\n"
                "ANSWER:"
            ),
        },
    ]

    result = _ollama_chat(messages)
    if result["ok"]:
        return {"ok": True, "answer": result["text"]}
    return result
