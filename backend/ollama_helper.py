"""
StudyMate AI — Ollama / Local AI Helper
Talks to the locally running Ollama service.
If Ollama is not available, functions return a structured error
instead of crashing — the rest of the backend continues to work.
"""

import json
import logging
import requests
from config import Config

logger = logging.getLogger(__name__)

# Timeout in seconds for AI inference calls
INFERENCE_TIMEOUT = 120


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
    1. Try exact configured model from Config.OLLAMA_MODEL.
    2. Try prefix/tag-insensitive match (e.g. 'gemma4' -> 'gemma4:latest').
    3. If configured model is not found, fallback to first available installed model.
    4. Otherwise return Config.OLLAMA_MODEL.
    """
    configured = (Config.OLLAMA_MODEL or "").strip()
    available = get_available_models()
    if not available:
        return configured or "gemma4:latest"

    # 1. Exact match
    if configured in available:
        return configured

    # 2. Tag-insensitive match (e.g. 'gemma4' matching 'gemma4:latest')
    for m in available:
        base_name = m.split(":")[0]
        if configured == base_name or (":" in configured and configured.split(":")[0] == base_name):
            return m

    # 3. Fallback to first available installed model
    return available[0]


def _ollama_generate(prompt: str) -> dict:
    """
    Call the Ollama /api/generate endpoint.
    Returns {"ok": True, "text": "..."} or {"ok": False, "error": "..."}.
    """
    base_url = get_base_url()
    model_to_use = resolve_model()
    url = f"{base_url}/api/generate"
    payload = {
        "model": model_to_use,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=INFERENCE_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            return {"ok": True, "text": data.get("response", "").strip(), "model": model_to_use}

        # Ollama returned non-200 (e.g. 404 model not found)
        error_msg = f"HTTP {resp.status_code}"
        try:
            err_json = resp.json()
            if "error" in err_json:
                error_msg = err_json["error"]
        except Exception:
            if resp.text:
                error_msg = resp.text

        available = get_available_models()
        avail_str = ", ".join(available) if available else "none"
        return {
            "ok": False,
            "error": f"Ollama error: {error_msg}. Configured: '{Config.OLLAMA_MODEL}', Available in Ollama: [{avail_str}]",
        }

    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "error": (
                "Ollama is not running. Start it with: ollama serve  "
                "Then pull a model: ollama pull gemma4"
            ),
        }
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Ollama inference timed out. Try a smaller model or increase timeout."}
    except Exception as exc:
        return {"ok": False, "error": f"Ollama error: {exc}"}


def check_ollama() -> dict:
    """Return a status dict describing whether Ollama + the model are available."""
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
    prompt = (
        "You are a study assistant. Summarize the following notes clearly and concisely "
        "in plain English. Focus on the key ideas. Do not add information not present in the notes.\n\n"
        f"NOTES:\n{text}\n\nSUMMARY:"
    )
    result = _ollama_generate(prompt)
    if result["ok"]:
        return {"ok": True, "summary": result["text"]}
    return result


def explain(topic: str, context: str = "") -> dict:
    ctx_part = f"\n\nAdditional context:\n{context}" if context.strip() else ""
    prompt = (
        f"Explain the concept '{topic}' in simple, clear language suitable for a student.{ctx_part}\n\n"
        "Respond in this exact JSON format (no markdown, just JSON):\n"
        '{"explanation": "...", "key_points": ["...", "..."], "example": "..."}'
    )
    result = _ollama_generate(prompt)
    if not result["ok"]:
        return result
    try:
        # Try to parse JSON from the model's response
        raw = result["text"]
        # Strip any wrapping markdown code fences if present
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
        parsed = json.loads(raw)
        return {"ok": True, **parsed}
    except (json.JSONDecodeError, IndexError):
        # Model didn't return clean JSON — return raw as explanation
        return {"ok": True, "explanation": result["text"], "key_points": [], "example": ""}


def generate_quiz(topic: str, num_questions: int = 5, difficulty: str = "medium") -> dict:
    prompt = (
        f"Create a {difficulty} difficulty multiple-choice quiz about '{topic}'. "
        f"Generate exactly {num_questions} questions. Each question must have exactly 4 options "
        f"and one correct answer.\n\n"
        "Respond in this exact JSON format (no markdown):\n"
        '{"questions": [{"question": "...", "options": ["A", "B", "C", "D"], "correct_index": 0}, ...]}'
    )
    result = _ollama_generate(prompt)
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
        return {"ok": True, "questions": parsed.get("questions", [])}
    except (json.JSONDecodeError, IndexError):
        return {"ok": False, "error": "The AI returned an unexpected format. Please try again."}


def ask_notes(notes: str, question: str) -> dict:
    prompt = (
        "You are a study assistant. Answer the student's question using ONLY the notes provided. "
        "If the answer is not in the notes, say so clearly.\n\n"
        f"NOTES:\n{notes}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    result = _ollama_generate(prompt)
    if result["ok"]:
        return {"ok": True, "answer": result["text"]}
    return result
