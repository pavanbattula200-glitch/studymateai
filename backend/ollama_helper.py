"""
StudyMate AI — Ollama / Local AI Helper
Connects to locally running Ollama via the /api/chat endpoint.
Uses the model specified in .env (e.g. gemma4:latest).
All operations run 100% locally and offline without external dependencies.
"""

import json
import logging
import re
import sys
import time
import requests
from config import Config

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# Timeout in seconds for AI inference calls (local models on CPU can take 30-90s)
INFERENCE_TIMEOUT = 180

# In-memory cache for available models to avoid spamming Ollama on every health poll
_models_cache = {"timestamp": 0.0, "models": []}
CACHE_TTL = 30.0  # seconds


def get_base_url() -> str:
    return Config.OLLAMA_BASE_URL.rstrip("/")


def get_available_models(force_refresh: bool = False) -> list[str]:
    """Fetch list of model names currently installed in local Ollama with TTL caching."""
    now = time.time()
    if not force_refresh and (now - _models_cache["timestamp"] < CACHE_TTL) and _models_cache["models"]:
        return _models_cache["models"]

    try:
        resp = requests.get(f"{get_base_url()}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
            _models_cache["timestamp"] = now
            _models_cache["models"] = models
            return models
    except Exception:
        pass
    return _models_cache["models"]


def get_loaded_models() -> list[dict]:
    """Fetch currently loaded models in Ollama memory via /api/ps."""
    try:
        resp = requests.get(f"{get_base_url()}/api/ps", timeout=3)
        if resp.status_code == 200:
            return resp.json().get("models", [])
    except Exception:
        pass
    return []


def is_model_warm(model_name: str | None = None) -> bool:
    """Check if the model is currently active/warm in Ollama memory."""
    target = model_name or resolve_model()
    target_base = target.split(":")[0].lower()
    for m in get_loaded_models():
        name = (m.get("name") or m.get("model") or "").lower()
        if name == target.lower() or name.split(":")[0] == target_base:
            return True
    return False


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

    # Substring / family match (e.g. 'gemma' or 'gemma4' matches 'gemma4:latest')
    for m in available:
        if configured.lower() in m.lower() or m.lower().startswith(configured.lower()):
            return m

    # Fallback to first available installed model
    return available[0]


def warmup_model() -> bool:
    """
    Pre-load the model into memory with a 60-minute keep-alive.
    Can be run in a background thread on startup so subsequent inferences start immediately.
    """
    try:
        model = resolve_model()
        if is_model_warm(model):
            return True
        resp = requests.post(
            f"{get_base_url()}/api/chat",
            json={"model": model, "keep_alive": "60m"},
            timeout=60,
        )
        return resp.status_code == 200
    except Exception as exc:
        logger.warning("Model warm-up skipped or failed: %s", exc)
        return False


def _clean_ai_text(text: str) -> str:
    """Clean internal thinking blocks or whitespace from the response."""
    if not text:
        return ""
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _ollama_chat(
    messages: list[dict],
    format_type: str | None = None,
    num_predict: int = 1024,
    temperature: float = 0.3,
) -> dict:
    """
    Call the Ollama /api/chat endpoint with stream: false, think: false, and keep_alive: 60m.
    Disabling thinking prevents Gemma4 from consuming all tokens on internal reasoning
    and returning an empty content block.
    Returns {"ok": True, "text": "...", "model": "..."} or {"ok": False, "error": "..."}.
    """
    base_url = get_base_url()
    model_to_use = resolve_model()
    url = f"{base_url}/api/chat"

    payload = {
        "model": model_to_use,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "60m",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if format_type:
        payload["format"] = format_type

    try:
        resp = requests.post(url, json=payload, timeout=INFERENCE_TIMEOUT)

        if resp.status_code == 200:
            data = resp.json()
            message_obj = data.get("message", {})
            raw_content = message_obj.get("content") or ""
            content = _clean_ai_text(raw_content)

            # Fallback in case a model still placed output in thinking
            if not content and message_obj.get("thinking"):
                content = _clean_ai_text(message_obj.get("thinking"))

            if not content:
                return {
                    "ok": False,
                    "error": "The local Gemma4 model returned an empty response. Please try again with more detailed notes.",
                }
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

        available = get_available_models(force_refresh=True)
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
            "error": (
                "Ollama response timed out. The local model took too long to complete. "
                "Try summarizing a shorter section of notes."
            ),
        }
    except Exception as exc:
        return {"ok": False, "error": f"Ollama request error: {exc}"}


def check_ollama() -> dict:
    """Return status dict describing whether Ollama, model, and memory state are ready."""
    try:
        available = get_available_models()
        active = resolve_model()
        model_ready = bool(available and (active in available or any(active.split(":")[0] in m for m in available)))
        warm = is_model_warm(active)
        return {
            "ollama_running": True,
            "model_ready": model_ready,
            "model_warm": warm,
            "configured_model": Config.OLLAMA_MODEL,
            "active_model": active,
            "available_models": available,
        }
    except Exception:
        return {
            "ollama_running": False,
            "model_ready": False,
            "model_warm": False,
            "configured_model": Config.OLLAMA_MODEL,
            "active_model": Config.OLLAMA_MODEL,
            "available_models": [],
        }


def get_model_details() -> dict:
    """Detailed model status for health and settings endpoints."""
    check = check_ollama()
    loaded_list = get_loaded_models()
    active = check["active_model"]
    active_base = active.split(":")[0].lower()
    warm_info = next(
        (m for m in loaded_list if active_base in (m.get("name") or m.get("model") or "").lower()),
        None
    )
    return {
        **check,
        "is_loaded": bool(warm_info),
        "details": warm_info or {},
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

    result = _ollama_chat(messages, num_predict=1500)
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

    result = _ollama_chat(messages, format_type="json", num_predict=600)
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
        if isinstance(parsed, dict):
            return {
                "ok": True,
                "explanation": parsed.get("explanation", ""),
                "key_points": parsed.get("key_points", []) if isinstance(parsed.get("key_points"), list) else [],
                "example": parsed.get("example", ""),
            }
        return {
            "ok": True,
            "explanation": str(parsed),
            "key_points": [],
            "example": "",
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

    result = _ollama_chat(messages, format_type="json", num_predict=800)
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

        raw_questions = []
        if isinstance(parsed, list):
            raw_questions = parsed
        elif isinstance(parsed, dict):
            raw_questions = parsed.get("questions", [])
            if not isinstance(raw_questions, list):
                raw_questions = []

        clean_questions = []
        for q in raw_questions:
            if isinstance(q, dict) and "question" in q:
                opts = q.get("options", [])
                if isinstance(opts, list) and len(opts) >= 2:
                    try:
                        c_idx = int(q.get("correct_index", 0))
                    except (ValueError, TypeError):
                        c_idx = 0
                    c_idx = max(0, min(c_idx, len(opts) - 1))
                    clean_questions.append({
                        "question": str(q["question"]),
                        "options": [str(opt) for opt in opts],
                        "correct_index": c_idx,
                    })

        if clean_questions:
            return {"ok": True, "questions": clean_questions}
        return {"ok": False, "error": "The AI model returned an invalid quiz structure. Please try again."}
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

    result = _ollama_chat(messages, num_predict=500)
    if result["ok"]:
        return {"ok": True, "answer": result["text"]}
    return result
