"""
StudyMate AI — Full Summarization Flow Integration Test
Tests the complete end-to-end flow:
1. Health check & model status
2. Authentication / login
3. Summarization request with actual text
4. Study history logging
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app import create_app

app = create_app()
client = app.test_client()
client.testing = True

print("=" * 60)
print("TEST 1: Health & Model Status")
r = client.get("/api/health")
print("Health Status:", r.status_code, r.get_json())
assert r.status_code == 200
assert r.get_json()["status"] == "ok"

r = client.get("/api/model/status")
print("Model Status:", r.status_code, r.get_json())
assert r.status_code == 200

print("\nTEST 2: Login")
# Register or login as test user
r = client.post("/api/auth/register", json={"username": "testuser", "password": "password123"})
if r.status_code != 201:
    r = client.post("/api/auth/login", json={"username": "testuser", "password": "password123"})
print("Auth Status:", r.status_code, r.get_json())
assert r.status_code in (200, 201)

print("\nTEST 3: Summarize Notes with local Gemma")
sample_notes = (
    "Machine learning is a branch of artificial intelligence and computer science "
    "which focuses on the use of data and algorithms to imitate the way that humans learn, "
    "gradually improving its accuracy. Core categories include supervised learning, "
    "unsupervised learning, and reinforcement learning."
)
r = client.post("/api/summarize", json={"text": sample_notes})
print("Summarize Status:", r.status_code)
data = r.get_json()
print("Summarize Response keys:", list(data.keys()))
assert r.status_code == 200
assert "summary" in data
assert len(data["summary"]) > 20
print("\nGenerated Summary Preview:\n", data["summary"][:200], "...\n")

print("\nTEST 4: Verify Study History Record")
r = client.get("/api/history")
print("History Status:", r.status_code)
hist = r.get_json().get("history", [])
print(f"Total history entries: {len(hist)}")
assert any(h["type"] == "summarize" for h in hist)

print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
