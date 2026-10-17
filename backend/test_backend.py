"""
StudyMate AI — Backend Test Suite
Run with: python test_backend.py
Tests all auth and history endpoints using Flask's built-in test client.
No external test framework needed.
"""

import sys
import json
import tempfile
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Point the database at a temp file so tests never touch the real DB
os.environ["DATABASE_PATH"] = tempfile.mktemp(suffix=".db")
# Use a fixed secret for reproducible sessions
os.environ["SECRET_KEY"] = "test-secret-key-for-tests-only"

from app import create_app  # noqa: E402  (import after env setup)

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  {status} {label}", f"- {detail}" if detail else "")
    results.append(condition)


def run_tests():
    app = create_app()
    client = app.test_client()
    # Enable cookies so sessions persist across requests in tests
    client.testing = True
    print("\n-- Health ---------------------------------------------")
    r = client.get("/api/health")
    check("GET /api/health returns 200", r.status_code == 200)
    data = r.get_json()
    check("Health response has 'status: ok'", data.get("status") == "ok")

    print("\n-- Registration ---------------------------------------")
    r = client.post("/api/auth/register",
                    json={"username": "alice", "password": "secret123"},
                    content_type="application/json")
    check("Register new user -> 201", r.status_code == 201, str(r.get_json()))

    r2 = client.post("/api/auth/register",
                     json={"username": "alice", "password": "another"},
                     content_type="application/json")
    check("Duplicate username -> 409", r2.status_code == 409, str(r2.get_json()))

    r3 = client.post("/api/auth/register",
                     json={"username": "ab", "password": "secret123"},
                     content_type="application/json")
    check("Too-short username -> 400", r3.status_code == 400)

    r4 = client.post("/api/auth/register",
                     json={"username": "bob", "password": "12345"},
                     content_type="application/json")
    check("Too-short password -> 400", r4.status_code == 400)

    print("\n-- Login -----------------------------------------------")
    # Fresh client for login tests (alice is already registered above)
    c2 = app.test_client()

    r = c2.post("/api/auth/login",
                json={"username": "alice", "password": "secret123"},
                content_type="application/json")
    check("Login correct credentials -> 200", r.status_code == 200, str(r.get_json()))
    data = r.get_json()
    check("Login response has 'user.username'",
          data.get("user", {}).get("username") == "alice")
    check("Password hash not returned in login",
          "password" not in str(data))

    r_bad = c2.post("/api/auth/login",
                    json={"username": "alice", "password": "wrong"},
                    content_type="application/json")
    check("Login wrong password -> 401", r_bad.status_code == 401)

    r_no = c2.post("/api/auth/login",
                   json={"username": "nobody", "password": "x"},
                   content_type="application/json")
    check("Login unknown user -> 401", r_no.status_code == 401)

    print("\n-- Session ---------------------------------------------")
    # c2 should still be logged in as alice from successful login above
    r = c2.get("/api/auth/me")
    check("GET /api/auth/me while logged in -> 200", r.status_code == 200)
    check("me() returns correct username",
          r.get_json().get("user", {}).get("username") == "alice")

    # Unauthenticated client
    c3 = app.test_client()
    r = c3.get("/api/auth/me")
    check("GET /api/auth/me unauthenticated -> 401", r.status_code == 401)

    print("\n-- Logout ----------------------------------------------")
    r = c2.post("/api/auth/logout")
    check("POST /api/auth/logout -> 200", r.status_code == 200)
    r = c2.get("/api/auth/me")
    check("After logout /me -> 401", r.status_code == 401)

    print("\n-- History access control ------------------------------")
    # Register and login as two users
    ca = app.test_client()
    cb = app.test_client()
    ca.post("/api/auth/register",
            json={"username": "user_a", "password": "passA123"},
            content_type="application/json")
    cb.post("/api/auth/register",
            json={"username": "user_b", "password": "passB123"},
            content_type="application/json")

    # Both should see their own (empty) history
    ra = ca.get("/api/history")
    rb = cb.get("/api/history")
    check("User A can GET /api/history -> 200", ra.status_code == 200)
    check("User B can GET /api/history -> 200", rb.status_code == 200)
    check("User A history starts empty", ra.get_json().get("history") == [])
    check("User B history starts empty", rb.get_json().get("history") == [])

    # Unauthenticated client cannot access history
    cx = app.test_client()
    rx = cx.get("/api/history")
    check("Unauthenticated /api/history -> 401", rx.status_code == 401)

    print("\n-- Protected route (summarize without login) -----------")
    cx2 = app.test_client()
    r = cx2.post("/api/summarize",
                 json={"text": "some notes"},
                 content_type="application/json")
    check("POST /api/summarize without login -> 401", r.status_code == 401)

    print()


if __name__ == "__main__":
    run_tests()
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    if passed < total:
        print(f"\n{total - passed} test(s) FAILED - check output above.")
        sys.exit(1)
    else:
        print("\nAll tests passed successfully.")
        sys.exit(0)
