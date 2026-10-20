"""The shared access code.

The engine has to be reachable from a browser for a teacher to use it, and
an open document endpoint is an invitation to have strangers' files run
through it.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from ct_engine.api.access import HEADER
from ct_engine.config import get_settings

CODE = "correct horse battery staple"


@pytest.fixture
def locked(monkeypatch):
    """A client for an engine that has an access code set."""
    monkeypatch.setenv("CT_ACCESS_PASSPHRASE", CODE)
    get_settings.cache_clear()
    import ct_engine.main as main

    importlib.reload(main)
    with TestClient(main.app) as client:
        yield client
    get_settings.cache_clear()


def test_without_a_code_the_engine_is_open(client):
    """A developer on localhost configures nothing."""
    assert client.get("/api/languages").status_code == 200
    assert client.get("/api/health").json()["access_required"] is False


def test_a_locked_engine_refuses_an_unauthenticated_request(locked):
    r = locked.get("/api/languages")
    assert r.status_code == 401
    assert r.json()["code"] == "access_required"


def test_the_wrong_code_is_refused(locked):
    assert locked.get("/api/languages", headers={HEADER: "guess"}).status_code == 401


def test_the_right_code_is_accepted(locked):
    r = locked.get("/api/languages", headers={HEADER: CODE})
    assert r.status_code == 200 and r.json()["languages"]


def test_uploading_is_gated(locked):
    """The expensive, abusable endpoint is the whole point of the gate."""
    r = locked.post(
        "/api/documents",
        files={"file": ("x.pdf", b"%PDF-1.4 whatever", "application/pdf")},
    )
    assert r.status_code == 401


def test_health_stays_open_so_a_host_can_probe_it(locked):
    """A platform's liveness check cannot present a code."""
    r = locked.get("/api/health")
    assert r.status_code == 200
    assert r.json()["access_required"] is True


def test_health_never_reveals_the_code(locked):
    body = locked.get("/api/health").text
    assert CODE not in body


def test_a_refusal_never_echoes_what_was_sent(locked):
    """An error message is not a place to leak a secret."""
    r = locked.get("/api/languages", headers={HEADER: "a-guessed-value"})
    assert "a-guessed-value" not in r.text
    assert r.json()["diagnostic"] is None


def test_a_route_added_later_is_gated_by_default(locked):
    """The gate is middleware, so forgetting to annotate a route is safe."""
    paths = [
        r.path
        for r in locked.app.routes
        if getattr(r, "path", "").startswith("/api/")
    ]
    ungated = [p for p in paths if "{" not in p and p != "/api/health"]
    assert ungated, "no fixed API paths to check"
    for path in ungated:
        assert locked.get(path).status_code in (401, 405), path


def test_a_get_may_carry_the_code_in_the_query(locked):
    """A browser cannot put a header on an `<img src>`."""
    assert locked.get("/api/languages?code=" + CODE.replace(" ", "%20")).status_code in (
        200,
        401,
    )
    from urllib.parse import quote

    assert locked.get(f"/api/languages?code={quote(CODE)}").status_code == 200


def test_a_query_code_does_not_unlock_an_upload(locked):
    """State-changing work needs the header, where it stays out of logs."""
    from urllib.parse import quote

    r = locked.post(
        f"/api/documents?code={quote(CODE)}",
        files={"file": ("x.pdf", b"%PDF-1.4 whatever", "application/pdf")},
    )
    assert r.status_code == 401


def test_a_refusal_is_readable_by_the_browser_that_caused_it(locked):
    """A 401 must carry CORS headers, or the page sees a network error.

    Starlette applies the most recently added middleware outermost. With
    the gate registered after CORSMiddleware it sat in front of it, so
    refusals reached the browser with no Access-Control-Allow-Origin, were
    blocked, and showed as "could not reach the service" rather than "wrong
    code". Every API test still passed, because TestClient does not enforce
    CORS -- so the ordering is asserted here instead.
    """
    origin = "http://localhost:3000"
    r = locked.get("/api/languages", headers={"Origin": origin})
    assert r.status_code == 401
    assert r.headers.get("access-control-allow-origin") == origin


def test_the_gate_runs_inside_cors_not_in_front_of_it():
    """Stated as an ordering fact, so a reordering edit fails here."""
    import ct_engine.main as main
    from starlette.middleware.cors import CORSMiddleware

    classes = [m.cls for m in main.app.user_middleware]
    assert CORSMiddleware in classes, classes
    # Outermost first: CORS must come before the access gate.
    assert classes.index(CORSMiddleware) < len(classes) - 1, (
        "CORSMiddleware must be outermost so refusals carry CORS headers"
    )
