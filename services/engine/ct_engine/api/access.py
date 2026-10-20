"""Shared-passphrase access control.

The engine has to be reachable from a browser on the public internet for a
teacher to use it, and an open document-processing endpoint is an invitation
to have someone else's files run through it. A district login is the right
long-term answer; one shared passphrase is the smallest thing that actually
closes the door, and it needs no accounts, no database and no password
resets.

Deliberate properties:

  * OFF when unset. A developer running the engine on localhost should not
    have to configure anything, and a deployment that forgets to set a
    passphrase should fail loudly in the health check rather than silently
    refuse every request.
  * Compared with `secrets.compare_digest`, so the check cannot be timed.
  * The health endpoint stays open, because a host's own liveness probe
    cannot present a passphrase -- and it reports whether the door is
    locked, so a misconfigured deployment is visible.
  * The passphrase is never logged and never returned, not even in a
    diagnostic.

This is a shared secret, not authentication: it identifies nobody, and
everyone who has it is the same person as far as the engine is concerned.
Anything that needs to know WHO uploaded a document needs real accounts.
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request

from ..config import get_settings

# Sent by the web app. A custom header rather than `Authorization`, so it is
# never confused with a real credential and never forwarded by a proxy that
# thinks it understands bearer tokens.
HEADER = "X-CT-Access"

# Accepted on GET ONLY, and only because a browser cannot attach a header to
# an `<img src>` or a download link, which is how page previews, thumbnails
# and the finished file are fetched. Rewriting all of those as blob fetches
# would remove the need, at the cost of loading every thumbnail through
# JavaScript and holding it in memory.
#
# The trade-off being accepted: a code in a query string is visible in
# browser history and in the server's access log. It is not accepted for
# uploads or edits -- anything that changes state or costs real work
# requires the header, where it stays out of both.
QUERY_PARAM = "code"

# Paths reachable without the passphrase. Kept as small as possible.
OPEN_PATHS = frozenset({"/", "/api/health", "/docs", "/openapi.json", "/redoc"})


def access_required() -> bool:
    return bool(get_settings().access_passphrase)


def check_access(request: Request) -> None:
    """Raise 401 unless this request carries the right passphrase."""
    expected = get_settings().access_passphrase
    if not expected:
        return
    if request.method == "OPTIONS":
        return  # CORS preflight carries no custom headers
    if request.url.path in OPEN_PATHS:
        return

    presented = request.headers.get(HEADER, "")
    if not presented and request.method == "GET":
        presented = request.query_params.get(QUERY_PARAM, "")
    if not presented or not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "This service needs the class access code.",
                # No hint about the expected value, and no echo of what was
                # sent -- an error message is not a place to leak a secret.
                "diagnostic": None,
                "code": "access_required",
            },
        )
