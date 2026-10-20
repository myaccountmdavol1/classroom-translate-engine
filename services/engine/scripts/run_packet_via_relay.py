"""Translate a PDF with the native engine, every batch through the web relay.

Phase 1 of docs/superpowers/plans/2026-10-19-browser-engine-pdf.md: before
any browser work, prove that the engine translates a real document through
apps/web's /api/mt relay exactly as it does with its own keys. The engine
here holds NO key: DeepL and Google are the relay providers
(ct_browser/relay.py), and the class code and Origin are sent explicitly,
since no browser is there to send Origin.

Two commands:

  fake-deepl   A stand-in DeepL for local runs: answers "[ES] " + text and
               counts characters. Start it first, then start the web app
               with CT_MT_TEST_DEEPL_URL pointing at it (honoured only
               outside production), so no real allowance is spent.

  run          Upload, read and translate one PDF through the relay, then
               save the translated PDF and print a JSON report: pages,
               failures, and every relay request with its size (the
               largest batch sets the relay's character cap).

Local check against the fake (three terminals, from the repo root):

  services/engine/.venv/bin/python services/engine/scripts/run_packet_via_relay.py \\
      fake-deepl --port 8790

  # a scratch copy of apps/web, because Next allows one dev server per folder;
  # the relay answers only in a build that runs the engine in the browser
  NEXT_PUBLIC_ENGINE_URL=/engine \\
  CT_DEEPL_API_KEY=fake:fx CT_ACCESS_PASSPHRASE=test-code \\
  CT_MT_TEST_DEEPL_URL=http://127.0.0.1:8790 npx next dev -p 3419

  services/engine/.venv/bin/python services/engine/scripts/run_packet_via_relay.py \\
      run --pdf "Unit 1 PACKET.pdf" --relay http://localhost:3419/api/mt \\
      --code test-code --out /tmp/relay-run

For the real run, start the web app with the real DeepL key and class code
(and no CT_MT_TEST_* variables) and give the real code; the report's
"chars" total is what DeepL should bill.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextvars
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent


# --- The stand-in DeepL --------------------------------------------------------
def serve_fake_deepl(port: int) -> None:
    stats = {"requests": 0, "chars": 0, "max_chars": 0, "max_items": 0, "sizes": []}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - http.server's naming
            if self.path == "/stats":
                with lock:
                    self._send(200, stats)
            else:
                self._send(404, {"message": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v2/translate":
                self._send(404, {"message": "not found"})
                return
            if not self.headers.get("Authorization", "").startswith("DeepL-Auth-Key "):
                self._send(403, {"message": "Authorization failed"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            texts = body.get("text") or []
            chars = sum(len(t) for t in texts)
            with lock:
                stats["requests"] += 1
                stats["chars"] += chars
                stats["max_chars"] = max(stats["max_chars"], chars)
                stats["max_items"] = max(stats["max_items"], len(texts))
                stats["sizes"].append([len(texts), chars])
            self._send(200, {"translations": [
                {"detected_source_language": "EN", "text": f"[ES] {t}"} for t in texts
            ]})

        def log_message(self, *_args) -> None:  # bodies are never logged
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"fake DeepL on http://127.0.0.1:{port} (GET /stats for counts)", flush=True)
    server.serve_forever()


# --- The run ------------------------------------------------------------------
class _ContextPool(cf.ThreadPoolExecutor):
    """Runs each job in the context it was submitted from, as the browser's
    job pool does, so a translation job carries the class code."""

    def submit(self, fn, /, *args, **kwargs):
        return super().submit(contextvars.copy_context().run, fn, *args, **kwargs)


def run(args: argparse.Namespace) -> int:
    pdf = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = tempfile.mkdtemp(prefix="ct-relay-run-")
    relay_url = args.relay.rstrip("/")
    origin = args.origin or relay_url.split("/api/", 1)[0]

    # Before the engine is imported: its settings are read once.
    os.environ.update({
        "CT_STORAGE_ROOT": storage,
        "CT_TRANSLATION_PROVIDER": args.providers,
        "CT_RELAY_URL": relay_url,
        "CT_RELAY_ORIGIN": origin,
        "CT_LOG_LEVEL": "WARNING",
        # This engine is a local one with no door of its own.
        "CT_ACCESS_PASSPHRASE": "",
    })
    # No key on this side, whatever a developer's .env holds: the relay
    # providers never read one, and this makes sure nothing else can.
    for name in ("DEEPL_API_KEY", "GOOGLE_TRANSLATE_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY"):
        os.environ[f"CT_{name}"] = ""
    sys.path.insert(0, str(ENGINE_ROOT))

    import httpx
    from fastapi.testclient import TestClient

    from ct_browser import relay
    from ct_engine.api import routes
    from ct_engine.api.deps import get_job_runner, get_manifest_store
    from ct_engine.main import app

    relay.install()
    relay.ACCESS_CODE.set(args.code)

    # Every relay request, with its size, for the report.
    sent: list[dict] = []
    real = httpx.HTTPTransport()

    class Counting(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content or b"{}")
            texts = body.get("text") or body.get("q") or []
            started = time.perf_counter()
            response = real.handle_request(request)
            sent.append({
                "service": request.url.path.rsplit("/", 1)[-1],
                "items": len(texts),
                "chars": sum(len(t) for t in texts),
                "longest_item": max((len(t) for t in texts), default=0),
                "status": response.status_code,
                "seconds": round(time.perf_counter() - started, 3),
            })
            return response

    relay._transport = Counting()

    runner = get_job_runner()
    runner._pool = _ContextPool(max_workers=1, thread_name_prefix="ct-job")
    manifests = get_manifest_store()
    client = TestClient(app)

    def wait(document_id: str, job_id: str, until) -> object:
        deadline = time.monotonic() + args.timeout
        while True:
            job = manifests.load_job(document_id, job_id)
            if job is not None and until(job):
                return job
            if time.monotonic() > deadline:
                raise SystemExit(f"timed out; job is {job.stage if job else None}")
            time.sleep(0.25)

    t0 = time.perf_counter()
    with pdf.open("rb") as fh:
        up = client.post("/api/documents", files={"file": (pdf.name, fh, "application/pdf")})
    up.raise_for_status()
    uploaded = up.json()
    document_id = uploaded["document_id"]
    analysis = wait(
        document_id, uploaded["job_id"],
        lambda j: j.is_terminal or j.stage.value == "REVIEW_READY",
    )
    t_analysis = time.perf_counter() - t0
    if analysis.stage.value != "REVIEW_READY":
        print(json.dumps({"analysis": analysis.stage.value, "error": analysis.error}))
        return 1

    # The engine's own translate route, called here so the class code set
    # above is the context the job is queued from.
    started = routes.start_translation(
        document_id, routes.TranslateRequest(target_language=args.target)
    )
    job = wait(document_id, started.job_id, lambda j: j.is_terminal)
    t_total = time.perf_counter() - t0

    manifest = manifests.load_manifest(document_id)
    report = {
        "document": pdf.name,
        "pages": len(manifest.pages) if manifest else None,
        "stage": job.stage.value,
        "error": job.error,
        "page_errors": [e.model_dump(mode="json")
                        for e in job.page_errors],
        "pages_translated": sum(1 for p in manifest.pages if p.translated) if manifest else None,
        "analysis_seconds": round(t_analysis, 1),
        "total_seconds": round(t_total, 1),
        "relay": {
            "requests": len(sent),
            "failed": sum(1 for s in sent if s["status"] != 200),
            "chars": sum(s["chars"] for s in sent if s["status"] == 200),
            "largest_batch_chars": max((s["chars"] for s in sent), default=0),
            "largest_batch_items": max((s["items"] for s in sent), default=0),
            "largest_item_chars": max((s["longest_item"] for s in sent), default=0),
            "by_service": {
                name: sum(1 for s in sent if s["service"] == name) for name in ("deepl", "google")
            },
        },
    }
    if job.stage.value == "COMPLETE":
        download = client.get(f"/api/documents/{document_id}/download")
        if download.status_code == 200:
            target = out_dir / f"{pdf.stem}.{args.target}.pdf"
            target.write_bytes(download.content)
            report["saved"] = str(target)
    (out_dir / "relay-requests.json").write_text(json.dumps(sent, indent=1))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    runner._pool.shutdown(wait=True)
    return 0 if job.stage.value == "COMPLETE" and report["relay"]["failed"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    fake = sub.add_parser("fake-deepl", help="serve a stand-in DeepL")
    fake.add_argument("--port", type=int, default=8790)

    go = sub.add_parser("run", help="translate a PDF through the relay")
    go.add_argument("--pdf", required=True)
    go.add_argument("--relay", required=True, help="e.g. http://localhost:3419/api/mt")
    go.add_argument("--code", required=True, help="the class access code the relay expects")
    go.add_argument("--origin", help="defaults to the relay's own origin")
    go.add_argument("--target", default="es")
    go.add_argument("--providers", default="deepl,google")
    go.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "ct-relay-run"))
    go.add_argument("--timeout", type=float, default=900.0)

    args = parser.parse_args()
    if args.command == "fake-deepl":
        serve_fake_deepl(args.port)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
