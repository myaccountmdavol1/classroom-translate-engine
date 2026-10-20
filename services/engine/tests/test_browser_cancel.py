"""Stop reaching a running job in the browser (ct_browser/cancel.py), run natively.

In the browser the page's Stop cannot reach Python while a job runs, so the
job's own checkpoints ask the page. Here the page is a function; the job is
run exactly as the worker runs it (one run_next_job entry).
"""

from __future__ import annotations

import json

import pytest

from ct_browser import cancel
from ct_browser.boot import BrowserEngine
from ct_engine.jobs.runner import JobRunner
from ct_engine.providers.registry import TRANSLATION_PROVIDERS

from test_browser_executor import _Stub, call, drain, upload


class _Page:
    """Answers ctCancelRequested: yes once `after` questions have been asked."""

    def __init__(self, after: int | None = None, fail: bool = False) -> None:
        self.after = after
        self.fail = fail
        self.asked: list[str] = []

    def __call__(self, job_id: str) -> bool:
        self.asked.append(job_id)
        if self.fail:
            raise RuntimeError("the page did not answer")
        return self.after is not None and len(self.asked) > self.after


@pytest.fixture
def make_engine(monkeypatch):
    monkeypatch.setitem(TRANSLATION_PROVIDERS, "local", _Stub)
    # Every checkpoint asks, so the test does not depend on the clock.
    monkeypatch.setattr(cancel, "CHECK_INTERVAL_S", 0.0)
    _Stub.codes_seen = []
    made: list[BrowserEngine] = []

    def make(page: _Page) -> BrowserEngine:
        eng = BrowserEngine(post=lambda raw: eng.posts.append(json.loads(raw)), cancel_checker=page)
        eng.posts = []
        eng.start()
        made.append(eng)
        return eng

    yield make
    for eng in made:
        eng.stop()


def _translate(engine, doc):
    status, job = call(engine, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
    assert status == 200, job
    return job


def test_a_stop_from_the_page_ends_a_running_translation_cancelled(make_engine):
    page = _Page()
    engine = make_engine(page)
    doc = upload(engine, "13_rotated_pages.pdf")
    drain(engine)
    asked_in_analysis = len(page.asked)
    assert asked_in_analysis > 0  # analysis asks too, and was never stopped
    job = _translate(engine, doc)

    # She presses Stop once the translation has passed a few checkpoints.
    page.after = len(page.asked) + 2
    out = engine.run_next_job()
    assert "error" not in out, out
    status, final = call(engine, "GET", f"/api/documents/{doc}/job")
    assert final["job_id"] == job["job_id"]
    assert final["stage"] == "CANCELLED", final
    assert set(page.asked) >= {job["job_id"]}
    # The snapshot the host serves says so too.
    posted = [p["job"] for p in engine.posts if p["type"] == "job" and p["document_id"] == doc]
    assert posted[-1]["stage"] == "CANCELLED"
    # The engine tidied up as after any stop: the document can be translated again.
    _translate(engine, doc)
    page.after = None
    drain(engine)
    status, again = call(engine, "GET", f"/api/documents/{doc}/job")
    assert again["stage"] == "COMPLETE"


def test_a_page_that_cannot_answer_never_stops_a_job(make_engine):
    engine = make_engine(_Page(fail=True))
    doc = upload(engine)
    drain(engine)
    _translate(engine, doc)
    drain(engine)
    status, job = call(engine, "GET", f"/api/documents/{doc}/job")
    assert job["stage"] == "COMPLETE"


def test_the_page_is_asked_at_most_every_interval(monkeypatch):
    page = _Page()
    cancel.install(page)
    try:
        monkeypatch.setattr(cancel, "CHECK_INTERVAL_S", 3600.0)

        class _Runner:
            _cancel_requested: set = set()

        class _Job:
            job_id = "job_x"

        for _ in range(50):
            JobRunner._check_cancel(_Runner(), _Job())
        assert page.asked == ["job_x"]
    finally:
        cancel.uninstall()


def test_stop_puts_the_engines_own_checkpoint_back(make_engine):
    original = JobRunner._check_cancel
    engine = make_engine(_Page())
    assert JobRunner._check_cancel is not original
    engine.stop()
    assert JobRunner._check_cancel is original
