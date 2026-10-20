"""The store journal and picking jobs up again in the browser (ct_browser), run natively.

In the browser the page keeps the lasting copy of every document in OPFS by
applying the journal's entries in order. These tests replay the journal the
same way into a fresh folder: the whole journal must give back the engine's
own store, and ANY prefix of it -- what OPFS holds if the tab dies at that
moment -- must be a store the engine can open and carry on from, exactly as
it carries on after a server restart.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ct_browser.boot import MEMORY_GAVE_UP, BrowserEngine
from ct_browser.journal import StoreJournal
from ct_engine.jobs.runner import JobRunner
from ct_engine.model.document import DocumentManifest
from ct_engine.model.enums import JobStage
from ct_engine.model.job import DocumentJob
from ct_engine.providers.registry import TRANSLATION_PROVIDERS
from ct_engine.storage.local_fs import LocalFileSystemStore

from test_browser_executor import _Stub, call, drain, upload


# ---- The journal on its own ------------------------------------------------------
class Recorder:
    """Stands in for the worker: reads each written file at once, as it does."""

    def __init__(self) -> None:
        self.entries: list[tuple[int, str, str, bytes | None]] = []

    def __call__(self, seq: int, op: str, key: str, path: str | None) -> None:
        data = Path(path).read_bytes() if path else None
        self.entries.append((seq, op, key, data))


def replay(entries, root: Path) -> LocalFileSystemStore:
    """What the host does with the entries, into a folder instead of OPFS."""
    root.mkdir(parents=True, exist_ok=True)
    for _seq, op, key, data in entries:
        path = root / key
        if op == "put":
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name("." + path.name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        elif op == "delete":
            path.unlink(missing_ok=True)
        elif op == "delete_prefix":
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    return LocalFileSystemStore(root)


def files_under(root: Path, prefix: str) -> dict[str, bytes]:
    base = root / prefix
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(base.rglob("*"))
        if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".tmp")
    }


def test_each_write_is_told_once_in_order_and_failed_writes_are_not(tmp_path):
    store = LocalFileSystemStore(tmp_path / "s")
    rec = Recorder()
    journal = StoreJournal(store, rec)
    journal.install()
    store.put_bytes("documents/d/a.bin", b"one")
    store.put_text("documents/d/b.txt", "two")  # calls put_bytes inside: told once
    scratch = store.scratch_dir() / "upload"
    scratch.write_bytes(b"three")
    store.put_file("documents/d/c.pdf", scratch)
    store.put_bytes("documents/d/a.bin", b"one again")
    store.delete("documents/d/b.txt")
    store.put_bytes("documents/e/x", b"x")
    store.delete_prefix("documents/e/")
    with pytest.raises(Exception):
        store.put_bytes("../escape", b"no")
    assert [(s, op, k, d) for s, op, k, d in rec.entries] == [
        (1, "put", "documents/d/a.bin", b"one"),
        (2, "put", "documents/d/b.txt", b"two"),
        (3, "put", "documents/d/c.pdf", b"three"),
        (4, "put", "documents/d/a.bin", b"one again"),
        (5, "delete", "documents/d/b.txt", None),
        (6, "put", "documents/e/x", b"x"),
        (7, "delete_prefix", "documents/e", None),
    ]
    journal.uninstall()
    assert "put_bytes" not in vars(store)
    store.put_bytes("documents/d/after", b"z")
    assert len(rec.entries) == 7


def test_a_page_that_cannot_hear_never_stops_a_write(tmp_path):
    store = LocalFileSystemStore(tmp_path / "s")

    def deaf(*_args):
        raise RuntimeError("the page went away")

    journal = StoreJournal(store, deaf)
    journal.install()
    store.put_text("documents/d/m.json", "{}")
    assert store.get_text("documents/d/m.json") == "{}"
    assert journal.failures == 1 and "went away" in journal.last_error


def test_a_write_whose_file_cannot_be_found_keeps_its_number_as_lost(tmp_path, monkeypatch):
    """The page must be able to tell a missing write from no write: every
    write has its number, and one the worker could not read is told as
    "lost" (the page then stops trusting its copy of that document)."""
    store = LocalFileSystemStore(tmp_path / "s")
    rec = Recorder()
    journal = StoreJournal(store, rec)
    journal.install()
    store.put_bytes("documents/d/a.bin", b"one")
    real = store.local_path
    monkeypatch.setattr(
        store, "local_path", lambda key: None if key.endswith("b.bin") else real(key)
    )
    store.put_bytes("documents/d/b.bin", b"two")
    store.put_bytes("documents/d/c.bin", b"three")
    assert [(s, op, k) for s, op, k, _d in rec.entries] == [
        (1, "put", "documents/d/a.bin"),
        (2, "lost", "documents/d/b.bin"),
        (3, "put", "documents/d/c.bin"),
    ]
    assert journal.failures == 1 and "b.bin" in journal.last_error


# ---- The whole engine, with a fresh store of its own -------------------------------------
@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """Point the routes and the job runner at an empty store for this test."""
    import ct_engine.api.deps as deps

    holder = {"store": LocalFileSystemStore(tmp_path / "engine")}
    monkeypatch.setattr(deps, "get_store", lambda: holder["store"])
    deps.get_job_runner.cache_clear()
    deps.get_manifest_store.cache_clear()
    yield holder
    deps.get_job_runner.cache_clear()
    deps.get_manifest_store.cache_clear()


def start_engine(rec=None, posts=None) -> BrowserEngine:
    eng = BrowserEngine(
        post=(lambda raw: posts.append(json.loads(raw))) if posts is not None else None,
        journal=rec,
    )
    eng.start()
    return eng


@pytest.fixture
def stub_translator(monkeypatch):
    monkeypatch.setitem(TRANSLATION_PROVIDERS, "local", _Stub)
    _Stub.codes_seen = []


def _referenced_keys(raw: str, doc: str) -> set[str]:
    found: set[str] = set()

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str) and v.startswith(f"documents/{doc}/"):
            found.add(v)

    walk(json.loads(raw))
    return found


def test_the_journal_replayed_is_the_store_and_every_prefix_is_usable(
    tmp_path, fresh_store, stub_translator
):
    rec = Recorder()
    eng = start_engine(rec)
    try:
        doc = upload(eng)
        drain(eng)
        call(eng, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
        drain(eng)
        status, blocks = call(eng, "GET", f"/api/documents/{doc}/pages/1/blocks")
        first = blocks["blocks"][0]["id"]
        status, edited = call(
            eng, "PATCH", f"/api/documents/{doc}/pages/1/blocks/{first}",
            json_body={"text": "Texto del maestro"},
        )
        assert status == 200, edited
        engine_root = fresh_store["store"].root
    finally:
        eng.stop()

    seqs = [e[0] for e in rec.entries]
    assert seqs == list(range(1, len(seqs) + 1))
    assert all(e[2].startswith(f"documents/{doc}/") for e in rec.entries)

    # The whole journal gives back the engine's own files, byte for byte.
    whole = replay(rec.entries, tmp_path / "opfs-whole")
    assert files_under(whole.root, f"documents/{doc}") == files_under(
        engine_root, f"documents/{doc}"
    )

    # Every prefix that ends on a manifest or job write: it parses, and
    # every key it names is already there (the engine writes a file before
    # the record that points at it).
    checked = 0
    for i, (_seq, op, key, data) in enumerate(rec.entries):
        if op != "put" or not key.endswith(".json") or "/export/" in key:
            continue
        root = tmp_path / f"opfs-{i}"
        replay(rec.entries[: i + 1], root)
        raw = (root / key).read_text()
        if key.endswith("/manifest.json"):
            DocumentManifest.model_validate_json(raw)
        elif "/jobs/" in key:
            DocumentJob.model_validate_json(raw)
        else:
            continue
        missing = [k for k in _referenced_keys(raw, doc) if not (root / k).is_file()]
        assert not missing, f"after entry {i} ({key}) these are named but absent: {missing}"
        shutil.rmtree(root)
        checked += 1
    assert checked > 10


def _cut_mid_translation(entries, doc, stages=("TRANSLATING", "RECONSTRUCTING"), first=False) -> int:
    """The entry after which the tab 'dies': a job write part-way through
    translating, with some pages already saved."""
    candidates = []
    for i, (_s, op, key, data) in enumerate(entries):
        if op == "put" and f"documents/{doc}/jobs/" in key and key.endswith(".json"):
            job = json.loads(data)
            if job["settings"]["target_language"] not in (None, "und") and job["stage"] in stages:
                candidates.append(i)
    assert candidates, "no mid-translation job write found"
    return candidates[0] if first else candidates[len(candidates) // 2]


def test_a_tab_that_dies_mid_translation_is_picked_up_from_what_was_written(
    tmp_path, fresh_store, stub_translator
):
    rec = Recorder()
    eng = start_engine(rec)
    try:
        doc = upload(eng, "13_rotated_pages.pdf")
        drain(eng)
        call(eng, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
        drain(eng)
        status, done = call(eng, "GET", f"/api/documents/{doc}/job")
        assert done["stage"] == "COMPLETE"
    finally:
        eng.stop()

    cut = _cut_mid_translation(rec.entries, doc)
    opfs = replay(rec.entries[: cut + 1], tmp_path / "opfs")
    # A new worker boots on what the browser kept (the page puts it in MEMFS).
    fresh_store["store"] = opfs
    import ct_engine.api.deps as deps

    deps.get_job_runner.cache_clear()
    deps.get_manifest_store.cache_clear()
    posts: list[dict] = []
    rec2 = Recorder()
    eng2 = start_engine(rec2, posts)
    try:
        resumed = eng2.resume(None)
        assert resumed == [done["job_id"]]
        resumed_post = [p for p in posts if p["type"] == "job" and p["document_id"] == doc]
        assert resumed_post and resumed_post[-1]["resume_count"] == 1
        drain(eng2)
        status, job = call(eng2, "GET", f"/api/documents/{doc}/job")
        assert job["stage"] == "COMPLETE", job
        assert job["job_id"] == done["job_id"]
        status, pdf = call(eng2, "GET", f"/api/documents/{doc}/download")
        assert status == 200 and bytes(pdf[:5]) == b"%PDF-"
        # The second worker's writes are journalled too, starting again at 1.
        assert rec2.entries and rec2.entries[0][0] == 1
    finally:
        eng2.stop()


def test_a_stop_she_made_before_the_tab_closed_ends_the_job_on_resume(
    tmp_path, fresh_store, stub_translator
):
    rec = Recorder()
    eng = start_engine(rec)
    try:
        doc = upload(eng, "13_rotated_pages.pdf")
        drain(eng)
        call(eng, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
        drain(eng)
        job_id = call(eng, "GET", f"/api/documents/{doc}/job")[1]["job_id"]
    finally:
        eng.stop()
    opfs = replay(rec.entries[: _cut_mid_translation(rec.entries, doc) + 1], tmp_path / "opfs")
    fresh_store["store"] = opfs
    import ct_engine.api.deps as deps

    deps.get_job_runner.cache_clear()
    deps.get_manifest_store.cache_clear()
    eng2 = start_engine(Recorder())
    try:
        assert eng2.resume(None, stops=[[doc, job_id]]) == []
        assert not eng2.has_pending_job()
        status, job = call(eng2, "GET", f"/api/documents/{doc}/job")
        assert job["stage"] == "CANCELLED"
        assert not opfs.exists(f"documents/{doc}/jobs/{job_id}.cancel")
    finally:
        eng2.stop()


def test_resume_for_named_documents_leaves_the_others_alone(
    tmp_path, fresh_store, stub_translator
):
    eng = start_engine(Recorder())
    try:
        a = upload(eng)
        b = upload(eng)
        drain(eng)
        # Two translations wait; the store says both are unfinished.
        call(eng, "POST", f"/api/documents/{a}/translate", json_body={"target_language": "es"})
        call(eng, "POST", f"/api/documents/{b}/translate", json_body={"target_language": "es"})
        # Already queued in this worker: never queued twice.
        assert eng.resume([a]) == []
        # A document's job left unfinished by an earlier worker (its entry
        # is gone from this pool) is picked up when it is named.
        b_job = call(eng, "GET", f"/api/documents/{b}/job")[1]["job_id"]
        assert eng.runner.cancel(b) is True  # drops it from the pool...
        job = eng.manifests.load_job(b, b_job)
        job.stage = JobStage.TRANSLATING  # ...and leave it looking interrupted
        eng.manifests.save_job(job)
        assert eng.resume([a]) == []
        assert eng.resume([b]) == [b_job]
        drain(eng)
        for d in (a, b):
            assert call(eng, "GET", f"/api/documents/{d}/job")[1]["stage"] == "COMPLETE"
    finally:
        eng.stop()


def test_running_out_of_memory_pauses_the_job_for_a_fresh_worker(
    fresh_store, stub_translator, monkeypatch
):
    original = JobRunner._translation_stages
    calls = {"n": 0}

    def short_of_memory(self, job):
        calls["n"] += 1
        if calls["n"] == 1:
            raise MemoryError("out of memory")
        return original(self, job)

    monkeypatch.setattr(JobRunner, "_translation_stages", short_of_memory)
    posts: list[dict] = []
    eng = start_engine(Recorder(), posts)
    try:
        doc = upload(eng)
        drain(eng)
        call(eng, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
        out = eng.run_next_job()
        assert out.get("recycle") is True and "error" not in out
        status, job = call(eng, "GET", f"/api/documents/{doc}/job")
        assert not job["is_terminal"] and job["error"] is None
        assert all(p["job"]["stage"] != "FAILED" for p in posts if p["type"] == "job")
    finally:
        eng.stop()
    # A fresh worker on the same store picks it up.
    import ct_engine.api.deps as deps

    deps.get_job_runner.cache_clear()
    deps.get_manifest_store.cache_clear()
    eng2 = start_engine(Recorder())
    try:
        assert eng2.resume([doc]) == [job["job_id"]]
        drain(eng2)
        status, job = call(eng2, "GET", f"/api/documents/{doc}/job")
        assert job["stage"] == "COMPLETE"
    finally:
        eng2.stop()


def test_running_out_of_memory_too_often_fails_in_plain_words(
    fresh_store, stub_translator, monkeypatch
):
    from ct_engine.jobs import runner as runner_mod

    def always_short(self, job):
        raise MemoryError("out of memory")

    monkeypatch.setattr(JobRunner, "_translation_stages", always_short)
    eng = start_engine(Recorder())
    try:
        doc = upload(eng)
        drain(eng)
        call(eng, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
        job_id = call(eng, "GET", f"/api/documents/{doc}/job")[1]["job_id"]
        job = eng.manifests.load_job(doc, job_id)
        job.resume_count = runner_mod.MAX_RESUME_ATTEMPTS
        eng.manifests.save_job(job)
        out = eng.run_next_job()
        assert out.get("recycle") is True
        status, job = call(eng, "GET", f"/api/documents/{doc}/job")
        assert job["stage"] == "FAILED" and job["error"] == MEMORY_GAVE_UP
    finally:
        eng.stop()


def test_a_resumed_translation_carries_the_class_code_it_is_given(
    tmp_path, fresh_store, stub_translator
):
    rec = Recorder()
    eng = start_engine(rec)
    try:
        doc = upload(eng, "13_rotated_pages.pdf")
        drain(eng)
        call(eng, "POST", f"/api/documents/{doc}/translate", json_body={"target_language": "es"})
        drain(eng)
    finally:
        eng.stop()
    cut = _cut_mid_translation(rec.entries, doc, stages=("TRANSLATING",), first=True)
    opfs = replay(rec.entries[: cut + 1], tmp_path / "opfs")
    fresh_store["store"] = opfs
    import ct_engine.api.deps as deps
    from ct_browser.relay import ACCESS_CODE

    deps.get_job_runner.cache_clear()
    deps.get_manifest_store.cache_clear()
    _Stub.codes_seen = []
    eng2 = start_engine(Recorder())
    try:
        assert len(eng2.resume(None, code="class-42")) == 1
        assert ACCESS_CODE.get() is None
        drain(eng2)
        assert _Stub.codes_seen and set(_Stub.codes_seen) == {"class-42"}
    finally:
        eng2.stop()
