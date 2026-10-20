"""A patched download keeps its fonts cut down, as a full rebuild does."""

from __future__ import annotations


def test_patched_export_stays_near_the_full_rebuild_size(store, monkeypatch):
    from ct_engine.pipeline import reconstruct_stage as R
    from ct_engine.pipeline.edit import apply_edit
    from test_edit import first_editable, prepared

    manifest = prepared(store, "03_math_worksheet.pdf")
    if len(manifest.pages) < 2:
        manifest = prepared(store, "10_mixed_orientation.pdf")
    assert len(manifest.pages) >= 2
    target = first_editable(manifest, 1)
    apply_edit(manifest, 1, target.id, "ZEBRA QUOKKA", store=store, target_lang="es")

    full, _ = R.rebuild_document(
        store.get_bytes(manifest.source_key), manifest.pages, target_lang="es"
    )

    def no_full_rebuild(*_a, **_k):
        raise AssertionError("the whole document was rebuilt")

    monkeypatch.setattr(R, "rebuild_document", no_full_rebuild)
    export = R.rebuild_export_only(manifest, store=store, target_lang="es")

    size = store.local_path(export.export_key).stat().st_size
    assert export.byte_size == size
    # Unsubset, the swapped-in page carries the whole font file again.
    assert size < len(full) * 1.25 + 10_000, (size, len(full))
    # No stray scratch files left behind.
    assert not list(store.scratch_dir().glob("*.pdf"))


def test_subset_that_changes_the_text_is_thrown_away(tmp_path, monkeypatch):
    import fitz

    from ct_engine.pipeline import reconstruct_stage as R

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Hola mundo", fontname="helv")
    path = tmp_path / "in.pdf"
    doc.save(str(path))
    doc.close()
    original = path.read_bytes()

    real_get_text = fitz.Page.get_text
    calls = {"n": 0}

    def drifting(self, *a, **k):
        calls["n"] += 1
        text = real_get_text(self, *a, **k)
        return text if calls["n"] == 1 else text + "?"

    monkeypatch.setattr(fitz.Page, "get_text", drifting)
    result = R._subset_fonts_file(path, tmp_path)

    assert result == path
    assert path.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == ["in.pdf"]


def test_download_serves_the_file_it_measured(store, monkeypatch):
    """An export replaced while a download is under way does not mix the
    old length with the new bytes: the open file is the one served."""
    import asyncio
    import os
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from ct_engine.api import routes

    key = "documents/abc/export/translated.pdf"
    old = b"%PDF-old " + b"a" * 300_000
    store.put_bytes(key, old, content_type="application/pdf")
    manifest = SimpleNamespace(
        office_source_key=None, content_type="application/pdf",
        filename="Fractions.pdf",
    )
    job = SimpleNamespace(
        export_key=key, settings=SimpleNamespace(target_language="es"),
    )
    monkeypatch.setattr(routes, "_fresh_export", lambda _id: (manifest, job))
    monkeypatch.setattr(routes, "get_object_store", lambda: store)

    response = routes.download_translated("abc")
    # A patched export lands on the same path before the body is sent.
    fd, name = tempfile.mkstemp(dir=str(store.scratch_dir()))
    os.write(fd, b"%PDF-new")
    os.close(fd)
    store.put_file(key, Path(name))

    async def body() -> bytes:
        parts = [chunk async for chunk in response.body_iterator]
        return b"".join(parts)

    sent = asyncio.run(body())
    assert sent == old
    assert response.headers["content-length"] == str(len(old))
