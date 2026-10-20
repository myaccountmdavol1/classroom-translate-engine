"""A page is only called translated once its translations are on disk.

The job is saved twice a second, the manifest every few seconds. If the job
said a page was done before the manifest holding it was saved, a restart in
between would skip that page on resume -- and when the document had been
translated before, into another language, the manifest still says
`translated`, so the old language would be exported without a word.
"""

from __future__ import annotations

import time

from conftest import corpus_file
from ct_engine.model.enums import JobStage
from ct_engine.pipeline.manifest_store import ManifestStore
from ct_engine.providers.translation.base import (
    TranslationProvider,
    TranslationResult,
)


class _Prefix(TranslationProvider):
    name = "prefix"

    def supports(self, source_lang, target_lang):
        return True

    def translate(self, segments, *, source_lang, target_lang):
        return [
            TranslationResult(segment_id=s.segment_id, text=f"es {s.text}", provider=self.name)
            for s in segments
        ]


def test_the_job_never_claims_a_page_the_saved_manifest_lacks(client, monkeypatch):
    provider = _Prefix()
    monkeypatch.setattr("ct_engine.api.routes.get_translation_provider", lambda: provider)
    monkeypatch.setattr("ct_engine.jobs.runner.get_translation_provider", lambda: provider)
    # The manifest is saved only at the end of the stage: every job save
    # before that is a moment a restart would find.
    monkeypatch.setattr("ct_engine.jobs.runner._MANIFEST_SAVE_INTERVAL", 1e9)

    on_disk: dict[str, set[int]] = {}
    broken: list[str] = []
    real_save_manifest = ManifestStore.save_manifest
    real_save_job = ManifestStore.save_job

    def save_manifest(self, manifest):
        real_save_manifest(self, manifest)
        on_disk[manifest.document_id] = {p.index for p in manifest.pages if p.translated}

    def save_job(self, job):
        claimed = set(job.completed_pages.get(JobStage.TRANSLATING.value, []))
        missing = claimed - on_disk.get(job.document_id, set())
        if missing:
            broken.append(f"{job.stage}: pages {sorted(missing)} claimed, not saved")
        real_save_job(self, job)

    monkeypatch.setattr(ManifestStore, "save_manifest", save_manifest)
    monkeypatch.setattr(ManifestStore, "save_job", save_job)

    up = client.post(
        "/api/documents",
        files={"file": ("w.pdf", corpus_file("01_clean_worksheet.pdf"), "application/pdf")},
    ).json()
    doc_id = up["document_id"]
    deadline = time.time() + 60
    while client.get(f"/api/documents/{doc_id}/job").json()["stage"] != "REVIEW_READY":
        assert time.time() < deadline
        time.sleep(0.05)

    r = client.post(
        f"/api/documents/{doc_id}/translate",
        json={"target_language": "es", "translate_image_text": False},
    )
    assert r.status_code == 200, r.text
    deadline = time.time() + 120
    while True:
        job = client.get(f"/api/documents/{doc_id}/job").json()
        if job["stage"] in ("COMPLETE", "FAILED"):
            break
        assert time.time() < deadline
        time.sleep(0.05)

    assert job["stage"] == "COMPLETE", job
    assert not broken, broken
    # And the pages are marked done in the end, so a later resume skips them.
    from ct_engine.api.deps import get_manifest_store

    saved = get_manifest_store().load_job(doc_id, job["job_id"])
    assert saved.completed_pages.get(JobStage.TRANSLATING.value)
