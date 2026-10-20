"""Portable local projects. Import always creates a new document namespace."""
from __future__ import annotations

import io
import json
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import PurePosixPath

from ..model.document import DocumentManifest
from ..model.job import DocumentJob
from ..storage import keys as K
from ..util.hashing import sha256_bytes

MAX_PACKAGE_BYTES = 200 * 1024 * 1024


def export_project(document_id, store):
    out = io.BytesIO()
    prefix = K.doc_prefix(document_id) + "/"
    asset_keys = [key for key in store.list_prefix(prefix) if "/repair-proposals/" not in key]
    if len(asset_keys) > 9999 or sum(store.size(key) for key in asset_keys) > MAX_PACKAGE_BYTES - 1024:
        raise ValueError("This project is too large to save as one package. Download the translated document instead.")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps({"schema_version": 1, "document_id": document_id}))
        for key in asset_keys:
            archive.writestr(key.removeprefix(prefix), store.get_bytes(key))
    return out.getvalue()


def import_project(data, store):
    if len(data) > MAX_PACKAGE_BYTES:
        raise ValueError("That project is larger than the 200 MB limit.")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > 10000 or sum(i.file_size for i in infos) > MAX_PACKAGE_BYTES:
            raise ValueError("That project contains too much data to open.")
        names = [i.filename for i in infos]
        if len(set(names)) != len(names):
            raise ValueError("That project contains duplicate files.")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
                raise ValueError("That project contains an invalid file path.")
        header = json.loads(archive.read("project.json"))
        if header.get("schema_version") != 1:
            raise ValueError("This project version is not supported yet.")
        old_id = header["document_id"]
        if not re.fullmatch(r"[a-f0-9]{32}", old_id):
            raise ValueError("That project has an invalid identifier.")
        new_id = uuid.uuid4().hex
        old_prefix, new_prefix = K.doc_prefix(old_id) + "/", K.doc_prefix(new_id) + "/"
        assets = {name: archive.read(name) for name in names if name != "project.json"}
        raw = json.loads(assets["manifest.json"])
        if raw.get("schema_version") != 1 or raw.get("document_id") != old_id:
            raise ValueError("That project's document does not match its package.")

        def remap(value, field=""):
            if isinstance(value, dict):
                return {k: remap(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [remap(v, field) for v in value]
            if isinstance(value, str) and field.endswith("key") and value:
                if not value.startswith(old_prefix) or value[len(old_prefix):] not in assets:
                    raise ValueError("That project is missing a document asset.")
                return new_prefix + value[len(old_prefix):]
            return value

        raw = remap(raw)
        raw["document_id"] = new_id
        raw["created_at"] = raw["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest = DocumentManifest.model_validate(raw)
        source = assets[manifest.source_key[len(new_prefix):]]
        if sha256_bytes(source) != manifest.sha256:
            raise ValueError("The original document does not match its saved checksum.")
        if manifest.original_upload_key:
            original = assets[manifest.original_upload_key[len(new_prefix):]]
            if sha256_bytes(original) != manifest.original_sha256:
                raise ValueError("The original image does not match its saved checksum.")
        assets["manifest.json"] = manifest.model_dump_json().encode()
        for name in list(assets):
            if name.startswith("jobs/") and name.endswith(".json"):
                raw_job = remap(json.loads(assets[name]))
                raw_job["document_id"] = new_id
                job = DocumentJob.model_validate(raw_job)
                if job.stage.value not in ("COMPLETE", "REVIEW_READY", "FAILED", "CANCELLED"):
                    raise ValueError("Wait for processing to finish before saving a project.")
                job.created_at = job.updated_at = datetime.now(timezone.utc)
                assets[name] = job.model_dump_json().encode()
    try:
        for name, payload in assets.items():
            store.put_bytes(new_prefix + name, payload)
    except Exception:
        store.delete_prefix(K.doc_prefix(new_id))
        raise
    return manifest
