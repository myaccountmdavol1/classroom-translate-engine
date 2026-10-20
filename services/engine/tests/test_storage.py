"""Storage: correctness, atomicity, and path-traversal resistance."""

from __future__ import annotations

import pytest

from ct_engine.storage import keys as K
from ct_engine.storage.base import InvalidKeyError, ObjectNotFoundError


def test_round_trip_bytes_and_text(store):
    store.put_bytes("documents/abc/source.pdf", b"hello")
    assert store.get_bytes("documents/abc/source.pdf") == b"hello"
    store.put_text("documents/abc/manifest.json", '{"a":1}')
    assert store.get_text("documents/abc/manifest.json") == '{"a":1}'
    assert store.size("documents/abc/source.pdf") == 5


def test_overwrite_replaces_content(store):
    store.put_bytes("documents/abc/x.bin", b"first")
    store.put_bytes("documents/abc/x.bin", b"second-longer")
    assert store.get_bytes("documents/abc/x.bin") == b"second-longer"


def test_missing_object_raises(store):
    with pytest.raises(ObjectNotFoundError):
        store.get_bytes("documents/abc/nope.bin")
    assert not store.exists("documents/abc/nope.bin")


@pytest.mark.parametrize(
    "bad",
    ["../../etc/passwd", "/etc/passwd", "documents/../../x", "", "a//b",
     "documents/abc/../../../secret"],
)
def test_path_traversal_is_blocked(store, bad):
    with pytest.raises((InvalidKeyError, ObjectNotFoundError)):
        store.get_bytes(bad)
    with pytest.raises((InvalidKeyError, ObjectNotFoundError)):
        store.put_bytes(bad, b"x")


def test_list_and_delete_prefix(store):
    for i in range(3):
        store.put_bytes(f"documents/d1/pages/{i:04d}/source.png", b"x")
    store.put_bytes("documents/d2/source.pdf", b"y")
    assert len(list(store.list_prefix("documents/d1"))) == 3
    assert store.delete_prefix("documents/d1") == 3
    assert list(store.list_prefix("documents/d1")) == []
    # An unrelated document is untouched.
    assert store.exists("documents/d2/source.pdf")


def test_no_temp_files_leak_into_listings(store):
    store.put_bytes("documents/d3/a.bin", b"x")
    listed = list(store.list_prefix("documents/d3"))
    assert listed == ["documents/d3/a.bin"]
    assert not any(k.endswith(".tmp") for k in listed)


def test_key_helpers_are_consistent():
    doc = "abc123"
    assert K.source_key(doc) == "documents/abc123/source.pdf"
    assert K.manifest_key(doc).startswith(K.doc_prefix(doc))
    assert K.page_source_render_key(doc, 7) == "documents/abc123/pages/0007/source.png"
    # Zero padding keeps page keys lexically sortable past page 9.
    assert K.page_prefix(doc, 12) > K.page_prefix(doc, 9)
    assert K.asset_key(doc, "deadbeef", ".PNG").endswith("deadbeef.png")
