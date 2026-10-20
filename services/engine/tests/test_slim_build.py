"""The deployment without the offline translation model.

The free hosting tier has 512 MB of RAM. The engine fits only because the
image is built with LOCAL_TRANSLATION=0, which leaves out torch and the
Argos model -- 3 GB of image and ~320 MB of RAM. Measured peak for a
37-page packet on that build is 286 MB.

`main.py` imports the Argos provider module unconditionally, so if that
module ever grows a top-level `import argostranslate` the slim image stops
booting -- in production, not here. These tests simulate the dependency
being absent so that mistake fails in CI instead.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest
from fastapi.testclient import TestClient

ABSENT = {"argostranslate", "torch", "stanza"}


@pytest.fixture
def without_offline_model(monkeypatch):
    """Make the offline translation dependencies unimportable."""
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in ABSENT:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    for module in list(sys.modules):
        if module.split(".")[0] in ABSENT:
            monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(builtins, "__import__", guard)
    yield


def test_the_engine_boots_without_the_offline_model(
    without_offline_model, monkeypatch
):
    monkeypatch.setenv("CT_TRANSLATION_PROVIDER", "deepl")
    monkeypatch.setenv("CT_DEEPL_API_KEY", "not-a-real-key")
    from ct_engine.config import get_settings

    get_settings.cache_clear()
    import ct_engine.main as main

    importlib.reload(main)
    with TestClient(main.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["translation_provider"] == "deepl"
        assert client.get("/api/languages").status_code == 200
    get_settings.cache_clear()


def test_a_remote_provider_still_builds(without_offline_model, monkeypatch):
    monkeypatch.setenv("CT_DEEPL_API_KEY", "not-a-real-key")
    from ct_engine.config import get_settings
    from ct_engine.providers.registry import get_translation_provider

    get_settings.cache_clear()
    provider = get_translation_provider("deepl")
    assert provider.name == "deepl"
    assert provider.sends_data_externally is True
    get_settings.cache_clear()


def test_the_offline_provider_refuses_loudly_rather_than_silently(
    without_offline_model,
):
    """A deploy that names 'local' by mistake must fail, not pretend."""
    from ct_engine.providers.registry import get_translation_provider
    from ct_engine.providers.translation.base import TranslationSegment

    provider = get_translation_provider("local")
    assert provider.supports("en", "es") is False
    with pytest.raises(Exception) as caught:
        provider.translate(
            [TranslationSegment(segment_id="a", text="hello")],
            source_lang="en",
            target_lang="es",
        )
    assert "not installed" in str(caught.value).lower()


def test_ocr_does_not_depend_on_the_offline_model(without_offline_model):
    """Scanned pages must still work on the slim image."""
    from ct_engine.providers.registry import OCR_PROVIDERS

    assert "local" in OCR_PROVIDERS
