"""Words lettered in capitals in a Word picture are translated in capitals."""

from __future__ import annotations

import pytest

from ct_engine.pipeline.translate import match_capitals


@pytest.mark.parametrize(
    "source, translated, expected",
    [
        ("MAY", "Mayo", "MAYO"),
        ("DAy OF THE DORK-WIT", "Día del idiota", "DÍA DEL IDIOTA"),
        ("Notes From the", "Notas de", "Notas de"),
        ("12 - 3!", "12 - 3!", "12 - 3!"),
    ],
)
def test_capitals_follow_the_source(source, translated, expected):
    assert match_capitals(source, translated) == expected
