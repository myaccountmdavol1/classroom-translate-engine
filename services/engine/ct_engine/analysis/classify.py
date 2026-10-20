"""Semantic classification of text blocks (spec sec. 21).

The role drives translation behaviour (an ANSWER_CHOICE is translated
differently from a READING_PASSAGE), typography during reconstruction, and
what QA expects to find. A wrong role changes handling silently, so every
rule here requires positive evidence and the default is OTHER.

Confidence is recorded alongside the role. Low-confidence roles are kept for
the reviewer but must not drive destructive decisions.
"""

from __future__ import annotations

import re

from ..model.document import Page
from ..model.enums import SemanticRole
from ..model.objects import TextObject
from ..translate.protection import (
    has_translatable_content,
    is_page_number,
    mask_text,
    split_leading_label,
)

_DIRECTION_LEAD = re.compile(
    r"^\s*(directions?|instructions?|directions to the student|part\s+[A-Z0-9]+\s*[-:—]|"
    r"base your answers?|read the following|using the|for each)\b",
    re.IGNORECASE,
)
_RESPONSE_LABEL = re.compile(
    r"^\s*(name|date|period|class|teacher|student|score|grade|hour|section)\s*:",
    re.IGNORECASE,
)
_QUESTION_LEAD = re.compile(r"^\s*\d{1,3}\s*[.)]?\s+\S")
_CHOICE_LEAD = re.compile(r"^\s*(\(\s*[A-Da-d1-9]\s*\)|[A-Da-d][.)])\s+\S")
_CAPTION_LEAD = re.compile(
    r"^\s*(figure|fig\.|table|chart|graph|diagram|photograph|photo|image|map|source)\b"
    r"[\s.:0-9]",
    re.IGNORECASE,
)


def _body_font_size(page: Page) -> float:
    """Character-weighted modal font size: the page's body text size."""
    weights: dict[float, int] = {}
    for t in page.text_objects:
        size = round(t.style.font_size, 1)
        if size <= 0:
            continue
        weights[size] = weights.get(size, 0) + max(1, t.char_count)
    if not weights:
        return 10.0
    return max(weights.items(), key=lambda kv: kv[1])[0]


def classify_block(
    block: TextObject,
    page: Page,
    *,
    body_size: float,
    is_first_page: bool,
) -> tuple[SemanticRole, float]:
    text = block.flat_text.strip()
    if not text:
        return SemanticRole.OTHER, 0.0

    size = block.style.font_size or body_size
    top_frac = block.bbox.y0 / page.height if page.height else 0.5
    bottom_frac = block.bbox.y1 / page.height if page.height else 0.5
    line_count = len(block.lines)
    words = len(text.split())

    # --- Unambiguous structural roles first ---
    if is_page_number(text):
        return SemanticRole.PAGE_NUMBER, 0.95

    if _RESPONSE_LABEL.match(text):
        return SemanticRole.RESPONSE_LABEL, 0.9

    # A block sitting inside an image is a label on that artwork.
    for img in page.image_objects:
        if img.bbox.overlap_fraction_of_self(block.bbox) > 0.9 or (
            block.bbox.overlap_fraction_of_self(img.bbox) > 0.85
        ):
            return SemanticRole.IMAGE_LABEL, 0.75

    # A block inside a detected table is a cell.
    for table in page.table_objects:
        if block.bbox.overlap_fraction_of_self(table.bbox) > 0.8:
            return SemanticRole.TABLE_CELL, 0.8

    # Fully protected content (a bare equation) is a formula.
    masked = mask_text(text)
    if masked.is_fully_protected and masked.has_protection:
        return SemanticRole.FORMULA, 0.85
    if not has_translatable_content(text):
        return SemanticRole.OTHER, 0.3

    # --- Position-based roles ---
    if bottom_frac > 0.93 and line_count <= 2 and words <= 12:
        return SemanticRole.FOOTER, 0.75
    if top_frac < 0.06 and line_count <= 2 and words <= 12 and size <= body_size * 1.1:
        return SemanticRole.HEADER, 0.7

    # --- Lead-pattern roles ---
    if _CHOICE_LEAD.match(text) and words <= 40:
        return SemanticRole.ANSWER_CHOICE, 0.85
    if _DIRECTION_LEAD.match(text):
        return SemanticRole.DIRECTION, 0.9
    if _CAPTION_LEAD.match(text) and words <= 30:
        return SemanticRole.CAPTION, 0.7
    if _QUESTION_LEAD.match(text):
        label, body = split_leading_label(text)
        if label and body.strip():
            # A numbered line ending in a question mark, or any numbered
            # prompt of reasonable length, is a question.
            confidence = 0.9 if text.rstrip().endswith("?") else 0.75
            return SemanticRole.QUESTION, confidence

    # --- Typographic roles ---
    if size >= body_size * 1.45 and words <= 16:
        # The largest text near the top of page one is the document title.
        if is_first_page and top_frac < 0.25:
            return SemanticRole.TITLE, 0.8
        return SemanticRole.HEADING, 0.7
    if size >= body_size * 1.12 and words <= 12 and block.style.bold:
        return SemanticRole.HEADING, 0.65

    # --- Bulk prose ---
    if line_count >= 3 and words >= 40:
        return SemanticRole.READING_PASSAGE, 0.6

    return SemanticRole.OTHER, 0.2


def classify_page(page: Page, *, is_first_page: bool = False) -> None:
    """Assign a semantic role to every text block on the page, in place."""
    if not page.text_objects:
        return
    body = _body_font_size(page)
    for block in page.text_objects:
        role, confidence = classify_block(
            block, page, body_size=body, is_first_page=is_first_page
        )
        block.role = role
        block.role_confidence = round(confidence, 2)
    page.diagnostics.extra["body_font_size"] = body
