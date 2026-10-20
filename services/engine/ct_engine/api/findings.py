"""Grouping page checks into a queue a teacher can work down.

The engine runs ten layout checks on every page and then throws the result
into a single line of the report -- "review suggested on pages 2, 3, 4,
10..." -- which tells nobody what to do. The same data grouped by kind is a
to-do list: every overlap, then every shrunk block, then every label it
could not read.

No new detection happens here. This is presentation of findings that
already exist.
"""

from __future__ import annotations

from ..model.document import DocumentManifest

# Teacher-facing names for the checks. A code with no entry falls back to
# the finding's own message, so a check added later still reaches the queue
# instead of silently disappearing from it.
#
# That fallback is a safety net, not a design: a group's title is decided
# once, from the FIRST finding of that kind anywhere in the document, and
# then shown above every row in the group. Most per-finding messages name
# their own page and count -- so on the real packet one group was headed
# "6 line(s) on page 3 were hard to read and may be wrong." with rows for
# page 3 and page 10 underneath it, and `ocr_failed` spliced a provider
# error string into the heading. Every code the engine can actually emit at
# warning or error severity therefore has an entry here; a test enumerates
# the source to keep that true. The per-finding messages are left alone --
# they are exactly what is wanted INSIDE the group, on the row for the page
# they describe.
TITLES: dict[str, str] = {
    "collision_text": "Text overlaps other text",
    "collision_response_area": "Text covers space for writing",
    "font_reduced": "Text was made smaller to fit",
    "text_overflow": "Text runs outside its box",
    "out_of_bounds": "Text falls outside the page",
    "image_aspect_changed": "A picture may be stretched",
    "bilingual_pairing_dropped": "Both languages would not fit",
    "font_unavailable": "A typeface could not be loaded",
    "font_weight_unavailable": "Bold or italic text was drawn plain",
    "shaping_unavailable": "This script could not be shaped correctly",
    "scanned_text_not_replaced": "This section still shows the original text",
    "image_label_unreadable": "A label in a picture was too unclear to read",
    "original_kept": "Original text kept",
    "source_time_uncertain": "Check this time in the original",
    "numeric_content_changed": "Check dates, times, and numbers",
    # Reading a scanned page.
    "ocr_render_failed": "A page could not be prepared for reading",
    "ocr_failed": "A page could not be read",
    "ocr_empty": "No text was found on a page",
    "ocr_low_confidence": "Some lines were hard to read and may be wrong",
    "ocr_discarded": "Some marks were too unclear to read",
    "image_text_unreadable": "Text inside a picture could not be read",
    "image_text_not_read": "Words in pictures were not read",
    "image_text_curved": "Curved words in a picture were left in English",
    "image_text_steep": "Steeply turned words in a picture were left in English",
    "image_text_turned": "Slanted words in a picture are now drawn along the slant",
    "image_text_tilt_kept_straight": "Slanted words you placed by hand were left straight",
    "turned_text_drawn_level": "Slanted words were translated but drawn level",
    # Translating.
    "translation_failed": "A page could not be translated",
    "segment_untranslated": "Left in English to keep a formula safe",
    "translation_unavailable": "The translation service did not translate this text",
    "untranslated_text": "Some text was left in the original language",
    # Rebuilding the page.
    "protected_content_lost": "A formula or value did not survive translation",
    "math_script_flattened": "An exponent or subscript is now on the line",
    "text_not_drawn": "Translated text could not be drawn on the page",
    "page_reflowed": "Text was moved down to make room",
    "list_number_changed": "A number in the text was replaced by the list's number",
    "reconstruction_failed": "A page kept its original text",
    "image_not_preserved": "A picture is missing from the translation",
    "office_picture_not_replaced": "A picture kept its original words in the saved file",
    # Word, PowerPoint and Excel files, which are edited in place.
    "docx_anchor_lost": "A paragraph could not be found in the Word file",
    "docx_formatting_collapsed": "Bold or italic inside a paragraph was lost",
    "pptx_anchor_lost": "Text could not be found in the PowerPoint file",
    "pptx_formatting_collapsed": "Bold or italic inside a text box was lost",
    "pptx_text_may_not_fit": "Translated text may not fit its box",
    "xlsx_anchor_lost": "A cell could not be found in the Excel file",
    "xlsx_formatting_collapsed": "Bold or italic inside a cell was lost",
}

# Findings that already say why a block was kept in its original language.
_SAYS_WHY_KEPT = frozenset({"ocr_low_confidence", "image_text_curved", "image_text_steep"})

# Worst first, so the queue opens on what matters.
_SEVERITY_ORDER = {"error": 0, "warning": 1}


# Checks that are recorded but are not a teacher's to-do: nothing she can do
# changes them, and on a real 37-page packet they were most of 63 items,
# burying the few that mattered. Filtered here, when the list is built, so
# documents translated before this rule benefit too.
NOT_ACTIONABLE = frozenset({
    "font_weight_unavailable",   # no bold face for the script; not fixable
    "image_label_unreadable",    # decoration or a smudge inside a picture
    "image_text_unreadable",
    "ocr_discarded",             # marks too unclear to be text
    "page_reflowed",             # a layout note; overlaps are checked directly
    "bilingual_pairing_dropped",
})
# Recomputed from the page every time, so a stored result from an older rule
# never lingers.
RECOMPUTED = frozenset({"numeric_content_changed", "source_time_uncertain"})
_READ_AS_TEXT = frozenset({"TRANSLATED", "MANUAL_EDIT"})


def _actionable(page, finding):
    """The finding as a teacher should see it, or None to leave it out."""
    if finding.code in NOT_ACTIONABLE or finding.code in RECOMPUTED:
        return None
    if finding.code == "ocr_low_confidence":
        # A misread line matters only if it was translated: an equation
        # read shakily is left exactly as it was printed.
        kept = [
            oid for oid in finding.object_ids
            if (obj := page.find_text_object(oid)) is not None
            and obj.translation_status.value in _READ_AS_TEXT
            # She has checked or corrected its English: nothing left to do.
            and obj.approved_source_text is None
        ]
        if not kept:
            return None
        return finding.model_copy(update={"object_ids": kept})
    return finding


def group_findings(manifest: DocumentManifest) -> list[dict]:
    """Every actionable page check, grouped by kind."""
    groups: dict[str, dict] = {}

    for page in manifest.pages:
        from ..translate.value_checks import value_findings
        stored = [f for f in (_actionable(page, f) for f in page.qa.findings) if f]
        # A line kept in English because it was hard to read is one thing to
        # check, not two: "hard to read" says why, so "original kept" for
        # the same line is dropped (it listed every such line twice). The
        # same goes for words kept because they curve or are turned steeply.
        unsure = {
            oid for f in stored if f.code in _SAYS_WHY_KEPT for oid in f.object_ids
        }
        stored = [
            f for f in stored
            if not (f.code == "original_kept" and set(f.object_ids) <= unsure)
        ]
        for finding in [*stored, *value_findings(page)]:
            if finding.severity not in _SEVERITY_ORDER:
                continue  # informational; not a to-do
            group = groups.get(finding.code)
            if group is None:
                group = groups[finding.code] = {
                    "code": finding.code,
                    "title": TITLES.get(finding.code, finding.message),
                    "severity": finding.severity,
                    "pages": [],
                    "items": [],
                }
            # An error anywhere makes the whole group an error, so one
            # serious instance is not hidden behind a pile of warnings.
            if _SEVERITY_ORDER[finding.severity] < _SEVERITY_ORDER[group["severity"]]:
                group["severity"] = finding.severity
            if page.number not in group["pages"]:
                group["pages"].append(page.number)
            group["items"].append(
                {
                    "page": page.number,
                    "object_ids": list(finding.object_ids),
                    "message": finding.message,
                }
            )

    return sorted(
        groups.values(),
        key=lambda g: (_SEVERITY_ORDER[g["severity"]], -len(g["items"])),
    )



def finding_keys(groups):
    return {f'{group["code"]}:{item["page"]}:' + (
        ",".join(sorted(item["object_ids"])) if item["object_ids"] else item["message"]
    ) for group in groups for item in group["items"]}
