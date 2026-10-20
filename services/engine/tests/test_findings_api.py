"""The problem queue behind the editor's right-hand rail.

The engine runs ten layout checks and then discards the result into one
undifferentiated report line. This groups them so a teacher can work down
one kind at a time.
"""

from __future__ import annotations

import pathlib

from ct_engine.api import findings as findings_module
from ct_engine.api.findings import group_findings
from ct_engine.model.document import DocumentManifest, Page, PageQA, QAFinding


def _manifest(pages: list[Page]) -> DocumentManifest:
    return DocumentManifest(
        document_id="d",
        filename="x.pdf",
        content_type="application/pdf",
        byte_size=1,
        sha256="a" * 64,
        source_key="k",
        pages=pages,
    )


def _page(number: int, findings: list[QAFinding]) -> Page:
    page = Page(index=number - 1, number=number, width=612, height=792)
    page.qa = PageQA(findings=findings)
    return page


def _overlap() -> QAFinding:
    return QAFinding(
        severity="warning",
        code="collision_text",
        message="Two blocks of text on this page may overlap.",
        object_ids=["o1", "o2"],
    )


def test_findings_are_grouped_by_kind_across_pages():
    manifest = _manifest([_page(2, [_overlap()]), _page(4, [_overlap()])])
    groups = group_findings(manifest)
    assert len(groups) == 1
    assert groups[0]["code"] == "collision_text"
    assert groups[0]["pages"] == [2, 4]
    assert len(groups[0]["items"]) == 2


def test_a_group_carries_a_teacher_facing_title():
    manifest = _manifest(
        [_page(1, [QAFinding(severity="warning", code="font_reduced", message="m")])]
    )
    title = group_findings(manifest)[0]["title"]
    assert title == "Text was made smaller to fit"
    assert "_" not in title and "=" not in title


def test_informational_findings_are_not_in_the_queue():
    """The rail is a to-do list. Only things worth acting on belong in it."""
    manifest = _manifest(
        [_page(1, [QAFinding(severity="info", code="fyi", message="Just so you know.")])]
    )
    assert group_findings(manifest) == []


def test_an_unknown_code_still_appears_with_its_own_message():
    """A check added later must not vanish from the queue."""
    manifest = _manifest(
        [
            _page(
                1,
                [
                    QAFinding(
                        severity="warning",
                        code="brand_new_check",
                        message="Something specific happened.",
                    )
                ],
            )
        ]
    )
    groups = group_findings(manifest)
    assert len(groups) == 1
    assert groups[0]["title"] == "Something specific happened."


def test_groups_are_ordered_worst_first():
    manifest = _manifest(
        [
            _page(
                1,
                [
                    QAFinding(severity="warning", code="font_reduced", message="m"),
                    QAFinding(severity="error", code="out_of_bounds", message="m"),
                ],
            )
        ]
    )
    assert [g["severity"] for g in group_findings(manifest)] == ["error", "warning"]


def test_every_finding_the_engine_can_raise_has_a_title():
    """A group's heading must not be one finding's own message.

    The title is chosen once, from the first finding of that kind in the
    document, and then shown over every row in the group. Most per-finding
    messages name their own page and count, so on the real packet a group
    headed "6 line(s) on page 3 were hard to read and may be wrong." had a
    page-10 row under it -- and `ocr_failed` put a provider error string in
    the heading. This walks the engine's own source rather than a hand-kept
    list, so a new check cannot be added without a name for it.
    """
    import ast

    from ct_engine.api.findings import TITLES

    root = pathlib.Path(findings_module.__file__).parent.parent
    missing: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None
            )
            if name != "QAFinding":
                continue
            kwargs = {
                kw.arg: kw.value
                for kw in node.keywords
                if isinstance(kw.value, ast.Constant)
            }
            code = kwargs.get("code")
            severity = kwargs.get("severity")
            if code is None or severity is None:
                continue  # built from a variable; nothing to check here
            if severity.value not in ("warning", "error"):
                continue  # informational findings never reach the rail
            if code.value not in TITLES:
                missing[code.value] = f"{path.relative_to(root)}:{node.lineno}"

    assert not missing, f"finding codes with no teacher-facing title: {missing}"


def test_an_unknown_document_is_a_404_not_a_500(client):
    assert client.get("/api/documents/nope/findings").status_code == 404


def test_a_block_reports_the_colour_behind_it():
    """The editor paints this behind a block while its text is edited.

    A white patch on a coloured banner is worse than no patch at all.
    """
    from ct_engine.api.routes import _block_out
    from ct_engine.model.enums import ObjectKind
    from ct_engine.model.geometry import NormRect, Rect
    from ct_engine.model.objects import TextObject

    obj = TextObject(
        id="o1",
        kind=ObjectKind.TEXT,
        bbox=Rect(x0=0, y0=0, x1=10, y1=10),
        norm_bbox=NormRect(x0=0, y0=0, x1=1, y1=1),
    )
    out = _block_out(obj)
    assert "background_color" in out
    assert out["background_color"] is None  # not measured until a rebuild

    obj.background_color = "#1a2b3c"
    assert _block_out(obj)["background_color"] == "#1a2b3c"


# --- Only what a teacher can act on ----------------------------------------
# A real 37-page packet showed 63 items, mostly "bold drawn plain", "a label in
# a picture was too unclear" and "lines hard to read" on equations that are
# left untranslated anyway. None of those is something to fix.
def _text(oid, status):
    from ct_engine.model.enums import ObjectKind, TextSource
    from ct_engine.model.geometry import Rect
    from ct_engine.model.objects import TextLine, TextObject, TextSpan
    r = Rect(x0=10, y0=10, x1=100, y1=20)
    return TextObject(id=oid, kind=ObjectKind.TEXT, bbox=r, norm_bbox=r.normalized(612, 792),
                      source=TextSource.OCR, translation_status=status,
                      lines=[TextLine(bbox=r, spans=[TextSpan(text="x", bbox=r)])])


def test_unactionable_checks_are_not_listed():
    page = _page(1, [
        QAFinding(severity="warning", code=c, message=c, object_ids=["a"])
        for c in ("font_weight_unavailable", "image_label_unreadable",
                  "ocr_discarded", "page_reflowed", "bilingual_pairing_dropped")
    ])
    assert group_findings(_manifest([page])) == []


def test_hard_to_read_lines_count_only_when_they_were_translated():
    from ct_engine.model.enums import TranslationStatus
    page = _page(1, [QAFinding(severity="warning", code="ocr_low_confidence",
                               message="m", object_ids=["eq", "prose"])])
    page.text_objects = [_text("eq", TranslationStatus.PROTECTED),
                         _text("prose", TranslationStatus.TRANSLATED)]
    groups = group_findings(_manifest([page]))
    assert [i["object_ids"] for g in groups for i in g["items"]] == [["prose"]]
    page.text_objects = [_text("eq", TranslationStatus.PROTECTED)]
    assert group_findings(_manifest([page])) == []


def test_stored_number_checks_are_replaced_by_the_current_rule():
    page = _page(1, [QAFinding(severity="warning", code="numeric_content_changed",
                               message="old rule", object_ids=["gone"])])
    assert group_findings(_manifest([page])) == []
