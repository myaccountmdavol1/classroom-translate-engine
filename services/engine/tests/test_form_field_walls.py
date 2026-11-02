"""A fillable form's own fields are walls to a growing text box.

On the FirstDue360 safety form each checklist row has Yes / No / N/A
checkboxes and a Notes box to its right, all of them form fields, not drawn
lines. The first pass widened the Spanish rows straight across the
checkboxes, and nothing flagged it."""

from __future__ import annotations

from ct_engine.model.document import Page
from ct_engine.model.enums import ObjectKind, TextSource
from ct_engine.model.geometry import Rect
from ct_engine.model.objects import FormFieldObject, TextLine, TextObject, TextSpan
from ct_engine.reconstruct.builder import _obstacles
from ct_engine.reconstruct.fitting import available_growth_right


def _rect(box):
    return Rect(x0=box[0], y0=box[1], x1=box[2], y1=box[3])


def _row(box, text):
    r = _rect(box)
    return TextObject(
        id="row", kind=ObjectKind.TEXT, bbox=r, norm_bbox=r.normalized(612, 792),
        source=TextSource.NATIVE, lines=[TextLine(bbox=r, spans=[TextSpan(text=text, bbox=r)])],
    )


def _field(name, box, kind):
    r = _rect(box)
    return FormFieldObject(
        id=name, bbox=r, norm_bbox=r.normalized(612, 792), field_name=name, field_type=kind,
    )


def _room(page, row):
    return available_growth_right(row.bbox, page.bounds, _obstacles(page, row.id))


def test_widening_stops_at_a_form_checkbox():
    row = _row((40, 408, 257, 420), "Smoke alarms are installed near sleeping areas.")
    page = Page(
        index=0, number=1, width=612, height=792, text_objects=[row],
        form_fields=[
            _field("yes", (289, 411, 299, 421), "CheckBox"),
            _field("note", (376, 408, 572, 424), "Text"),
        ],
    )
    assert row.bbox.x1 + _room(page, row) <= 289


def test_a_zero_size_widget_is_not_a_wall():
    row = _row((40, 408, 257, 420), "Name")
    page = Page(
        index=0, number=1, width=612, height=792, text_objects=[row],
        form_fields=[_field("ghost", (300, 412, 300, 412), "Text")],
    )
    assert row.bbox.x1 + _room(page, row) > 300


def test_a_field_the_words_sit_in_does_not_stop_them():
    row = _row((40, 408, 257, 420), "Name")
    page = Page(
        index=0, number=1, width=612, height=792, text_objects=[row],
        form_fields=[_field("name", (36, 404, 400, 424), "Text")],
    )
    assert _room(page, row) > 100
