"""Fix layout: more room for the blocks a page's checks flagged.

The first translation fits each block with modest room (48pt down, about
twice as wide) and then shrinks it. On a form whose translation runs long
that leaves a list of "made smaller" and "runs outside its box" items a
teacher would otherwise drag out one by one. Fix layout marks each flagged
block `tidy_room`, so the builder gives it all the empty room beside and
below it, then lets a column of same-size rows share one size, and keeps a
change only where it did not make the page worse.

The marks are saved on the blocks, not applied once: every later rebuild
(any edit on the page) lays them out the same way, and the page's undo
snapshot carries them, so one Undo puts the page back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model.document import DocumentManifest, Page
from ..model.enums import TextSource, TranslationStatus
from ..model.objects import TextObject
from ..storage.base import ObjectStore
from .edit import EditError, _rerender, _restore_snapshot, push_history

# The checks Fix layout answers. Others ("check this time") need a reader.
LAYOUT_CODES = frozenset({"text_overflow", "font_reduced", "collision_text"})

# Rows of one column: left edges this close, sizes this close.
SAME_COLUMN = 2.0
SAME_SIZE = 0.25
# A shared size below this share of the rows' own size would itself be
# flagged "made smaller to fit" (the builder flags below 0.8).
SHARE_FLOOR = 0.80

_DRAWN = (TranslationStatus.TRANSLATED, TranslationStatus.MANUAL_EDIT)


@dataclass
class FixLayoutResult:
    changed: list[Page] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)


def _eligible(obj: TextObject) -> bool:
    """A block the engine placed, upright, outside pictures, sized by it."""
    return (
        obj.manual_bbox is None
        and not obj.is_manual_block
        and not obj.render_angle
        and obj.source is not TextSource.IMAGE_OCR
        and not obj.fill_box
        and not obj.manual_fit
        and not obj.manual_size_locked
        and not obj.keep_original
        and not obj.suppressed
        and obj.translation_status in _DRAWN
    )


def _layout_keys(page: Page) -> set[tuple[str, tuple[str, ...]]]:
    return {
        (f.code, tuple(sorted(f.object_ids)))
        for f in page.qa.findings
        if f.code in LAYOUT_CODES
    }


def _flagged(page: Page) -> set[str]:
    return {i for _, ids in _layout_keys(page) for i in ids}


def _review_key(page: Page, code: str, ids: tuple[str, ...]) -> str:
    """The key the review list uses for an item (see api/findings.py)."""
    return f"{code}:{page.number}:" + ",".join(ids)


def _unreviewed_ids(manifest: DocumentManifest, page: Page) -> set[str]:
    """Blocks named by a layout item she has not already checked off. Items
    she accepted are hers to leave as they are."""
    reviewed = set(manifest.reviewed_findings)
    return {
        i
        for code, ids in _layout_keys(page)
        if _review_key(page, code, ids) not in reviewed
        for i in ids
    }


def share_sizes(page: Page, flagged: set[str]) -> bool:
    """Let a column of same-size rows that holds a flagged block share one
    size: the smallest any of them was drawn at, if that is close enough to
    their own size not to be flagged itself. True when any block changed."""
    rows = sorted(
        (
            o for o in page.text_objects
            if _eligible(o) and o.translated_font_size and len(o.lines) == 1
        ),
        key=lambda o: (o.style.font_size, o.bbox.x0, o.bbox.y0),
    )
    # One-line rows with the same left edge and size, then split into runs
    # where each row sits close under the one above (a stacked list, not two
    # unrelated rows that happen to share a margin).
    columns: list[list[TextObject]] = []
    for obj in rows:
        last = columns[-1][-1] if columns else None
        if (
            last is not None
            and abs(obj.style.font_size - columns[-1][0].style.font_size) <= SAME_SIZE
            and abs(obj.bbox.x0 - columns[-1][0].bbox.x0) <= SAME_COLUMN
        ):
            columns[-1].append(obj)
        else:
            columns.append([obj])
    groups: list[list[TextObject]] = []
    for column in columns:
        run: list[TextObject] = []
        for obj in sorted(column, key=lambda o: o.bbox.y0):
            if run and obj.bbox.y0 - run[-1].bbox.y1 > 3 * run[-1].bbox.height:
                groups.append(run)
                run = []
            run.append(obj)
        groups.append(run)

    changed = False
    for group in groups:
        if len(group) < 2 or not any(o.id in flagged for o in group):
            continue
        target = min(o.translated_font_size for o in group)
        own = min(o.style.font_size for o in group) or target
        if target < own * SHARE_FLOOR:
            continue
        for obj in group:
            if obj.translated_font_size > target + SAME_SIZE:
                obj.tidy_font_size = target
                changed = True
    return changed


def tidy_page(
    manifest: DocumentManifest, page_number: int, *, store: ObjectStore, target_lang: str
) -> bool:
    """Give the flagged blocks on one page more room. True when the page
    changed (one undo step recorded); False when there was nothing to do or
    nothing helped (no step recorded, the page as it was)."""
    page = manifest.page(page_number - 1)
    if page is None:
        raise EditError(f"Page {page_number} does not exist in this document.")
    before = _layout_keys(page)
    flagged = _unreviewed_ids(manifest, page)
    targets = [
        o for o in page.text_objects
        if o.id in flagged and _eligible(o) and not o.tidy_room
    ]
    if not targets:
        return False
    sizes = {o.id: o.translated_font_size or 0.0 for o in targets}
    shared_before = {o.id: o.tidy_font_size for o in page.text_objects}

    # What a revert must hand back exactly: a try that helped nothing is not
    # an edit, so it must not cost her a Redo, her oldest Undo step, or the
    # items she had already checked off on this page.
    saved_history = list(page.edit_history)
    saved_redo = list(page.redo_history)
    saved_reviewed = list(manifest.reviewed_findings)

    push_history(page)
    snapshot = page.edit_history[-1]

    def put_back() -> None:
        _restore_snapshot(manifest, page, snapshot, store=store, target_lang=target_lang)
        page.edit_history[:] = saved_history
        page.redo_history[:] = saved_redo
        manifest.reviewed_findings[:] = saved_reviewed

    try:
        for obj in targets:
            obj.tidy_room = True
        _rerender(manifest, page, None, store=store, target_lang=target_lang)
        if share_sizes(page, flagged):
            _rerender(manifest, page, None, store=store, target_lang=target_lang)

        # A block whose new room made an overlap it did not have goes back.
        new = _layout_keys(page) - before
        worse = {i for code, ids in new if code == "collision_text" for i in ids}
        worse &= {o.id for o in targets}
        if worse:
            for obj in targets:
                if obj.id in worse:
                    obj.tidy_room = False
            _rerender(manifest, page, None, store=store, target_lang=target_lang)

        # A block none of whose items cleared, and whose words are no bigger,
        # was only moved about: it goes back as it was.
        cleared = before - _layout_keys(page)
        idle = {
            o.id for o in targets
            if o.tidy_room
            and (o.translated_font_size or 0.0) <= sizes[o.id] + 0.1
            and not any(o.id in ids for _, ids in cleared)
        }
        if idle:
            for obj in targets:
                if obj.id in idle:
                    obj.tidy_room = False
                    obj.tidy_font_size = shared_before[obj.id]
            _rerender(manifest, page, None, store=store, target_lang=target_lang)

        after = _layout_keys(page)
        bigger = any(
            (o.translated_font_size or 0.0) > sizes[o.id] + 0.1
            for o in targets if o.tidy_room
        )
        kept = not (after - before or not (before - after or bigger))
    except Exception:
        # Put back once; if that fails too, its error is the one to show.
        put_back()
        raise
    if not kept:
        # Worse somewhere, or nothing better: the page as it was.
        put_back()
        return False

    # The rebuild forgot every checked-off item on this page. Give back the
    # ones Fix layout did not touch; an item about a block it changed is a
    # new layout she should look at again.
    changed_ids = {
        o.id for o in page.text_objects
        if (o.tidy_room and o.id in sizes)
        or o.tidy_font_size != shared_before.get(o.id)
    }
    stale = {
        _review_key(page, code, ids)
        for code, ids in before
        if changed_ids.intersection(ids)
    }
    manifest.reviewed_findings[:] = [k for k in saved_reviewed if k not in stale]
    return True


def fix_layout(
    manifest: DocumentManifest, pages: list[int], *, store: ObjectStore, target_lang: str
) -> FixLayoutResult:
    """Fix layout on each page asked for. A page that fails is put back and
    named; the others keep their fixes."""
    result = FixLayoutResult()
    for number in sorted(set(pages)):
        try:
            if tidy_page(manifest, number, store=store, target_lang=target_lang):
                result.changed.append(manifest.page(number - 1))
        except Exception:
            # tidy_page has already put the page back; one page's surprise
            # must not lose the others' fixes.
            result.failed.append(number)
    return result
