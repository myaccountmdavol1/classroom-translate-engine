"""Translation protection layer (spec sec. 20 / 23 / 24).

Some content must survive translation byte-identical: equations, chemical
formulas, URLs, question numbers, answer-choice labels. If `3x + 7 = 22`
becomes `3x + 7 = 22.` or `H₂O` becomes `H2O`, the assessment is damaged.

Mechanism: protected spans are replaced by placeholders before translation
and restored afterward.

PLACEHOLDER FORMAT -- chosen empirically, do not change casually.

    #N#   (e.g. #0#, #1#)

Machine translation mangles most sentinel formats. Measured against Argos NMT
with three placeholders per sentence across es/fr/zh/ar/ru:

    #N#        survived all five languages, in order
    [[N]]      lost a bracket in zh, dropped tokens in ru
    NNNxNNN    corrupted in zh
    <x1/>      corrupted even in es
    {{TOK_1}}  underscore became a space in es

Any new format must be re-measured the same way, and any provider's output is
verified and repaired below regardless -- a provider that mangles a
placeholder must fail its segment loudly, never emit a corrupted formula.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- Placeholder plumbing -------------------------------------------------
PLACEHOLDER_TMPL = "#{}#"
# Tolerant matcher used when repairing provider output: accepts stray spaces
# introduced by tokenizers, e.g. "# 3 #".
_PLACEHOLDER_RE = re.compile(r"#\s*(\d{1,3})\s*#")


def placeholder(index: int) -> str:
    return PLACEHOLDER_TMPL.format(index)


# --- Detection patterns ---------------------------------------------------
# Ordered by priority: earlier patterns win when spans overlap, because the
# more specific categories appear first.

_URL = re.compile(r"\b(?:https?://|www\.)[^\s,;)\]]+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Chemical formulas: element sequences that carry a digit or a Unicode
# subscript. The digit requirement is what stops ordinary words like "He" or
# "In" from being protected.
# Leading elements may carry no count ("CO2" is C then O2), so the count is
# optional per element. The requirement that SOME digit be present lives in
# _looks_like_chemistry, which is what keeps "USA" and "PDF" out.
_CHEM = re.compile(r"\b(?:[A-Z][a-z]?[\d₀-₉]*){1,8}\b")

# Mathematical expressions are found by scanning token runs rather than with
# one greedy regex. A greedy character-class match absorbs the prose that
# follows an equation ("3x + 7 = 22 for x"), and the whole match is then
# rejected as prose -- losing the equation entirely. Token runs bound the
# match at the first non-mathematical word.
_MATH_CHAR_RE = re.compile(
    r"^[0-9A-Za-z\u03b1-\u03c9\u0391-\u03a9"
    r"\+\-\*/=<>^_(){}\[\].,:\u00b7\u00d7\u00f7\u2212\u2264\u2265\u2260"
    r"\u2248\u221a\u03c0\u00b0\u00b2\u00b3\u00b9\u2070-\u209f\u2032\u2033|"
    # Raised and lowered letters recovered from Word scripts (a\u1d50, x\u207f, v\u2093).
    r"\u02b0-\u02b8\u02e0-\u02e4\u1d2c-\u1d6a\u1d9b-\u1dbf\u2c7c\u2c7d]+$"
)
_RELATIONS = "=<>\u2264\u2265\u2260\u2248"
# ASCII hyphen-minus is included: it is the minus sign in virtually every
# classroom document, and excluding it split "5(y - 2) = 35" into fragments,
# corrupting the expression. Prose hyphens ("well-known", "home - it") are
# excluded by the run guards instead, which require a digit or a single-letter
# variable and reject runs shorter than three characters.
_OPERATORS = "+*/^-\u00d7\u00f7\u2212\u00b7\u22c5"

# A raised or lowered character (x\u00b2, H\u2082O, a\u1d50) or a Greek letter bound to a
# number or variable (2\u03c0r, \u03b8\u2081) makes a term mathematical on its own, with no
# operator in sight. Without this, "x\u00b2" in a sentence went to the translator.
_SCRIPT_RE = re.compile(
    "[\u00b2\u00b3\u00b9\u2070-\u209f\u02b0-\u02b8\u02e0-\u02e4"
    "\u1d2c-\u1d6a\u1d9b-\u1dbf\u2c7c\u2c7d]"
)
_GREEK_TERM_RE = re.compile(r"^(?=.*[\u0391-\u03c9])(?:.*[0-9A-Za-z].*|.)$")


def _is_standalone_term(tok: str) -> bool:
    return bool(_SCRIPT_RE.search(tok) or _GREEK_TERM_RE.match(tok))

# Standalone algebraic terms such as 3x, 2y², x₁ -- a coefficient bound to a
# variable, which must not be reordered or localized.
_ALGEBRA_TERM = re.compile(r"\b\d+\s?[a-zA-Z](?![a-zA-Z])[²³⁰-₟]?\b")

# Measurements: a number plus a unit symbol. Units are conventionally kept in
# the source form on classroom material.
_MEASUREMENT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?"
    r"(?:mm|cm|km|m|kg|mg|g|lb|oz|ml|mL|L|l|in|ft|yd|mi|"
    r"°C|°F|°|K|N|J|W|V|A|Hz|Pa|mol|s|ms|min|hr?|%)\b"
)

# Citations and identifiers: "(Smith, 2019)", "ISBN 0-306", "Fig. 3", codes.
# Fill-in blanks ("Name: ______", dotted leaders). These ARE the student's
# response space: a translator that drops them destroys somewhere to write
# (spec sec. 28), which is exactly what happened before they were protected.
_BLANK = re.compile(r"_{2,}|\.{4,}|\u2026{2,}|\u2014{3,}")

_CITATION = re.compile(r"\([A-Z][A-Za-z.\-]+(?:\s+et\s+al\.)?,?\s+\d{4}[a-z]?\)")
_IDENTIFIER = re.compile(r"\b(?:[A-Z]{2,}[-\s]?\d{2,}|\d{2,}[-][A-Z0-9]{2,})\b")

# An existing "#12#" in the source would be indistinguishable from one of our
# placeholders, so it is protected first and restored verbatim.
_LITERAL_SENTINEL = re.compile(r"#\s*\d{1,3}\s*#")

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("sentinel", _LITERAL_SENTINEL),
    ("blank", _BLANK),
    ("url", _URL),
    ("email", _EMAIL),
    ("citation", _CITATION),
    ("chemistry", _CHEM),
    ("algebra", _ALGEBRA_TERM),
    ("measurement", _MEASUREMENT),
    ("identifier", _IDENTIFIER),
]

# --- Leading labels -------------------------------------------------------
# Question numbers and answer-choice labels lead a block and must keep their
# exact form and position: renumbering a question breaks the answer key.
_LEADING_QUESTION = re.compile(r"^\s*(\d{1,3}[.)]?)\s+")
_LEADING_CHOICE = re.compile(r"^\s*(\(?[A-Da-d1-9]\)|\(\d\)|[A-Da-d][.)])\s+")
_PAGE_NUMBER_ONLY = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.IGNORECASE)


@dataclass
class ProtectedSpan:
    start: int
    end: int
    text: str
    category: str
    index: int = -1

    @property
    def token(self) -> str:
        return placeholder(self.index)


@dataclass
class MaskResult:
    """Text with protected content replaced by placeholders."""

    masked_text: str
    spans: list[ProtectedSpan] = field(default_factory=list)

    @property
    def has_protection(self) -> bool:
        return bool(self.spans)

    @property
    def is_fully_protected(self) -> bool:
        """True when nothing translatable remains.

        Such a block is passed through unchanged rather than sent to a
        translation engine that would only have placeholders to work with.
        """
        stripped = _PLACEHOLDER_RE.sub("", self.masked_text)
        return not any(ch.isalpha() for ch in stripped)


@dataclass
class RestoreResult:
    text: str
    restored: int = 0
    repaired: list[int] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def find_protected_spans(text: str) -> list[ProtectedSpan]:
    """Locate every span that must survive translation unchanged.

    Conservative by construction: a false positive leaves a phrase
    untranslated (visible, fixable in review), while a false negative can
    silently corrupt a formula.
    """
    claimed: list[tuple[int, int]] = []
    found: list[ProtectedSpan] = []

    from .value_checks import CLOCK, NUMERIC_DATE
    for category, pattern in (("time", CLOCK), ("date", NUMERIC_DATE)):
        for match in pattern.finditer(text):
            if not any(_overlaps(match.span(), c) for c in claimed):
                claimed.append(match.span())
                found.append(ProtectedSpan(start=match.start(), end=match.end(),
                    text=match.group(), category=category))

    # Equations are claimed first so a whole expression wins over the
    # individual algebraic terms inside it.
    for start, end in find_math_spans(text):
        if any(_overlaps((start, end), c) for c in claimed):
            continue
        claimed.append((start, end))
        found.append(
            ProtectedSpan(start=start, end=end, text=text[start:end], category="math")
        )

    for category, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(0)
            start, end = m.span()
            # Trim whitespace the greedy math pattern may have absorbed.
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())
            start += lead
            end -= trail
            snippet = text[start:end]
            if not snippet.strip():
                continue
            if category == "math" and not _looks_like_math(snippet):
                continue
            if category == "chemistry" and not _looks_like_chemistry(snippet):
                continue
            if any(_overlaps((start, end), c) for c in claimed):
                continue
            claimed.append((start, end))
            found.append(
                ProtectedSpan(start=start, end=end, text=snippet, category=category)
            )

    found.sort(key=lambda s: s.start)
    for i, span in enumerate(found):
        span.index = i
    return found



def _is_math_token(tok: str) -> bool:
    """True when a whitespace-delimited token belongs to an expression."""
    if not tok or not _MATH_CHAR_RE.match(tok):
        return False
    letters = re.findall(r"[A-Za-z]{2,}", tok)
    # Multi-letter runs are prose UNLESS they are known function/unit names or
    # short variable products. "mc\u00b2" is m\u00b7c\u00b2 and must be protected; treating it
    # as a word let a translator rewrite it to "mc2", silently changing the
    # formula. Common short English words are excluded so prose still breaks
    # the run.
    for w in letters:
        low = w.lower()
        if low in _MATH_WORDS:
            continue
        if len(w) <= 3 and low not in _SHORT_WORDS:
            continue
        return False
    has_digit = any(c.isdigit() for c in tok)
    has_symbol = any(c in _RELATIONS + _OPERATORS for c in tok)
    is_variable = bool(re.fullmatch(r"[A-Za-z][\u00b2\u00b3\u2070-\u209f]?", tok))
    return (has_digit or has_symbol or is_variable or _is_standalone_term(tok)
            or tok.lower() in _MATH_WORDS)


def find_math_spans(text: str) -> list[tuple[int, int]]:
    """Maximal runs of mathematical tokens that form a real expression."""
    spans: list[tuple[int, int]] = []
    run_start: int | None = None
    run_end = 0
    pos = 0

    def flush(start: int, end: int) -> None:
        snippet = text[start:end].strip()
        standalone = any(_is_standalone_term(t) for t in snippet.split())
        if len(snippet) < (2 if standalone else 3):
            return
        if standalone:
            real_start = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            spans.append((real_start, real_start + len(snippet)))
            return
        has_relation = any(c in _RELATIONS for c in snippet)
        has_operator = any(c in _OPERATORS for c in snippet)
        has_digit = any(c.isdigit() for c in snippet)
        # An expression needs a relation or operator plus something to act on.
        if not (has_relation or has_operator):
            return
        if not (has_digit or re.search(r"\b[A-Za-z]\b", snippet)):
            return
        # A lone variable beside a bare operator is not an expression: "Part
        # A - Answer all questions" was being protected as math, which left
        # the whole direction untranslated. Require a digit or a relation.
        if not (has_digit or has_relation):
            return
        # Re-anchor onto the trimmed snippet so surrounding spaces are excluded.
        real_start = start + (len(text[start:end]) - len(text[start:end].lstrip()))
        spans.append((real_start, real_start + len(snippet)))

    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        # Trailing sentence punctuation is not part of the expression.
        core = tok.rstrip(".,;:?!")
        trimmed = len(tok) - len(core)
        if core and _is_math_token(core):
            if run_start is None:
                run_start = m.start()
            run_end = m.end() - trimmed
        else:
            if run_start is not None:
                flush(run_start, run_end)
                run_start = None
        pos = m.end()
    if run_start is not None:
        flush(run_start, run_end)
    return spans


def _looks_like_math(s: str) -> bool:
    """Reject prose that merely contains a hyphen or slash."""
    stripped = s.strip()
    if len(stripped) < 2:
        return False
    has_relation = any(c in stripped for c in "=<>≤≥≠≈")
    has_operator = any(c in _OPERATORS for c in stripped)
    has_digit = any(c.isdigit() for c in stripped)
    # Single-letter variables are mathematical; multi-letter words are prose.
    words = re.findall(r"[A-Za-z]{2,}", stripped)
    prose_words = [w for w in words if w.lower() not in _MATH_WORDS]
    if prose_words:
        return False
    if has_relation and (has_digit or re.search(r"\b[a-zA-Z]\b", stripped)):
        return True
    # "3 + 4" with no relation is still an expression to preserve.
    return has_operator and has_digit


# Short English words that must NOT be mistaken for variable products.
_SHORT_WORDS = {
    "a", "an", "the", "is", "are", "was", "of", "to", "in", "it", "and", "or",
    "for", "at", "by", "on", "be", "as", "if", "we", "do", "so", "no", "up",
    "my", "me", "he", "she", "his", "her", "you", "all", "but", "not", "can",
    "has", "had", "how", "why", "who", "one", "two", "six", "ten", "own",
    "get", "out", "use", "see", "may", "now", "any", "new", "add", "per",
}

# Function and unit names that may legitimately appear inside an expression.
_MATH_WORDS = {
    "sin", "cos", "tan", "log", "ln", "exp", "sqrt", "abs", "max", "min",
    "pi", "cm", "mm", "km", "kg", "mol", "sec",
}


def _looks_like_chemistry(s: str) -> bool:
    stripped = s.strip()
    if len(stripped) < 2 or len(stripped) > 24:
        return False
    if not any(c.isdigit() or "₀" <= c <= "₉" for c in stripped):
        return False
    # Must read as element symbols: capital, optional lowercase, optional count.
    return bool(re.fullmatch(r"(?:[A-Z][a-z]?[\d₀-₉]*)+", stripped))


def mask_text(text: str) -> MaskResult:
    """Replace protected spans with placeholders."""
    spans = find_protected_spans(text)
    if not spans:
        return MaskResult(masked_text=text, spans=[])

    out: list[str] = []
    cursor = 0
    for span in spans:
        out.append(text[cursor : span.start])
        out.append(span.token)
        cursor = span.end
    out.append(text[cursor:])
    return MaskResult(masked_text="".join(out), spans=spans)


def restore_text(translated: str, spans: list[ProtectedSpan]) -> RestoreResult:
    """Put protected content back, repairing tokenizer damage where possible.

    Returns a report rather than raising: the caller decides whether a segment
    with missing placeholders should fail (it should) while the rest of the
    document continues.
    """
    if not spans:
        return RestoreResult(text=translated)

    by_index = {s.index: s for s in spans}
    seen: set[int] = set()
    repaired: list[int] = []

    def sub(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        span = by_index.get(idx)
        if span is None:
            # A placeholder index we never issued: leave the text as-is rather
            # than inventing content.
            return m.group(0)
        seen.add(idx)
        if m.group(0) != span.token:
            repaired.append(idx)
        return span.text

    result = _PLACEHOLDER_RE.sub(sub, translated)
    missing = sorted(set(by_index) - seen)

    # --- Tolerant repair pass -------------------------------------------
    # Measured NMT damage modes (Argos en->fr/zh/ar): a delimiter is eaten
    # ("#0#" -> "0#" or "#0"), or "#" is translated into a word and one
    # delimiter survives. Both keep the index adjacent to a surviving "#",
    # which is specific enough to repair safely.
    #
    # Only indices still missing are repaired, and only where a "#" survives
    # on one side -- never a bare digit, which would rewrite real numbers in
    # the translation (a catastrophic edit on a math worksheet).
    for idx in list(missing):
        span = by_index[idx]
        # Digit-adjacency rather than \b: CJK characters are word characters,
        # so "\b" never matches in "\u4e3a0#" and the repair silently missed every
        # Chinese segment. Guarding against neighbouring digits is what
        # actually prevents matching inside a longer number.
        one_sided = re.compile(rf"(?:#\s*{idx}(?!\d)|(?<!\d){idx}\s*#)")
        m = one_sided.search(result)
        if m is None:
            continue
        result = result[: m.start()] + span.text + result[m.end() :]
        seen.add(idx)
        repaired.append(idx)
        missing.remove(idx)

    return RestoreResult(
        text=result,
        restored=len(seen),
        repaired=sorted(set(repaired)),
        missing=missing,
    )


# --- Block-level decisions ------------------------------------------------
def is_page_number(text: str) -> bool:
    return bool(_PAGE_NUMBER_ONLY.match(text.strip()))


def split_leading_label(text: str) -> tuple[str, str]:
    """Separate a leading question number or answer letter from the body.

    Returns (label, remainder). The label is never sent for translation, so
    "3." cannot become "3.°" or be reordered, and numbering stays aligned with
    the answer key (spec sec. 24).
    """
    stripped = text.lstrip()
    offset = len(text) - len(stripped)
    for pattern in (_LEADING_CHOICE, _LEADING_QUESTION):
        m = pattern.match(stripped)
        if m:
            end = m.end()
            return text[: offset + end], stripped[end:]
    return "", text


def has_translatable_content(text: str) -> bool:
    """True when there is natural language worth sending to a translator."""
    masked = mask_text(text).masked_text
    without_tokens = _PLACEHOLDER_RE.sub("", masked)
    letters = [c for c in without_tokens if c.isalpha()]
    return len(letters) >= 2


def peel_edges(text: str) -> tuple[str, str, str]:
    """Split leading/trailing protected spans out of the translatable core.

    "Name: ______" is on virtually every worksheet. Masking the blank to a
    placeholder puts it at the mercy of the translator, and when the
    placeholder does not survive the whole line falls back to English.

    Because the blank sits at the edge, it never needs to move: peel it off,
    translate "Name:", and re-attach. That removes the placeholder risk for
    the most common pattern in the corpus entirely.

    Returns (prefix, core, suffix); core may be empty if the whole string was
    protected.
    """
    spans = find_protected_spans(text)
    if not spans:
        return "", text, ""

    prefix = ""
    suffix = ""
    start, end = 0, len(text)

    first = spans[0]
    if first.start == 0:
        prefix = text[: first.end]
        start = first.end

    last = spans[-1]
    if last.end == len(text) and last.start >= start:
        suffix = text[last.start :]
        end = last.start

    core = text[start:end]
    # Keep the whitespace that separated them so the line rebuilds exactly.
    return prefix, core, suffix


# --- Symbols the translator mangles ---------------------------------------

# A standalone ampersand: surrounded by whitespace, so "AT&T", "R&D" and
# "Barnes&Noble" are left alone -- those are names, not conjunctions.
_STANDALONE_AMP = re.compile(r"(?<=\s)&(?=\s)|^&(?=\s)|(?<=\s)&$")

# Entities an engine emits when it HTML-escapes its own output.
_ENTITY = re.compile(r"&(?:amp|quot|apos|lt|gt|nbsp|#\d{2,4}|#x[0-9a-fA-F]{2,4});")

_ENTITY_TEXT = {
    "&amp;": "&",
    "&quot;": '"',
    "&apos;": "'",
    "&lt;": "<",
    "&gt;": ">",
    "&nbsp;": " ",
}


def normalize_for_translation(text: str) -> str:
    """Rewrite symbols the translation engine handles badly.

    Currently one rule: a standalone `&` becomes the word "and".

    Measured against Argos on real worksheet titles. Left alone, `&` comes
    back as a literal `&amp;` or `&quot;` -- the engine HTML-escapes its own
    output -- and `&quot;` is worse than it looks, because the ampersand is
    gone rather than escaped. Masking it as a placeholder was tried and is
    worse still: `"Mrs. Rivera & Mrs. O'Connor"` came back as `"Sra. Rivera
    #0#"`, silently dropping a name. Substituting the word lets the engine
    translate the conjunction properly -- `"Causa y efecto"`, `"Taller de
    lectura y escritura"` -- which is what the ampersand meant.

    The ampersand is NOT restored afterwards. A translated conjunction is a
    faithful rendering of it, and there is no way to know which word in the
    output was the one substituted.

    Whitespace at the edges is preserved, because the caller re-attaches
    peeled protected content against it.
    """
    return _STANDALONE_AMP.sub("and", text)


def repair_entities(translated: str, source: str) -> str:
    """Undo HTML escaping the translation engine applied to its own output.

    A last line of defence, not the main fix: `normalize_for_translation`
    removes the common cause. But a literal `&amp;` reaching a student is
    never acceptable, whatever produced it, so any entity that was not in
    the source is decoded. An entity the source really contained is left
    exactly as it was -- a worksheet that teaches HTML must keep saying
    `&amp;`.
    """
    if "&" not in translated:
        return translated

    def sub(match: re.Match) -> str:
        entity = match.group(0)
        if entity in source:
            return entity  # the source said it; keep it
        lowered = entity.lower()
        if lowered in _ENTITY_TEXT:
            return _ENTITY_TEXT[lowered]
        # Numeric form: "&#233;" (decimal) or "&#xe9;" (hex).
        code = lowered[2:-1]
        try:
            value = int(code[1:], 16) if code.startswith("x") else int(code)
            return chr(value)
        except (ValueError, OverflowError):
            return entity

    return _ENTITY.sub(sub, translated)


def is_math_dominant(text: str) -> bool:
    """True when a line is an equation with at most a stray misread word.

    Guards against a specific, damaging failure seen on a real scanned
    homework page. Tesseract read `-51 + 15x - 11x = 41` as
    `(5) "91+ ox - 11x = 41`. Because "ox" is a real English word, the line
    looked translatable, so it was translated -- to `"91+ buey - 11x = 41` --
    and drawn over the original scan. The five equations beside it, which
    happened to contain no word-shaped misreads, were left alone and stayed
    correct.

    Altering the numbers in a maths question is the worst thing this program
    can do (spec sec. 24), and a recognition error must never be the thing
    that triggers it. So a line that is overwhelmingly an expression is
    treated as an expression, and the one or two stray short tokens in it are
    assumed to be misreads rather than words worth translating.

    Deliberately narrow, because a word problem must still translate:
      * a relation (=, <, >, ...) must be present -- this is an equation,
        not prose containing a number
      * at least two thirds of the tokens must be mathematical
      * every non-mathematical token must be at most three characters, so
        "Solve for x: 3x + 7 = 22" and "The sum of 3 and 4 is 7" both fail
    """
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if len(tokens) < 3:
        return False
    if not any(c in _RELATIONS for c in text):
        return False

    math_tokens = 0
    for tok in tokens:
        stripped = tok.strip("()[].,;:")
        if not stripped:
            math_tokens += 1
            continue
        if _is_math_token(stripped) or stripped.isdigit():
            math_tokens += 1
        elif len(re.sub(r"[^A-Za-z]", "", stripped)) > 3:
            return False  # a real word: this is prose
    return math_tokens / len(tokens) >= 2 / 3
