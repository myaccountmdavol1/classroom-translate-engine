"""Recovering translations of short, context-free labels.

Diagram labels are the worst possible input for machine translation: two
title-case nouns with no sentence around them. Measured with the offline
engine, en->es:

    "Cell Wall"     -> "Cell Wall"          (unchanged; no translation)
    "cell wall"     -> "pared"              (partial; lost a word)
    "the cell wall" -> "la pared celular"   (correct)

So when a label comes back identical to what went in, it is retried inside a
minimal noun-phrase frame and the article is removed afterwards.

Two properties keep this safe:

  * It only runs when the first attempt returned the source unchanged, so a
    translation that already worked is never touched, and there is no cost on
    the common path.
  * The article is stripped only when the leading word is a known article for
    the TARGET language and the source had no article of its own. Languages
    without articles (Chinese, Japanese, Korean, Russian, Arabic) simply pass
    through with nothing removed.
"""

from __future__ import annotations

import re

# Definite and indefinite articles by language base. Only these are ever
# removed, and only from the start of a result.
_ARTICLES: dict[str, frozenset[str]] = {
    "es": frozenset({"el", "la", "los", "las", "un", "una"}),
    "fr": frozenset({"le", "la", "les", "un", "une", "l'"}),
    "pt": frozenset({"o", "a", "os", "as", "um", "uma"}),
    "it": frozenset({"il", "lo", "la", "i", "gli", "le", "un", "uno", "una"}),
    "de": frozenset({"der", "die", "das", "ein", "eine"}),
    "nl": frozenset({"de", "het", "een"}),
    "ca": frozenset({"el", "la", "els", "les"}),
    "ht": frozenset({"la", "an", "nan"}),
}

_ENGLISH_ARTICLES = ("the ", "a ", "an ")
# Beyond this many words a label has enough context to translate on its own.
MAX_FRAMED_WORDS = 5


def should_frame(text: str) -> bool:
    """True when `text` is short enough to be worth retrying framed."""
    stripped = text.strip()
    if not stripped or len(stripped) < 3:
        return False
    words = stripped.split()
    if len(words) > MAX_FRAMED_WORDS:
        return False
    # Sentences and questions already carry context.
    if stripped[-1] in ".?!:;":
        return False
    if not any(c.isalpha() for c in stripped):
        return False
    # An existing article means the phrase is already framed.
    return not stripped.lower().startswith(_ENGLISH_ARTICLES)


def frame(text: str) -> str:
    """Wrap a bare label in a minimal noun phrase."""
    return "the " + text.strip().lower()


def unframe(translated: str, target_lang: str, *, original: str) -> str:
    """Remove the article the frame introduced, and restore capitalization."""
    result = translated.strip()
    if not result:
        return result

    base = (target_lang or "").split("-")[0].lower()
    articles = _ARTICLES.get(base, frozenset())
    if articles:
        first, _, rest = result.partition(" ")
        if rest and first.lower().rstrip("'") in {a.rstrip("'") for a in articles}:
            result = rest.strip()
        elif "'" in first[:2].lower() and first.lower().startswith("l'"):
            result = first[2:] + (" " + rest if rest else "")

    # Mirror the source's capitalization: "Cell Wall" -> "Pared celular",
    # "cell wall" -> "pared celular". Title case is not reproduced word by
    # word because most target languages do not title-case common nouns.
    source = original.strip()
    if source[:1].isupper() and result[:1].islower():
        result = result[0].upper() + result[1:]
    elif source[:1].islower() and result[:1].isupper():
        result = result[0].lower() + result[1:]
    return result


def looks_untranslated(source: str, result: str) -> bool:
    """Did the provider return the input essentially unchanged?"""
    def norm(s: str) -> str:
        return re.sub(r"[\s\W_]+", "", s).lower()

    return norm(source) == norm(result)
