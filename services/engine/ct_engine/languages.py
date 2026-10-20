"""Target-language registry.

Ordered for a US K-12 district: the languages most often needed for home-
language communication come first. Extending the list is a data change, not a
code change (spec sec. 6).

Each entry carries what the rest of the pipeline needs to do its job
correctly: writing direction for the RTL layout path, script for font
fallback, and an expansion factor the layout engine uses to predict whether
translated text will still fit (spec sec. 25).
"""

from __future__ import annotations

from pydantic import BaseModel


class Language(BaseModel):
    code: str           # BCP-47, e.g. "zh-Hans", "pt-BR"
    base: str           # ISO-639-1 base, e.g. "zh", "pt"
    english_name: str
    native_name: str
    rtl: bool = False
    script: str = "Latin"
    # Typical length change vs. English, measured in characters. Used to
    # anticipate overflow before rendering, never to alter the translation.
    expansion_factor: float = 1.0
    # Whether the script needs complex shaping (ligatures, joining forms).
    complex_shaping: bool = False
    notes: str | None = None


LANGUAGES: list[Language] = [
    Language(code="es", base="es", english_name="Spanish", native_name="Español",
             expansion_factor=1.25),
    Language(code="zh-Hans", base="zh", english_name="Simplified Chinese",
             native_name="简体中文", script="Han", expansion_factor=0.55,
             notes="Used in mainland China and Singapore."),
    Language(code="zh-Hant", base="zh", english_name="Traditional Chinese",
             native_name="繁體中文", script="Han", expansion_factor=0.55,
             notes="Used in Taiwan, Hong Kong and Macau."),
    Language(code="ar", base="ar", english_name="Arabic", native_name="العربية",
             rtl=True, script="Arabic", expansion_factor=1.0, complex_shaping=True),
    Language(code="ht", base="ht", english_name="Haitian Creole",
             native_name="Kreyòl Ayisyen", expansion_factor=1.2),
    Language(code="pt-BR", base="pt", english_name="Portuguese (Brazil)",
             native_name="Português (Brasil)", expansion_factor=1.25),
    Language(code="pt-PT", base="pt", english_name="Portuguese (Portugal)",
             native_name="Português (Portugal)", expansion_factor=1.25),
    Language(code="fr", base="fr", english_name="French", native_name="Français",
             expansion_factor=1.3),
    Language(code="bn", base="bn", english_name="Bengali", native_name="বাংলা",
             script="Bengali", expansion_factor=1.1, complex_shaping=True),
    Language(code="ur", base="ur", english_name="Urdu", native_name="اردو",
             rtl=True, script="Arabic", expansion_factor=1.0, complex_shaping=True),
    Language(code="ru", base="ru", english_name="Russian", native_name="Русский",
             script="Cyrillic", expansion_factor=1.2),
    Language(code="ko", base="ko", english_name="Korean", native_name="한국어",
             script="Hangul", expansion_factor=0.8),
    Language(code="ja", base="ja", english_name="Japanese", native_name="日本語",
             script="Japanese", expansion_factor=0.65),
    Language(code="vi", base="vi", english_name="Vietnamese",
             native_name="Tiếng Việt", expansion_factor=1.15),
    Language(code="pl", base="pl", english_name="Polish", native_name="Polski",
             expansion_factor=1.2),
    Language(code="he", base="he", english_name="Hebrew", native_name="עברית",
             rtl=True, script="Hebrew", expansion_factor=0.9, complex_shaping=True),
    Language(code="uk", base="uk", english_name="Ukrainian",
             native_name="Українська", script="Cyrillic", expansion_factor=1.2),
]

BY_CODE: dict[str, Language] = {lang.code: lang for lang in LANGUAGES}


def get_language(code: str | None) -> Language | None:
    if not code:
        return None
    if code in BY_CODE:
        return BY_CODE[code]
    # Accept a bare base code ("zh") by returning the most common variant,
    # so an API caller that does not know about script tags still works.
    for lang in LANGUAGES:
        if lang.base == code:
            return lang
    return None


def is_supported(code: str | None) -> bool:
    return get_language(code) is not None


# Display names for detected source languages. Detection returns ISO-639-1,
# which may not be a target we support, so this map is intentionally wider.
SOURCE_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "es": "Spanish", "zh": "Chinese", "ar": "Arabic",
    "ht": "Haitian Creole", "pt": "Portuguese", "fr": "French", "bn": "Bengali",
    "ur": "Urdu", "ru": "Russian", "ko": "Korean", "ja": "Japanese",
    "vi": "Vietnamese", "pl": "Polish", "he": "Hebrew", "uk": "Ukrainian",
    "de": "German", "it": "Italian", "hi": "Hindi", "th": "Thai",
    "el": "Greek", "tr": "Turkish", "nl": "Dutch", "so": "Somali",
    "sw": "Swahili", "fa": "Persian", "ta": "Tamil", "te": "Telugu",
    "gu": "Gujarati", "pa": "Punjabi", "ne": "Nepali", "my": "Burmese",
    "km": "Khmer", "lo": "Lao", "am": "Amharic", "yo": "Yoruba",
    "id": "Indonesian", "tl": "Tagalog", "ro": "Romanian", "cs": "Czech",
}


def source_language_name(code: str | None) -> str | None:
    if not code:
        return None
    return SOURCE_LANGUAGE_NAMES.get(code, code.upper())
