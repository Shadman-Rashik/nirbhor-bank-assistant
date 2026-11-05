"""Small text helpers shared by every module (no third-party dependencies)."""
from __future__ import annotations

import re
import unicodedata

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_COMMA_IN_NUMBER = re.compile(r"(?<=\d),(?=\d)")
BN_CHAR = re.compile(r"[ঀ-৿]")
LATIN_CHAR = re.compile(r"[A-Za-z]")
LATIN_WORD = re.compile(r"[a-z][a-z']*")

_STOPWORDS = frozenset(
    "a an the is are was were am be do does did i me my you your we our of to in on for "
    "how what when where which who can could will would should if it its this that there "
    "any have has with from at by and or about get".split()
)


def normalize_text(text: str) -> str:
    """NFC-normalise, convert Bangla digits to ASCII and drop commas inside numbers."""
    text = unicodedata.normalize("NFC", text).translate(_BN_DIGITS)
    return _COMMA_IN_NUMBER.sub("", text)


def stem(word: str) -> str:
    """Very small English stemmer: enough to match loan/loans, cheque/cheques, fees/fee."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def english_tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lower-cased, stemmed Latin-script tokens."""
    words = LATIN_WORD.findall(normalize_text(text).lower())
    if drop_stopwords:
        words = [w for w in words if w not in _STOPWORDS]
    return [stem(w) for w in words]


def latin_words(text: str) -> list[str]:
    """Lower-cased raw Latin words (no stemming, no stopword removal)."""
    return LATIN_WORD.findall(text.lower())


def digit_groups(text: str) -> set[str]:
    """Every run of digits in the text, after normalisation (3:00 -> {'3','00'})."""
    return set(re.findall(r"\d+", normalize_text(text)))
