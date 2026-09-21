"""Helpers for decoding uploaded text files safely.

The DD review flow accepts SQL uploads from desktop tools that sometimes
save plain text in encodings other than UTF-8. We try a small, ordered
set of common text encodings and only fail if none of them produce
readable text.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_COMMON_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

# Unicode whitespace variants that tools like SSMS/Word silently introduce
# when text is pasted in from a formatted source (EN/EM/THIN/HAIR spaces,
# NBSP, narrow NBSP, ideographic space, ogham space mark, medium mathematical
# space). None of these are `str.isprintable()` in Python (they're category
# Zs, not the ASCII 0x20 exception `isprintable()` special-cases), and left
# alone they can also silently break downstream regexes that match on a
# literal ASCII space. Collapsed to a plain space right after decoding.
_EXOTIC_WHITESPACE_RE = re.compile(
    "[" + chr(0x00A0) + chr(0x1680) + chr(0x202F) + chr(0x205F) + chr(0x3000)
    + chr(0x2000) + "-" + chr(0x200A) + "]"
)


def _normalize_exotic_whitespace(text: str) -> str:
    return _EXOTIC_WHITESPACE_RE.sub(" ", text)


@dataclass(frozen=True)
class DecodedText:
    text: str
    encoding: str


def decode_text_bytes(data: bytes) -> DecodedText:
    """Decode bytes into text using common SQL-friendly encodings."""
    last_error: UnicodeDecodeError | None = None
    for encoding in _COMMON_ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        if _looks_like_text(text):
            return DecodedText(text=_normalize_exotic_whitespace(text), encoding=encoding)

    for encoding in ("utf-16", "utf-16le", "utf-16be"):
        if not _looks_like_utf16(data, encoding):
            continue
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        if _looks_like_text(text):
            return DecodedText(text=_normalize_exotic_whitespace(text), encoding=encoding)

    if last_error is not None:
        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            "Could not decode uploaded file as readable text using common encodings",
        )
    raise UnicodeDecodeError(
        "utf-8",
        data,
        0,
        min(len(data), 1),
        "Could not decode uploaded file as readable text using common encodings",
    )


_NON_TEXT_UNICODE_CATEGORIES = {"Cc", "Cf", "Co", "Cs", "Cn"}


def _looks_like_text(text: str) -> bool:
    """True if `text` looks like real decoded text rather than garbage from
    trying the wrong encoding.

    Judges on actual control/unassigned/surrogate characters (Unicode
    categories Cc/Cf/Co/Cs/Cn), not `str.isprintable()` -- `isprintable()`
    also rejects category Zs (space separator) characters other than the
    ASCII 0x20 space, which wrongly flagged legitimate text that uses
    Unicode space variants (EN SPACE, NBSP, etc. -- common paste artefacts
    from SSMS/Word) as undecodable.
    """
    if not text:
        return True
    if "\x00" in text:
        return False
    bad = sum(
        1 for ch in text
        if unicodedata.category(ch) in _NON_TEXT_UNICODE_CATEGORIES and ch not in "\n\r\t"
    )
    return bad / len(text) <= 0.02


def _looks_like_utf16(data: bytes, encoding: str) -> bool:
    if encoding == "utf-16":
        return data.startswith((b"\xff\xfe", b"\xfe\xff"))
    if encoding == "utf-16le":
        return data.startswith(b"\xff\xfe")
    if encoding == "utf-16be":
        return data.startswith(b"\xfe\xff")
    return False
