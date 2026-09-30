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


_BOMS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16"),  # UTF-16 LE (SSMS "Unicode")
    (b"\xfe\xff", "utf-16"),  # UTF-16 BE
)


def _bomless_utf16_encoding(data: bytes) -> str | None:
    """Detect UTF-16 without a BOM from its NUL-byte pattern.

    ASCII-range SQL saved as UTF-16 LE has a NUL in (almost) every odd byte;
    BE has them in even bytes. Real single-byte text has essentially none.
    """
    sample = data[:4096]
    if len(sample) < 4:
        return None
    even_nuls = sample[0::2].count(0)
    odd_nuls = sample[1::2].count(0)
    half = len(sample) / 2
    if odd_nuls >= 0.4 * half and even_nuls <= 0.05 * half:
        return "utf-16-le"
    if even_nuls >= 0.4 * half and odd_nuls <= 0.05 * half:
        return "utf-16-be"
    return None


def decode_text_bytes(data: bytes) -> DecodedText:
    """Decode bytes into text using common SQL-friendly encodings.

    A byte-order mark decides the encoding outright. Without one, UTF-16 is
    recognised by its NUL-byte pattern before any single-byte codec is tried:
    cp1252/latin-1 accept every byte, so trying them first would turn a
    UTF-16 file into NUL-interleaved text instead of failing.
    """
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            try:
                text = data.decode(encoding)
            except UnicodeDecodeError:
                break  # a truncated / corrupt file: fall back to the heuristics below
            return DecodedText(text=_normalize_exotic_whitespace(text), encoding=encoding)
    bomless = _bomless_utf16_encoding(data)
    if bomless:
        try:
            text = data.decode(bomless)
        except UnicodeDecodeError:
            pass
        else:
            if _looks_like_text(text):
                return DecodedText(text=_normalize_exotic_whitespace(text), encoding=bomless)

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


def read_sql_file(path) -> str:
    """Read a SQL file from disk in whatever encoding it was saved in."""
    from pathlib import Path

    return decode_text_bytes(Path(path).read_bytes()).text


def normalize_sql_text(text: str) -> str:
    """Repair SQL text that was already decoded with the wrong codec.

    Text arriving through the API is a ``str`` decoded by the client. A client
    that read a UTF-16 file as latin-1/cp1252 delivers ``"U\\x00P\\x00…"``
    (with ``"ÿþ"`` in front); that is re-encoded and decoded as UTF-16. A
    stray BOM character left by a UTF-8-sig or UTF-16 decode is dropped.
    """
    if not text:
        return text
    if "\x00" in text:
        for codec in ("latin-1", "cp1252"):
            try:
                raw = text.encode(codec)
            except UnicodeEncodeError:
                continue
            try:
                repaired = decode_text_bytes(raw).text
            except UnicodeDecodeError:
                continue
            if "\x00" not in repaired:
                text = repaired
                break
        text = text.replace("\x00", "")
    return text.lstrip("﻿")


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
