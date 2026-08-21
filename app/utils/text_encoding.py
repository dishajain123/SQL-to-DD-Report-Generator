"""Helpers for decoding uploaded text files safely.

The DD review flow accepts SQL uploads from desktop tools that sometimes
save plain text in encodings other than UTF-8. We try a small, ordered
set of common text encodings and only fail if none of them produce
readable text.
"""
from __future__ import annotations

from dataclasses import dataclass


_COMMON_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


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
            return DecodedText(text=text, encoding=encoding)

    for encoding in ("utf-16", "utf-16le", "utf-16be"):
        if not _looks_like_utf16(data, encoding):
            continue
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        if _looks_like_text(text):
            return DecodedText(text=text, encoding=encoding)

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


def _looks_like_text(text: str) -> bool:
    if not text:
        return True
    if "\x00" in text:
        return False
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return printable / max(len(text), 1) >= 0.85


def _looks_like_utf16(data: bytes, encoding: str) -> bool:
    if encoding == "utf-16":
        return data.startswith((b"\xff\xfe", b"\xfe\xff"))
    if encoding == "utf-16le":
        return data.startswith(b"\xff\xfe")
    if encoding == "utf-16be":
        return data.startswith(b"\xfe\xff")
    return False
