"""Offset-preserving SQL lexical views shared by parsing and derivation.

Quoted strings/identifiers and nested block comments are opaque to scanners.
Every whitespace character is preserved so splitlines() remains aligned even
with CR-only or Unicode line separators inside comments and literals.
"""
from __future__ import annotations

import re
from functools import lru_cache


def protected_spans(text: str):
    """Yield (start, end, kind) for comments and quoted tokens."""
    i, n = 0, len(text)
    while i < n:
        start = i
        pair = text[i:i + 2]
        if pair == '--':
            i += 2
            while i < n and text[i] not in '\r\n':
                i += 1
            yield start, i, 'comment'
        elif pair == '/*':
            depth = 1
            i += 2
            while i < n and depth:
                pair = text[i:i + 2]
                if pair == '/*':
                    depth += 1
                    i += 2
                elif pair == '*/':
                    depth -= 1
                    i += 2
                else:
                    i += 1
            yield start, i, 'comment'
        elif text[i] in "'\"[":
            closing = ']' if text[i] == '[' else text[i]
            i += 1
            while i < n:
                if text[i] == closing:
                    i += 1
                    if i < n and text[i] == closing:
                        i += 1
                        continue
                    break
                i += 1
            yield start, i, 'quoted'
        else:
            i += 1


@lru_cache(maxsize=32)
def mask_sql(text: str, *, quotes: bool = True) -> str:
    out, previous = [], 0
    for start, end, kind in protected_spans(text):
        if kind == 'comment' or quotes:
            out.append(text[previous:start])
            out.append(''.join(ch if ch.isspace() else ' ' for ch in text[start:end]))
            previous = end
    out.append(text[previous:])
    return ''.join(out)


_SPACED_COMPARISON = re.compile(r'([<>!])([ \t]+)(=)|(<)([ \t]+)(>)')


# ``A. SRCASSETCLASSALT_KEY`` -- SQL Server accepts blanks around the dot of a
# qualified name; every downstream alias/column scanner expects ``A.COL``.
_SPACED_QUALIFIER = re.compile(r'((?<![\w.])[A-Za-z_#][\w#]*|\])\.([ \t]+)([A-Za-z_][\w#]*)')


def normalize_comparison_spacing(text: str) -> tuple[str, list[str]]:
    """Recover split comparison tokens and spaced qualifier dots in code only,
    recording every repair.

    Keep offsets stable by moving the intervening spaces after the operator
    (or after the column name for ``A. COL``).
    Never merge across comments/newlines or change quoted values/identifiers.
    This is an explicit parser recovery, not a claim about server acceptance.
    """
    notes: list[str] = []
    if _SPACED_COMPARISON.search(text):
        masked = mask_sql(text)
        out, previous = [], 0
        for match in _SPACED_COMPARISON.finditer(text):
            if masked[match.start():match.end()] != match.group(0):
                continue
            first, gap, last = match.group(1, 2, 3) if match.group(1) else match.group(4, 5, 6)
            out.extend((text[previous:match.start()], first + last + gap))
            notes.append(f"Comparison spacing normalized at offset {match.start()}: {match.group(0)!r} -> {first + last!r}")
            previous = match.end()
        out.append(text[previous:])
        text = ''.join(out)
    if _SPACED_QUALIFIER.search(text):
        masked = mask_sql(text)
        out, previous = [], 0
        for match in _SPACED_QUALIFIER.finditer(text):
            if masked[match.start():match.end()] != match.group(0):
                continue
            qualifier, gap, column = match.group(1, 2, 3)
            out.extend((text[previous:match.start()], f"{qualifier}.{column}{gap}"))
            notes.append(f"Qualifier spacing normalized at offset {match.start()}: {match.group(0)!r} -> {qualifier + '.' + column!r}")
            previous = match.end()
        out.append(text[previous:])
        text = ''.join(out)
    return text, notes
