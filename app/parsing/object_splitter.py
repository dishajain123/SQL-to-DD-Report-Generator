"""Splits an uploaded SQL file into individual object units.

Handles both cases discussed in the architecture: a single file containing
multiple CREATE PROCEDURE/FUNCTION/TRIGGER/VIEW statements, and a file that
already holds exactly one object.
"""
from __future__ import annotations

import re
import uuid

from app.models.core import Dialect, ObjectType, SQLObject

_OBJECT_NAME_RE = r'(?:"[^"]+"|\[[^\]]+\]|[A-Za-z0-9_]+)'
_QUALIFIED_OBJECT_NAME_RE = rf"{_OBJECT_NAME_RE}(?:\s*\.\s*{_OBJECT_NAME_RE})*"
_OBJECT_START_RE = re.compile(
    r"CREATE\s+(?:OR\s+(?:REPLACE|ALTER)\s+)?"
    r"(PROCEDURE|FUNCTION|TRIGGER|VIEW)\s+"
    rf"({_QUALIFIED_OBJECT_NAME_RE})",
    re.IGNORECASE,
)


def split_objects(sql_text: str, source_file: str, dialect: Dialect) -> list[SQLObject]:
    """Split raw SQL text into one SQLObject per CREATE ... statement found.

    If no CREATE OBJECT statement is found at all, the whole file is treated
    as a single unclassified object so the pipeline still has something to
    work with rather than silently dropping content or mislabeling it.
    """
    matches = list(_OBJECT_START_RE.finditer(sql_text))
    if not matches:
        return [
            SQLObject(
                object_id=str(uuid.uuid4()),
                name=_derive_name(source_file),
                object_type=ObjectType.UNKNOWN,
                dialect=dialect,
                raw_sql=sql_text.strip(),
                source_file=source_file,
            )
        ]

    objects: list[SQLObject] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(sql_text)
        body = sql_text[start:end].strip()

        obj_type_raw = match.group(1).upper()
        name_raw = _normalize_object_name(match.group(2))
        # Strip schema qualifier (PRO.DPD_Calculation -> DPD_Calculation) but
        # keep the full qualified name available in raw_sql for traceability.
        name = name_raw.split(".")[-1]

        objects.append(
            SQLObject(
                object_id=str(uuid.uuid4()),
                name=name,
                object_type=ObjectType(obj_type_raw),
                dialect=dialect,
                raw_sql=body,
                source_file=source_file,
            )
        )
    return objects


def _derive_name(source_file: str) -> str:
    base = source_file.rsplit("/", 1)[-1]
    base = re.sub(r"\.sql$", "", base, flags=re.IGNORECASE)
    return base


def _normalize_object_name(name: str) -> str:
    parts = []
    for part in re.split(r"\s*\.\s*", name.strip()):
        part = part.strip().strip('"')
        if part.startswith("[") and part.endswith("]"):
            part = part[1:-1]
        parts.append(part)
    return ".".join(parts)
