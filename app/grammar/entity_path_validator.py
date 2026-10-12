"""Reject malformed entity paths in compiled 4X formulas."""
from __future__ import annotations

import re

from app.derivation.v2.sql_text import bare_ident, is_ephemeral_sql_alias

# Permanent entities must never appear with a session ``#`` prefix (only ``##`` globals / #temps).
PHYSICAL_DERIVATION_ENTITY_NAMES = frozenset(
    {
        "ACCOUNTCAL",
        "CUSTOMERCAL",
        "COBORROWERCAL",
        "ADVACRESTRUCTURECAL",
        "RESTR_NPA",
        "DIMPRODUCT",
        "DIMPARAMETER",
        "DIMSOURCE",
        "DIMSOURCEDB",
        "DIMACCOUNTTYPE",
        "DIMPRODUCTTYPE",
        "ADVACBASICDETAIL",
        "ACLRUNNINGPROCESSSTATUS",
    }
)

_PHYSICAL_ENTITY_NAMES = PHYSICAL_DERIVATION_ENTITY_NAMES


def is_physical_derivation_entity(name: str) -> bool:
    """True for global interface / dimension entities that must never use session ``#``."""
    bare = bare_ident(name or "").upper().lstrip("#")
    return bare in _PHYSICAL_ENTITY_NAMES


_PATH_REF_RE = re.compile(
    r'"(?P<p1>#?[^"]+)"(?:\.(?:"(?P<p2>#?[^"]+)"(?:\.(?:"(?P<p3>#?[^"]+)")?)?))?',
)


def validate_entity_paths_in_expression(expression: str) -> list[str]:
    """Return human-readable errors for invalid ``"Ent"."Rel"."Col"`` paths."""
    if not expression:
        return []
    errors: list[str] = []
    seen: set[str] = set()
    for match in _PATH_REF_RE.finditer(expression):
        p1 = bare_ident(match.group("p1") or "")
        p2 = bare_ident(match.group("p2") or "") if match.group("p2") else ""
        p3 = bare_ident(match.group("p3") or "") if match.group("p3") else ""
        key = match.group(0)
        if key in seen:
            continue
        seen.add(key)
        if p1.startswith("#") and not p1.startswith("##"):
            base = p1.lstrip("#").upper()
            if base in _PHYSICAL_ENTITY_NAMES:
                errors.append(
                    f'invalid session-temp prefix on physical entity: "{p1}" in {key}'
                )
        # Only flag statement aliases in the relationship hop (``"Ent"."C"."Col"``),
        # not column names on two-segment refs (``"AccountCal"."FinalNpaDt"``).
        if p2 and p3 and is_ephemeral_sql_alias(p2):
            errors.append(f"SQL subquery alias must not appear in entity path: {key}")
    return errors
