"""Derivation Option classification and display formatting for DD export.

Internal AST compilation may emit quoted execution parameters (``"@TIMEKEY"``)
so formulas pass the 4X grammar validator; presentation export strips those
quotes for operators and regulatory reviewers.
"""
from __future__ import annotations

import re
from typing import Any

from app.models.core import DerivationOption

_SYSTEM_SEEDS = frozenset(
    {
        "N",
        "Y",
        "0",
        "1",
        "0.00",
        "NULL",
        '""',
        '"N"',
        '"Y"',
    }
)

_NUMERIC_LITERAL_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_QUOTED_PARAM_RE = re.compile(r'"(@[A-Za-z0-9_]+)"')


def format_expression_syntax(expression: str) -> str:
    """Strip presentation quotes from ``@`` execution parameters."""
    if not expression:
        return expression
    return _QUOTED_PARAM_RE.sub(r"\1", expression)


def classify_derivation_option(
    expression: str,
    *,
    ast: dict[str, Any] | None = None,
) -> DerivationOption:
    """Every exported derivation is a Formula Expression.

    System seeds and direct assignments are no longer separate options:
    seeds are filtered out (see ``is_static_seed``) and pass-throughs are
    expressed as a formula whose body is the mapping expression.
    """
    return DerivationOption.FORMULA_EXPRESSION


def is_static_seed(expression: str, *, ast: dict[str, Any] | None = None) -> bool:
    """True for constant initialisations ("N", 0, NULL, ...) with no logic."""
    if isinstance(ast, dict) and ast.get("type") == "LITERAL":
        return True
    text = format_expression_syntax((expression or "").strip())
    if not text:
        return False
    bare = text.strip('"').strip("'")
    return bare in _SYSTEM_SEEDS or text in _SYSTEM_SEEDS or bool(_NUMERIC_LITERAL_RE.match(text))
