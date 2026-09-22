"""Post-generation DD helpers shared by orchestration and export.

Kept separate from the v2 AST pipeline so report/export filters and
duplicate-key flagging do not depend on the retired LLM derivation engine.
"""
from __future__ import annotations

from app.models.core import DDRow, DDStatus


def is_non_derivable_expression(expression: str) -> bool:
    """True when the formula is a bare NULL/0/1 with no derivation logic."""
    text = (expression or "").strip()
    if not text:
        return False
    return text.upper() in {"NULL", "0", "1"}


def should_omit_dd_row_from_presentation(expression: str) -> bool:
    """Omit rows that have nothing meaningful to show stakeholders."""
    text = (expression or "").strip()
    if not text:
        return True
    return is_non_derivable_expression(text)


# Backward-compatible private aliases used by older call sites.
_should_omit_dd_row_from_presentation = should_omit_dd_row_from_presentation
_is_non_derivable_expression = is_non_derivable_expression


def flag_duplicate_dd_rows(dd_rows: list[DDRow]) -> list[DDRow]:
    """Flag DD rows that share the same entity/column/effective-date key.

    Does not drop or rewrite formulas — only status, confidence, and
    validation_errors are updated so reviewers (or merge_dd_rows) can see
    the conflict instead of a silent last-write-wins overwrite.
    """
    key_to_rows: dict[tuple[str, str, object], list[DDRow]] = {}
    for row in dd_rows:
        key = (row.entity_name, row.column_name, row.effective_start_date)
        key_to_rows.setdefault(key, []).append(row)

    for rows in key_to_rows.values():
        if len(rows) < 2:
            continue
        distinct_chains = sorted({r.source_chain_id for r in rows})
        for row in rows:
            other_chains = [c for c in distinct_chains if c != row.source_chain_id] or distinct_chains
            row.status = DDStatus.PENDING_REVIEW
            row.confidence = min(row.confidence, 0.3)
            row.validation_errors.append(
                f'Another derivation for "{row.entity_name}"."{row.column_name}" '
                f"effective {row.effective_start_date} was generated from a "
                f"different source ({', '.join(other_chains)}). Multiple "
                "procedures/statements write this column for this "
                "effective date -- reconcile into a single formula (for "
                "example, guard each with its own row-scoping condition) "
                "before accepting any of them."
            )
    return dd_rows
