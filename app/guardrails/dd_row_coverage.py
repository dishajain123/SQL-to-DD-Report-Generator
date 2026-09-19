"""Surface write-coverage caveats without wiping valid platform formulas.

Reviewed gold output for samples 01–17 exports grammar-valid composed
expressions as ACTIVE. Set-based MERGEs, procedure branches, temp staging,
prior-value self-references, and same-procedure column dependencies may still
need a platform workflow — those caveats belong in advisory notes and the QA
ledger, not as blanked UNSUPPORTED rows.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from app.models.core import DDRow, DDStatus, ReviewState, StructuralInfo
from app.parsing.coverage_ledger import CoverageLedger, WriteKind, build_coverage_ledger

_COLUMN_REF = re.compile(r'"([^"]+)"\s*\.\s*"([^"]+)"')


def _name(value: str) -> str:
    return (value or "").split(".")[-1].strip().strip('"').strip("[]").upper()


def mark_ledger_coverage(
    ledger: CoverageLedger,
    rows: Iterable[DDRow],
    entity_name_map: dict[str, str] | None = None,
) -> CoverageLedger:
    """Mark a write covered only when every assigned column has a valid DD row."""
    mapping = {_name(k): _name(v) for k, v in (entity_name_map or {}).items()}
    valid_rows = [
        row for row in rows
        if ledger.object_id in row.source_object_ids
        and row.display_derivation_expression
        and not row.validation_errors
        and row.review_state in {ReviewState.GENERATED, ReviewState.APPROVED}
    ]
    for entry in ledger.entries:
        if entry.kind not in {WriteKind.ROW_FORMULA, WriteKind.DECISION_TABLE}:
            continue
        target = _name(entry.target_table)
        entity = mapping.get(target, target)
        entry.covered_by_dd = bool(entry.columns) and all(
            any(
                _name(row.entity_name) == entity
                and _name(row.column_name) == _name(column)
                for row in valid_rows
            )
            for column in entry.columns
        )
    ledger.dd_coverage_checked = True
    return ledger


def flag_rows_with_uncovered_writes(
    rows: Iterable[DDRow],
    info: StructuralInfo,
    source_sql: str,
    entity_name_map: dict[str, str] | None = None,
) -> None:
    """Attach coverage / dependency advisories; keep valid formulas exportable.

    Never demotes ACTIVE/GENERATED rows or blanks their expressions solely
    because the source write is MERGE / procedure-branch / temp staging, the
    formula preserves a prior column value, or it references another column
    written in the same procedure. The companion QA ledger still lists those
    operations for manual workflow review.
    """
    mapping = {_name(k): _name(v) for k, v in (entity_name_map or {}).items()}
    ledger = build_coverage_ledger(info, source_sql=source_sql)
    rows = list(rows)

    def _advise_advisory(row: DDRow, reasons: list[str]) -> None:
        row.advisory_notes = list(dict.fromkeys([*(row.advisory_notes or []), *reasons]))

    def _withhold_empty(row: DDRow, reasons: list[str]) -> None:
        row.validation_errors = list(dict.fromkeys([*row.validation_errors, *reasons]))
        row.review_state = ReviewState.UNSUPPORTED
        row.status = DDStatus.PENDING_REVIEW
        row.display_derivation_expression = ""
        row.decision_table_json = None
        row.conditional_json = None

    written = {
        (mapping.get(_name(entry.target_table), _name(entry.target_table)), _name(column))
        for entry in ledger.entries
        for column in entry.columns
    }

    for row in rows:
        if row.source_object_ids and info.object_id not in row.source_object_ids:
            continue
        candidates = {
            _name(entry.target_table)
            for entry in ledger.entries
            if any(_name(col) == _name(row.column_name) for col in entry.columns)
        }
        matching_tables = {
            _name(entry.target_table)
            for entry in ledger.entries
            if any(_name(col) == _name(row.column_name) for col in entry.columns)
            and (
                _name(entry.target_table) == _name(row.entity_name)
                or mapping.get(_name(entry.target_table)) == _name(row.entity_name)
            )
        }
        if not matching_tables and len(candidates) == 1:
            matching_tables = candidates

        reasons: list[str] = []
        if not matching_tables and row.source_object_ids:
            reasons.append(
                "Generated row cannot be linked unambiguously to a source write target"
            )
        for entry in ledger.entries:
            if _name(entry.target_table) not in matching_tables:
                continue
            if not any(_name(col) == _name(row.column_name) for col in entry.columns):
                continue
            if entry.kind not in {WriteKind.ROW_FORMULA, WriteKind.DECISION_TABLE}:
                reasons.append(
                    f"stmt #{entry.statement_index} {entry.statement_type} to "
                    f"{entry.target_table} requires {entry.kind.value} coverage"
                )

        expression = row.display_derivation_expression or ""
        own_reference = re.compile(
            rf'(?i)"{re.escape(row.entity_name)}"\s*\.\s*"{re.escape(row.column_name)}"'
        )
        if own_reference.search(expression):
            reasons.append(
                "Formula reads its own target column; prior-row state or write order "
                "may require an explicit platform workflow"
            )

        for entity, column in _COLUMN_REF.findall(expression):
            ref = (_name(entity), _name(column))
            if ref not in written:
                continue
            if ref == (_name(row.entity_name), _name(row.column_name)):
                continue
            reasons.append(
                f"Formula depends on {entity}.{column}, which this procedure "
                "also writes; spot-check ordered workflow semantics"
            )

        if not reasons:
            continue

        has_exportable_formula = bool(expression.strip()) and row.review_state in {
            ReviewState.GENERATED,
            ReviewState.APPROVED,
            ReviewState.NEEDS_REVIEW,
        }
        if has_exportable_formula and row.status in {DDStatus.ACTIVE, DDStatus.PENDING_REVIEW}:
            _advise_advisory(row, reasons)
            continue

        _withhold_empty(row, reasons)
