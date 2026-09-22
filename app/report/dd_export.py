"""Architecture step 18: DD CSV + Excel Export.

Writes DD rows using the exact column schema observed in the platform's
Derivations export. CSV remains the round-trippable merge artifact; Excel
(.xlsx) is the operator-facing deliverable matching the patched sample
export structure.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from openpyxl import Workbook, load_workbook

from app.models.core import DDRow, ReviewState
from app.utils import db
from app.utils.identity import canonical_expression_key, canonical_logical_name

COLUMNS = [
    "Entity Name",
    "Column Name",
    "Column Type",
    "Derivation Option",
    "Display Derivation Expression",
    "Effective Start Date",
    "Status",
    "Data Type",
    "Decision Table Json",
    "Conditional Json",
]

_COLUMN_KEYS = {
    "Entity Name": "entity_name",
    "Column Name": "column_name",
    "Column Type": "column_type",
    "Derivation Option": "derivation_option",
    "Display Derivation Expression": "display_derivation_expression",
    "Effective Start Date": "effective_start_date",
    "Status": "status",
    "Data Type": "data_type",
    "Decision Table Json": "decision_table_json",
    "Conditional Json": "conditional_json",
}


def dd_row_to_dict(dd: DDRow) -> dict:
    return {
        "entity_name": dd.entity_name,
        "column_name": dd.column_name,
        "column_type": dd.column_type.value,
        "derivation_option": dd.derivation_option.value,
        "display_derivation_expression": dd.display_derivation_expression,
        "effective_start_date": dd.effective_start_date.strftime("%d-%m-%Y"),
        "status": dd.status.value,
        "data_type": dd.data_type,
        "decision_table_json": dd.decision_table_json or "",
        "conditional_json": dd.conditional_json or "",
    }


def _row_dict_to_dd_row(row: dict) -> DDRow:
    """Rebuild a DDRow from a stored job row.

    Prefer the full `row_json` payload (data type, decision table, advisory
    notes, source refs, review_state, …) and overlay only the reviewed
    expression/status from the narrow dd_rows columns.
    """
    payload: dict = {}
    raw_json = row.get("row_json")
    if raw_json:
        try:
            payload = json.loads(raw_json) if isinstance(raw_json, str) else dict(raw_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = {}

    merged = dict(payload)
    # Narrow table columns win for fields that review can mutate.
    if row.get("expression") is not None and str(row.get("expression")).strip() != "":
        merged["display_derivation_expression"] = row.get("expression")
    if row.get("status"):
        merged["status"] = row.get("status")
    if row.get("confidence") is not None:
        merged["confidence"] = row.get("confidence")
    if row.get("entity_name"):
        merged["entity_name"] = row.get("entity_name")
    if row.get("column_name"):
        merged["column_name"] = row.get("column_name")
    if row.get("chain_id"):
        merged["source_chain_id"] = row.get("chain_id")

    effective_start = merged.get("effective_start_date") or row.get("effective_start_date")
    if isinstance(effective_start, date):
        effective_date = effective_start
    else:
        text = str(effective_start)
        try:
            effective_date = date.fromisoformat(text)
        except ValueError:
            # Reviewed exports sometimes store dd-mm-YYYY.
            try:
                day, month, year = text.split("-")
                effective_date = date(int(year), int(month), int(day))
            except Exception:
                effective_date = date.today()

    validation_errors = merged.get("validation_errors") or row.get("validation_errors") or []
    if isinstance(validation_errors, str):
        try:
            validation_errors = json.loads(validation_errors)
        except json.JSONDecodeError:
            validation_errors = [validation_errors]

    advisory_notes = merged.get("advisory_notes") or []
    if isinstance(advisory_notes, str):
        try:
            advisory_notes = json.loads(advisory_notes)
        except json.JSONDecodeError:
            advisory_notes = [advisory_notes]

    from app.models.core import ReviewState

    review_state_raw = merged.get("review_state") or ReviewState.GENERATED.value
    try:
        review_state = ReviewState(review_state_raw)
    except ValueError:
        review_state = ReviewState.GENERATED

    return DDRow(
        entity_name=str(merged.get("entity_name", "")),
        column_name=str(merged.get("column_name", "")),
        column_type=merged.get("column_type")
        if isinstance(merged.get("column_type"), str)
        else str(merged.get("column_type", "Physical")),
        derivation_option=merged.get("derivation_option")
        if isinstance(merged.get("derivation_option"), str)
        else str(merged.get("derivation_option", "Formula Expression")),
        display_derivation_expression=str(
            merged.get("display_derivation_expression") or merged.get("expression") or ""
        ),
        effective_start_date=effective_date,
        status=merged.get("status")
        if isinstance(merged.get("status"), str)
        else str(merged.get("status", "PENDING_REVIEW")),
        review_state=review_state,
        data_type=str(merged.get("data_type") or ""),
        decision_table_json=merged.get("decision_table_json") or None,
        conditional_json=merged.get("conditional_json") or None,
        business_meaning=str(merged.get("business_meaning") or ""),
        source_chain_id=str(merged.get("source_chain_id") or merged.get("chain_id") or ""),
        source_object_ids=list(merged.get("source_object_ids") or []),
        source_statement_refs=list(merged.get("source_statement_refs") or []),
        source_statement_sql=list(merged.get("source_statement_sql") or []),
        confidence=float(merged.get("confidence") or 0.0),
        validation_errors=list(validation_errors),
        advisory_notes=list(advisory_notes),
    )


def _row_key(row: dict) -> tuple:
    return (
        canonical_logical_name(str(row["entity_name"])),
        canonical_logical_name(str(row["column_name"])),
        row["effective_start_date"],
    )


def _row_signature(row: dict) -> tuple:
    return (
        canonical_logical_name(str(row["entity_name"])),
        canonical_logical_name(str(row["column_name"])),
        canonical_expression_key(str(row["display_derivation_expression"])),
        row["column_type"],
        row["derivation_option"],
        row["status"],
        row["data_type"],
        row["decision_table_json"],
        row["conditional_json"],
    )


def _dedupe_equivalent_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for row in rows:
        signature = _row_signature(row)
        existing = grouped.get(signature)
        if existing is None:
            grouped[signature] = dict(row)
            continue
        if row["effective_start_date"] < existing["effective_start_date"]:
            existing["effective_start_date"] = row["effective_start_date"]
    return list(grouped.values())


def _read_existing_dd_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            rows.append(
                {_COLUMN_KEYS[h]: (row_dict.get(h) or "") for h in COLUMNS if h in _COLUMN_KEYS}
            )
    return rows


def _read_existing_dd_xlsx(path: Path) -> list[dict]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    try:
        header = [str(c).strip() if c is not None else "" for c in next(rows_iter)]
    except StopIteration:
        wb.close()
        return []

    rows: list[dict] = []
    for raw in rows_iter:
        row_dict = {header[i]: (raw[i] if i < len(raw) and raw[i] is not None else "") for i in range(len(header))}
        rows.append({_COLUMN_KEYS[h]: str(row_dict.get(h) or "") for h in COLUMNS if h in _COLUMN_KEYS})
    wb.close()
    return rows


def read_existing_dd_csv(path: str | Path) -> list[dict]:
    """Read a previously exported DD CSV back into plain dict rows."""
    path = Path(path)
    if not path.exists():
        return []

    return _read_existing_dd_csv(path)


def read_existing_dd_excel(path: str | Path) -> list[dict]:
    """Read a prior CSV or XLSX Derivations export."""
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        return _read_existing_dd_xlsx(path)
    return _read_existing_dd_csv(path)


def merge_dd_rows(existing: list[dict], new_rows: list[DDRow]) -> list[dict]:
    """New rows always win over an existing row with the same
    (entity, column, effective_start_date) key. Existing rows with no
    matching new row are preserved unchanged."""
    new_dicts = [dd_row_to_dict(r) for r in new_rows]
    new_keys = {_row_key(r) for r in new_dicts}
    preserved = [r for r in existing if _row_key(r) not in new_keys]
    return _dedupe_equivalent_rows(preserved + new_dicts)


def _write_dd_csv(merged: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in merged:
            writer.writerow({header: row.get(_COLUMN_KEYS[header], "") for header in COLUMNS})
    return output_path


def _write_dd_xlsx(merged: list[dict], output_path: Path) -> Path:
    """Write Derivations rows to Excel matching the sample export schema.

    Formatting mirrors the platform Derivations CSV/XLSX shape: frozen
    header row, bold headers, sensible column widths, and text-wrapped
    expression cells so operators can review formulas in-grid.
    """
    from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Derivations"
    ws.append(COLUMNS)

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    for col_idx in range(1, len(COLUMNS) + 1):
        cell = ws.cell(1, col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = thin

    body_align = Alignment(vertical="top", wrap_text=True)
    for row in merged:
        values = [row.get(_COLUMN_KEYS[header], "") for header in COLUMNS]
        ws.append(values)
        for col_idx in range(1, len(COLUMNS) + 1):
            cell = ws.cell(ws.max_row, col_idx)
            cell.alignment = body_align
            cell.border = thin

    widths = {
        "Entity Name": 22,
        "Column Name": 28,
        "Column Type": 14,
        "Derivation Option": 20,
        "Display Derivation Expression": 80,
        "Effective Start Date": 18,
        "Status": 14,
        "Data Type": 12,
        "Decision Table Json": 24,
        "Conditional Json": 20,
    }
    for idx, header in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = widths.get(header, 16)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(output_path)
    return output_path


def export_dd_rows(
    dd_rows: list[DDRow], output_path: str | Path, existing_dd_path: str | Path | None = None
) -> Path:
    from app.derivation.dd_postprocess import should_omit_dd_row_from_presentation

    presentable = [
        row
        for row in dd_rows
        if row.review_state in {ReviewState.GENERATED, ReviewState.APPROVED}
        and not row.validation_errors
        and not should_omit_dd_row_from_presentation(row.display_derivation_expression or "")
    ]
    if existing_dd_path is not None:
        existing = read_existing_dd_excel(existing_dd_path)
        # A newly analysed source write can invalidate a previously exported
        # formula. Do not silently resurrect that old formula simply because
        # the replacement row was withheld by coverage validation.
        presentable_ids = {id(row) for row in presentable}
        withheld_keys = {
            _row_key(dd_row_to_dict(row))
            for row in dd_rows
            if id(row) not in presentable_ids
        }
        existing = [row for row in existing if _row_key(row) not in withheld_keys]
        merged = merge_dd_rows(existing, presentable)
    else:
        merged = _dedupe_equivalent_rows([dd_row_to_dict(r) for r in presentable])

    output_path = Path(output_path)
    if output_path.suffix.lower() in {".xlsx", ".xlsm"}:
        return _write_dd_xlsx(merged, output_path)
    return _write_dd_csv(merged, output_path)


def export_dd_rows_csv(
    dd_rows: list[DDRow], output_path: str | Path, existing_dd_path: str | Path | None = None
) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".csv", ""}:
        output_path = output_path.with_suffix(".csv")
    return export_dd_rows(dd_rows, output_path, existing_dd_path=existing_dd_path)


def export_dd_rows_excel(
    dd_rows: list[DDRow], output_path: str | Path, existing_dd_path: str | Path | None = None
) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".xlsx", ".xlsm"}:
        output_path = output_path.with_suffix(".xlsx")
    return export_dd_rows(dd_rows, output_path, existing_dd_path=existing_dd_path)


def export_reviewed_dd_rows_for_job(
    job_id: str, output_path: str | Path, db_path: str | None = None
) -> Path:
    rows = db.get_dd_rows_for_job(job_id, db_path=db_path)
    dd_rows = [_row_dict_to_dd_row(dict(row)) for row in rows]
    return export_dd_rows(dd_rows, output_path)


def export_reviewed_dd_rows_for_job_csv(
    job_id: str, output_path: str | Path, db_path: str | None = None
) -> Path:
    rows = db.get_dd_rows_for_job(job_id, db_path=db_path)
    dd_rows = [_row_dict_to_dd_row(dict(row)) for row in rows]
    return export_dd_rows_csv(dd_rows, output_path)


def export_reviewed_dd_rows_for_job_excel(
    job_id: str, output_path: str | Path, db_path: str | None = None
) -> Path:
    rows = db.get_dd_rows_for_job(job_id, db_path=db_path)
    dd_rows = [_row_dict_to_dd_row(dict(row)) for row in rows]
    return export_dd_rows_excel(dd_rows, output_path)


def write_qa_coverage_report(
    dd_rows: list[DDRow],
    output_path: str | Path,
    *,
    coverage_markdown: str = "",
    job_id: str = "",
    blockers: list[str] | None = None,
) -> Path:
    """Companion QA report: never distribute platform CSV without this while blockers remain."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blockers = list(blockers or [])
    ready = not blockers and all(
        getattr(r, "review_state", None) and r.review_state.value == "APPROVED" for r in dd_rows
    ) if dd_rows else False

    lines = [
        f"# QA / coverage report{f' — {job_id}' if job_id else ''}",
        "",
        f"Ready to present: **{'yes' if ready else 'no'}**",
        f"Rows: {len(dd_rows)}",
        "",
        "| Entity | Column | Review state | Platform status | Confidence | Expression | Validation | Advisory | Source refs |",
        "|--------|--------|--------------|-----------------|------------|------------|------------|----------|-------------|",
    ]
    for row in dd_rows:
        expr = (row.display_derivation_expression or "").replace("|", "\\|").replace("\n", " ")
        if len(expr) > 120:
            expr = expr[:117] + "..."
        lines.append(
            f"| {row.entity_name} | {row.column_name} | "
            f"{getattr(row.review_state, 'value', row.review_state)} | {row.status.value} | "
            f"{row.confidence:.3f} | `{expr}` | "
            f"{'; '.join(row.validation_errors) or '—'} | "
            f"{'; '.join(row.advisory_notes) or '—'} | "
            f"{'; '.join(row.source_statement_refs) or '—'} |"
        )
    if blockers:
        lines.extend(["", "## Blockers", ""])
        for b in blockers:
            lines.append(f"- {b}")
    if coverage_markdown:
        lines.extend(["", "## Write coverage ledger", "", coverage_markdown])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path

    dd_rows = [_row_dict_to_dd_row(dict(row)) for row in rows]
    return export_dd_rows_excel(dd_rows, output_path)
