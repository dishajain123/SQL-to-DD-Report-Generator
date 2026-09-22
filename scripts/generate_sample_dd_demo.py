#!/usr/bin/env python3
"""Generate DD CSV + Excel + report for samples 01–18 via Derivation Engine v2.

Uses the same v2 AST path as the live API/Streamlit pipeline.

    python scripts/generate_sample_dd_demo.py
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from app.derivation.v2.pipeline import generate_for_sql
from app.derivation.v2.sql_text import (
    bare_ident,
    extract_insert_select,
    extract_merge_matched_updates,
    extract_update_statements,
    normalize_table_name,
)
from app.derivation.v2.phase2_mutation_folder import _iter_set_assignments
from app.grammar.validator import validate_expression
from app.models.core import DDStatus, ReviewState
from app.parsing.write_inventory_scan import read_sql_file
from app.report.dd_export import export_dd_rows_csv, export_dd_rows_excel
from app.utils.config import settings
from app.utils.text_encoding import decode_text_bytes

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples" / "sql"
OUT = ROOT / "output" / "demo_01_to_18_v2"


def _collect_pairs(sql: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for stmt in extract_update_statements(sql):
        head = (stmt.get("head") or "").strip()
        m = re.match(r"(?is)^(\[?#?#?[A-Za-z0-9_\.]+\]?)", head)
        table = normalize_table_name(m.group(1)) if m else ""
        if not table or len(table) <= 2:
            continue
        for assign in _iter_set_assignments(stmt.get("set_clause") or ""):
            pairs.add((table, assign["column"]))
    for ins in extract_insert_select(sql, temps_only=False):
        table = normalize_table_name(ins.get("target") or "")
        for col in (ins.get("cols") or "").split(","):
            name = bare_ident(col)
            if table and name:
                pairs.add((table, name))
    for merge in extract_merge_matched_updates(sql):
        table = normalize_table_name(merge.get("target") or "")
        for assign in _iter_set_assignments(merge.get("set_clause") or ""):
            if table:
                pairs.add((table, assign["column"]))
    return pairs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = sorted(
        p for p in SAMPLES.glob("*.sql") if p.name[:2].isdigit() and 1 <= int(p.name[:2]) <= 18
    )
    entity_map = dict(getattr(settings, "entity_name_map", {}) or {})

    for path in samples:
        sql = read_sql_file(path) if hasattr(path, "read_bytes") else decode_text_bytes(path.read_bytes()).text
        try:
            sql = read_sql_file(path)
        except Exception:
            sql = decode_text_bytes(path.read_bytes()).text

        rows = []
        for entity, column in sorted(_collect_pairs(sql)):
            row, debug = generate_for_sql(
                sql,
                entity,
                column,
                entity_map=entity_map or None,
                source_chain_id=path.stem,
                source_object_ids=[path.name],
            )
            formula = debug.get("formula") or ""
            if formula and validate_expression(formula).valid:
                row.status = DDStatus.ACTIVE
                row.review_state = ReviewState.GENERATED
            rows.append(row)

        sample_out = OUT / path.stem
        sample_out.mkdir(parents=True, exist_ok=True)
        export_dd_rows_csv(rows, sample_out / "dd_export.csv")
        export_dd_rows_excel(rows, sample_out / "dd_export.xlsx")
        print(f"{path.name}: {len(rows)} row(s) -> {sample_out}")


if __name__ == "__main__":
    main()
