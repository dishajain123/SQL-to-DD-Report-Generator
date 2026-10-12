#!/usr/bin/env python3
"""Print Entity/Column/Formula rows for S01, S02, S06 sequenced procedures."""
from __future__ import annotations

import json
from pathlib import Path

from app.derivation.dd_postprocess import should_omit_dd_row_from_presentation
from app.derivation.v2.pipeline import generate_dd_rows_for_chains
from app.models.core import CanonicalModel, LineageChain
from app.parsing.dialect import detect_dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object
from app.parsing.write_inventory_scan import read_sql_file
from app.report.dd_export import is_exportable_row
from app.utils.entity_name_map import build_entity_name_map_for_tables

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "06_S01_PRO.Reference_Period_Calculation.StoredProcedure.sql",
    "07_S02_PRO.DPD_Calculation.StoredProcedure.sql",
    "11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql",
]


def _generate(name: str):
    path = ROOT / "samples/sql/PRO_SPs_Sequenced" / name
    sql = read_sql_file(path)
    object_list = split_objects(sql, path.name, detect_dialect(sql))
    for index, obj in enumerate(object_list):
        obj.object_id = f"{path.stem}:{index}"
    objects = {o.object_id: o for o in object_list}
    infos = {oid: analyze_object(obj) for oid, obj in objects.items()}
    mapping = build_entity_name_map_for_tables(
        [t for info in infos.values() for t in info.tables_read + info.tables_written]
    )
    chains = [LineageChain(chain_id=path.stem, object_ids=list(objects), order=list(objects))]
    model = CanonicalModel(
        chain_id=path.stem,
        job_id=path.stem,
        object_ids=list(objects),
        technical_summary="export",
        business_summary="",
    )
    return generate_dd_rows_for_chains(
        chains, [model], objects, infos, None, entity_name_map=mapping
    )


def main() -> None:
    report = {}
    for name in FILES:
        dd_rows = _generate(name)
        serial = []
        for row in sorted(dd_rows, key=lambda r: (r.entity_name or "", r.column_name or "")):
            serial.append(
                {
                    "entity": row.entity_name,
                    "column": row.column_name,
                    "formula": (row.display_derivation_expression or "").strip(),
                    "errors": row.validation_errors,
                    "data_type": row.data_type,
                    "exportable": is_exportable_row(row, should_omit_dd_row_from_presentation),
                }
            )
        report[name] = serial

    out_path = ROOT / "output" / "export_three_procs_derivations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, rows in report.items():
        exportable = sum(1 for r in rows if r.get("exportable"))
        print(f"\n=== {name} ({len(rows)} generated, {exportable} exportable to Excel) ===")
        for r in rows:
            f = r["formula"]
            preview = f if len(f) <= 120 else f[:117] + "…"
            flags = []
            if r["errors"]:
                flags.append("ERR")
            if not r.get("exportable"):
                flags.append("SKIP_EXPORT")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            print(f"  {r['entity']}.{r['column']}{suffix}: {preview or '(empty)'}")


if __name__ == "__main__":
    main()
