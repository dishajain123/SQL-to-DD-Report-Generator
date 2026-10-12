"""Ensure full-procedure DD generation yields exportable business rows (not Excel-filtered away)."""
from __future__ import annotations

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

_REPO = Path(__file__).resolve().parents[3]


def _rows_for_file(filename: str):
    path = _REPO / "samples/sql/PRO_SPs_Sequenced" / filename
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
        technical_summary="export coverage",
        business_summary="",
    )
    return generate_dd_rows_for_chains(
        chains, [model], objects, infos, None, entity_name_map=mapping
    )


def _exportable(rows) -> list:
    return [r for r in rows if is_exportable_row(r, should_omit_dd_row_from_presentation)]


def test_s06_generates_exportable_core_npa_date_columns():
    rows = _rows_for_file("11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql")
    exportable = _exportable(rows)
    keys = {(r.entity_name or "").upper().replace("##", ""), (r.column_name or "").upper()) for r in exportable}
    for entity, column in (
        ("ACCOUNTCAL", "FINALNPADT"),
        ("ACCOUNTCAL", "INITIALNPADT"),
        ("CUSTOMERCAL", "SYSNPA_DT"),
        ("CUSTOMERCAL", "DEGREASON"),
        ("RESTR_NPA", "UCIFENTITYID"),
        ("ACLRUNNINGPROCESSSTATUS", "COMPLETED"),
    ):
        assert (entity, column) in keys, f"missing exportable {entity}.{column}; have {sorted(keys)[:20]}…"


def test_s01_generates_exportable_acl_completed():
    rows = _rows_for_file("06_S01_PRO.Reference_Period_Calculation.StoredProcedure.sql")
    exportable = _exportable(rows)
    keys = {(r.entity_name or "").upper(), (r.column_name or "").upper()) for r in exportable}
    assert ("ACLRUNNINGPROCESSSTATUS", "COMPLETED") in keys


def test_s02_generates_exportable_dpd_and_acl_columns():
    rows = _rows_for_file("07_S02_PRO.DPD_Calculation.StoredProcedure.sql")
    exportable = _exportable(rows)
    keys = {(r.entity_name or "").upper().replace("##", ""), (r.column_name or "").upper()) for r in exportable}
    for entity, column in (
        ("ACCOUNTCAL", "DPD_INTSERVICE"),
        ("ACCOUNTCAL", "DPD_OVERDUE"),
        ("ADVACRESTRUCTURECAL", "DPD_MAXFIN"),
        ("ACLRUNNINGPROCESSSTATUS", "COMPLETED"),
    ):
        assert (entity, column) in keys, f"missing exportable {entity}.{column}"
