"""S15 UpdateProvisionKey — ProvisionAlt_Key should compile when mutations fold."""
from __future__ import annotations

from pathlib import Path

from app.derivation.v2.pipeline import generate_dd_rows_for_chains
from app.grammar.validator import validate_expression
from app.models.core import CanonicalModel, LineageChain
from app.parsing.dialect import detect_dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object
from app.parsing.write_inventory_scan import read_sql_file
from app.utils.entity_name_map import build_entity_name_map_for_tables

_REPO = Path(__file__).resolve().parents[3]
S15 = _REPO / "samples/sql/PRO_SPs_Sequenced/23_S15_PRO.UpdateProvisionKey_AccountWise.StoredProcedure.sql"


def _rows():
    sql = read_sql_file(S15)
    object_list = split_objects(sql, S15.name, detect_dialect(sql))
    for index, obj in enumerate(object_list):
        obj.object_id = f"{S15.stem}:{index}"
    objects = {o.object_id: o for o in object_list}
    infos = {oid: analyze_object(obj) for oid, obj in objects.items()}
    mapping = build_entity_name_map_for_tables(
        [t for info in infos.values() for t in info.tables_read + info.tables_written]
    )
    chains = [LineageChain(chain_id=S15.stem, object_ids=list(objects), order=list(objects))]
    model = CanonicalModel(
        chain_id=S15.stem,
        job_id=S15.stem,
        object_ids=list(objects),
        technical_summary="s15 test",
        business_summary="",
    )
    return generate_dd_rows_for_chains(
        chains, [model], objects, infos, None, entity_name_map=mapping
    )


def test_s15_provision_alt_key_has_valid_formula():
    matches = [
        r
        for r in _rows()
        if (r.column_name or "").upper() == "PROVISIONALT_KEY"
        and (r.entity_name or "").upper().replace("##", "") == "ACCOUNTCAL"
    ]
    assert matches, "missing AccountCal.ProvisionAlt_Key row"
    row = matches[0]
    expr = (row.display_derivation_expression or "").strip()
    assert expr, f"empty formula; errors={row.validation_errors}"
    assert " BETWEEN LowerDPD" not in expr or "UpperDPD" in expr, (
        "truncated BETWEEN predicate leaked into assignment"
    )
    vr = validate_expression(expr)
    assert vr.valid, vr.error
