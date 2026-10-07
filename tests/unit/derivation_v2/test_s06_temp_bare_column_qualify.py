"""Bare column qualification onto ``#temp`` join aliases (S06 DPD pass)."""
from pathlib import Path

from app.derivation.v2.phase1_lineage import build_lineage_map
from app.derivation.v2.phase2_mutation_folder import (
    MutationSourceIndex,
    _augment_written_cols_with_join_temp_schemas,
    _qualify_foreign_bare_columns,
    _written_columns_by_table,
    fold_column_mutations,
)
from app.derivation.v2.sql_text import extract_update_statements

_REPO = Path(__file__).resolve().parents[3]
_S06 = _REPO / "samples/sql/PRO_SPs_Sequenced/11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql"


def test_refperiodnpa_in_dpd_update_qualifies_to_temptablenpa_join():
    sql = _S06.read_text(encoding="utf-8", errors="replace")
    lineage = build_lineage_map(sql, None)
    idx = MutationSourceIndex.build(sql)
    dpd_stmt = next(
        s
        for s in extract_update_statements(sql)
        if "REFPERIODNPA" in (s.get("set_clause") or "") and "#TEMPTABLENPA" in (s.get("from_clause") or "").upper()
    )
    from app.derivation.v2.phase2_mutation_folder import _parse_update_sources

    alias_map, _ = _parse_update_sources(
        (dpd_stmt.get("head") or "").strip(),
        (dpd_stmt.get("from_clause") or "").strip(),
        lineage,
        None,
    )
    written_cols = _augment_written_cols_with_join_temp_schemas(
        _written_columns_by_table(idx, lineage, None), alias_map, lineage
    )
    set_clause = dpd_stmt.get("set_clause") or ""
    qualified = _qualify_foreign_bare_columns(
        set_clause, alias_map, ["##AccountCal"], written_cols
    )
    assert qualified
    assert "B.REFPERIODNPA" in qualified.upper().replace(" ", "")


def test_customer_sysnpa_dt_has_chronological_mutations():
    sql = _S06.read_text(encoding="utf-8", errors="replace")
    lineage = build_lineage_map(sql, None)
    muts = fold_column_mutations(sql, "CustomerCal", "SysNPA_Dt", lineage, None)
    assert len(muts) >= 2
    assert any("MIN(" in (m.assigned_expression or "").upper() for m in muts)
