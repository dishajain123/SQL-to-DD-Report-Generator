"""S06 regressions: GROUP BY derived-table roll-ups and ``CASE … END alias`` temp schemas."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.phase1_lineage import build_lineage_map
from app.derivation.v2.phase2_mutation_folder import fold_column_mutations
from app.derivation.v2.pipeline import generate_for_sql
from app.derivation.v2.sql_text import parse_select_list
from app.grammar.validator import validate_expression

_REPO = Path(__file__).resolve().parents[3]
_S06 = _REPO / "samples/sql/PRO_SPs_Sequenced/11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql"


def _sql() -> str:
    return _S06.read_text(encoding="utf-8", errors="replace")


def test_parse_select_list_reads_alias_after_case_end():
    items = parse_select_list(
        "A.CustomerAcID, "
        "CASE WHEN isnull(A.X,0)>=isnull(A.Y,0) THEN isnull(a.X,0) ELSE isnull(a.Y,0) END AS REFPERIODNPA, "
        "CASE WHEN isnull(A.X,0)>0 THEN A.X ELSE 0 END DPD_X, "
        "CASE WHEN isnull(A.X,0)>0 THEN A.X ELSE isnull(A.Z,0) END"
    )
    aliases = [alias for _q, _c, alias, _raw in items]
    assert aliases == [None, "REFPERIODNPA", "DPD_X", None]


def test_temp_schemas_include_case_projection_columns():
    lineage = build_lineage_map(_sql(), None)
    npa_cols = {c.upper() for c in lineage.temp_table_columns.get("#TEMPTABLENPA", [])}
    assert {"CUSTOMERACID", "REFPERIODNPA"} <= npa_cols
    dpd_cols = {c.upper() for c in lineage.temp_table_columns.get("#TEMPTABLEDPD", [])}
    assert {"DPD_INTSERVICE", "DPD_OVERDRAWN", "DPD_STOCKSTMT"} <= dpd_cols


def test_customer_sysnpa_dt_groupby_list_is_not_alias_resolved():
    lineage = build_lineage_map(_sql(), None)
    muts = fold_column_mutations(_sql(), "CustomerCal", "SysNPA_Dt", lineage, None)
    assert muts
    for mut in muts:
        expr = mut.assigned_expression or ""
        assert '.["' not in expr, expr
        assert ", PUI_CAL" not in expr, expr
    assert any("PUI_CAL::NPA_DATE" in (m.assigned_expression or "") for m in muts)


def test_customer_sysnpa_dt_is_derived_and_valid():
    row, debug = generate_for_sql(_sql(), "##CUSTOMERCAL", "SysNPA_Dt", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    assert formula
    assert not row.validation_errors, row.validation_errors
    assert validate_expression(formula).passed, validate_expression(formula).errors
    assert "Untranslated" not in formula
    assert not re.search(r"\b(MIN|MAX|SUM|COUNT)\s*\(", formula, re.I), formula[:500]
    assert "PUI_CAL" in formula.upper() or "NPA_DATE" in formula.upper()


def test_defer_customer_rollup_arm_unit():
    from app.derivation.v2.ast_optimize import _defer_customer_rollup_arms

    def cond(name):
        return {"type": "COLUMN_REF", "entity": "A", "column": name}

    def col(name):
        return {"type": "COLUMN_REF", "entity": "A", "column": name}

    sys_copy = {"type": "COLUMN_REF", "entity": "AccountCal", "relationship": "CustomerCal", "column": "SysNPA_Dt"}
    null = {"type": "LITERAL", "value_type": "NULL", "value": None}
    chain = {
        "type": "IF_THEN_ELSE", "condition": cond("c1"), "then_branch": col("alwys"),
        "else_branch": {
            "type": "IF_THEN_ELSE", "condition": cond("c2"), "then_branch": sys_copy,
            "else_branch": {
                "type": "IF_THEN_ELSE", "condition": cond("c3"), "then_branch": col("dpd"),
                "else_branch": {
                    "type": "IF_THEN_ELSE", "condition": cond("c4"), "then_branch": null,
                    "else_branch": col("keep"),
                },
            },
        },
    }
    out = _defer_customer_rollup_arms(chain, "FinalNpaDt")
    thens = []
    node = out
    while node.get("type") == "IF_THEN_ELSE":
        thens.append(node["then_branch"].get("column") or node["then_branch"].get("type"))
        node = node["else_branch"]
    assert thens == ["alwys", "dpd", "SysNPA_Dt", "LITERAL"]
    # Self-target (CustomerCal.SysNPA_Dt) is left untouched.
    assert _defer_customer_rollup_arms(chain, "SysNPA_Dt") is chain


def test_final_npa_dt_refperiodnpa_resolves_to_temptablenpa():
    _, debug = generate_for_sql(_sql(), "AccountCal", "FinalNpaDt", llm_client=None)
    formula = (debug.get("formula") or "").replace(" ", "")
    assert formula
    assert '"#TEMPTABLENPA"."REFPERIODNPA"' in formula.upper()
    assert '"ACCOUNTCAL"."REFPERIODNPA"' not in formula.upper()
