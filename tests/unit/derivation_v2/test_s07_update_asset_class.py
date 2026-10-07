"""Structural regressions for PRO Update_AssetClass (S07)."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import FORMULA_CHAR_BUDGET
from app.derivation.v2.phase3_ast_generator import parse_sql_expression_to_ast
from app.derivation.v2.pipeline import generate_for_sql
from app.derivation.v2.sql_text import is_staging_derivation_entity
from app.grammar.validator import validate_expression

_REPO = Path(__file__).resolve().parents[3]
_S07 = _REPO / "samples/sql/PRO_SPs_Sequenced/12_S07_PRO.Update_AssetClass.StoredProcedure.sql"
_INLINE_IF_CMP = re.compile(
    r"\(IF\s*\([^)]+\)\s*THEN\s*\([^)]+\)\s*ELSE\s*\([^)]+\)\)\s*[<>!=]",
    re.I,
)
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)


def _sql() -> str:
    return _S07.read_text(encoding="utf-8", errors="replace")


def _row_formula(entity: str, column: str) -> tuple[object, str, list]:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    return row, formula, debug.get("mutations") or []


def _assert_hygiene(formula: str, column: str) -> None:
    assert formula, column
    assert len(formula) <= FORMULA_CHAR_BUDGET, f"{column} length {len(formula)}"
    assert _INLINE_IF_CMP.search(formula) is None, f"{column}: {formula[:500]}"
    assert not _AGG_RE.search(formula), f"{column}: {formula[:500]}"
    assert "__UNRESOLVED_SUBQUERY_PREDICATE__" not in formula
    assert validate_expression(formula).passed, (column, validate_expression(formula).errors)


def test_dbt_dt_los_in_predicate_keeps_relationship_hop():
    cond = (
        "(ISNULL(B.FlgDeg,'N')='Y' AND ISNULL(B.FlgProcessing,'N')='N') AND "
        "SysAssetClassAlt_Key IN(SELECT AssetClassAlt_Key FROM DimAssetClass "
        "WHERE AssetClassShortName='LOS' AND EffectiveFromTimeKey<=@TIMEKEY "
        "AND EffectiveToTimeKey>=@TIMEKEY)"
    )
    node = parse_sql_expression_to_ast(
        cond,
        default_entity="CustomerCal",
        target_column="DbtDt",
        as_condition=True,
    )
    formula = compile_ast_to_4x_string(
        node, target_entity="CustomerCal", target_column="DbtDt"
    )
    assert '"LOS"' in formula or '."LOS".' in formula
    assert "SysAssetClassAlt_Key" in formula or "SYSASSETCLASSALT_KEY" in formula.upper()


def test_dim_lookup_subqueries_discriminate_short_names():
    for short in ("SUB", "DB1", "DB2", "DB3", "LOS"):
        node = parse_sql_expression_to_ast(
            f"(SELECT AssetClassAlt_Key FROM DimAssetClass "
            f"WHERE AssetClassShortName='{short}' "
            f"AND EffectiveFromTimeKey<=@TIMEKEY AND EffectiveToTimeKey>=@TIMEKEY)",
            default_entity="CustomerCal",
            target_column="SysAssetClassAlt_Key",
        )
        assert node["relationship"] == short
        formula = compile_ast_to_4x_string(
            node, target_entity="CustomerCal", target_column="SysAssetClassAlt_Key"
        )
        assert f'"CustomerCal"."{short}"."AssetClassAlt_Key"' == formula
        assert validate_expression(formula).passed


def test_s07_customer_sys_asset_class_alt_key_aging_and_fraud_precedence():
    row, formula, mutations = _row_formula("##CustomerCal", "SysAssetClassAlt_Key")
    if not formula:
        row, formula, mutations = _row_formula("CustomerCal", "SysAssetClassAlt_Key")
    assert not getattr(row, "validation_errors", None), getattr(row, "validation_errors", None)
    _assert_hygiene(formula, "SysAssetClassAlt_Key")
    assert len(mutations) >= 2
    upper = formula.upper()
    assert "ADDDAY(" in upper or "PERIOD(" in upper
    assert "SYSNPA" in upper or "SysNPA" in formula
    for token in ("SUB", "DB1", "DB2", "DB3"):
        assert f'."{token}".' in formula or f'."{token}"' in formula
    assert '"LOS"' in formula or '."LOS".' in formula
    assert "870" in formula or "SplCatg" in formula
    idx_sub = formula.find('"SUB"')
    fraud_markers = [formula.find("870"), formula.upper().find("SPLCATG")]
    idx_fraud = min(i for i in fraud_markers if i >= 0) if any(i >= 0 for i in fraud_markers) else -1
    if idx_fraud >= 0 and idx_sub >= 0:
        assert idx_fraud < idx_sub, "fraud LOS override must be outermost over SUB/DB aging"


def test_s07_account_final_asset_class_from_customer():
    row, formula, _ = _row_formula("##AccountCal", "FinalAssetClassAlt_Key")
    if not formula:
        row, formula, _ = _row_formula("AccountCal", "FinalAssetClassAlt_Key")
    assert not getattr(row, "validation_errors", None), getattr(row, "validation_errors", None)
    _assert_hygiene(formula, "FinalAssetClassAlt_Key")
    assert "SysAssetClassAlt_Key" in formula or "SYSASSETCLASSALT_KEY" in formula.upper()
    assert "NORMAL" in formula.upper()
    assert "FlgDeg" in formula or "FLGDEG" in formula.upper()
    assert "FinalAssetClassAlt_Key" in formula or "FINALASSETCLASSALT_KEY" in formula.upper()


def test_s07_mutation_index_includes_account_final_asset_class():
    from app.parsing.structural_analysis import analyze_object
    from app.models.core import ObjectType, SQLObject
    from app.parsing.dialect import detect_dialect
    from app.derivation.v2.phase2_mutation_folder import MutationSourceIndex, _written_columns_by_table
    from app.derivation.v2.phase1_lineage import build_lineage_map

    sql = _sql()
    obj = SQLObject(
        object_id="s07",
        name="Update_AssetClass",
        object_type=ObjectType.PROCEDURE,
        source_file="12_S07",
        raw_sql=sql,
        dialect=detect_dialect(sql),
    )
    info = analyze_object(obj)
    idx = MutationSourceIndex.build(sql)
    lineage = build_lineage_map(sql, None)
    written = _written_columns_by_table(idx, lineage, None)
    account_tables = [t for t in written if "ACCOUNTCAL" in t.upper()]
    assert account_tables
    assert "FINALASSETCLASSALT_KEY" in written[account_tables[0]]
    cols = info.columns_written_by_table or {}
    account_cols = {
        c.upper()
        for t, cs in cols.items()
        if "ACCOUNTCAL" in t.upper()
        for c in cs
    }
    assert "FINALASSETCLASSALT_KEY" in account_cols or "FINALASSETCLASSALT_KEY" in written[
        account_tables[0]
    ]


def test_s07_customer_dbt_dt_and_deg_date():
    _, dbt, _ = _row_formula("##CustomerCal", "DbtDt")
    if not dbt:
        _, dbt, _ = _row_formula("CustomerCal", "DbtDt")
    if dbt:
        _assert_hygiene(dbt, "DbtDt")
        assert "NULL" in dbt.upper()
        assert "ADDDAY(" in dbt.upper() or "SysNPA" in dbt
        assert '"LOS"' in dbt or '."LOS".' in dbt
        assert "DimAssetClass.LOS" not in dbt
        assert dbt.count('"LOS"."AssetClassAlt_Key"') + dbt.count('."LOS"."AssetClassAlt_Key"') <= 1
        compact = dbt.replace(" ", "")
        assert "ADDDAY(" in dbt.upper()
        # Pass-1 NULL reset must not sit under the same guard as ADDDAY (CASE fall-through).
        assert not (
            "THEN(IF(AND(" in compact
            and "THEN(NULL)ELSEIF(" in compact.upper()
            and '"LOS"' not in dbt
        )
    _, deg, _ = _row_formula("##CustomerCal", "DegDate")
    if not deg:
        _, deg, _ = _row_formula("CustomerCal", "DegDate")
    if deg:
        _assert_hygiene(deg, "DegDate")
        assert "ProcessDate" in deg or "@ProcessDate" in deg or "TIMEKEY" in deg.upper()
        assert "ELSEIF" not in deg.upper()
        compact = deg.replace(" ", "")
        guard = 'COALESCE("CUSTOMERCAL"."FlgDeg"'
        assert compact.count(guard) == 1 or compact.count(guard.lower()) == 1


def test_s07_sys_asset_class_uses_addday_on_sys_npa_dt_not_raw_plus():
    _, formula, _ = _row_formula("##CustomerCal", "SysAssetClassAlt_Key")
    if not formula:
        _, formula, _ = _row_formula("CustomerCal", "SysAssetClassAlt_Key")
    assert formula
    compact = formula.replace(" ", "").replace('"', "").upper()
    assert "SYSNPA_DT+" not in compact and "SYSNPA_DT+@" not in compact
    assert "ADDDAY(" in formula.upper()
    assert "SYSNPA" in formula.upper()


def test_s07_acl_completed_try_catch():
    row, formula, _ = _row_formula("ACLRUNNINGPROCESSSTATUS", "COMPLETED")
    _assert_hygiene(formula, "COMPLETED")
    assert "Update_AssetClass" in formula
    catch = row.exception_handler_expression or ""
    assert catch
    assert "Update_AssetClass" in catch


def test_s07_staging_cte_not_export_target():
    assert is_staging_derivation_entity("#CTE_CustomerWiseBalance")
    assert is_staging_derivation_entity("CTE_CustomerWiseBalance")
    assert not is_staging_derivation_entity("##CustomerCal")
    assert not is_staging_derivation_entity("##AccountCal")
