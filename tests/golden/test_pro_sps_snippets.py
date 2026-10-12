"""Golden regressions for S01 / S02 / S06 sequenced procedures (engine-level)."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression

_REPO = Path(__file__).resolve().parents[2]
_S01 = _REPO / "samples/sql/PRO_SPs_Sequenced/06_S01_PRO.Reference_Period_Calculation.StoredProcedure.sql"
_S02 = _REPO / "samples/sql/PRO_SPs_Sequenced/07_S02_PRO.DPD_Calculation.StoredProcedure.sql"
_S06 = _REPO / "samples/sql/PRO_SPs_Sequenced/11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql"


def _formula(sql_path: Path, entity: str, column: str) -> tuple[str, list[str]]:
    row, debug = generate_for_sql(
        sql_path.read_text(encoding="utf-8", errors="replace"),
        entity,
        column,
        llm_client=None,
    )
    formula = debug.get("formula") or row.display_derivation_expression or ""
    return formula, list(row.validation_errors or [])


def test_s02_dpd_max_fin_emits_max_aggregate_not_self_isempty():
    formula, errors = _formula(_S02, "PRO.AdvAcRestructureCal", "DPD_MaxFin")
    assert not errors, errors
    assert re.search(r"\bMAX\s*\(", formula, re.I), formula[:500]
    assert "ISEMPTY" not in formula.replace(" ", "").upper() or "DPD_MAXFIN" not in formula


def test_s06_customer_sysnpa_dt_has_min_aggregate():
    formula, errors = _formula(_S06, "##CUSTOMERCAL", "SysNPA_Dt")
    assert not errors, errors
    assert re.search(r"\bMIN\s*\(", formula, re.I), formula[:500]
    assert validate_expression(formula).passed


def test_s06_adv_ac_flgdeg_uses_dimparameter_not_self_entity():
    formula, errors = _formula(_S06, "AdvAcRestructureCal", "FlgDeg")
    assert not errors, errors
    assert '"DimParameter"."ParameterShortNameEnum"' in formula
    assert '"AdvAcRestructureCal"."ParameterShortNameEnum"' not in formula


def test_s06_no_hash_accountcal_entity_paths():
    formula, errors = _formula(_S06, "RESTR_NPA", "UcifEntityID")
    assert not errors, errors
    assert formula, "RESTR_NPA.UcifEntityID must export a formula"
    assert '"#AccountCal"' not in formula
    assert '"AccountCal"' in formula or '"ACCOUNTCAL"' in formula.upper()
    assert validate_expression(formula).passed


def test_s02_dimproduct_not_nested_under_accountcal():
    sql = _S02.read_text(encoding="utf-8", errors="replace")
    row, debug = generate_for_sql(sql, "##ACCOUNTCAL", "DPD_IntService", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    assert formula
    assert '"AccountCal"."DimProduct"' not in formula.replace(" ", "")
    assert '"DIMPRODUCT"' in formula.upper() or '"DimProduct"' in formula


def test_s01_advacbasicdetail_top_level_path():
    formula, errors = _formula(_S01, "AdvAcBasicDetail", "RefPeriodMax")
    assert not errors, errors
    assert '"AdvAcBasicDetail"' in formula or '"ADVACBASICDETAIL"' in formula.upper()
    assert '"AccountCal"."AdvAcBasicDetail"' not in formula.replace(" ", "")


def test_s06_degreason_no_ephemeral_alias_in_path():
    formula, errors = _formula(_S06, "##CUSTOMERCAL", "DegReason")
    assert not errors, errors
    assert formula, "CustomerCal.DegReason must not be dropped from export"
    assert '."C".' not in formula
    assert "STRING_AGG" not in formula.upper()
    assert "DEFAULT_REASON" in formula.upper() or "PUI" in formula.upper()
    assert "CONCAT(" in formula.upper()
    assert re.search(r"CONCAT\([^)]+\)\s*\+", formula, re.I) is None


def test_s02_dpd_overdue_timekey_nesting():
    formula, errors = _formula(_S02, "##ACCOUNTCAL", "DPD_Overdue")
    assert not errors, errors
    assert "26372" in formula
    assert "26267" in formula
    idx_outer = formula.find("26267")
    idx_inner = formula.find("26372")
    assert idx_outer >= 0 and idx_inner > idx_outer, "26372 must nest inside 26267 arm"


def test_s02_dpd_overdue_floor_uses_max_not_coalesce_guard():
    formula, errors = _formula(_S02, "##ACCOUNTCAL", "DPD_Overdue")
    assert not errors, errors
    assert re.search(r"\bMAX\s*\([^)]*,\s*0\s*\)", formula.replace(" ", ""), re.I) or "MAX(" in formula.upper()
