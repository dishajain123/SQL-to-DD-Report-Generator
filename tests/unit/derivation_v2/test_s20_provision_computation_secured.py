"""Structural regressions for PRO ProvisionComputationSecured (S20)."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.ast_optimize import FORMULA_CHAR_BUDGET
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression

_REPO = Path(__file__).resolve().parents[3]
_S20 = (
    _REPO
    / "samples/sql/PRO_SPs_Sequenced/28_S20_PRO.ProvisionComputationSecured.StoredProcedure.sql"
)
_INLINE_IF_CMP = re.compile(
    r"\(IF\s*\([^)]+\)\s*THEN\s*\([^)]+\)\s*ELSE\s*\([^)]+\)\)\s*[<>!=]",
    re.I,
)
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)


def _sql() -> str:
    return _S20.read_text(encoding="utf-8", errors="replace")


def _row_formula(column: str, entity: str = "##AccountCal") -> tuple[object, str, list]:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    if not formula and entity.startswith("##"):
        row, debug = generate_for_sql(_sql(), "AccountCal", column, llm_client=None)
        formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    return row, formula, debug.get("mutations") or []


def _assert_hygiene(formula: str, column: str) -> None:
    assert formula, column
    assert len(formula) <= FORMULA_CHAR_BUDGET, f"{column} length {len(formula)}"
    assert _INLINE_IF_CMP.search(formula) is None, f"{column}: {formula[:500]}"
    assert not _AGG_RE.search(formula), f"{column}: {formula[:500]}"
    assert "__UNRESOLVED_SUBQUERY_PREDICATE__" not in formula
    assert validate_expression(formula).passed, (column, validate_expression(formula).errors)


def test_s20_secured_amt_reset_join_case_and_nonpositive_clamp():
    row, formula, mutations = _row_formula("SECUREDAMT")
    assert not getattr(row, "validation_errors", None), getattr(row, "validation_errors", None)
    _assert_hygiene(formula, "SECUREDAMT")
    assert len(mutations) >= 4
    upper = formula.upper()
    assert "USEDRV" in upper or "UsedRV" in formula
    assert "LOS" in upper
    assert "FLGPROCESSING" in upper or "FlgProcessing" in formula
    assert "BALANCE" in upper or "Balance" in formula
    assert "FINALASSETCLASSALT_KEY" in upper or "FinalAssetClassAlt_Key" in formula
    assert "THEN(0)" in formula.replace(" ", "")


def test_s20_prov_secured_corporate_common_branch_and_rbi_rate():
    row, formula, mutations = _row_formula("PROVSECURED")
    assert not getattr(row, "validation_errors", None), getattr(row, "validation_errors", None)
    _assert_hygiene(formula, "PROVSECURED")
    assert len(mutations) >= 4
    assert "Corporate Common" in formula or "CORPORATE COMMON" in formula.upper()
    assert "ProvPerSecured" in formula or "PROVPERSECURED" in formula.upper()
    _, bank, _ = _row_formula("BANKPROVSECURED")
    _assert_hygiene(bank, "BANKPROVSECURED")
    _, rbi, _ = _row_formula("RBIPROVSECURED")
    _assert_hygiene(rbi, "RBIPROVSECURED")
    assert "RBIPROVISION" in rbi.upper() or "RBI" in rbi.upper()


def test_s20_corporate_common_uses_usedrv_times_provpersecured_not_self_reference():
    for column in ("PROVSECURED", "BANKPROVSECURED"):
        _, formula, _ = _row_formula(column)
        compact = formula.replace(" ", "").upper()
        # The Corporate Common arm must compute USEDRV * ProvPerSecured ...
        assert '"ACCOUNTCAL"."PROVPERSECURED"' in compact, (column, formula[:600])
        assert compact.count('"ACCOUNTCAL"."PROVPERSECURED"') >= 2, column  # Seg + SegStd arms
        # ... and must not return the target column itself (circular tautology).
        assert f'THEN("ACCOUNTCAL"."{column.upper()}")' not in compact, (column, formula[:600])
        assert (
            'THEN(COALESCE("ACCOUNTCAL"."USEDRV",0)*COALESCE("ACCOUNTCAL"."PROVPERSECURED",0))'
            in compact
        ), (column, formula[:800])


def test_s20_no_redundant_nested_coalesce_zero_comparisons():
    for column in ("PROVSECURED", "BANKPROVSECURED"):
        _, formula, _ = _row_formula(column)
        assert "COALESCE(COALESCE(" not in formula.replace(" ", "").upper(), (column, formula[:800])
    _, rbi, _ = _row_formula("RBIPROVSECURED")
    _assert_hygiene(rbi, "RBIPROVSECURED")


def test_terminal_else_rule_only_collapses_prior_value_chains():
    from app.derivation.v2.ast_optimize import _terminal_else_reads_column

    clamp = {
        "type": "IF_THEN_ELSE",
        "condition": {"type": "COLUMN_REF", "entity": "ACCOUNTCAL", "column": "X"},
        "then_branch": {"type": "COLUMN_REF", "entity": "ACCOUNTCAL", "column": "USEDRV"},
        "else_branch": {"type": "LITERAL", "value_type": "NUMBER", "value": 0},
    }
    prior = dict(clamp, else_branch={"type": "COLUMN_REF", "entity": "AccountCal", "column": "ProvSecured"})
    assert not _terminal_else_reads_column(clamp, "##AccountCal", "PROVSECURED")
    assert _terminal_else_reads_column(prior, "##AccountCal", "PROVSECURED")


def test_s20_npa_vs_std_provision_paths_by_asset_class_key():
    """Two joined provision tables: Seg (key>1) vs SegStd (key=1)."""
    _, prov, _ = _row_formula("PROVSECURED")
    assert prov
    assert "Provision" in prov or "PROVISION" in prov.upper()
    idx_gt = prov.upper().find(">1")
    idx_eq = prov.upper().find("==1")
    assert idx_gt >= 0 or idx_eq >= 0, prov[:400]


def test_s20_acl_completed_try_catch():
    row, debug = generate_for_sql(
        _sql(), "ACLRUNNINGPROCESSSTATUS", "COMPLETED", llm_client=None
    )
    formula = debug.get("formula") or row.display_derivation_expression or ""
    _assert_hygiene(formula, "COMPLETED")
    assert "ProvisionComputationSecured" in formula
    catch = row.exception_handler_expression or ""
    assert catch
    assert "ProvisionComputationSecured" in catch
