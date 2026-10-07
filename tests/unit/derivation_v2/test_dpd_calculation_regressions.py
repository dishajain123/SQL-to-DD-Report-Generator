"""PRO.DPD_Calculation — AdvAcRestructureCal + DPD column fold regressions."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import FORMULA_CHAR_BUDGET, optimize_expression_ast
from app.derivation.v2.phase2_mutation_folder import _try_dedupe_coalesce_negative_clamp
from app.derivation.v2.phase3_ast_generator import _is_cross_column_null_default_guard
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression

_SIMPLE_ACCOUNT_CAL_UPDATE = """
UPDATE ##AccountCal SET ContiExcessDt = NULL WHERE OverdueDays > 0;
"""

_RESTR_SQL = """
;WITH CTE_FIN_DPD AS (
    SELECT AccountEntityID, DPD_IntService DPD FROM ##AccountCal WHERE ISNULL(DPD_IntService,0)>0
)
UPDATE B
    SET B.DPD_MaxFin=A.DPD_MaxFin
FROM (SELECT AccountEntityID, MAX(DPD) DPD_MaxFin FROM CTE_FIN_DPD GROUP BY AccountEntityID) a
    INNER JOIN PRO.AdvAcRestructureCal B ON A.AccountEntityID = B.AccountEntityId;

;WITH CTE_NONFIN_DPD AS (
    SELECT AccountEntityID, DPD_StockStmt DPD FROM ##AccountCal WHERE ISNULL(DPD_StockStmt,0)>0
)
UPDATE B
    SET B.DPD_MaxNonFin=A.DPD_MaxNonFin
FROM (SELECT AccountEntityID, MAX(DPD) DPD_MaxNonFin FROM CTE_NONFIN_DPD GROUP BY AccountEntityID) a
    INNER JOIN PRO.AdvAcRestructureCal B ON A.AccountEntityID = B.AccountEntityId;

UPDATE PRO.AdvAcRestructureCal SET DPD_MaxNonFin=0 WHERE DPD_MaxNonFin IS NULL;
UPDATE PRO.AdvAcRestructureCal SET DPD_MaxFin=0 WHERE DPD_MaxNonFin IS NULL;
"""

_DPD_INT_SQL = """
UPDATE ##AccountCal SET DPD_IntService=0;
if @TIMEKEY >26267
    UPDATE A SET A.DPD_IntService = CASE WHEN @TIMEKEY >26384
        THEN (CASE WHEN A.IntNotServicedDt IS NOT NULL
            THEN DATEDIFF(DAY,A.IntNotServicedDt,@ProcessDate)+2 ELSE 0 END)
        ELSE (CASE WHEN A.IntNotServicedDt IS NOT NULL
            THEN DATEDIFF(DAY,A.IntNotServicedDt,@ProcessDate)+1 ELSE 0 END)
    END
    FROM ##AccountCal A;
UPDATE ##AccountCal SET DPD_IntService=0 WHERE isnull(DPD_IntService,0)<0;
UPDATE A SET DPD_IntService=0
FROM ##ACCOUNTCAL A
INNER JOIN DIMPRODUCT C ON A.ProductAlt_Key=C.ProductAlt_Key
WHERE C.EffectiveFromTimeKey<=@TIMEKEY AND C.EffectiveToTimeKey>=@TIMEKEY
AND (ISNULL(C.Aqua_Scheme,'N')='Y' AND ISNULL(C.SchemeType,'')='ODA');
"""

_AQUA_ONLY_DEBIT_SQL = """
UPDATE A SET DebitSinceDt=NULL
FROM ##ACCOUNTCAL A
INNER JOIN DIMPRODUCT C ON A.ProductAlt_Key=C.ProductAlt_Key
WHERE C.EffectiveFromTimeKey<=@TIMEKEY AND C.EffectiveToTimeKey>=@TIMEKEY
AND (ISNULL(C.Aqua_Scheme,'N')='Y' AND ISNULL(C.SchemeType,'')='ODA');
"""

_AQUA_MERGE_SQL = """
MERGE INTO PRO.AccountCal_Stg T
USING (
    SELECT A.AccountEntityID
    FROM PRO.AccountCal_Stg A
    INNER JOIN DIMPRODUCT C ON A.ProductAlt_Key = C.ProductAlt_Key
    WHERE C.EffectiveFromTimeKey <= @TIMEKEY AND C.EffectiveToTimeKey >= @TIMEKEY
      AND (ISNULL(C.Aqua_Scheme,'N') = 'Y' AND ISNULL(C.SchemeType,'') = 'ODA')
) S ON (T.AccountEntityID = S.AccountEntityID)
WHEN MATCHED THEN UPDATE SET T.DebitSinceDt = NULL;
"""

_PROCESS_STATUS_SQL = """
UPDATE PRO.ACLRUNNINGPROCESSSTATUS
SET COMPLETED='Y', ERRORDATE=NULL, ERRORDESCRIPTION=NULL, COUNT=ISNULL(COUNT,0)+1
WHERE RUNNINGPROCESSNAME='DPD_Calculation';
UPDATE PRO.ACLRUNNINGPROCESSSTATUS
SET COMPLETED='N', ERRORDATE=GETDATE(), ERRORDESCRIPTION=ERROR_MESSAGE(), COUNT=ISNULL(COUNT,0)+1
WHERE RUNNINGPROCESSNAME='DPD_Calculation';
"""


def test_guarded_update_does_not_fail_regex_in_set_based_skip_check():
    """Regression: ``^(?i)`` in re patterns raised re.error for every column."""
    row, _ = generate_for_sql(
        _SIMPLE_ACCOUNT_CAL_UPDATE, "AccountCal", "ContiExcessDt", llm_client=None
    )
    errors = row.validation_errors or []
    assert not any("global flags not at the start" in e for e in errors), errors
    assert row.display_derivation_expression, errors


def test_adv_ac_restructure_columns_are_self_isempty_not_subquery_alias():
    for column in ("DPD_MaxFin", "DPD_MaxNonFin"):
        row, _ = generate_for_sql(
            _RESTR_SQL, "PRO.AdvAcRestructureCal", column, llm_client=None
        )
        expr = row.display_derivation_expression or ""
        assert expr, column
        assert '"A"' not in expr, expr
        if column.upper() == "DPD_MAXFIN":
            assert "DPD_MaxNonFin" not in expr, expr
            assert 'ISEMPTY("AdvAcRestructureCal"."DPD_MaxFin")' in expr.replace(" ", ""), expr
        assert column.upper() in expr.upper(), expr
        assert "ISEMPTY" in expr, expr
        assert validate_expression(expr).passed, validate_expression(expr).errors


def test_resolved_dimproduct_isnull_predicate_is_not_cross_column_skip():
    """Phase-2 WHERE text uses ``ISNULL(Entity::Rel::Col,…)`` — must still fold."""
    cond = (
        "AccountCal::DIMPRODUCT::EffectiveFromTimeKey<=@TIMEKEY AND "
        "AccountCal::DIMPRODUCT::EffectiveToTimeKey>=@TIMEKEY AND "
        "(ISNULL(AccountCal::DIMPRODUCT::Aqua_Scheme,'N')='Y' AND "
        "ISNULL(AccountCal::DIMPRODUCT::SchemeType,'')='ODA')"
    )
    assert not _is_cross_column_null_default_guard(cond, "IntNotServicedDt")
    assert _is_cross_column_null_default_guard("DPD_MaxFin IS NULL", "DPD_MaxNonFin")


def test_int_not_serviced_dt_retains_aqua_scheme_after_sentinel_nulling():
    sql = """
UPDATE ##AccountCal SET IntNotServicedDt = NULL
WHERE (IntNotServicedDt='1900-01-01' OR IntNotServicedDt='01/01/1900');
UPDATE A SET A.IntNotServicedDt=NULL
FROM ##ACCOUNTCAL A
INNER JOIN DIMPRODUCT C ON A.ProductAlt_Key=C.ProductAlt_Key
WHERE C.EffectiveFromTimeKey<=@TIMEKEY AND C.EffectiveToTimeKey>=@TIMEKEY
AND (ISNULL(C.Aqua_Scheme,'N')='Y' AND ISNULL(C.SchemeType,'')='ODA');
"""
    row, _ = generate_for_sql(sql, "AccountCal", "IntNotServicedDt", llm_client=None)
    expr = row.display_derivation_expression or ""
    assert expr
    assert "Aqua_Scheme" in expr or "SchemeType" in expr, expr


def test_merge_using_row_predicate_strips_trailing_source_alias():
    from app.derivation.v2.sql_text import extract_merge_matched_updates, extract_merge_using_row_predicate

    merges = extract_merge_matched_updates(_AQUA_MERGE_SQL)
    assert merges
    pred = extract_merge_using_row_predicate(merges[0]["using_body"])
    assert pred
    assert "Aqua_Scheme" in pred or "SchemeType" in pred


def test_merge_aqua_scheme_predicate_folds_into_formula():
    row, _ = generate_for_sql(
        _AQUA_MERGE_SQL, "AccountCal_Stg", "DebitSinceDt", llm_client=None
    )
    expr = row.display_derivation_expression or ""
    assert expr
    assert "Aqua_Scheme" in expr or "SchemeType" in expr, expr
    assert expr.strip() != '"AccountCal_Stg"."DebitSinceDt"', expr


def test_aqua_scheme_override_folds_into_target_column_formula():
    row, _ = generate_for_sql(
        _AQUA_ONLY_DEBIT_SQL, "ACCOUNTCAL", "DebitSinceDt", llm_client=None
    )
    expr = row.display_derivation_expression or ""
    errors = row.validation_errors or []
    assert not any("_is_coalesce_or_isnull_of_self" in e for e in errors), errors
    assert expr
    assert "Aqua_Scheme" in expr or "SchemeType" in expr, expr
    assert "NULL" in expr.upper(), expr
    assert expr.strip() != '"ACCOUNTCAL"."DebitSinceDt"', expr


def test_process_status_completed_includes_running_process_filter():
    row, _ = generate_for_sql(
        _PROCESS_STATUS_SQL, "ACLRUNNINGPROCESSSTATUS", "COMPLETED", llm_client=None
    )
    expr = row.display_derivation_expression or ""
    assert expr
    assert "RUNNINGPROCESSNAME" in expr.upper(), expr
    assert '"Y"' in expr or "Y" in expr, expr
    assert expr.strip() != '"ACLRUNNINGPROCESSSTATUS"."COMPLETED"', expr


def test_dpd_int_service_does_not_duplicate_datediff_for_negative_clamp():
    row, _ = generate_for_sql(
        _DPD_INT_SQL, "##ACCOUNTCAL", "DPD_IntService", llm_client=None
    )
    expr = row.display_derivation_expression or ""
    errors = row.validation_errors or []
    assert not any("_is_coalesce_or_isnull_of_self" in e for e in errors), errors
    assert expr
    assert "COALESCE(IF(" not in expr.replace(" ", ""), expr
    assert not re.search(
        r'COALESCE\s*\(\s*"[^"]+"\s*\.\s*"DPD_IntService"\s*,\s*0\s*\)\s*<',
        expr,
        flags=re.I,
    ), f"negative clamp must test derived value, not input column: {expr}"
    # Clamp must be row-level IF (no set-based MAX wrapper in the export).
    upper = expr.upper()
    assert "MAX(" not in upper, expr
    assert "MIN(" not in upper, expr
    assert "SUM(" not in upper, expr
    timekey_hits = len(re.findall(r"@TIMEKEY\"\s*>\s*26267", expr))
    assert timekey_hits <= 2, f"redundant TIMEKEY guard copy in formula: {timekey_hits} in {expr}"
    datediff_count = len(re.findall(r"DATEDIFF\s*\(", expr, flags=re.I))
    assert datediff_count <= 4, f"expected at most 4 DATEDIFF arms, got {datediff_count}: {expr}"
    assert validate_expression(expr).passed, validate_expression(expr).errors


_FULL_DPD_SQL_CANDIDATES = (
    Path(__file__).resolve().parents[3]
    / "samples"
    / "sql"
    / "PRO_SPs_Sequenced"
    / "07_S02_PRO.DPD_Calculation.StoredProcedure.sql",
    Path(__file__).resolve().parents[3]
    / "samples"
    / "sql"
    / "PRO_DPD_Calculation_StoredProcedure_2.sql",
    Path(__file__).resolve().parents[3]
    / "output"
    / "047_job-cb9992900d"
    / "source"
    / "0001.sql",
    Path(__file__).resolve().parents[3]
    / "output"
    / "046_job-aced3dd3f7"
    / "source"
    / "0001.sql",
    Path(__file__).resolve().parents[3]
    / "output"
    / "050_job-1dbd76edf3"
    / "source"
    / "0001.sql",
)


def test_full_dpd_procedure_does_not_hit_invalid_inline_regex_flags():
    sql_path = next((p for p in _FULL_DPD_SQL_CANDIDATES if p.is_file()), None)
    if sql_path is None:
        pytest.skip("no full DPD procedure SQL fixture on disk")
    sql = sql_path.read_text(encoding="utf-8", errors="replace")
    row, _ = generate_for_sql(sql, "AccountCal", "DPD_IntService", llm_client=None)
    errors = row.validation_errors or []
    assert not any("global flags not at the start" in e for e in errors), errors
    assert not any("_is_coalesce_or_isnull_of_self" in e for e in errors), errors
    assert row.display_derivation_expression, errors


def test_full_dpd_procedure_aqua_nulling_folds_into_date_columns():
    sql_path = next((p for p in _FULL_DPD_SQL_CANDIDATES if p.is_file()), None)
    if sql_path is None:
        pytest.skip("no full DPD procedure SQL fixture on disk")
    sql = sql_path.read_text(encoding="utf-8", errors="replace")
    row, _ = generate_for_sql(sql, "AccountCal", "IntNotServicedDt", llm_client=None)
    expr = row.display_derivation_expression or ""
    assert expr
    assert "Aqua_Scheme" in expr or "SchemeType" in expr or "DIMPRODUCT" in expr.upper(), expr


_S02 = (
    Path(__file__).resolve().parents[3]
    / "samples/sql/PRO_SPs_Sequenced/07_S02_PRO.DPD_Calculation.StoredProcedure.sql"
)
_INLINE_IF_CMP = re.compile(
    r"\(IF\s*\([^)]+\)\s*THEN\s*\([^)]+\)\s*ELSE\s*\([^)]+\)\)\s*[<>!=]",
    re.I,
)
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)
_S02_DPD_COLS = (
    "DPD_IntService",
    "DPD_NoCredit",
    "DPD_Overdrawn",
    "DPD_Overdue",
    "DPD_Renewal",
    "DPD_StockStmt",
    "DPD_PrincOverdue",
    "DPD_IntOverdueSince",
    "DPD_OtherOverdueSince",
)
_S02_DATE_COLS = (
    "IntNotServicedDt",
    "LastCrDate",
    "ContiExcessDt",
    "OverDueSinceDt",
    "ReviewDueDt",
    "StockStDt",
    "DebitSinceDt",
)


def _assert_formula_hygiene(expr: str, column: str) -> None:
    assert expr, column
    assert len(expr) <= FORMULA_CHAR_BUDGET, f"{column} length {len(expr)}"
    assert "IF(IF(" not in expr, f"{column}: {expr[:400]}"
    assert _INLINE_IF_CMP.search(expr) is None, f"{column}: {expr[:400]}"
    assert not _AGG_RE.search(expr), f"{column}: {expr[:400]}"
    assert validate_expression(expr).passed, (column, validate_expression(expr).errors)


def test_distributed_negative_clamp_collapses_to_floor_without_if_comparison():
    """``(IF(c) THEN v ELSE 0) < 0`` must not remain a comparison wrapping IF."""
    cond = {
        "type": "BINARY_OP",
        "operator": ">",
        "left": {"type": "VARIABLE_REF", "name": "@TIMEKEY"},
        "right": {"type": "LITERAL", "value_type": "NUMBER", "value": 26267},
    }
    value = {
        "type": "BINARY_OP",
        "operator": "+",
        "left": {
            "type": "FUNCTION_CALL",
            "function_name": "DATEDIFF",
            "arguments": [
                {"type": "COLUMN_REF", "entity": "AccountCal", "column": "IntNotServicedDt"},
                {"type": "VARIABLE_REF", "name": "@ProcessDate"},
                {"type": "LITERAL", "value_type": "STRING", "value": "DAY"},
            ],
        },
        "right": {"type": "LITERAL", "value_type": "NUMBER", "value": 2},
    }
    zero = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    deriv = {
        "type": "IF_THEN_ELSE",
        "condition": cond,
        "then_branch": value,
        "else_branch": zero,
    }
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": {
            "type": "BINARY_OP",
            "operator": "<",
            "left": deriv,
            "right": zero,
        },
        "then_branch": zero,
        "else_branch": deriv,
    }
    distributed = {
        "type": "IF_THEN_ELSE",
        "condition": {
            "type": "IF_THEN_ELSE",
            "condition": cond,
            "then_branch": {
                "type": "BINARY_OP",
                "operator": "<",
                "left": value,
                "right": zero,
            },
            "else_branch": {
                "type": "BINARY_OP",
                "operator": "<",
                "left": zero,
                "right": zero,
            },
        },
        "then_branch": zero,
        "else_branch": deriv,
    }
    assert _try_dedupe_coalesce_negative_clamp(ast) is not None
    assert _try_dedupe_coalesce_negative_clamp(distributed) is not None
    formula = compile_ast_to_4x_string(
        optimize_expression_ast(ast, target_entity="AccountCal", target_column="DPD_IntService"),
        target_entity="AccountCal",
        target_column="DPD_IntService",
    )
    _assert_formula_hygiene(formula, "DPD_IntService")
    assert "COALESCE(IF(" not in formula.replace(" ", "")
    assert "26267" in formula
    assert "DATEDIFF(" in formula.upper()


def test_s02_file_exportable_dpd_formulas_are_valid():
    sql = _S02.read_text(encoding="utf-8", errors="replace")
    aqua_cols = {"DPD_INTSERVICE", "DPD_NOCREDIT", "DPD_STOCKSTMT"}
    for column in _S02_DPD_COLS:
        row, debug = generate_for_sql(sql, "##ACCOUNTCAL", column, llm_client=None)
        formula = debug.get("formula") or row.display_derivation_expression or ""
        assert not row.validation_errors, f"{column}: {row.validation_errors}"
        _assert_formula_hygiene(formula, column)
        assert "DATEDIFF(" in formula.upper(), column
        if column.upper() != "DPD_OVERDRAWN":
            # Overdrawn uses DATEDIFF+1 on both TIMEKEY arms, so the procedural
            # IF/ELSE collapses as degenerate once the payloads agree.
            assert "26267" in formula, column
        if column.upper() in aqua_cols:
            assert "Aqua_Scheme" in formula or "SchemeType" in formula, column
        if column.upper() == "DPD_OVERDUE":
            assert "26372" in formula, formula[:500]
            assert "SourceAlt_Key" in formula or "SOURCEALT_KEY" in formula.upper(), formula[:500]
        if column.upper() == "DPD_INTSERVICE":
            assert "26384" in formula, formula[:500]
            assert "Aqua_Scheme" in formula or "SchemeType" in formula
        if column.upper() == "DPD_NOCREDIT":
            assert ">= 90" in formula or ">=90" in formula.replace(" ", "") or "> 90" in formula
            assert "DebitSinceDt" in formula or "DEBITSINCEDT" in formula.upper()


def test_s02_file_date_sentinels_and_aqua_nulling():
    sql = _S02.read_text(encoding="utf-8", errors="replace")
    aqua_dates = {"INTNOTSERVICEDDT", "LASTCRDATE", "STOCKSTDT", "DEBITSINCEDT"}
    for column in _S02_DATE_COLS:
        row, debug = generate_for_sql(sql, "##ACCOUNTCAL", column, llm_client=None)
        formula = debug.get("formula") or row.display_derivation_expression or ""
        assert not row.validation_errors, f"{column}: {row.validation_errors}"
        _assert_formula_hygiene(formula, column)
        if column.upper() in aqua_dates:
            assert "Aqua_Scheme" in formula or "SchemeType" in formula, column
        if column.upper() != "DEBITSINCEDT":
            assert "1900" in formula, column


def test_s02_acl_completed_scoped_to_dpd_calculation():
    sql = _S02.read_text(encoding="utf-8", errors="replace")
    row, debug = generate_for_sql(
        sql, "ACLRUNNINGPROCESSSTATUS", "COMPLETED", llm_client=None
    )
    formula = debug.get("formula") or ""
    _assert_formula_hygiene(formula, "COMPLETED")
    assert "DPD_Calculation" in formula
    catch = row.exception_handler_expression or debug.get("exception_handler_formula") or ""
    assert catch
    assert "DPD_Calculation" in catch
    assert 'THEN("N")' in catch or "THEN(\"N\")" in catch


def test_s02_restructure_max_columns_skip_set_based_subquery_alias():
    sql = _S02.read_text(encoding="utf-8", errors="replace")
    for column in ("DPD_MaxFin", "DPD_MaxNonFin"):
        row, debug = generate_for_sql(
            sql, "PRO.AdvAcRestructureCal", column, llm_client=None
        )
        formula = debug.get("formula") or row.display_derivation_expression or ""
        _assert_formula_hygiene(formula, column)
        assert '"A"' not in formula, formula
        assert "ISEMPTY" in formula, formula
        if column.upper() == "DPD_MAXFIN":
            assert "DPD_MaxNonFin" not in formula, formula

