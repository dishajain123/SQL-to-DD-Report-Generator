"""Unit tests for Derivation Engine v2 (AST pipeline)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.phase1_lineage import build_lineage_map
from app.derivation.v2.phase2_mutation_folder import fold_column_mutations
from app.derivation.v2.phase3_ast_generator import (
    build_ast_from_mutations,
    parse_sql_expression_to_ast,
)
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = (
    ROOT
    / "samples"
    / "sql"
    / "v2_fixtures"
    / "PRO_Final_AssetClass_Npadate_MOC_StoredProcedure.sql"
)


def test_ast_compiler_numeric_decimal_literal_not_path():
    """Decimals must stay raw numbers, never ``\"1\".\"10\"`` paths."""
    node = {"type": "LITERAL", "value_type": "NUMBER", "value": "1.10"}
    assert compile_ast_to_4x_string(node) == "1.10"

    for value_type in ("NUMBER", "FLOAT", "INT", "DECIMAL"):
        assert compile_ast_to_4x_string(
            {"type": "LITERAL", "value_type": value_type, "value": "1.05"}
        ) == "1.05"

    assert compile_ast_to_4x_string(
        {"type": "LITERAL", "value_type": "NUMBER", "value": 1.10}
    ) in {"1.1", "1.10"}

    # STRING stays quoted
    assert (
        compile_ast_to_4x_string(
            {"type": "LITERAL", "value_type": "STRING", "value": "ALWYS_STD"}
        )
        == '"ALWYS_STD"'
    )

    # Mis-tagged COLUMN_REF from ``1.10`` must still compile as a number
    assert (
        compile_ast_to_4x_string(
            {
                "type": "COLUMN_REF",
                "entity": "1",
                "relationship": None,
                "column": "10",
            }
        )
        == "1.10"
    )

    # Parser must not turn decimals into COLUMN_REF
    parsed = parse_sql_expression_to_ast("1.10", default_entity="AccountCal")
    assert parsed["type"] == "LITERAL"
    assert parsed["value_type"] == "NUMBER"
    compiled = compile_ast_to_4x_string(parsed)
    assert compiled in {"1.1", "1.10"}
    assert '"' not in compiled
    assert compiled != '"1"."10"'


def test_ast_compiler_elseif_chain_validates():
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": {
            "type": "FUNCTION_CALL",
            "function_name": "ISEMPTY",
            "arguments": [
                {
                    "type": "COLUMN_REF",
                    "entity": "##AccountCal",
                    "relationship": None,
                    "column": "NPA_Date",
                }
            ],
        },
        "then_branch": {"type": "LITERAL", "value_type": "NUMBER", "value": 1},
        "else_branch": {
            "type": "IF_THEN_ELSE",
            "condition": {
                "type": "MEMBERSHIP_OP",
                "operator": "IN",
                "column": {
                    "type": "COLUMN_REF",
                    "entity": "##AccountCal",
                    "relationship": "##CUSTOMERCAL",
                    "column": "CustSegment",
                },
                "values": ["SMA1", "SMA2"],
            },
            "then_branch": {"type": "LITERAL", "value_type": "NUMBER", "value": 2},
            "else_branch": {
                "type": "COLUMN_REF",
                "entity": "##AccountCal",
                "relationship": None,
                "column": "AssetClassAlt_Key",
            },
        },
    }
    formula = compile_ast_to_4x_string(ast)
    assert "ELSEIF" in formula
    assert "ELSEIFIF" not in formula
    assert 'IN ["SMA1", "SMA2"]' in formula or 'IN ["SMA1","SMA2"]' in formula.replace(" ", "")
    result = validate_expression(formula)
    assert result.valid, result.error


def test_phase1_maps_local_temp_to_global_root():
    sql = FIXTURE.read_text(encoding="utf-8")
    lineage = build_lineage_map(sql)
    assert any(e.upper().endswith("ACCOUNTCAL") or e.startswith("##") for e in lineage.root_entities)
    # #AssetClassWork columns should resolve toward ##AccountCal
    keys = [k for k in lineage.columns if k.upper().startswith("#ASSETCLASSWORK.")]
    assert keys, "expected #AssetClassWork column mappings"
    sample = lineage.columns[keys[0]]
    assert "ACCOUNTCAL" in sample.entity.upper() or sample.entity.startswith("##")


def test_phase2_collects_final_asset_class_updates():
    sql = FIXTURE.read_text(encoding="utf-8")
    lineage = build_lineage_map(sql)
    mutations = fold_column_mutations(
        sql, "##AccountCal", "FinalAssetClassAlt_Key", lineage
    )
    assert len(mutations) >= 1
    assert any(m.where_clause for m in mutations) or len(mutations) >= 1


def test_join_to_plain_table_uses_its_real_name_as_relationship():
    """Regression: a JOIN to a plain physical/dimension table (no ## prefix
    in the source SQL, e.g. DimProduct) must keep that real name as the
    COLUMN_REF relationship -- not get a fabricated "##" prefix invented
    just because other joins in this codebase's sample corpus happen to
    target genuine ## global-temp entities."""
    sql = """
    UPDATE A
    SET A.ASSET_NORM = 'CONDI_STD'
    FROM ##ACCOUNTCAL A
    INNER JOIN DimProduct P ON P.ProductAlt_Key = A.ProductAlt_Key
    WHERE P.ProductGroup = 'FDSEC'
    """
    _, debug = generate_for_sql(sql, "##ACCOUNTCAL", "ASSET_NORM", llm_client=None)
    formula = debug["formula"]
    assert '"ACCOUNTCAL"."DimProduct"."ProductGroup"' in formula
    assert "##DimProduct" not in formula
    assert validate_expression(formula).valid


def test_join_to_global_temp_keeps_its_prefix_and_local_temp_traces_to_root():
    """A join to a genuine global temp table (##CUSTOMERCAL) keeps that
    ## spelling as the relationship. A join to a LOCAL temp table
    (#CustRisk) must NOT surface the throwaway local name at all -- Phase 1
    lineage traces it back to the real physical table it was built from
    (PRO.CustomerMaster) and that's what appears as the relationship."""
    sql_global = """
    UPDATE A
    SET A.ASSET_NORM = 'CONDI_STD'
    FROM ##ACCOUNTCAL A
    INNER JOIN ##CUSTOMERCAL B ON A.CustomerID = B.CustomerID
    WHERE B.RiskFlag = 'Y'
    """
    _, d_global = generate_for_sql(sql_global, "##ACCOUNTCAL", "ASSET_NORM", llm_client=None)
    assert '"ACCOUNTCAL"."##CUSTOMERCAL"."RiskFlag"' in d_global["formula"]
    assert validate_expression(d_global["formula"]).valid

    sql_local = """
    SELECT CustomerID, RiskFlag INTO #CustRisk FROM PRO.CustomerMaster WHERE Active = 'Y'

    UPDATE A
    SET A.ASSET_NORM = 'CONDI_STD'
    FROM ##ACCOUNTCAL A
    INNER JOIN #CustRisk C ON A.CustomerID = C.CustomerID
    WHERE C.RiskFlag = 'Y'
    """
    _, d_local = generate_for_sql(sql_local, "##ACCOUNTCAL", "ASSET_NORM", llm_client=None)
    assert '"ACCOUNTCAL"."CustomerMaster"."RiskFlag"' in d_local["formula"]
    assert "#CustRisk" not in d_local["formula"]
    assert validate_expression(d_local["formula"]).valid


def test_phase3_between_not_split_on_and():
    node = parse_sql_expression_to_ast(
        "AccountCal::DaysPastDue BETWEEN 61 AND 90",
        default_entity="AccountCal",
        as_condition=True,
    )
    assert node["type"] == "BINARY_OP"
    assert node["operator"] == "AND"
    assert node["left"]["operator"] == ">="
    assert node["right"]["operator"] == "<="
    formula = compile_ast_to_4x_string(node)
    assert "BETWEEN" not in formula
    assert validate_expression(formula).valid


def test_phase3_numeric_increment_is_binary_op_not_addday():
    """COUNT = ISNULL(COUNT,0)+1 must be BINARY_OP +, never ADDDAY."""
    from app.derivation.v2.phase3_ast_generator import _sanitize_addday_misuse

    ast = parse_sql_expression_to_ast(
        "ISNULL(COUNT, 0) + 1",
        default_entity="ACLRUNNINGPROCESSSTATUS",
        target_column="COUNT",
    )
    assert ast["type"] == "BINARY_OP"
    assert ast["operator"] == "+"
    assert ast["left"]["type"] == "FUNCTION_CALL"
    assert ast["left"]["function_name"] == "COALESCE"
    assert ast["right"]["type"] == "LITERAL"
    assert ast["right"]["value_type"] == "NUMBER"
    assert str(ast["right"]["value"]) in {"1", "1.0"}
    compiled = compile_ast_to_4x_string(ast)
    assert "ADDDAY" not in compiled
    assert "+" in compiled
    assert validate_expression(compiled).valid

    # LLM-style misuse: ADDDAY around a numeric COALESCE must be rewritten.
    bad = {
        "type": "FUNCTION_CALL",
        "function_name": "ADDDAY",
        "arguments": [
            {
                "type": "FUNCTION_CALL",
                "function_name": "COALESCE",
                "arguments": [
                    {
                        "type": "COLUMN_REF",
                        "entity": "ACLRUNNINGPROCESSSTATUS",
                        "relationship": None,
                        "column": "COUNT",
                    },
                    {"type": "LITERAL", "value_type": "NUMBER", "value": 0},
                ],
            },
            {"type": "LITERAL", "value_type": "NUMBER", "value": 1},
        ],
    }
    fixed = _sanitize_addday_misuse(bad, "COUNT")
    assert fixed["type"] == "BINARY_OP"
    assert fixed["operator"] == "+"
    assert "ADDDAY" not in compile_ast_to_4x_string(fixed)


def test_phase3_dateadd_maps_to_addday():
    ast = parse_sql_expression_to_ast(
        "DATEADD(DAY, 1, ProcessDate)",
        default_entity="AccountCal",
        target_column="ProcessDate",
    )
    assert ast["type"] == "FUNCTION_CALL"
    assert ast["function_name"] == "ADDDAY"
    assert compile_ast_to_4x_string(ast).startswith("ADDDAY(")


def test_phase3_is_null_maps_to_isempty():
    node = parse_sql_expression_to_ast(
        "W.NPA_Date IS NULL", default_entity="##AccountCal", as_condition=True
    )
    assert node["type"] == "FUNCTION_CALL"
    assert node["function_name"] == "ISEMPTY"


def test_phase3_outer_if_else_preserves_branch_order():
    """ELSE UPDATE must not wipe IF / ELSE IF arms for BucketWorsened."""
    sql = (ROOT / "samples" / "sql" / "07_DPD_Bucket_Classification.sql").read_text(
        encoding="utf-8"
    )
    lineage = build_lineage_map(sql)
    mutations = fold_column_mutations(sql, "LoanAccountCal", "BucketWorsened", lineage)
    assert len(mutations) >= 3
    assert {m.control_branch_kind for m in mutations} >= {"IF", "ELSEIF", "ELSE"}
    assert len({m.control_branch_group for m in mutations}) == 1

    row, debug = generate_for_sql(sql, "LoanAccountCal", "BucketWorsened")
    formula = debug["formula"]
    assert "ELSEIF" in formula
    assert formula.startswith("IF(")
    assert formula != '"N"'
    assert validate_expression(formula).valid
    # First IF arm (grace → 'N') appears before the worsened 'Y' arm.
    assert formula.index('THEN("N")') < formula.index('THEN("Y")')


def _grace_condition(**extra):
    return {
        "type": "BINARY_OP",
        "operator": ">=",
        "left": {
            "type": "COLUMN_REF",
            "entity": "LoanAccountCal",
            "relationship": None,
            "column": "LastPaymentDueDate",
        },
        "right": {"type": "VARIABLE_REF", "name": "@GraceWindowStart"},
        **extra,
    }


def test_prune_collapses_nested_identical_condition():
    from app.derivation.v2.phase2_mutation_folder import prune_redundant_ast

    self_ref = {
        "type": "COLUMN_REF",
        "entity": "LoanAccountCal",
        "relationship": None,
        "column": "BucketWorsened",
    }
    n_lit = {"type": "LITERAL", "value_type": "STRING", "value": "N"}
    node = {
        "type": "IF_THEN_ELSE",
        # Outer EXISTS guard carries projection metadata the WHERE lacks.
        "condition": _grace_condition(_dependency_refs=["LoanAccountCal.LastPaymentDueDate"]),
        "then_branch": {
            "type": "IF_THEN_ELSE",
            "condition": _grace_condition(),
            "then_branch": n_lit,
            "else_branch": self_ref,
        },
        "else_branch": self_ref,
    }
    pruned = prune_redundant_ast(node)
    assert pruned["then_branch"] == n_lit
    assert pruned["else_branch"] == self_ref

    # A different inner condition is a real guard and must survive.
    node["then_branch"]["condition"] = {**_grace_condition(), "operator": "<"}
    assert prune_redundant_ast(node)["then_branch"]["type"] == "IF_THEN_ELSE"


def test_bucket_worsened_has_no_nested_duplicate_guard():
    sql = (ROOT / "samples" / "sql" / "07_DPD_Bucket_Classification.sql").read_text(
        encoding="utf-8"
    )
    lineage = build_lineage_map(sql)
    mutations = fold_column_mutations(sql, "LoanAccountCal", "BucketWorsened", lineage)
    ast = build_ast_from_mutations(mutations, "LoanAccountCal", "BucketWorsened")

    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "IF_THEN_ELSE":
            inner = node.get("then_branch")
            if isinstance(inner, dict) and inner.get("type") == "IF_THEN_ELSE":
                strip = lambda c: {k: v for k, v in (c or {}).items() if not k.startswith("_")}
                assert strip(inner.get("condition")) != strip(node.get("condition"))
        for value in node.values():
            walk(value)

    walk(ast)
    _, debug = generate_for_sql(sql, "LoanAccountCal", "BucketWorsened")
    assert validate_expression(debug["formula"]).valid


def test_dpd_bucket_classification_verification_formulas():
    """Golden checks from the v2 bug-fix pass against sample 07."""
    sql = (ROOT / "samples" / "sql" / "07_DPD_Bucket_Classification.sql").read_text(
        encoding="utf-8"
    )

    _, adj = generate_for_sql(sql, "LoanAccountCal", "AdjustedPenalty")
    assert "* 1.1" in adj["formula"]
    assert '"1"."' not in adj["formula"]
    assert validate_expression(adj["formula"]).valid

    _, grace = generate_for_sql(sql, "LoanAccountCal", "GracePeriodApplied")
    g = grace["formula"]
    assert 'THEN("Y")' in g
    # The ELSEIF/ELSE arms of this IF/ELSE chain never assign
    # GracePeriodApplied at all, so those code paths must preserve the
    # column's existing value rather than null it out.
    assert 'ELSE("LoanAccountCal"."GracePeriodApplied")' in g
    assert '"@GraceWindowStart"' in g
    assert "LoanAccountCal\".\"@GraceWindowStart\"" not in g
    assert 'GracePeriodApplied" == "Y"' not in g
    assert validate_expression(g).valid

    _, count = generate_for_sql(sql, "ACLRUNNINGPROCESSSTATUS", "COUNT")
    c = count["formula"]
    # ACLRUNNINGPROCESSSTATUS is a shared table with one row per process
    # name; both the TRY and CATCH increments guard on
    # RUNNINGPROCESSNAME = 'DPD_Bucket_Classification', and that guard must
    # survive folding -- collapsing it away would increment every process's
    # counter, not just this one.
    assert 'RUNNINGPROCESSNAME" == "DPD_Bucket_Classification"' in c
    assert 'COALESCE("ACLRUNNINGPROCESSSTATUS"."COUNT", 0) + 1' in c
    assert 'ELSE("ACLRUNNINGPROCESSSTATUS"."COUNT")' in c
    assert validate_expression(c).valid


def test_ast_compiler_at_variable_and_numeric_literals():
    from app.derivation.v2.ast_compiler import compile_ast_to_4x_string

    assert (
        compile_ast_to_4x_string(
            {"type": "VARIABLE_REF", "name": "@GraceWindowStart"}
        )
        == '"@GraceWindowStart"'
    )
    assert (
        compile_ast_to_4x_string(
            {
                "type": "COLUMN_REF",
                "entity": "LoanAccountCal",
                "relationship": None,
                "column": "@ProcessDate",
            }
        )
        == '"@ProcessDate"'
    )
    assert (
        compile_ast_to_4x_string(
            {"type": "LITERAL", "value_type": "NUMBER", "value": "1.10"}
        )
        == "1.10"
    )
    assert (
        compile_ast_to_4x_string(
            {"type": "LITERAL", "value_type": "STRING", "value": "Y"}
        )
        == '"Y"'
    )


def test_end_to_end_final_asset_class_validates():
    sql = FIXTURE.read_text(encoding="utf-8")
    row, debug = generate_for_sql(
        sql,
        "##AccountCal",
        "FinalAssetClassAlt_Key",
        llm_client=None,
    )
    assert debug["mutations"], "expected UPDATE mutations for FinalAssetClassAlt_Key"
    assert debug["formula"]
    result = validate_expression(debug["formula"])
    assert result.valid, f"{result.error}\nformula={debug['formula']}\nast={debug['ast']}"
    assert row.entity_name
    assert row.column_name == "FinalAssetClassAlt_Key"
    assert row.display_derivation_expression == debug["formula"]
    assert "ELSEIF" in debug["formula"] or debug["formula"].startswith("IF(")


def test_sample_01_asset_class_generates_valid_expression():
    sql = (ROOT / "samples" / "sql" / "01_NPA_Classification_Simple.sql").read_text(
        encoding="utf-8"
    )
    row, debug = generate_for_sql(
        sql,
        "AccountCal",
        "AssetClass",
        llm_client=None,
    )
    assert debug["mutations"], "expected UPDATE mutations for AssetClass"
    assert debug["formula"]
    result = validate_expression(debug["formula"])
    assert result.valid, f"{result.error}\nformula={debug['formula']}"


def test_in_select_projects_predicate_as_complete_formula():
    """IN (SELECT …) should project WHERE + preserve deps as a complete formula."""
    from app.models.core import ReviewState

    sql = (ROOT / "samples" / "sql" / "05_NPA_Movement_Audit_Log.sql").read_text(
        encoding="utf-8"
    )
    row, debug = generate_for_sql(sql, "CustomerCal", "MultiAccountMovementFlag")
    formula = debug["formula"]
    assert validate_expression(formula).valid, formula
    assert "CustomerClassMovementHistory" in formula
    assert "TimeKey" in formula
    # Grammar-valid outputs are complete — not pending human review.
    assert row.review_state == ReviewState.GENERATED
    assert row.status.value == "ACTIVE" or str(row.status) == "ACTIVE"
    assert row.source_statement_sql, "raw SQL fragment must be preserved for audit"
    assert "MultiAccountMovementFlag" in row.source_statement_sql[0]
    meta = debug["metadata"]
    assert "Review Required" not in meta
    assert meta.get("Mutation SQL Fragments")


def test_not_in_subquery_negates_the_projected_predicate():
    """Regression: ``NOT IN (SELECT ...)`` and ``IN (SELECT ...)`` were
    generating the exact same condition -- the projected row predicate was
    reused as-is regardless of the NOT, silently dropping the negation."""
    in_node = parse_sql_expression_to_ast(
        "ID IN (SELECT ID FROM Rules WHERE Enabled = 1)",
        default_entity="Accounts",
        as_condition=True,
    )
    not_in_node = parse_sql_expression_to_ast(
        "ID NOT IN (SELECT ID FROM Rules WHERE Enabled = 1)",
        default_entity="Accounts",
        as_condition=True,
    )
    in_formula = compile_ast_to_4x_string(in_node)
    not_in_formula = compile_ast_to_4x_string(not_in_node)
    assert in_formula != not_in_formula
    assert not_in_formula.startswith("NOT(")
    assert validate_expression(in_formula).valid
    assert validate_expression(not_in_formula).valid


def test_merge_when_matched_creates_mutations():
    sql = (ROOT / "samples" / "sql" / "07_DPD_Bucket_Classification.sql").read_text(
        encoding="utf-8"
    )
    from app.derivation.v2.phase1_lineage import build_lineage_map
    from app.derivation.v2.phase2_mutation_folder import fold_column_mutations
    from app.derivation.v2.sql_text import extract_merge_matched_updates
    from app.models.core import ReviewState

    assert extract_merge_matched_updates(sql), "expected MERGE WHEN MATCHED in sample 07"
    lineage = build_lineage_map(sql)
    muts = fold_column_mutations(sql, "DpdBucketHistory", "DpdBucket", lineage)
    assert muts, "MERGE WHEN MATCHED should yield mutations for DpdBucket"
    assert any("MERGE" in (m.raw_sql or "").upper() for m in muts)

    row, debug = generate_for_sql(sql, "DpdBucketHistory", "DpdBucket")
    assert validate_expression(debug["formula"]).valid
    assert row.review_state == ReviewState.GENERATED
    assert row.source_statement_sql
    assert any("MERGE" in frag.upper() for frag in row.source_statement_sql)


def test_error_message_maps_to_variable_token():
    from app.models.core import ReviewState
    from app.derivation.v2.phase3_ast_generator import parse_sql_expression_to_ast
    from app.derivation.v2.ast_compiler import compile_ast_to_4x_string

    node = parse_sql_expression_to_ast(
        "ERROR_MESSAGE()", default_entity="RunStatus", target_column="ErrorDescription"
    )
    assert node.get("type") == "VARIABLE_REF"
    assert node.get("name") == "@ErrorMessage"
    formula = compile_ast_to_4x_string(node)
    assert formula == '"@ErrorMessage"'
    assert validate_expression(formula).valid

    sql = (ROOT / "samples" / "sql" / "01_NPA_Classification_Simple.sql").read_text(
        encoding="utf-8"
    )
    row, debug = generate_for_sql(sql, "RunStatus", "ErrorDescription")
    assert validate_expression(debug["formula"]).valid
    # Valid formula → GENERATED, not NEEDS_REVIEW
    assert row.review_state == ReviewState.GENERATED


def test_select_distinct_projection_parses_column_name():
    """``SELECT DISTINCT col`` must not capture DISTINCT as the column name."""
    from app.derivation.v2.sql_text import parse_select_list as _parse_select_list

    projections = _parse_select_list("DISTINCT UcifEntityID")
    assert len(projections) == 1
    src_qual, src_col, alias, raw = projections[0]
    assert src_col == "UcifEntityID"
    assert src_col != "DISTINCT"

    # TOP n and ALL are handled the same way.
    assert _parse_select_list("TOP 10 AccountId")[0][1] == "AccountId"
    assert _parse_select_list("TOP (5) AccountId")[0][1] == "AccountId"
    assert _parse_select_list("ALL AccountId")[0][1] == "AccountId"


def test_select_distinct_cte_lineage_resolves_correctly():
    """A CTE built from ``SELECT DISTINCT col`` must trace to the real column."""
    sql = """
    CREATE PROCEDURE PRO.Test_CTE_Distinct @TIMEKEY INT AS
    BEGIN
        ;WITH CTE_NPA_UCIFID_CUST AS (
            SELECT DISTINCT UcifEntityID FROM ##CUSTOMERCAL
            WHERE SysAssetClassAlt_Key > 1
            GROUP BY UcifEntityID
        )
        UPDATE A SET A.ASSET_NORM = 'CONDI_STD'
        FROM ##ACCOUNTCAL A
        INNER JOIN CTE_NPA_UCIFID_CUST B ON A.UcifEntityID = B.UcifEntityID
        WHERE A.ASSET_NORM = 'ALWYS_STD'
    END
    """
    lineage = build_lineage_map(sql)
    ref = lineage.resolve_column("CTE_NPA_UCIFID_CUST", "UcifEntityID")
    assert ref.column == "UcifEntityID"
    assert ref.column != "DISTINCT"
    assert "CUSTOMERCAL" in ref.entity.upper()


def test_chained_self_referential_passes_collapse_to_flat_or():
    """Regression: several ``UPDATE ... SET Col = 'X' WHERE Col = 'Y'`` passes
    (same guard, same assigned value, each restricted to a different
    JOIN/cohort population) must fold into one flat ``OR(...)`` guard instead
    of nesting each pass's own guard around the full accumulated prior AST.

    This is the exact shape from PRO.Final_AssetClass_Npadate: three
    consecutive UPDATEs set ##ACCOUNTCAL.ASSET_NORM = 'CONDI_STD' wherever it
    is currently 'ALWYS_STD', each gated by a different CTE/cohort join.
    Naive prior-value substitution nests an IF_THEN_ELSE as a comparison
    operand on every pass after the first, producing a formula that grows
    with every additional pass and previously failed Lark grammar
    validation -- which silently dropped the row from the DD export
    (app/report/dd_export.py's ``export_dd_rows`` filters out any row with
    ``validation_errors``).
    """
    sql = """
    ;WITH CTE_NPA_UCIFID AS (
        SELECT UcifEntityID FROM ##ACCOUNTCAL WHERE FinalAssetClassAlt_Key > 1 GROUP BY UcifEntityID
    )
    UPDATE A SET A.ASSET_NORM = 'CONDI_STD' FROM ##ACCOUNTCAL A
    INNER JOIN CTE_NPA_UCIFID B ON A.UcifEntityID = B.UcifEntityID
    WHERE ASSET_NORM = 'ALWYS_STD'

    ;WITH CTE_NPA_UCIFID_CUST AS (
        SELECT DISTINCT UcifEntityID FROM ##CUSTOMERCAL WHERE SysAssetClassAlt_Key > 1 GROUP BY UcifEntityID
    )
    UPDATE A SET A.ASSET_NORM = 'CONDI_STD' FROM ##ACCOUNTCAL A
    INNER JOIN CTE_NPA_UCIFID_CUST B ON A.UcifEntityID = B.UcifEntityID
    WHERE A.ASSET_NORM = 'ALWYS_STD'

    UPDATE A SET A.ASSET_NORM = 'CONDI_STD' FROM ##ACCOUNTCAL A
    INNER JOIN coborrowercal B ON A.UCIF_ID = B.UCIC
    WHERE A.ASSET_NORM = 'ALWYS_STD'
    """
    row, debug = generate_for_sql(sql, "##ACCOUNTCAL", "ASSET_NORM", llm_client=None)
    formula = debug["formula"]
    assert formula
    result = validate_expression(formula)
    assert result.valid, f"{result.error}\nformula={formula}"
    # Collapsed to a flat OR, not nested IF-as-comparison-operand passes --
    # a nested fold would repeat 'THEN("CONDI_STD")ELSE(' at least 3 times.
    assert formula.count('THEN("CONDI_STD")') == 1
    assert "OR(" in formula
    assert len(formula) < 1000
    assert row.display_derivation_expression == formula
    assert not row.validation_errors


def test_chained_self_referential_passes_with_different_values_do_not_collapse():
    """When chained self-referential passes assign DIFFERENT values, the
    flat-OR collapse must not apply (it would conflate two distinct
    outcomes) -- the general prior-value-substitution fold still runs, and
    the result must still validate."""
    sql = """
    UPDATE A SET A.ASSET_NORM = 'CONDI_STD' FROM ##ACCOUNTCAL A
    WHERE ASSET_NORM = 'ALWYS_STD'

    UPDATE A SET A.ASSET_NORM = 'OTHER_VALUE' FROM ##ACCOUNTCAL A
    WHERE ASSET_NORM = 'ALWYS_STD'
    """
    row, debug = generate_for_sql(sql, "##ACCOUNTCAL", "ASSET_NORM", llm_client=None)
    formula = debug["formula"]
    assert formula
    result = validate_expression(formula)
    assert result.valid, f"{result.error}\nformula={formula}"
    assert '"CONDI_STD"' in formula
    assert '"OTHER_VALUE"' in formula


def test_isempty_guarded_chain_collapses_to_elseif_priority_cascade():
    """Class B: a chain of ``UPDATE ... SET Col = 'literal' WHERE ... AND
    Col IS NULL`` passes, each assigning a DIFFERENT non-empty literal,
    must flatten into one priority-ordered ELSEIF cascade (earliest pass
    wins) instead of nesting the full accumulated tree into each guard.

    Sound because ISEMPTY(self) can only go true -> false once any pass in
    the chain writes a non-empty literal -- a later same-shaped guard can
    never re-match that row, so evaluating every guard against the
    ORIGINAL state (chronological order, first match wins) reproduces the
    real UPDATE-by-UPDATE execution exactly.
    """
    sql = """
    UPDATE A SET A.DegReason = 'REASON_ONE' FROM ##ACCOUNTCAL A
    WHERE A.FlgOne = 'Y' AND A.DegReason IS NULL

    UPDATE A SET A.DegReason = 'REASON_TWO' FROM ##ACCOUNTCAL A
    WHERE A.FlgTwo = 'Y' AND A.DegReason IS NULL
    """
    row, debug = generate_for_sql(sql, "##ACCOUNTCAL", "DegReason", llm_client=None)
    formula = debug["formula"]
    assert formula
    result = validate_expression(formula)
    assert result.valid, f"{result.error}\nformula={formula}"
    assert not row.validation_errors
    assert "ELSEIF" in formula
    # Earliest pass (REASON_ONE) must be the outer/first-checked arm.
    assert formula.index('"REASON_ONE"') < formula.index('"REASON_TWO"')
    # Flat cascade, not nested IF-as-comparison-operand: each guard's own
    # ISEMPTY check appears once per arm, not duplicated by substitution.
    assert formula.count("ISEMPTY(") == 2


def test_degreason_style_value_dependent_chain_does_not_collapse():
    """Class C: PRO.Final_AssetClass_Npadate's real DEGREASON shape --
    steps 1-2 share an ISEMPTY guard with DIFFERENT values (NULL then a
    literal, so Class B's non-empty-literal requirement rules out the
    first), step 3's guard reads a SPECIFIC VALUE step 2 just wrote (order
    -dependent, never flattenable), and steps 3 & 4 both assign the exact
    same COLUMN_REF (##CustomerCal.DegReason) -- which must NOT trigger
    Class A's flat OR, because a shared value only collapses safely when
    it's a literal (see ``_try_collapse_class_a``'s docstring); a shared
    column read has no such guarantee if anything else writes that source
    column in between.

    None of this should silently collapse -- the general prior-value
    substitution fold must still run for all of it, and the result must
    still be a valid, if larger, formula.
    """
    sql = """
    UPDATE A SET A.DegReason = NULL FROM ##ACCOUNTCAL A
    WHERE A.FlgDeg = 'Y' AND A.DegReason IS NULL

    UPDATE A SET A.DegReason = 'PERCOLATION BY OTHER ACCOUNT' FROM ##ACCOUNTCAL A
    WHERE A.FlgDeg = 'Y' AND A.DegReason IS NULL

    UPDATE A SET A.DegReason = B.DegReason FROM ##ACCOUNTCAL A
    INNER JOIN ##CustomerCal B ON A.SourceSystemCustomerID = B.SourceSystemCustomerID
    WHERE A.DegReason = 'PERCOLATION BY OTHER ACCOUNT' AND A.FlgDeg = 'N'

    UPDATE A SET A.DegReason = B.DegReason FROM ##ACCOUNTCAL A
    INNER JOIN ##CustomerCal B ON A.SourceSystemCustomerID = B.SourceSystemCustomerID
    WHERE A.FlgProcessing = 'N' AND A.FlgDeg = 'N' AND B.DegReason IS NOT NULL AND A.DegReason IS NULL
    """
    row, debug = generate_for_sql(sql, "##ACCOUNTCAL", "DegReason", llm_client=None)
    formula = debug["formula"]
    assert formula
    result = validate_expression(formula)
    assert result.valid, f"{result.error}\nformula={formula}"
    assert not row.validation_errors
    # Steps 3 & 4 both assign CustomerCal.DegReason. A wrongful Class A
    # collapse would merge them into one shared THEN (a single OR'd
    # condition), leaving only one THEN-position occurrence of that
    # relationship path; the correct, uncollapsed fold keeps each pass's
    # own copy, so it appears at least twice.
    assert formula.upper().count("CUSTOMERCAL") >= 2


def test_like_with_literal_pattern_maps_to_membership_op():
    """A fixed-literal LIKE pattern must map onto the grammar's native
    CONTAINS/BEGINSWITH/ENDSWITH — the 4X grammar has no LIKE token."""
    contains_node = parse_sql_expression_to_ast(
        "ReviewReason LIKE '%UNDERCOVER%'",
        default_entity="ProvisionCoverageSummary",
        as_condition=True,
    )
    assert contains_node["type"] == "MEMBERSHIP_OP"
    assert contains_node["operator"] == "CONTAINS"
    assert contains_node["values"] == ["UNDERCOVER"]
    formula = compile_ast_to_4x_string(contains_node)
    assert "CONTAINS" in formula
    assert validate_expression(formula).valid, formula

    begins_node = parse_sql_expression_to_ast(
        "Reason LIKE 'SEVERE%'", default_entity="CollectionsQueue", as_condition=True
    )
    assert begins_node["type"] == "MEMBERSHIP_OP"
    assert begins_node["operator"] == "BEGINSWITH"
    assert validate_expression(compile_ast_to_4x_string(begins_node)).valid

    ends_node = parse_sql_expression_to_ast(
        "Reason LIKE '%OVERDUE'", default_entity="CollectionsQueue", as_condition=True
    )
    assert ends_node["type"] == "MEMBERSHIP_OP"
    assert ends_node["operator"] == "ENDSWITH"
    assert validate_expression(compile_ast_to_4x_string(ends_node)).valid


def test_not_like_with_literal_pattern_maps_to_doesnotcontains():
    node = parse_sql_expression_to_ast(
        "ReviewReason NOT LIKE '%UNDERCOVER%'",
        default_entity="ProvisionCoverageSummary",
        as_condition=True,
    )
    assert node["type"] == "MEMBERSHIP_OP"
    assert node["operator"] == "DOESNOTCONTAINS"
    formula = compile_ast_to_4x_string(node)
    assert "DOESNOTCONTAINS" in formula
    assert validate_expression(formula).valid, formula


def test_three_operand_string_concatenation_folds_left_associatively():
    """Regression: a chain with more than one operator of the same kind
    (e.g. ``'%' + A.Col + '%'``, two '+' signs / three operands) must fold
    left-associatively into nested CONCAT calls -- not fall through to
    the final "give up" fallback that wraps the whole raw SQL text
    (quotes, column reference and all) as a single opaque STRING literal.

    T-SQL's string "+" has no 4X equivalent (+ is numeric-only there), so
    each "+" between text operands compiles to FUNCTION_CALL CONCAT."""
    node = parse_sql_expression_to_ast(
        "'%' + A.NPA_Reason + '%'",
        default_entity="CustomerCal",
    )
    assert node["type"] == "FUNCTION_CALL"
    assert node["function_name"] == "CONCAT"
    # Outer call's right argument is the trailing '%' literal; its left
    # argument is itself the inner CONCAT('%', NPA_Reason) call.
    assert node["arguments"][1] == {"type": "LITERAL", "value_type": "STRING", "value": "%"}
    inner = node["arguments"][0]
    assert inner["type"] == "FUNCTION_CALL"
    assert inner["function_name"] == "CONCAT"
    assert inner["arguments"][1]["type"] == "COLUMN_REF"
    assert inner["arguments"][1]["column"] == "NPA_Reason"


def test_like_with_dynamic_pattern_falls_back_honestly():
    """A LIKE pattern built from column concatenation has no valid 4X
    representation — it must surface as a grammar-validation failure, not
    silently collapse into a wrong string literal."""
    node = parse_sql_expression_to_ast(
        "B.DegradeReason LIKE '%' + A.NPA_Reason + '%'",
        default_entity="CustomerCal",
        as_condition=True,
    )
    assert node["type"] == "FUNCTION_CALL"
    assert node["function_name"] == "__UNSUPPORTED_SQL__"
    assert "dynamic" in node["_validation_error"]
    import pytest
    with pytest.raises(ValueError, match="dynamic"):
        compile_ast_to_4x_string(node)



def test_npa_reason_staged_match_flag_compiles_without_dynamic_like():
    """PRO.Final_AssetClass_Npadate refactored shape: NPA_Reason uses a
    precomputed flag join, not LIKE '%' + column + '%' in the SET clause."""
    sql = """
    UPDATE A
    SET A.NPA_Reason = CASE WHEN M.Stg_NpaReason_MatchFlag = 1
        THEN A.NPA_Reason
        ELSE CONCAT(A.NPA_Reason, ',', B.DegradeReason) END
    FROM ##ACCOUNTCAL A
    INNER JOIN (SELECT DISTINCT CUSTOMERACID, DegradeReason FROM PRO.CoBorrowerCal) B
        ON A.CustomerAcID = B.CustomerACID
    INNER JOIN #NPA_Reason_Match M ON A.CustomerAcID = M.CustomerAcID
    WHERE B.PERC_FinalAssetClass_AltKey > 1
    """
    row, debug = generate_for_sql(sql, "##ACCOUNTCAL", "NPA_Reason", llm_client=None)
    formula = debug["formula"]
    assert formula
    result = validate_expression(formula)
    assert result.valid, f"{result.error}\nformula={formula}"
    assert not row.validation_errors
    assert "LIKE" not in formula.upper()
    assert "CONCAT" in formula or "concat" in formula.lower()


def test_membership_op_preserves_contains_operator_in_compiler():
    """Regression: the compiler used to force any non-IN/NOTIN operator
    back to bare IN, silently discarding CONTAINS/BEGINSWITH/etc."""
    node = {
        "type": "MEMBERSHIP_OP",
        "operator": "CONTAINS",
        "column": {
            "type": "COLUMN_REF",
            "entity": "CollectionsQueue",
            "relationship": None,
            "column": "Reason",
        },
        "values": ["OVERDUE"],
    }
    formula = compile_ast_to_4x_string(node)
    assert "CONTAINS" in formula
    assert formula != '"CollectionsQueue"."Reason" IN ["OVERDUE"]'
    assert validate_expression(formula).valid


def test_membership_op_value_that_is_a_column_ref_node_compiles_recursively():
    """Regression: a MEMBERSHIP_OP whose ``values`` list contains an
    uncompiled AST node dict (e.g. from a JSON-authored/LLM AST, or an
    ``IN (col, 'lit')`` mixed list) must compile that node through the real
    compiler, not fall through to str(dict) and quote the resulting Python
    repr as a bogus string literal."""
    node = {
        "type": "MEMBERSHIP_OP",
        "operator": "IN",
        "column": {
            "type": "COLUMN_REF",
            "entity": "CoBorrowerCal",
            "relationship": None,
            "column": "DegradeReason",
        },
        "values": [
            {
                "type": "COLUMN_REF",
                "entity": "ACCOUNTCAL",
                "relationship": None,
                "column": "NPA_Reason",
            },
            "CO_OBLIGANT",
        ],
    }
    formula = compile_ast_to_4x_string(node)
    assert "{'type':" not in formula
    assert '"ACCOUNTCAL"."NPA_Reason"' in formula
    assert formula == '"CoBorrowerCal"."DegradeReason" IN ["ACCOUNTCAL"."NPA_Reason", "CO_OBLIGANT"]'


def test_plain_scalar_subquery_resolves_to_column_ref():
    """A non-aggregate scalar lookup subquery must resolve to the real
    target column instead of collapsing into a STRING literal of the raw
    SQL text."""
    node = parse_sql_expression_to_ast(
        "(SELECT AssetClassAlt_Key FROM DimAssetClass "
        "WHERE AssetClassShortName='STD' AND EffectiveFromTimeKey<=@TIMEKEY "
        "AND EffectiveToTimeKey>=@TIMEKEY)",
        default_entity="CustomerCal",
        target_column="FinalAssetClassAlt_Key",
    )
    assert node["type"] == "COLUMN_REF"
    assert node["entity"] == "DimAssetClass"
    assert node["column"] == "AssetClassAlt_Key"
    formula = compile_ast_to_4x_string(node)
    assert formula == '"DimAssetClass"."AssetClassAlt_Key"'
    assert validate_expression(formula).valid


def test_unresolvable_exists_subquery_raises_instead_of_always_true():
    """Regression: an EXISTS(...) whose subquery can't be projected into a
    row-level predicate used to silently fall back to a grammar-valid
    ``1 == 1`` tautology -- broadening eligibility to every row instead of
    surfacing that the guard couldn't be resolved. It must now raise at
    compile time so the pipeline records a validation error."""
    node = parse_sql_expression_to_ast(
        "EXISTS (SELECT COUNT(*) FROM Rules GROUP BY RuleType HAVING COUNT(*) > 5)",
        default_entity="Accounts",
        as_condition=True,
    )
    assert node["type"] == "FUNCTION_CALL"
    assert node["function_name"] == "__UNRESOLVED_SUBQUERY_PREDICATE__"
    with pytest.raises(ValueError, match="EXISTS/IN subquery"):
        compile_ast_to_4x_string(node)


def test_plain_scalar_subquery_inside_case_else_branch():
    """The lookup subquery shape from PRO.Final_AssetClass_Npadate: a CASE
    ELSE branch falling back to a DimAssetClass key lookup."""
    case_sql = (
        "CASE WHEN A.Asset_Norm <> 'ALWYS_STD' THEN A.SysAssetClassAlt_Key "
        "ELSE (SELECT AssetClassAlt_Key FROM DimAssetClass "
        "WHERE AssetClassShortName='STD' AND EffectiveFromTimeKey<=@TIMEKEY "
        "AND EffectiveToTimeKey>=@TIMEKEY) END"
    )
    node = parse_sql_expression_to_ast(
        case_sql, default_entity="AccountCal", target_column="FinalAssetClassAlt_Key"
    )
    assert node["type"] == "IF_THEN_ELSE"
    assert node["else_branch"]["type"] == "COLUMN_REF"
    assert node["else_branch"]["entity"] == "DimAssetClass"
    formula = compile_ast_to_4x_string(node)
    assert '"DimAssetClass"."AssetClassAlt_Key"' in formula
    assert validate_expression(formula).valid, formula


@pytest.mark.parametrize(
    "sql,expected",
    [
        ("SUBSTRING(Code, 1, 3)", 'SUBSTR("AccountCal"."Code", 1, 3)'),
        ("TRIM(Code)", 'TRIM("AccountCal"."Code")'),
        ("LTRIM(Code)", 'TRIM("AccountCal"."Code")'),
        ("RTRIM(Code)", 'TRIM("AccountCal"."Code")'),
        ("REPLACE(Code, ',', ' ')", 'REPLACE("AccountCal"."Code", ",", " ")'),
        ("FLOOR(Balance)", 'FLOOR("AccountCal"."Balance")'),
        ("CEILING(Balance)", 'CEIL("AccountCal"."Balance")'),
        ("CAST(Balance AS DECIMAL(18, 4))", 'CONVERT("AccountCal"."Balance", "DECIMAL(18,4)")'),
        ("CONVERT(VARCHAR(10), Code)", 'CONVERT("AccountCal"."Code", "VARCHAR(10)")'),
        ("CONVERT(VARCHAR(10), OpenDate, 112)", 'CONVERT("AccountCal"."OpenDate", "VARCHAR(10)")'),
    ],
)
def test_tsql_string_and_math_functions_map_to_4x_equivalents(sql, expected):
    node = parse_sql_expression_to_ast(sql, default_entity="AccountCal")
    formula = compile_ast_to_4x_string(node)
    assert formula == expected
    assert validate_expression(formula).valid, formula


def test_4x_reference_function_conformance():
    """Every documented 4X function, AND/OR function form and membership
    operator must compile from an AST node into grammar-valid 4X syntax."""
    import re

    doc = (ROOT / "samples" / "platform_docs" / "4x_functions_operators.md").read_text(
        encoding="utf-8"
    )
    documented_functions = set(re.findall(r"`([A-Z]+)\(", doc))
    documented_membership_ops = set(re.findall(r"> ([A-Z]+) \[ListOfValues\]", doc))

    def col(name):
        return {"type": "COLUMN_REF", "entity": "AccountCal", "relationship": None, "column": name}

    def num(value):
        return {"type": "LITERAL", "value_type": "NUMBER", "value": value}

    def text(value):
        return {"type": "LITERAL", "value_type": "STRING", "value": value}

    def call(name, *args):
        return {"type": "FUNCTION_CALL", "function_name": name, "arguments": list(args)}

    def compare(op, left, right):
        return {"type": "BINARY_OP", "operator": op, "left": left, "right": right}

    # Argument shapes follow the documented signatures.
    function_cases = {
        "SUBSTR": call("SUBSTR", col("Code"), num(1), num(3)),
        "LOWER": call("LOWER", col("Code")),
        "UPPER": call("UPPER", col("Code")),
        "LEN": call("LEN", col("Code")),
        "CONVERT": call("CONVERT", col("Balance"), text("DECIMAL(18,2)")),
        "CONCAT": call("CONCAT", col("Code"), text("-"), col("SubCode")),
        "TRIM": call("TRIM", col("Code")),
        "REPLACE": call("REPLACE", col("Code"), text("-"), text(" ")),
        "SOM": call("SOM", col("OpenDate")),
        "EOM": call("EOM", col("OpenDate")),
        "DATEDIFF": call("DATEDIFF", col("OpenDate"), col("CloseDate"), text("DAY")),
        "TODATE": call("TODATE", text("2024-01-01")),
        "ADDDAY": call("ADDDAY", col("OpenDate"), num(30)),
        "ISEMPTY": call("ISEMPTY", col("Code")),
        "ISNOTEMPTY": call("ISNOTEMPTY", col("Code")),
        "MAX": call("MAX", col("Balance"), {"type": "LIST_LITERAL", "items": ["UCIF_ID"]}),
        "MIN": call("MIN", col("Balance"), {"type": "LIST_LITERAL", "items": ["UCIF_ID"]}),
        "ROUND": call("ROUND", col("Balance"), num(2)),
        "ABS": call("ABS", col("Balance")),
        "FLOOR": call("FLOOR", col("Balance"), num(1)),
        "CEIL": call("CEIL", col("Balance"), num(1)),
        "COALESCE": call("COALESCE", col("Balance"), num(0)),
        "SUM": call("SUM", col("Balance")),
        "COUNT": call("COUNT", num(1)),
    }
    assert not set(function_cases) - documented_functions, (
        "test covers functions missing from the 4X reference: "
        f"{sorted(set(function_cases) - documented_functions)}"
    )

    dpd_over_90 = compare(">", col("DPD"), num(90))
    is_active = compare("==", col("Status"), text("ACTIVE"))
    logical_cases = {
        "AND": compare("AND", dpd_over_90, is_active),
        "OR": compare("OR", dpd_over_90, compare("OR", is_active, call("ISEMPTY", col("Code")))),
    }

    membership_ops = ["IN", "NOTIN", "CONTAINS", "BEGINSWITH", "ENDSWITH", "DOESNOTCONTAINS"]
    assert set(membership_ops) <= documented_membership_ops
    membership_cases = {
        op: {"type": "MEMBERSHIP_OP", "operator": op, "column": col("Code"), "values": ["SMA1", "SMA2"]}
        for op in membership_ops
    }

    failures = []
    for name, node in function_cases.items():
        formula = compile_ast_to_4x_string(node)
        if not formula.startswith(f"{name}("):
            failures.append(f"{name}: compiled to unexpected shape {formula}")
        result = validate_expression(formula)
        if not result.valid:
            failures.append(f"{name}: {formula} -> {result.error}")
    for name, node in logical_cases.items():
        formula = compile_ast_to_4x_string(node)
        if not formula.startswith(f"{name}("):
            failures.append(f"{name}: expected function form, got {formula}")
        result = validate_expression(formula)
        if not result.valid:
            failures.append(f"{name}: {formula} -> {result.error}")
    for op, node in membership_cases.items():
        formula = compile_ast_to_4x_string(node)
        if f" {op} [" not in formula:
            failures.append(f"{op}: operator not preserved in {formula}")
        result = validate_expression(formula)
        if not result.valid:
            failures.append(f"{op}: {formula} -> {result.error}")
    assert not failures, "\n".join(failures)


def test_cast_to_date_still_feeds_addday_heuristic():
    node = parse_sql_expression_to_ast(
        "CAST(OpenDate AS DATE) + 30", default_entity="AccountCal"
    )
    assert node["function_name"] == "ADDDAY"
    assert node["arguments"][0]["function_name"] == "CONVERT"


def test_from_join_clause_handles_outer_joins_hints_and_three_part_names():
    from app.derivation.v2.sql_text import parse_from_join_clause

    parsed = parse_from_join_clause(
        "FROM DEMO.PRO.LoanAccount A WITH (NOLOCK) "
        "LEFT OUTER JOIN PRO.CustomerMaster B WITH (NOLOCK) ON A.CustomerId = B.CustomerId "
        "FULL OUTER JOIN PRO.Branch C ON C.BranchId = A.BranchId "
        "WHERE B.IsNpa = 'Y'"
    )
    assert parsed == [
        ("LoanAccount", "A", None),
        ("CustomerMaster", "B", "A.CustomerId = B.CustomerId"),
        ("Branch", "C", "C.BranchId = A.BranchId"),
    ]
    # A table hint without an alias must not become the alias "WITH".
    assert parse_from_join_clause("FROM PRO.T WITH (NOLOCK)") == [("T", None, None)]


def test_update_join_keys_feed_lineage_and_step_context():
    sql = """
    UPDATE A
    SET A.AssetClass = B.NewClass
    FROM PRO.LoanAccount A
    INNER JOIN PRO.CustomerMaster B ON A.CustomerId = B.CustomerId
    WHERE B.IsNpa = 'Y'
    """
    row, debug = generate_for_sql(sql, "LoanAccount", "AssetClass")
    assert validate_expression(debug["formula"]).valid
    assert '"LoanAccount"."CustomerMaster"."NewClass"' in debug["formula"]
    refs = debug["mutations"][0]["dependency_refs"]
    assert "LoanAccount.CustomerId" in refs and "CustomerMaster.CustomerId" in refs
    assert row.execution_steps[0].join_conditions == [
        "JOIN CustomerMaster ON LoanAccount.CustomerId = CustomerMaster.CustomerId"
    ]


@pytest.mark.parametrize(
    "reset", ["TRUNCATE TABLE #Stage", "DELETE FROM #Stage", "DELETE #Stage"]
)
def test_table_reset_starts_a_fresh_derivation_pass(reset):
    sql = f"""
    INSERT INTO #Stage (AccountId, Bucket) SELECT AccountId, 'OLD' FROM PRO.Acct
    UPDATE S SET S.Bucket = 'STALE' FROM #Stage S WHERE S.AccountId = '1'
    {reset}
    INSERT INTO #Stage (AccountId, Bucket) SELECT AccountId, 'NEW' FROM PRO.Acct WHERE Active = 'Y'
    TRUNCATE TABLE #Stage
    """
    row, debug = generate_for_sql(sql, "Stage", "Bucket")
    assert "OLD" not in debug["formula"] and "STALE" not in debug["formula"]
    assert '"NEW"' in debug["formula"]
    assert len(debug["mutations"]) == 1
    # The trailing cleanup TRUNCATE (after the last write) does not wipe it.
    assert any("cleared the table" in n for n in row.execution_steps[0].notes)


def test_filtered_delete_and_other_table_resets_do_not_reset_the_target():
    from app.derivation.v2.sql_text import extract_table_resets

    sql = """
    INSERT INTO #Stage (Bucket) SELECT 'OLD' FROM PRO.Acct
    DELETE FROM #Stage WHERE Bucket = 'X'
    DELETE S FROM #Stage S INNER JOIN PRO.Acct A ON A.Id = S.Id
    TRUNCATE TABLE #Other
    MERGE PRO.T AS T USING #Stage AS S ON T.Id = S.Id WHEN MATCHED THEN DELETE;
    UPDATE S SET S.Bucket = 'NEW' FROM #Stage S WHERE S.Id = 1
    """
    assert [r["table"] for r in extract_table_resets(sql)] == ["#Other"]
    _, debug = generate_for_sql(sql, "Stage", "Bucket")
    assert len(debug["mutations"]) == 2


def _sample_07() -> str:
    return (ROOT / "samples" / "sql" / "07_DPD_Bucket_Classification.sql").read_text(
        encoding="utf-8"
    )


def test_catch_handler_is_not_folded_into_the_main_formula():
    """TRY and CATCH updates guard on the same predicate; folding them as two
    sequential UPDATEs produced IF(p)THEN(catch)ELSEIF(p)THEN(try) — an
    unreachable success arm. The CATCH path must be kept separate."""
    sql = _sample_07()
    row, debug = generate_for_sql(sql, "ACLRUNNINGPROCESSSTATUS", "COMPLETED")
    formula = debug["formula"]
    assert formula == (
        'IF("ACLRUNNINGPROCESSSTATUS"."RUNNINGPROCESSNAME" == "DPD_Bucket_Classification")'
        'THEN("Y")ELSE("ACLRUNNINGPROCESSSTATUS"."COMPLETED")'
    )
    assert 'THEN("N")' in row.exception_handler_expression
    assert [s.scope for s in row.execution_steps] == ["Main", "Exception handler (CATCH)"]

    _, count = generate_for_sql(sql, "ACLRUNNINGPROCESSSTATUS", "COUNT")
    assert "COALESCE(IF(" not in count["formula"]
    assert "ELSEIF" not in count["formula"]
    assert validate_expression(count["formula"]).valid


def test_procedure_wide_exists_gate_stays_out_of_the_platform_formula():
    """IF EXISTS(...) gates are run-level and have no platform equivalent. The
    exported formula must use only real columns / parameters (row-level
    projection of each branch); the gate itself goes to procedural context."""
    sql = _sample_07()
    row, debug = generate_for_sql(sql, "LoanAccountCal", "BucketWorsened")
    formula = debug["formula"]
    assert "WorkflowGate" not in formula and "Gate " not in formula
    assert formula.startswith('IF("LoanAccountCal"."LastPaymentDueDate" >= "@GraceWindowStart")')
    assert formula.endswith('ELSE("N")')
    assert validate_expression(formula).valid
    assert [g.split(" := ")[0] for g in row.workflow_gates] == ["Gate 1", "Gate 2"]
    assert all("EXISTS" in gate for gate in row.workflow_gates)
    assert any(note.startswith("Procedural context:") for note in row.advisory_notes)

    _, grace = generate_for_sql(sql, "LoanAccountCal", "GracePeriodApplied")
    assert "Gate" not in grace["formula"]

    # Scalar variable conditions are already run-level; they stay inline.
    scalar_sql = """
    IF @TimeKey > 26267
    BEGIN
        UPDATE A SET A.Flag = 'Y' FROM PRO.AccountCal A WHERE A.Balance > 0
    END
    """
    scalar_row, scalar = generate_for_sql(scalar_sql, "AccountCal", "Flag")
    assert "@TimeKey" in scalar["formula"]
    assert not scalar_row.workflow_gates


def test_overwritten_not_applicable_bucket_is_reported_not_rewritten():
    """Step 2 sets DpdBucket='NOT_APPLICABLE' AND DpdDays=0 for NULL due
    dates; step 3 (WHERE DpdDays IS NOT NULL) then re-matches those rows and
    assigns 'CURRENT'. The formula must show the executed result and the
    overwrite must be surfaced, not silently hidden in a dead ELSEIF arm."""
    sql = _sample_07()
    row, debug = generate_for_sql(sql, "LoanAccountCal", "DpdBucket")
    assert validate_expression(debug["formula"]).valid
    steps = row.execution_steps
    assert [s.assigned_value for s in steps][0] == '"NOT_APPLICABLE"'
    assert len(steps) == 2
    assert steps[0].notes and "Overwritten by step 2" in steps[0].notes[0]
    assert '"CURRENT"' in steps[0].notes[0]
    assert "DpdDays = 0" in steps[0].notes[0]
    assert any("NOT_APPLICABLE" in note and "CURRENT" in note for note in row.advisory_notes)

    # A later step that does not re-match the earlier rows is not an overwrite.
    days_row, _ = generate_for_sql(sql, "LoanAccountCal", "DpdDays")
    assert not any(step.notes for step in days_row.execution_steps)


@pytest.mark.parametrize(
    "entity,column,expected",
    [
        ("LoanAccountCal", "DpdBucket", "String"),
        ("LoanAccountCal", "DpdDays", "Integer"),
        ("LoanAccountCal", "PenalInterestAmount", "Decimal"),
        ("LoanAccountCal", "BucketWorsened", "String"),
        ("DpdStaging", "AdjustedPenalty", "Decimal"),
        ("DpdStaging", "DpdBucket", "String"),
        ("DpdBucketHistory", "LastUpdatedDate", "Date"),
        ("CollectionsQueue", "EscalationDate", "Date"),
        ("ACLRUNNINGPROCESSSTATUS", "COUNT", "Integer"),
        ("ACLRUNNINGPROCESSSTATUS", "ERRORDATE", "Date"),
    ],
)
def test_data_type_is_inferred_from_values_not_name_fragments(entity, column, expected):
    row, _ = generate_for_sql(_sample_07(), entity, column)
    assert row.data_type == expected


def test_export_reconciles_data_type_against_formula_outputs():
    from app.report.dd_export import reconcile_data_type

    bucket = 'IF("T"."DpdDays" == 0)THEN("CURRENT")ELSE("BUCKET_90_PLUS")'
    assert reconcile_data_type("Decimal", bucket, "DpdBucket") == "String"
    assert reconcile_data_type("Integer", 'IF("T"."X" > 1)THEN(0)ELSE(1)', "DpdDays") == "Integer"
    assert reconcile_data_type("String", 'IF("T"."X" > 1)THEN(1)ELSE(0)', "RetryCount") == "Integer"
    # A "@Var" token is a variable, not text; no literal outputs → unchanged.
    assert reconcile_data_type("Date", 'IF("T"."X" > 1)THEN("@ProcessDate")ELSE("T"."D")', "D") == "Date"


def test_report_and_qa_list_rules_in_execution_order(tmp_path):
    from datetime import date as _date

    from app.models.core import (
        CanonicalModel,
        ColumnType,
        DDRow,
        DDStatus,
        DerivationOption,
        Dialect,
        Intent,
        JobPlan,
        ObjectType,
        SQLObject,
    )
    from app.report.dd_export import write_qa_coverage_report
    from app.report.report_generator import generate_report

    def row(entity, column, order):
        return DDRow(
            entity_name=entity,
            column_name=column,
            column_type=ColumnType.PHYSICAL,
            derivation_option=DerivationOption.FORMULA_EXPRESSION,
            display_derivation_expression=f'IF("{entity}"."K" == 1)THEN("A")ELSE("{entity}"."{column}")',
            effective_start_date=_date(2026, 1, 1),
            status=DDStatus.ACTIVE,
            data_type="String",
            source_chain_id="c1",
            source_object_ids=["obj-1"],
            execution_order=order,
        )

    # Deliberately alphabetical and entity-grouped input; execution order differs.
    rows = [row("Alpha", "Late", 300), row("Zulu", "First", 10), row("Alpha", "Middle", 150)]
    model = CanonicalModel(chain_id="c1", job_id="j", object_ids=["obj-1"],
                           technical_summary="", business_summary="")
    plan = JobPlan(job_id="j", intent=Intent.GENERATE_DD, company="x", platform="4X")
    obj = SQLObject(object_id="obj-1", name="P", object_type=ObjectType.PROCEDURE,
                    dialect=Dialect.SQLSERVER, raw_sql="", source_file="p.sql")
    text = generate_report(plan, [model], rows, tmp_path / "report.md",
                           objects={"obj-1": obj}).read_text(encoding="utf-8")
    positions = [text.index(f"#### Determine {c} (") for c in ("First", "Middle", "Late")]
    assert positions == sorted(positions)
    assert "| 1 | [Determine First (Zulu)]" in text

    qa = write_qa_coverage_report(rows, tmp_path / "qa.md").read_text(encoding="utf-8")
    qa_positions = [qa.index(f"| {e} | {c} |") for e, c in
                    (("Zulu", "First"), ("Alpha", "Middle"), ("Alpha", "Late"))]
    assert qa_positions == sorted(qa_positions)
