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
    from app.derivation.v2.phase1_lineage import _parse_select_list

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
    left-associatively into nested BINARY_OP nodes -- not fall through to
    the final "give up" fallback that wraps the whole raw SQL text
    (quotes, column reference and all) as a single opaque STRING literal."""
    node = parse_sql_expression_to_ast(
        "'%' + A.NPA_Reason + '%'",
        default_entity="CustomerCal",
    )
    assert node["type"] == "BINARY_OP"
    assert node["operator"] == "+"
    # Outer node's right operand is the trailing '%' literal; its left
    # operand is itself the inner ('%' + A.NPA_Reason) BINARY_OP.
    assert node["right"] == {"type": "LITERAL", "value_type": "STRING", "value": "%"}
    inner = node["left"]
    assert inner["type"] == "BINARY_OP"
    assert inner["operator"] == "+"
    assert inner["right"]["type"] == "COLUMN_REF"
    assert inner["right"]["column"] == "NPA_Reason"


def test_like_with_dynamic_pattern_falls_back_honestly():
    """A LIKE pattern built from column concatenation has no valid 4X
    representation — it must surface as a grammar-validation failure, not
    silently collapse into a wrong string literal."""
    node = parse_sql_expression_to_ast(
        "B.DegradeReason LIKE '%' + A.NPA_Reason + '%'",
        default_entity="CustomerCal",
        as_condition=True,
    )
    assert node["type"] == "BINARY_OP"
    assert node["operator"] == "LIKE"
    # The LHS/RHS must still resolve to real column refs, not be swallowed
    # into a raw string blob.
    assert node["left"]["type"] == "COLUMN_REF"
    formula = compile_ast_to_4x_string(node)
    assert not validate_expression(formula).valid


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
