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
    assert "ELSE(NULL)" in g
    assert '"@GraceWindowStart"' in g
    assert "LoanAccountCal\".\"@GraceWindowStart\"" not in g
    assert 'GracePeriodApplied" == "Y"' not in g
    assert validate_expression(g).valid

    _, count = generate_for_sql(sql, "ACLRUNNINGPROCESSSTATUS", "COUNT")
    c = count["formula"]
    assert c == 'COALESCE("ACLRUNNINGPROCESSSTATUS"."COUNT", 0) + 1' or (
        c.startswith("COALESCE(") and "+ 1" in c and "ISEMPTY" not in c and "ADDDAY" not in c
    )
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
