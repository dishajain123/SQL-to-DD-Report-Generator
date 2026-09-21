from app.derivation.dd_generator import (
    _assignment_sites,
    _compose_simple_assignment_expression,
    _find_snippet_in_procedure,
    _procedural_exclusive_branch_predicate,
)
from app.grammar.validator import validate_expression
from app.models.core import Dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object


def test_find_snippet_in_procedure_tolerates_collapsed_whitespace():
    # An assignment site's raw_sql can be reassembled from separately-
    # stripped statement fragments (an "ELSE" header folded onto the
    # UPDATE it guards), which collapses the original indentation between
    # them -- an exact substring search then wrongly reports "not found"
    # even though nothing about the underlying code changed.
    proc = "IF X THEN\n  NULL;\nELSE\n            UPDATE T SET a = 1;\nEND IF;"
    snippet = "ELSE\nUPDATE T SET a = 1;"  # indentation collapsed vs. proc
    assert proc.find(snippet) == -1  # exact search fails, as expected
    idx = _find_snippet_in_procedure(proc, snippet)
    assert idx >= 0
    assert proc[idx:].startswith("ELSE")


def test_oracle_bare_if_then_else_is_recognized_as_exclusive_branch():
    # Oracle PL/SQL has no BEGIN for an IF/ELSE arm (only THEN/END IF) --
    # the T-SQL-only BEGIN requirement previously meant this predicate was
    # never recognized for Oracle procedures at all.
    proc = (
        "IF p_TIMEKEY > 26267 THEN\n"
        "  UPDATE T SET a = 1;\n"
        "ELSE\n"
        "  UPDATE T SET a = 2;\n"
        "END IF;"
    )
    then_stage = "UPDATE T SET a = 1;"
    pred, idx, disqualified = _procedural_exclusive_branch_predicate(
        then_stage, proc, allow_opening_if=True
    )
    assert pred == "p_TIMEKEY > 26267"
    assert disqualified is False


def test_oracle_embedded_else_header_is_recognized():
    # The ELSE header is folded directly onto the front of the assignment
    # site's own raw_sql (CONTROL_FLOW_BLOCK), so it is no longer part of
    # the *preceding* text a backward window search would look at.
    proc = (
        "IF p_TIMEKEY > 26267 THEN\n"
        "  UPDATE T SET a = 1;\n"
        "ELSE\n"
        "            UPDATE T SET a = 2;\n"
        "END IF;"
    )
    else_stage = "ELSE\nUPDATE T SET a = 2;"
    pred, idx, disqualified = _procedural_exclusive_branch_predicate(
        else_stage, proc, search_from=0, allow_opening_if=False
    )
    assert pred == ""
    assert disqualified is False


def test_dpd_intservice_preserves_both_version_threshold_branches(dpd_calculation_sql):
    # End-to-end reproduction against the real corpus sample: the
    # deterministic composer used to silently discard the entire
    # `IF p_TIMEKEY > 26267 THEN ... ELSE ... END IF` THEN arm (including
    # the nested p_TIMEKEY > 26384 sub-branch and its +1/+2 day offsets),
    # composing only the ELSE arm as if there were no branching at all.
    objs = split_objects(dpd_calculation_sql, "PRO_DPD_Calculation_StoredProcedure_2.sql", Dialect.ORACLE)
    obj = objs[0]
    info = analyze_object(obj)

    sites = _assignment_sites(info, "DPD_IntService", target_table=None)
    composed = _compose_simple_assignment_expression(
        sites, "AccountCal_Stg", "DPD_IntService", procedure_sql=obj.raw_sql
    )

    assert composed is not None
    assert validate_expression(composed).valid
    # Both branches of the outer p_TIMEKEY > 26267 guard must be present.
    assert composed.startswith("IF(p_TIMEKEY > 26267)THEN(")
    # The nested p_TIMEKEY > 26384 sub-branch (with its own +1/+2 day
    # offsets) inside the THEN arm must survive composition too.
    assert "p_TIMEKEY > 26384" in composed
    assert "+ 2" in composed
    assert "+ 1" in composed
