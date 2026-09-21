from app.derivation.dd_generator import (
    _assignment_sites,
    _compose_simple_assignment_expression,
    _repair_self_referential_guard,
    _split_outer_if_then_else,
)
from app.grammar.validator import validate_expression
from app.models.core import Dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object


def test_split_outer_if_then_else_extracts_the_three_parts():
    parts = _split_outer_if_then_else("IF(A > 1)THEN(2)ELSE(3)")
    assert parts == ("A > 1", "2", "3")


def test_split_outer_if_then_else_rejects_elseif_chains():
    # ELSEIF chains aren't the single-clamp shape this repair understands;
    # must not be misparsed into a guard/then/else triple.
    assert _split_outer_if_then_else("IF(A>1)THEN(2)ELSEIF(A>0)THEN(1)ELSE(0)") is None


def test_split_outer_if_then_else_rejects_trailing_content():
    assert _split_outer_if_then_else("IF(A>1)THEN(2)ELSE(3) AND TRUE") is None


def test_repair_self_referential_guard_substitutes_else_branch_into_guard():
    # The DPD_IntService shape from the real corpus: a trailing negative
    # clamp whose guard reads the column's own (stale, stored) value
    # instead of the value this same formula computes in its ELSE arm.
    expr = (
        'IF(COALESCE("AccountCal_Stg"."DPD_IntService", 0) < 0)'
        "THEN(0)"
        'ELSE(IF(ISNOTEMPTY("AccountCal_Stg"."IntNotServicedDt"))'
        'THEN(("AccountCal_Stg"."var"."BUSINESS_DATE" - "AccountCal_Stg"."IntNotServicedDt"))'
        "ELSE(0))"
    )
    repaired, blocked = _repair_self_referential_guard(expr, "AccountCal_Stg", "DPD_IntService")
    assert blocked is False
    assert '"AccountCal_Stg"."DPD_IntService"' not in repaired
    assert validate_expression(repaired).valid
    # The guard now tests the actual computed value, not a stale reference.
    assert repaired.startswith('IF(COALESCE((IF(ISNOTEMPTY("AccountCal_Stg"."IntNotServicedDt"))')


def test_repair_self_referential_guard_is_a_noop_without_a_self_reference():
    expr = 'IF("A"."X" > 0)THEN(1)ELSE(0)'
    repaired, blocked = _repair_self_referential_guard(expr, "A", "Y")
    assert blocked is False
    assert repaired == expr


def test_repair_self_referential_guard_blocks_when_else_branch_also_self_references():
    # Nothing safe to substitute if the branch we'd substitute is itself
    # circular -- the guard stays provably circular, so this must block
    # rather than guess or silently leave it in.
    expr = 'IF("A"."X" > 0)THEN(1)ELSE("A"."X")'
    repaired, blocked = _repair_self_referential_guard(expr, "A", "X")
    assert blocked is True


def test_repair_self_referential_guard_leaves_non_simple_shapes_untouched():
    # A self-reference inside a deep ELSEIF/nested chain (e.g. a genuinely
    # recursive-style lineage formula) isn't the single-clamp shape this
    # repair targets. Existing advisory-only handling for that broader
    # case is intentional elsewhere in the pipeline -- this repair must
    # not block it.
    expr = 'IF("A"."X" > 5)THEN(1)ELSEIF("A"."X" > 0)THEN(2)ELSE(0)'
    repaired, blocked = _repair_self_referential_guard(expr, "A", "X")
    assert blocked is False
    assert repaired == expr


def test_repair_self_referential_guard_leaves_branch_only_self_reference_untouched():
    # Self-reference inside a branch (not the guard) of an otherwise
    # simple IF/THEN/ELSE is also outside this repair's target shape.
    expr = 'IF("A"."Y" > 0)THEN("A"."X")ELSE(0)'
    repaired, blocked = _repair_self_referential_guard(expr, "A", "X")
    assert blocked is False
    assert repaired == expr


def test_dpd_intservice_composition_from_real_sample_has_no_self_reference(dpd_calculation_sql):
    # End-to-end reproduction (no LLM) against the real corpus sample: the
    # deterministic composer alone used to produce a self-referential guard
    # for DPD_IntService's trailing negative-value clamp.
    objs = split_objects(dpd_calculation_sql, "PRO_DPD_Calculation_StoredProcedure_2.sql", Dialect.ORACLE)
    obj = objs[0]
    info = analyze_object(obj)

    sites = _assignment_sites(info, "DPD_IntService", target_table=None)
    composed = _compose_simple_assignment_expression(
        sites, "AccountCal_Stg", "DPD_IntService", procedure_sql=obj.raw_sql
    )
    assert composed is not None

    repaired, blocked = _repair_self_referential_guard(composed, "AccountCal_Stg", "DPD_IntService")
    assert blocked is False
    assert '"AccountCal_Stg"."DPD_IntService"' not in repaired
    assert validate_expression(repaired).valid
