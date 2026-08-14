from app.derivation.period_pruning import prune_expression_for_period
from app.grammar.validator import validate_expression


def test_prunes_simple_threshold_to_selected_branch_when_condition_true():
    expr = 'IF("p_TIMEKEY" > 100)THEN(1)ELSE(0)'
    pruned = prune_expression_for_period(expr, "p_TIMEKEY", 101)
    assert pruned == "1"
    assert validate_expression(pruned).valid


def test_prunes_simple_threshold_to_selected_branch_when_condition_false():
    expr = 'IF("p_TIMEKEY" > 100)THEN(1)ELSE(0)'
    pruned = prune_expression_for_period(expr, "p_TIMEKEY", 50)
    assert pruned == "0"


def test_prunes_nested_thresholds_to_the_correct_deep_branch():
    expr = (
        'IF("p_TIMEKEY" > 100)THEN('
        'IF("p_TIMEKEY" > 200)THEN("deep_true")ELSE("mid_true")'
        ')ELSE("outer_false")'
    )
    assert prune_expression_for_period(expr, "p_TIMEKEY", 50) == '"outer_false"'
    assert prune_expression_for_period(expr, "p_TIMEKEY", 150) == '"mid_true"'
    assert prune_expression_for_period(expr, "p_TIMEKEY", 250) == '"deep_true"'


def test_prunes_elseif_chain_correctly():
    expr = (
        'IF("p_TIMEKEY" > 300)THEN("late")'
        'ELSEIF("p_TIMEKEY" > 200)THEN("mid")'
        'ELSEIF("p_TIMEKEY" > 100)THEN("early")'
        'ELSE("earliest")'
    )
    assert prune_expression_for_period(expr, "p_TIMEKEY", 50) == '"earliest"'
    assert prune_expression_for_period(expr, "p_TIMEKEY", 150) == '"early"'
    assert prune_expression_for_period(expr, "p_TIMEKEY", 250) == '"mid"'
    assert prune_expression_for_period(expr, "p_TIMEKEY", 350) == '"late"'


def test_leaves_unrelated_conditions_completely_untouched():
    expr = (
        'IF(ISNOTEMPTY("A"."X"))THEN('
        'IF("p_TIMEKEY" > 100)THEN(("A"."Y") + 1)ELSE(("A"."Y") + IF("A"."SourceAlt_Key" == 6)THEN(0)ELSE(1))'
        ')ELSE(0)'
    )
    before = prune_expression_for_period(expr, "p_TIMEKEY", 50)
    assert "SourceAlt_Key" in before
    assert 'ISNOTEMPTY("A"."X")' in before
    assert validate_expression(before).valid

    after = prune_expression_for_period(expr, "p_TIMEKEY", 150)
    assert 'ISNOTEMPTY("A"."X")' in after
    assert "SourceAlt_Key" not in after
    assert validate_expression(after).valid


def test_leaves_expression_untouched_when_condition_references_a_different_variable():
    expr = 'IF("p_OTHERKEY" > 100)THEN(1)ELSE(0)'
    assert prune_expression_for_period(expr, "p_TIMEKEY", 150) == expr


def test_leaves_expression_untouched_when_condition_mixes_variable_with_something_else():
    expr = 'IF(AND("p_TIMEKEY" > 100, "A"."FLAG" == "Y"))THEN(1)ELSE(0)'
    assert prune_expression_for_period(expr, "p_TIMEKEY", 150) == expr


def test_returns_original_on_unparseable_input():
    garbage = "not a valid 4X expression((("
    assert prune_expression_for_period(garbage, "p_TIMEKEY", 100) == garbage


def test_returns_original_when_nothing_prunable_exists():
    expr = 'IF(ISNOTEMPTY("A"."X"))THEN(1)ELSE(0)'
    assert prune_expression_for_period(expr, "p_TIMEKEY", 100) == expr