"""Behavioral oracles: expected results follow SQL update semantics, not strings."""
from datetime import date
from pathlib import Path
import operator

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.derivation.v2.phase3_ast_generator import parse_sql_expression_to_ast
from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.phase4_metadata import extract_timekey_thresholds
from app.grammar.validator import validate_expression
from app.models.core import DDStatus

SAMPLES = Path(__file__).resolve().parents[3] / "samples/sql"


def evaluate(node, record):
    kind = node['type']
    if kind == 'LITERAL': return node['value']
    if kind == 'COLUMN_REF': return record[node['column'].lower()]
    if kind == 'VARIABLE_REF': return record[node['name'].lower()]
    if kind == 'IF_THEN_ELSE':
        return evaluate(node['then_branch'] if evaluate(node['condition'], record) else node['else_branch'], record)
    if kind == 'BINARY_OP':
        left = evaluate(node['left'], record); right = evaluate(node['right'], record)
        op = node['operator']
        if op == 'AND': return bool(left) and bool(right)
        if op == 'OR': return bool(left) or bool(right)
        if left is None or right is None: return None
        return {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv,
                '==': operator.eq, '!=': operator.ne, '>': operator.gt, '>=': operator.ge,
                '<': operator.lt, '<=': operator.le}[op](left, right)
    if kind == 'FUNCTION_CALL':
        args = [evaluate(a, record) for a in node['arguments']]
        if node['function_name'] == 'COALESCE': return next((v for v in args if v is not None), None)
        if node['function_name'] == 'ISEMPTY': return args[0] is None
        if node['function_name'] == 'ISNOTEMPTY': return args[0] is not None
    raise AssertionError(f'Unsupported oracle node: {node}')


def generated(sql, col='Flag'):
    return generate_for_sql(sql, 'AccountCal', col)


@pytest.mark.parametrize('dpd,initial,expected', [(89,5,5),(90,5,5),(91,5,1),(None,5,5)])
def test_conditional_update_preserves_unmatched_rows(dpd, initial, expected):
    _, debug = generated('UPDATE AccountCal SET Flag=1 WHERE DPD>90;')
    assert evaluate(debug['ast'], {'dpd':dpd,'flag':initial}) == expected


def test_repeated_unconditional_increments_are_not_deduplicated():
    _, debug = generated('UPDATE AccountCal SET Flag=Flag+1; UPDATE AccountCal SET Flag=Flag+1;')
    assert evaluate(debug['ast'], {'flag':5}) == 7


def test_mixed_write_order_is_source_order():
    row, debug = generated('INSERT INTO AccountCal(Flag) SELECT 1 FROM Source; UPDATE AccountCal SET Flag=2;')
    assert [m['operation'] for m in debug['mutations']] == ['INSERT','UPDATE']
    assert evaluate(debug['ast'], {'flag':5}) == 2
    assert row.status == DDStatus.ACTIVE


@pytest.mark.parametrize('mode,eligible,expected', [(1,0,8),(1,1,2),(0,0,9)])
def test_selected_if_arm_does_not_fall_into_else_when_where_is_false(mode,eligible,expected):
    sql = '''UPDATE AccountCal SET Flag=8;
    IF @Mode=1 BEGIN UPDATE AccountCal SET Flag=2 WHERE Eligible=1; END
    ELSE BEGIN UPDATE AccountCal SET Flag=9; END'''
    _, debug = generated(sql)
    assert evaluate(debug['ast'], {'@mode':mode,'eligible':eligible,'flag':77}) == expected


@pytest.mark.parametrize('expr,expected', [('10-3+2',9),('10/2*3',15),('(10-4)/2',3),('10-(3-2)',9)])
def test_arithmetic_order(expr,expected):
    ast = parse_sql_expression_to_ast(expr, default_entity='AccountCal')
    assert evaluate(ast,{}) == expected
    if expr == '(10-4)/2':
        assert compile_ast_to_4x_string(ast) == '(10 - 4) / 2'


@pytest.mark.parametrize('dpd,active,expected', [(70,1,True),(70,0,False),(91,1,False),(60,1,False)])
def test_between_with_other_predicates(dpd,active,expected):
    ast = parse_sql_expression_to_ast('Active=1 AND DPD BETWEEN 61 AND 90', default_entity='AccountCal', as_condition=True)
    assert bool(evaluate(ast, {'active':active,'dpd':dpd})) == expected


def test_comment_timekeys_are_not_effective_dates():
    assert not extract_timekey_thresholds('-- EXEC proc @TimeKey=25140\nWHERE A.EffectiveToTimeKey=49999')
    row, _ = generated('-- @TimeKey=25140\nUPDATE AccountCal SET Flag=1;')
    assert row.effective_start_date == date(1900,1,1)


@pytest.mark.parametrize('predicate', ['ID NOT IN (SELECT ID FROM Rules)', 'EXISTS (SELECT 1 FROM Rules WHERE Flag=1)'])
def test_subquery_predicate_ships_as_active_when_projected(predicate):
    row, debug = generated(f'UPDATE AccountCal SET Flag=1 WHERE {predicate};')
    assert validate_expression(debug['formula']).valid
    assert row.status == DDStatus.ACTIVE


def test_unknown_sql_is_not_quoted_as_a_valid_literal():
    node = parse_sql_expression_to_ast('x COLLATE Latin1_General_CI_AS', default_entity='AccountCal')
    with pytest.raises(ValueError, match='Untranslated SQL'):
        compile_ast_to_4x_string(node)


def test_no_mutation_is_visible_not_an_identity_success():
    row,_ = generated('UPDATE AccountCal SET OtherFlag=1;')
    assert row.status == DDStatus.PENDING_REVIEW
    assert any('No source assignment' in e for e in row.validation_errors)


@pytest.mark.parametrize('days,active,prior,expected', [(90,'ACTIVE','X','STANDARD'),(91,'ACTIVE','X','SUBSTANDARD'),
    (180,'ACTIVE','X','SUBSTANDARD'),(181,'ACTIVE','X','DOUBTFUL'),(365,'ACTIVE','X','DOUBTFUL'),
    (366,'ACTIVE','X','LOSS'),(None,'ACTIVE','X','LOSS'),(91,'CLOSED','X','X')])
def test_sample01_asset_class_boundaries(days,active,prior,expected):
    _, debug=generated((SAMPLES/'01_NPA_Classification_Simple.sql').read_text(),'AssetClass')
    assert evaluate(debug['ast'], {'dayspastdue':days,'accountstatus':active,'assetclass':prior}) == expected


@pytest.mark.parametrize('asset,secured,expected', [('STANDARD','N',.4),('SUBSTANDARD','N',25),
    ('DOUBTFUL','N',35),('LOSS','N',110),('DOUBTFUL','Y',25),('OTHER','N',10)])
def test_sample03_provision_loading_uses_calculated_base(asset,secured,expected):
    _, debug=generated((SAMPLES/'03_Provision_Percentage_Calculation.sql').read_text(),'ProvisionPct')
    assert evaluate(debug['ast'], {'assetclass':asset,'securedflag':secured,'provisionpct':999}) == expected


@pytest.mark.parametrize('balance,pct,prior,expected', [(100,25,999,25),(100,110,0,100),(0,25,50,0),(-1,25,50,-1)])
def test_sample03_cap_tests_new_provision_value(balance,pct,prior,expected):
    _, debug=generated((SAMPLES/'03_Provision_Percentage_Calculation.sql').read_text(),'ProvisionAmount')
    assert evaluate(debug['ast'], {'outstandingbalance':balance,'provisionpct':pct,'provisionamount':prior}) == expected
