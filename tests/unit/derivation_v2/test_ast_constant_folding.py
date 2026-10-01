"""Literal-only arithmetic and COALESCE(<literal>, …) compile to one constant."""
from datetime import date

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.versioning import resolve_timekey_to_date


def _num(value):
    return {"type": "LITERAL", "value_type": "NUMBER", "value": value}


def _col(entity, column):
    return {"type": "COLUMN_REF", "entity": entity, "relationship": None, "column": column}


def _coalesce(*args):
    return {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": list(args)}


def test_coalesce_literal_plus_one_folds_to_constant():
    node = {"type": "BINARY_OP", "operator": "+", "left": _coalesce(_num(0), _num(0)), "right": _num(1)}
    assert compile_ast_to_4x_string(node) == "1"


def test_coalesce_with_column_first_argument_is_not_folded():
    node = {"type": "BINARY_OP", "operator": "+", "left": _coalesce(_col("A", "COUNT"), _num(0)), "right": _num(1)}
    assert compile_ast_to_4x_string(node) == 'COALESCE("A"."COUNT", 0) + 1'


def test_division_is_never_folded():
    node = {"type": "BINARY_OP", "operator": "/", "left": _num(4), "right": _num(2)}
    assert compile_ast_to_4x_string(node) == "4 / 2"


def test_unmapped_timekey_resolves_to_plausible_year():
    # PRO.DPD_Calculation: `@TIMEKEY > 26267` is "IMPLEMENTED FROM 2021-12-01".
    resolved, is_real = resolve_timekey_to_date(26268)
    assert resolved == date(2021, 12, 1)
    assert is_real is False
