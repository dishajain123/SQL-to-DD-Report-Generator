"""Large-source regressions: bounded expansion, exact indexing, fail closed."""
import time
from dataclasses import asdict

import pytest

from app.derivation.v2.ast_limits import FormulaExpansionError, check_formula_expansion
from app.derivation.v2.phase1_lineage import build_lineage_map
from app.derivation.v2.phase2_mutation_folder import MutationSourceIndex, fold_column_mutations
from app.derivation.v2.pipeline import generate_for_sql
from app.guardrails.dd_row_coverage import source_statement_key
from app.guardrails.completeness import enforce_completeness
from app.models.core import DDStatus, Dialect, SQLObject
from app.parsing.structural_analysis import analyze_object


def test_exponential_shared_graph_is_detected_without_expanding():
    node = {"type": "COLUMN_REF", "entity": "T", "column": "X"}
    for _ in range(100):
        node = {"type": "BINARY_OP", "operator": "+", "left": node, "right": node}
    started = time.perf_counter()
    with pytest.raises(FormulaExpansionError, match="ordered execution steps"):
        check_formula_expansion(node)
    assert time.perf_counter() - started < 1


def test_long_self_dependent_chain_retains_all_evidence_and_steps():
    sql = '\n'.join(f'UPDATE T SET X = X + {i + 1} WHERE X < {i + 10};' for i in range(45))
    row, debug = generate_for_sql(sql, 'T', 'X')
    assert len(debug['mutations']) == 45
    assert len(row.source_statement_sql) == 45
    assert len(row.execution_steps) == 45
    assert row.status == DDStatus.PENDING_REVIEW
    assert any('Flat formula expansion' in e for e in row.validation_errors)
    assert not row.display_derivation_expression


def test_index_retains_every_assignment_order_and_rejects_wrong_source():
    sql = '''UPDATE T SET X = 1, Y = 2;
    IF @flag = 1 BEGIN UPDATE T SET X = X + 1 WHERE Y = 2; END;
    INSERT INTO T (X, Y) SELECT X, Y FROM S;
    UPDATE T SET Y = 3;'''
    index = MutationSourceIndex.build(sql)
    assert len(index.updates) == 3
    assert [item[0] for item in index.updates_by_column['X']] == [0, 1]
    for column in ['X', 'Y']:
        expected = fold_column_mutations(sql, 'T', column, build_lineage_map(sql))
        actual = fold_column_mutations(sql, 'T', column, build_lineage_map(sql), source_index=index)
        assert [asdict(m) for m in actual] == [asdict(m) for m in expected]
    with pytest.raises(ValueError, match='does not match'):
        fold_column_mutations(sql + ' ', 'T', 'X', build_lineage_map(sql), source_index=index)


def test_statement_identity_preserves_literal_values_and_token_boundaries():
    assert source_statement_key("UPDATE T SET X='a b';") != source_statement_key("UPDATE T SET X='AB';")
    assert source_statement_key("UPDATE T SET X='ab';") != source_statement_key("UPDATE T SET X='AB';")
    assert source_statement_key("UPDATE T SET X = 'a b'; --note") == source_statement_key("update t set x='a b'")
    assert source_statement_key('SELECT a b') != source_statement_key('SELECT ab')


def _object(sql):
    return SQLObject(object_id='test-object', source_file='test.sql', name='test',
                     object_type='PROCEDURE', dialect=Dialect.SQLSERVER, raw_sql=sql)


def test_missing_write_gates_valid_formula_and_retains_full_inventory():
    sql = 'UPDATE T SET X=1; UPDATE T SET Y=2;'
    obj = _object(sql)
    row, _ = generate_for_sql('UPDATE T SET X=1;', 'T', 'X', source_object_ids=[obj.object_id])
    evidence = enforce_completeness([row], {obj.object_id: obj}, {obj.object_id: analyze_object(obj)})
    assert not evidence['ready']
    assert row.status == DDStatus.PENDING_REVIEW
    assert row.display_derivation_expression == '1'
    assert len(evidence['objects'][obj.object_id]['writes']) == 2
    assert any('no validated DD row' in b for b in evidence['objects'][obj.object_id]['blockers'])


def test_fully_covered_simple_write_remains_active():
    sql = 'UPDATE T SET X=1;'
    obj = _object(sql)
    row, _ = generate_for_sql(sql, 'T', 'X', source_object_ids=[obj.object_id])
    evidence = enforce_completeness([row], {obj.object_id: obj}, {obj.object_id: analyze_object(obj)})
    assert evidence['ready']
    assert row.status == DDStatus.ACTIVE
