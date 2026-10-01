"""General regressions for the 12 failures in the supplied RBL procedure."""
import pytest

from app.models.core import Dialect, SQLObject, ObjectType
from app.parsing.sql_lex import mask_sql, normalize_comparison_spacing
from app.parsing.sql_parser import split_statements, parse_statement
from app.parsing.structural_analysis import analyze_object
from app.parsing.coverage_ledger import build_coverage_ledger
from app.parsing.write_inventory_scan import scan_expected_writes
from app.derivation.v2.sql_text import extract_update_statements, stripped_offset_to_line
from app.derivation.v2.pipeline import generate_for_sql


def parse_all(sql):
    return [parse_statement(s, i, Dialect.SQLSERVER)
            for i, s in enumerate(split_statements(sql, Dialect.SQLSERVER))]


# Preserve separate cases for every original failure, including the paired
# failures caused by cutting one nested comment into two invalid statements.
@pytest.mark.parametrize('sql, expected_types', [
    pytest.param("INSERT INTO ##ACCOUNTCAL (Balance) SELECT Balance FROM Accounts WHERE EndKey > = @TimeKey", ['INSERT'], id='46-insert-comparison'),
    pytest.param("SELECT PanNo INTO #Stage FROM Accounts GROUP BY PanNo\nALTER TABLE #Stage ADD SourceName VARCHAR(20)", ['SELECT', 'OTHER'], id='116-select-alter'),
    pytest.param("UPDATE ProcessMonitor SET Mode='COMPLETE'\n/* disabled\n/* nested */\nUPDATE Ghost SET X=1;\n*/", ['UPDATE'], id='247-nested-comment-opening'),
    pytest.param("/* disabled\n/* nested */\nINSERT INTO Ghost(X) SELECT 1\n*/\nINSERT INTO Monitor(X) SELECT 2", ['INSERT'], id='249-nested-comment-closing'),
    pytest.param("/* earlier comment\rline */\nSELECT CASE WHEN DueDt > ExtendedDt\nTHEN DueDt\nELSE ExtendedDt END DueDt\nINTO #Overdue FROM Facilities", ['SELECT'], id='281-case-after-cr-only-comment'),
    pytest.param("SELECT SecurityId INTO #Stock FROM Security WHERE StartKey < = @TimeKey", ['SELECT'], id='317-select-comparison'),
    pytest.param("UPDATE Accounts SET X=1 WHERE StartKey < = @TimeKey AND EndKey > = @TimeKey", ['UPDATE'], id='320-update-comparisons'),
    pytest.param("UPDATE A SET A.X=1 FROM Accounts A JOIN Security B ON B.Id=A.Id AND B.StartKey < = @TimeKey", ['UPDATE'], id='387-join-comparison'),
    pytest.param("INSERT INTO Monitor(X) SELECT 1\nEXEC PRO.RefreshAccounts", ['INSERT', 'OTHER'], id='392-insert-exec'),
    pytest.param("UPDATE Accounts SET DueDt=NULL\n/*\n/* seller update */\nUPDATE Ghost SET X=1\n*/", ['UPDATE'], id='405-nested-comment-opening'),
    pytest.param("/*\n/* seller update */\nUPDATE Ghost SET X=1\n*/\nUPDATE Accounts SET X=2", ['UPDATE'], id='407-nested-comment-closing'),
    pytest.param("INSERT INTO CoBorrower (Id) SELECT Id FROM SourceRows WHERE TimeKey=@TimeKey\nEXEC [dbo].[AddMissingAccounts]", ['INSERT', 'OTHER'], id='470-insert-exec'),
])
def test_each_original_failure_shape_parses_without_losing_live_statements(sql, expected_types):
    statements = parse_all(sql)
    assert [s.statement_type for s in statements] == expected_types
    assert all(s.parsed_ok for s in statements), [s.parse_error for s in statements]
    assert all('Ghost' not in s.tables_written for s in statements)


@pytest.mark.parametrize('separator', ['\r', '\n', '\r\n', '\v', '\f', '\x85', '\u2028', '\u2029'])
def test_mask_preserves_all_line_boundaries_and_offsets(separator):
    sql = f"/* outer{separator}/* nested */ end */\nSELECT 'a{separator}b''c', [odd]]name] FROM Accounts;"
    masked = mask_sql(sql)
    assert len(masked) == len(sql)
    assert len(masked.splitlines()) == len(sql.splitlines())
    assert [i for i,c in enumerate(sql) if c.isspace()] == [i for i,c in enumerate(masked) if sql[i].isspace() and c.isspace()]
    assert 'nested' not in masked
    assert 'FROM Accounts' in masked


def test_go_and_semicolons_inside_nested_comments_are_not_boundaries():
    sql = "UPDATE Accounts SET X=1\n/* outer\n/* inner */\nGO\nUPDATE Ghost SET X=2;\n*/\nUPDATE Accounts SET Y=3;"
    statements = parse_all(sql)
    assert [s.set_columns_by_table for s in statements] == [{'Accounts':['X']}, {'Accounts':['Y']}]
    assert [w.target_table for w in scan_expected_writes(sql)] == ['Accounts', 'Accounts']
    extracted = extract_update_statements(sql)
    assert len(extracted) == 2
    assert stripped_offset_to_line(sql, int(extracted[1]['start'])) == 7


def test_insert_exec_and_union_all_remain_single_inserts():
    sql = "INSERT INTO Accounts (X)\nEXEC dbo.ReadValues\nEXEC dbo.NextStep\nINSERT INTO Accounts (X)\nSELECT 1\nUNION ALL\nSELECT 2\nEXEC dbo.Done"
    parts = split_statements(sql, Dialect.SQLSERVER)
    assert len(parts) == 4
    assert 'ReadValues' in parts[0] and 'NextStep' not in parts[0]
    assert 'UNION ALL\nSELECT 2' in parts[2]
    assert 'Done' not in parts[2]


def test_normalization_changes_only_explicit_code_tokens_and_is_audited():
    sql = "UPDATE [a > = b] SET X='it''s < = text' /* > = untouched */ WHERE A > = 1 AND B <\t= 2 AND C < > 3"
    fixed, notes = normalize_comparison_spacing(sql)
    assert len(notes) == 3
    assert len(fixed) == len(sql)
    assert "[a > = b]" in fixed and "'it''s < = text'" in fixed and '/* > = untouched */' in fixed
    assert 'A >=  1' in fixed and 'B <=\t 2' in fixed and 'C <>  3' in fixed
    assert normalize_comparison_spacing("A > /* boundary */ = 1")[0] == "A > /* boundary */ = 1"
    assert normalize_comparison_spacing("A >\n= 1")[0] == "A >\n= 1"
    stmt = parse_all('UPDATE Accounts SET X=1 WHERE A > = 2')[0]
    assert stmt.raw_text.endswith('A > = 2')
    assert stmt.normalization_notes
    assert any('>=' in condition for condition in stmt.conditions)
    obj = SQLObject(object_id='obj', name='test', source_file='test.sql',
                    object_type=ObjectType.PROCEDURE, dialect=Dialect.SQLSERVER,
                    raw_sql=stmt.raw_text)
    ledger = build_coverage_ledger(analyze_object(obj), source_sql=obj.raw_sql)
    assert any('Comparison spacing normalized' in e for e in ledger.blockers)


def test_conditions_use_same_recovered_operator_as_structural_parser():
    row, debug = generate_for_sql('UPDATE Accounts SET X=1 WHERE Balance > = 10;', 'Accounts', 'X')
    assert '>=' in debug['mutations'][0]['where_clause']
    assert '>=' in row.display_derivation_expression


def test_genuine_invalid_sql_still_reports_a_parse_failure():
    bad = parse_all('UPDATE Accounts SET X = WHERE Id=1')[0]
    assert not bad.parsed_ok
    assert bad.parse_error
