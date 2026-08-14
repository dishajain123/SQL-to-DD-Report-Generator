from app.models.core import Dialect
from app.parsing.sql_parser import classify_statement, parse_statement, split_statements


def test_split_statements_respects_nested_parens():
    sql = "UPDATE t SET x = (SELECT MAX(y) FROM u WHERE z = 1); UPDATE t2 SET a = 1;"
    stmts = split_statements(sql)
    assert len(stmts) == 2


def test_split_statements_respects_quoted_semicolons():
    sql = "UPDATE t SET note = 'a;b'; UPDATE t2 SET x = 1;"
    stmts = split_statements(sql)
    assert len(stmts) == 2
    assert "a;b" in stmts[0]


def test_split_statements_handles_block_comment_before_statement():
    sql = "/* a comment; with a semicolon */\nMERGE INTO t USING u ON (t.id = u.id) WHEN MATCHED THEN UPDATE SET t.x = 1;"
    stmts = split_statements(sql)
    assert len(stmts) == 1
    assert classify_statement(stmts[0]) == "MERGE"


def test_classify_statement_skips_leading_block_comment():
    text = "/* note */\n   UPDATE t SET x = 1"
    assert classify_statement(text) == "UPDATE"


def test_classify_statement_control_flow():
    assert classify_statement("IF p_x > 1 THEN") == "CONTROL_FLOW"


def test_parse_statement_extracts_tables_and_columns():
    stmt = "UPDATE PRO.AccountCal_Stg SET DPD_Overdue = 0 WHERE FlgDeg = 'Y'"
    info = parse_statement(stmt, 0, Dialect.ORACLE)
    assert info.parsed_ok
    assert info.tables_written == ["AccountCal_Stg"]
    assert "DPD_Overdue" in info.columns


def test_parse_statement_merge_extracts_target_as_written():
    stmt = (
        "MERGE INTO PRO.AccountCal_Stg A USING "
        "(SELECT id FROM PRO.Other_Table) B ON (A.id = B.id) "
        "WHEN MATCHED THEN UPDATE SET A.x = 1"
    )
    info = parse_statement(stmt, 0, Dialect.ORACLE)
    assert info.parsed_ok
    assert "AccountCal_Stg" in info.tables_written
    assert "Other_Table" in info.tables_read


def test_parse_statement_flags_unparseable_sql():
    stmt = "UPDATE FROM WHERE ((("
    info = parse_statement(stmt, 0, Dialect.ORACLE)
    assert not info.parsed_ok
    assert info.parse_error is not None
