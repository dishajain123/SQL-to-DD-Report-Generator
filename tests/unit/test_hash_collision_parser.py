"""``PRO.X`` written from ``#X`` must stay a write to ``X``, not to ``#X``."""
from __future__ import annotations

from app.parsing.dialect import Dialect
from app.parsing.sql_parser import parse_statement


def _parse(sql: str):
    return parse_statement(sql, 0, Dialect.SQLSERVER)


def test_insert_into_permanent_table_from_same_named_temp_targets_the_permanent_table():
    info = _parse(
        "INSERT INTO PRO.AMH (CustomerAcID, Status)\n"
        "SELECT T.CustomerAcID, T.Status FROM #AMH T"
    )
    assert [t.upper() for t in info.tables_written] == ["AMH"], info.tables_written
    assert "#AMH" in info.tables_read


def test_insert_into_the_temp_keeps_its_hash():
    info = _parse(
        "INSERT INTO #AMH (CustomerAcID, Status)\n"
        "SELECT X.CustomerAcID, X.SMA_CLASS FROM ##AccountCal X"
    )
    assert [t.upper() for t in info.tables_written] == ["#AMH"], info.tables_written


def test_update_alias_resolves_to_permanent_table_when_a_same_named_temp_is_joined():
    info = _parse(
        "UPDATE AA SET EffectiveToTimeKey = 1\n"
        "FROM PRO.AMH AA LEFT JOIN #AMH B ON AA.CustomerAcID = B.CustomerAcID\n"
        "WHERE B.CustomerAcID IS NULL"
    )
    assert [t.upper() for t in info.tables_written] == ["AMH"], info.tables_written


def test_unambiguous_temp_still_gets_its_hash_back():
    info = _parse("UPDATE #STAGE SET X = 1 WHERE Y = 2")
    assert [t.upper() for t in info.tables_written] == ["#STAGE"], info.tables_written
