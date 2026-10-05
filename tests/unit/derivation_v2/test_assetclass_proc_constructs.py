"""Constructs used by PRO.InsertDataforAssetClassficationRBL that the SQL
scanners must read correctly: join hints, ``alias = expr`` projections,
chained CTEs, and trailing ``OPTION (...)`` hints."""
from app.derivation.v2.sql_text import (
    extract_cte_definitions,
    extract_update_statements,
    parse_from_join_clause_with_type,
    parse_select_list,
)


def test_join_hints_do_not_hide_the_join_type():
    joins = parse_from_join_clause_with_type(
        "FROM ##CUSTOMERCAL A LEFT hash JOIN DBO.AdvCustNPAdetail C ON C.CustomerEntityId = A.CustomerEntityId"
        " INNER MERGE JOIN dbo.X D ON D.Id = A.Id"
    )
    assert [(t, a, kw) for t, a, _on, kw in joins] == [
        ("##CUSTOMERCAL", "A", "FROM"),
        ("AdvCustNPAdetail", "C", "LEFT JOIN"),
        ("X", "D", "INNER JOIN"),
    ]
    assert "hash" not in (joins[1][2] or "").lower()
    assert "MERGE" not in (joins[1][2] or "").upper()


def test_alias_equals_projection_keeps_value_and_output_name():
    items = parse_select_list("ACCOUNTENTITYID = ABD.AccountEntityID, FLGDEG='N', X = CASE WHEN a=b THEN 1 ELSE 0 END")
    assert items[0][1:3] == ("AccountEntityID", "ACCOUNTENTITYID")
    assert items[1][2] == "FLGDEG" and items[1][3] == "'N'"
    assert items[2][2] == "X" and items[2][3].upper().startswith("CASE")


def test_chained_ctes_are_all_extracted():
    ctes = extract_cte_definitions(
        ";WITH CTE_A AS (SELECT CUSTOMERENTITYID FROM T1), CTE_B AS (SELECT CUSTOMERACID FROM CTE_A) "
        "UPDATE C SET X = 1"
    )
    assert [c["name"] for c in ctes] == ["CTE_A", "CTE_B"]


def test_option_hint_is_not_part_of_where_or_from():
    sql = "UPDATE A SET A.X = 1 FROM ##T A WHERE A.Y = 2 OPTION (MAXDOP 1)"
    stmt = extract_update_statements(sql)[0]
    assert "OPTION" not in (stmt["where_clause"] or "").upper()
    sql = "UPDATE A SET A.X = B.X FROM ##T A INNER JOIN S B ON A.Id = B.Id OPTION (MAXDOP 1)"
    stmt = extract_update_statements(sql)[0]
    assert "OPTION" not in (stmt["from_clause"] or "").upper()
