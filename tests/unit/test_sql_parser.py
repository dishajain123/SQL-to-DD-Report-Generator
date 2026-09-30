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


def test_split_statements_handles_go_batches_and_tsql_line_boundaries():
    sql = (
        "SET ANSI_NULLS ON\nGO\n"
        "IF OBJECT_ID('TEMPDB..#DPD') IS NOT NULL\n"
        " DROP TABLE #DPD\n"
        "SELECT a, b INTO #DPD FROM dbo.t\n"
        "UPDATE #DPD SET b = 0 WHERE ISNULL(b, 0) < 0\n"
    )
    stmts = split_statements(sql, Dialect.SQLSERVER)
    assert [classify_statement(stmt) for stmt in stmts[:5]] == ["OTHER", "CONTROL_FLOW", "OTHER", "SELECT", "UPDATE"]
    assert any("SELECT a, b INTO #DPD" in stmt for stmt in stmts)


def test_classify_statement_skips_leading_block_comment():
    text = "/* note */\n   UPDATE t SET x = 1"
    assert classify_statement(text) == "UPDATE"


def test_classify_statement_control_flow():
    assert classify_statement("IF p_x > 1 THEN") == "CONTROL_FLOW"


def test_classify_statement_cte_prefixed_update_is_not_select():
    # T-SQL allows a CTE to prefix UPDATE, not just SELECT. Misclassifying
    # this as SELECT (the "WITH implies SELECT" shortcut) makes the write
    # target resolve to nothing downstream, silently dropping the column.
    text = (
        ";WITH CTE_NPA_UCIFID AS "
        "(SELECT UcifEntityID FROM ##ACCOUNTCAL WHERE FinalAssetClassAlt_Key>1 GROUP BY UcifEntityID) "
        "UPDATE A SET A.ASSET_NORM='CONDI_STD' FROM ##ACCOUNTCAL A "
        "INNER JOIN CTE_NPA_UCIFID B ON A.UcifEntityID=B.UcifEntityID "
        "WHERE ASSET_NORM='ALWYS_STD'"
    )
    assert classify_statement(text) == "UPDATE"


def test_classify_statement_cte_prefixed_select_stays_select():
    text = "WITH cte AS (SELECT id FROM t) SELECT * FROM cte"
    assert classify_statement(text) == "SELECT"


def test_parse_statement_captures_cte_prefixed_update_write_target():
    stmt = (
        ";WITH CTE_NPA_UCIFID AS "
        "(SELECT UcifEntityID FROM ##ACCOUNTCAL WHERE FinalAssetClassAlt_Key>1 GROUP BY UcifEntityID) "
        "UPDATE A SET A.ASSET_NORM='CONDI_STD' FROM ##ACCOUNTCAL A "
        "INNER JOIN CTE_NPA_UCIFID B ON A.UcifEntityID=B.UcifEntityID "
        "WHERE ASSET_NORM='ALWYS_STD'"
    )
    info = parse_statement(stmt, 0, Dialect.SQLSERVER)
    assert info.statement_type == "UPDATE"
    assert any(t.lstrip("#").upper() == "ACCOUNTCAL" for t in info.tables_written)
    written_cols = {
        col.upper()
        for table, cols in info.set_columns_by_table.items()
        for col in cols
    }
    assert "ASSET_NORM" in written_cols


def test_parse_statement_extracts_tables_and_columns():
    stmt = "UPDATE PRO.AccountCal_Stg SET DPD_Overdue = 0 WHERE FlgDeg = 'Y'"
    info = parse_statement(stmt, 0, Dialect.ORACLE)
    assert info.parsed_ok
    assert info.tables_written == ["AccountCal_Stg"]
    assert "DPD_Overdue" in info.columns


def test_parse_statement_extracts_select_into_target_and_projection_columns():
    stmt = "SELECT a.AccountEntityID, CASE WHEN ISNULL(a.DPD_Overdrawn,0)>30 THEN 1 ELSE 0 END AS DPD_FLAG INTO #DPD FROM PRO.AccountCal a WHERE ISNULL(a.DPD_Overdrawn,0)>30"
    info = parse_statement(stmt, 0, Dialect.SQLSERVER)
    assert info.parsed_ok
    assert any(t.lstrip("#").upper() == "DPD" for t in info.tables_written)
    assert any(t.startswith("#") for t in info.tables_written) or "DPD" in info.tables_written
    assert any("DPD_FLAG" in cols for cols in info.set_columns_by_table.values())


def test_parse_statement_insert_records_target_including_temp():
    stmt = (
        "INSERT INTO #DpdStaging (AccountId, DpdBucket, FacilityType, AdjustedPenalty)\n"
        "SELECT A.AccountId, A.DpdBucket, A.FacilityType, A.PenalInterestAmount\n"
        "FROM PRO.LoanAccountCal A WHERE A.BucketWorsened = 'Y'"
    )
    info = parse_statement(stmt, 0, Dialect.SQLSERVER)
    assert info.tables_written
    assert any(t.upper().endswith("DPDSTAGING") for t in info.tables_written)
    assert any(t.startswith("#") for t in info.tables_written)


def test_parse_statement_keeps_merge_intact_with_when_clauses():
    stmt = (
        "MERGE PRO.DpdBucketHistory AS Target\n"
        "USING #DpdStaging AS Source\n"
        "ON Target.AccountId = Source.AccountId\n"
        "WHEN MATCHED THEN\n"
        "    UPDATE SET Target.DpdBucket = Source.DpdBucket\n"
        "WHEN NOT MATCHED BY TARGET THEN\n"
        "    INSERT (AccountId, DpdBucket) VALUES (Source.AccountId, Source.DpdBucket);"
    )
    stmts = split_statements(stmt, Dialect.SQLSERVER)
    assert len(stmts) == 1
    info = parse_statement(stmts[0], 0, Dialect.SQLSERVER)
    assert info.statement_type == "MERGE"
    assert info.parsed_ok
    assert any("DPDBUCKETHISTORY" in t.upper() for t in info.tables_written)


def test_parse_statement_handles_cte_wrapped_update():
    stmt = "WITH x AS (SELECT 1 AS id) UPDATE t SET a = 1 FROM x WHERE t.id = x.id"
    info = parse_statement(stmt, 0, Dialect.SQLSERVER)
    assert info.parsed_ok
    assert "t" in info.tables_written or "T" in info.tables_written


def test_split_statements_keeps_semicolon_cte_glued_to_its_update():
    """Real pattern from banking NPA/asset-classification procs:
    `;WITH cte AS (...) UPDATE A SET ... FROM ##T A INNER JOIN cte B ON ...`
    spread across many lines, immediately followed by an unrelated UPDATE.
    Splitting at the UPDATE line (as if it were a new statement) leaves the
    CTE definition orphaned -- a syntax error on its own -- and the UPDATE
    half loses the CTE, so its alias resolves as a fake physical table.
    """
    sql = (
        ";WITH CTE_NPA_UCIFID AS\n"
        "(SELECT UcifEntityID FROM ##ACCOUNTCAL\n"
        "WHERE FinalAssetClassAlt_Key>1\n"
        "GROUP BY UcifEntityID)\n"
        "\n"
        "UPDATE A SET A.ASSET_NORM='CONDI_STD' FROM ##ACCOUNTCAL A\n"
        "INNER JOIN CTE_NPA_UCIFID B ON A.UcifEntityID=B.UcifEntityID\n"
        "INNER JOIN DimProduct P ON P.EffectiveFromTimeKey<=@TIMEKEY\n"
        "AND P.EffectiveToTimeKey>=@TIMEKEY AND P.ProductAlt_Key=A.ProductAlt_Key\n"
        "AND P.ProductGroup='FDSEC'\n"
        "WHERE ASSET_NORM='ALWYS_STD'\n"
        "\n"
        "UPDATE B SET B.FinalNpaDt=A.SYSNPA_DT FROM ##CustomerCal A\n"
        "INNER JOIN ##ACCOUNTCAL B ON A.SourceSystemCustomerID=B.SourceSystemCustomerID\n"
        "WHERE ISNULL(B.ASSET_NORM,'NORMAL')<>'ALWYS_STD'\n"
    )
    stmts = split_statements(sql, Dialect.SQLSERVER)
    assert len(stmts) == 2, f"expected the CTE glued to its UPDATE plus one trailing UPDATE, got: {stmts}"
    assert stmts[0].upper().startswith(";WITH") or stmts[0].upper().startswith("WITH")
    assert "CTE_NPA_UCIFID" in stmts[0]
    assert stmts[1].strip().upper().startswith("UPDATE B")

    first = parse_statement(stmts[0], 0, Dialect.SQLSERVER)
    assert first.parsed_ok, first.parse_error
    assert any(t.upper().lstrip("#") == "ACCOUNTCAL" for t in first.tables_written)
    # The CTE alias must never be reported as a physical source table.
    assert not any(t.upper() == "CTE_NPA_UCIFID" for t in first.tables_read)

    second = parse_statement(stmts[1], 1, Dialect.SQLSERVER)
    assert second.parsed_ok, second.parse_error
    assert any(t.upper().lstrip("#") == "ACCOUNTCAL" for t in second.tables_written)


def test_split_statements_with_then_select_still_works():
    sql = (
        ";WITH x AS (SELECT 1 AS id)\n"
        "SELECT * FROM x\n"
        "UPDATE t SET y = 1 WHERE t.z = 2\n"
    )
    stmts = split_statements(sql, Dialect.SQLSERVER)
    assert len(stmts) == 2
    assert "SELECT * FROM x" in stmts[0]
    assert stmts[1].strip().upper().startswith("UPDATE T")


def test_parse_statement_resolves_update_alias_from_real_table():
    stmt = (
        "UPDATE A SET A.RestructureEligible = 'Y'\n"
        "FROM PRO.LoanAccountCal A WHERE A.OutstandingBalance IS NOT NULL"
    )
    info = parse_statement(stmt, 0, Dialect.SQLSERVER)
    assert any("LOANACCOUNTCAL" in t.upper() for t in info.tables_written)
    assert not any(t.upper() == "A" for t in info.tables_written)


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


def test_multi_table_update_join_parses_to_the_real_target():
    stmt = (
        "UPDATE A\n"
        "SET A.AssetClass = B.NewClass\n"
        "FROM PRO.LoanAccount A\n"
        "INNER JOIN PRO.CustomerMaster B ON A.CustomerId = B.CustomerId\n"
        "WHERE B.IsNpa = 'Y'"
    )
    info = parse_statement(stmt, 0, Dialect.SQLSERVER)
    assert info.parsed_ok, info.parse_error
    assert info.tables_written == ["LoanAccount"]
    assert info.set_columns_by_table == {"LoanAccount": ["AssetClass"]}
    assert "CustomerMaster" in info.join_tables
    assert any("CustomerId" in cond for cond in info.join_conditions)

    left = parse_statement(
        "UPDATE B SET B.Flag = 'Y' FROM PRO.LoanAccount A "
        "LEFT OUTER JOIN PRO.CustomerMaster B WITH (NOLOCK) ON A.CustomerId = B.CustomerId",
        0,
        Dialect.SQLSERVER,
    )
    assert left.parsed_ok, left.parse_error
    assert left.tables_written == ["CustomerMaster"]


def test_multi_table_delete_targets_the_aliased_table():
    info = parse_statement(
        "DELETE A FROM PRO.Stage X INNER JOIN PRO.Target A ON X.Id = A.Id",
        0,
        Dialect.SQLSERVER,
    )
    assert info.parsed_ok, info.parse_error
    assert info.tables_written == ["Target"]


def test_coverage_items_are_not_reported_as_unparseable():
    """Regression: every coverage-ledger blocker (IF branch, temp staging,
    MERGE, CATCH, source anomaly) was counted as an "unparseable statement",
    so well-formed UPDATE…JOINs were reported as parse failures."""
    from pathlib import Path

    from app.guardrails.structural_guardrails import check_structural_info
    from app.parsing.dialect import detect_dialect
    from app.parsing.object_splitter import split_objects
    from app.parsing.structural_analysis import analyze_object

    root = Path(__file__).resolve().parents[2]
    sql = (root / "samples" / "sql" / "07_DPD_Bucket_Classification.sql").read_text(encoding="utf-8")
    for obj in split_objects(sql, "07.sql", detect_dialect(sql)):
        info = analyze_object(obj)
        assert info.unsupported_constructs, "ledger items are still tracked"
        assert info.parse_failures == []
        assert not any("unparseable" in e for e in check_structural_info(info).errors)


def test_parse_statement_flags_unparseable_sql():
    stmt = "UPDATE FROM WHERE ((("
    info = parse_statement(stmt, 0, Dialect.ORACLE)
    assert not info.parsed_ok
    assert info.parse_error is not None


def test_parse_statement_captures_where_clause_as_a_condition():
    # Regression: a plain UPDATE ... WHERE with no IF/CASE at all is the
    # overwhelming majority shape in this corpus, and its WHERE clause
    # *is* the derivation condition -- the IF/CASE-WHEN regex extraction
    # alone never saw it, so it never reached the chunk/report condition
    # inventory.
    stmt = "UPDATE A SET A.X = 1 FROM T A WHERE A.Y = 'Y';"
    info = parse_statement(stmt, 0, Dialect.SQLSERVER)
    assert info.parsed_ok
    assert any("A.Y = 'Y'" in c for c in info.conditions)


def test_parse_statement_captures_merge_on_and_when_conditions():
    stmt = (
        "MERGE INTO T USING S ON (T.id = S.id)\n"
        "WHEN MATCHED AND S.flag = 1 THEN UPDATE SET T.x = S.x\n"
        "WHEN NOT MATCHED THEN INSERT (id) VALUES (S.id);"
    )
    info = parse_statement(stmt, 0, Dialect.SQLSERVER)
    assert info.parsed_ok
    assert any("T.id" in c and "S.id" in c for c in info.conditions)
    assert any("S.flag = 1" in c for c in info.conditions)


def test_parse_statement_does_not_extract_conditions_from_comments():
    # Comment-stripping already exists (_mask_comments_only) -- retired
    # logic left in a comment must not appear as a live condition.
    stmt = (
        "/* retired rule\n"
        "IF B.ProvisionRule IN ('OTHERS/BLANK') THEN X := 1; END IF;\n"
        "*/\n"
        "UPDATE T SET Y = CASE WHEN Z = 1 THEN 2 ELSE 0 END;"
    )
    info = parse_statement(stmt, 0, Dialect.ORACLE)
    assert not any("ProvisionRule" in c for c in info.conditions)
    assert any("Z = 1" in c for c in info.conditions)
