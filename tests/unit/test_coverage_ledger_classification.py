from app.models.core import Dialect
from app.parsing.coverage_ledger import WriteKind, build_coverage_ledger
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object


def _ledger_for(sql: str, dialect: Dialect = Dialect.SQLSERVER):
    obj = split_objects(sql, "test.sql", dialect)[0]
    info = analyze_object(obj)
    return build_coverage_ledger(info, source_sql=obj.raw_sql)


def test_update_join_on_equality_key_is_a_lookup_row_formula():
    # T-SQL routinely expresses a single-row update-with-dimension-lookup
    # this way -- a foreign-key reference, not a relational fan-out write.
    sql = (
        "CREATE PROCEDURE dbo.Test AS\nBEGIN\n"
        "UPDATE A SET A.X = D.Y FROM PRO.Fact A INNER JOIN PRO.Dim D "
        "ON A.DimId = D.DimId WHERE D.Active = 1\n"
        "END"
    )
    ledger = _ledger_for(sql)
    assert len(ledger.entries) == 1
    assert ledger.entries[0].kind == WriteKind.ROW_FORMULA
    assert any("lookup" in note.lower() for note in ledger.entries[0].notes)


def test_update_join_with_aggregate_stays_cross_row():
    sql = (
        "CREATE PROCEDURE dbo.Test AS\nBEGIN\n"
        "UPDATE A SET A.Total = SUM(D.Amount) FROM PRO.Fact A INNER JOIN PRO.Detail D "
        "ON A.Id = D.FactId GROUP BY A.Id\n"
        "END"
    )
    ledger = _ledger_for(sql)
    assert ledger.entries[0].kind == WriteKind.CROSS_ROW


def test_update_join_with_non_equality_condition_stays_cross_row():
    sql = (
        "CREATE PROCEDURE dbo.Test AS\nBEGIN\n"
        "UPDATE A SET A.X = D.Y FROM PRO.Fact A INNER JOIN PRO.Dim D "
        "ON A.StartDate <= D.EndDate\n"
        "END"
    )
    ledger = _ledger_for(sql)
    assert ledger.entries[0].kind == WriteKind.CROSS_ROW


def test_update_join_condition_with_inline_comment_noise_is_still_a_lookup():
    # A join condition rendered from the parsed tree can carry an inline
    # comment sqlglot attached to the AST node -- must not defeat the
    # equality-shape check.
    sql = (
        "CREATE PROCEDURE dbo.Test AS\nBEGIN\n"
        "UPDATE A SET A.X = D.Y FROM PRO.Fact A INNER JOIN PRO.Dim D "
        "ON A.DimId = D.DimId /* lookup */\n"
        "WHERE D.Active = 1\n"
        "END"
    )
    ledger = _ledger_for(sql)
    assert ledger.entries[0].kind == WriteKind.ROW_FORMULA


def test_global_temp_merged_into_persistent_table_is_not_temp_staging():
    sql = (
        "CREATE PROCEDURE dbo.Test AS\nBEGIN\n"
        "UPDATE A SET A.X = 1 FROM ##Working A WHERE A.Y = 1\n"
        "MERGE INTO PRO.RealTable T USING ##Working W ON T.Id = W.Id "
        "WHEN MATCHED THEN UPDATE SET T.X = W.X;\n"
        "END"
    )
    ledger = _ledger_for(sql)
    working_entry = next(e for e in ledger.entries if e.target_table == "##Working")
    assert working_entry.kind == WriteKind.ROW_FORMULA


def test_global_temp_with_no_merge_back_stays_temp_staging():
    sql = (
        "CREATE PROCEDURE dbo.Test AS\nBEGIN\n"
        "UPDATE A SET A.X = 1 FROM ##Scratch A WHERE A.Y = 1\n"
        "END"
    )
    ledger = _ledger_for(sql)
    assert ledger.entries[0].kind == WriteKind.TEMP_STAGING


def test_session_local_temp_is_never_treated_as_a_merge_source():
    # Only `##` global temp tables get this treatment -- a `#` session-
    # local scratch table stays TEMP_STAGING even if later merged.
    sql = (
        "CREATE PROCEDURE dbo.Test AS\nBEGIN\n"
        "UPDATE A SET A.X = 1 FROM #Scratch A WHERE A.Y = 1\n"
        "MERGE INTO PRO.RealTable T USING #Scratch W ON T.Id = W.Id "
        "WHEN MATCHED THEN UPDATE SET T.X = W.X;\n"
        "END"
    )
    ledger = _ledger_for(sql)
    scratch_entry = next(e for e in ledger.entries if e.target_table == "#Scratch")
    assert scratch_entry.kind == WriteKind.TEMP_STAGING
