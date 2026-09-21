from app.models.core import Dialect, ObjectType, SQLObject
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object


def test_dpd_calculation_structural_analysis(dpd_calculation_sql):
    objs = split_objects(dpd_calculation_sql, "PRO_DPD_Calculation_StoredProcedure_2.sql", Dialect.ORACLE)
    info = analyze_object(objs[0])

    assert "AccountCal_Stg" in info.tables_written
    assert info.confidence >= 0.9
    assert not info.has_dynamic_sql

    # This is the concrete claim made earlier in the conversation: the proc
    # contains version-threshold branches on p_TIMEKEY that should be
    # auto-detected without hardcoding the specific values.
    assert len(info.version_thresholds) >= 1
    assert all(t.variable.upper().endswith("TIMEKEY") for t in info.version_thresholds)
    detected_values = {t.value for t in info.version_thresholds}
    assert "26267" in detected_values


def test_npa_date_calculation_reads_accountcal_stg(npa_date_sql):
    objs = split_objects(npa_date_sql, "PRO_NPA_Date_Calculation_StoredProcedure_1.sql", Dialect.ORACLE)
    info = analyze_object(objs[0])
    assert "AccountCal_Stg" in info.tables_read


def test_object_with_no_dml_has_neutral_confidence():
    from app.models.core import Dialect, ObjectType, SQLObject

    obj = SQLObject(
        object_id="x",
        name="Empty_Proc",
        object_type=ObjectType.PROCEDURE,
        dialect=Dialect.ORACLE,
        raw_sql="BEGIN NULL; END;",
        source_file="x.sql",
    )
    info = analyze_object(obj)
    assert info.confidence == 1.0


def test_sqlserver_sm_marking_procedure_produces_written_columns(sma_marking_sql):
    objs = split_objects(
        sma_marking_sql,
        "PRO.SMA_MARKING_12122023.StoredProcedure.sql",
        Dialect.SQLSERVER,
    )
    info = analyze_object(objs[0])

    assert info.statements
    assert info.tables_written
    assert info.columns_written_by_table
    assert any(table.upper() == "#DPD" for table in info.tables_written)
    assert any(cols for cols in info.columns_written_by_table.values())


def _analyze(raw_sql: str, dialect: Dialect = Dialect.SQLSERVER):
    obj = SQLObject(
        object_id="x",
        name="Test_Proc",
        object_type=ObjectType.PROCEDURE,
        dialect=dialect,
        raw_sql=raw_sql,
        source_file="x.sql",
    )
    return analyze_object(obj)


def test_bare_tsql_timekey_parameter_is_detected_as_a_version_threshold():
    # Regression: the previous regex required at least one character before
    # the literal "TIMEKEY", so a T-SQL procedure's own bare `@TIMEKEY`
    # parameter (as opposed to a prefixed `@V_TIMEKEY`) was never detected
    # at all -- the single most common real rule-version cutover shape.
    sql = "CREATE PROCEDURE dbo.Test @TIMEKEY INT AS BEGIN\nIF @TIMEKEY > 26267\nBEGIN\n  UPDATE A SET x=1\nEND\nELSE\nBEGIN\n  UPDATE A SET x=2\nEND\nEND"
    info = _analyze(sql)
    assert [t.value for t in info.version_thresholds] == ["26267"]
    assert info.version_thresholds[0].variable.upper() == "TIMEKEY"


def test_scd2_effective_timekey_columns_are_not_mistaken_for_version_thresholds():
    # Regression: EffectiveToTimeKey/EffectiveFromTimeKey are SCD-2 row-
    # validity columns, always written qualified with a table alias in this
    # corpus -- not rule-versioning parameters -- and must not be detected
    # as thresholds just because they end in "TimeKey".
    sql = (
        "UPDATE ABD SET ABD.EffectiveToTimeKey = 49999 "
        "WHERE ABD.EffectiveFromTimeKey <= 20240101"
    )
    info = _analyze(sql)
    assert info.version_thresholds == []


def test_scd2_sentinel_is_excluded_even_if_written_through_a_bare_parameter():
    # Belt-and-braces: a genuine bare parameter compared against a
    # universal SCD-2 "current row" sentinel is still overwhelmingly more
    # likely to be a row-validity bound than an actual rule cutover.
    sql = "IF @ToTimeKey >= 49999 BEGIN SELECT 1 END"
    info = _analyze(sql)
    assert info.version_thresholds == []


def test_commented_out_usage_example_is_not_a_version_threshold():
    # Regression: nearly every procedure header carries a usage example
    # like `--exec [Pro].[DPD_Calculation] @timekey=25140;` in a comment --
    # without comment stripping, that gets misread as a real threshold.
    sql = "--exec [Pro].[DPD_Calculation] @timekey=25140;\nSELECT 1;"
    info = _analyze(sql)
    assert info.version_thresholds == []


def test_equality_assignment_is_not_a_version_threshold():
    # `@TIMEKEY = 26418` in real procedures is a debug/log assignment, not
    # a rule-version boundary -- cutovers are always inequalities.
    sql = "SET @TIMEKEY = 26418"
    info = _analyze(sql)
    assert info.version_thresholds == []


def test_bracket_quoted_exec_calls_are_detected():
    # Regression: bracket-quoted calls (`EXEC [PRO].[Foo]`) are the T-SQL
    # norm in this corpus, but the character class had no `[`/`]` at all,
    # so every bracket-quoted call was silently dropped from the lineage
    # graph's explicit-call edges.
    sql = (
        "EXEC [PRO].[GovtGuarAppropriation] @TIMEKEY=@TIMEKEY\n"
        "EXEC  PRO.GovtGurCoverAmount       @TIMEKEY=@TIMEKEY\n"
    )
    info = _analyze(sql)
    assert info.called_objects == ["GovtGuarAppropriation", "GovtGurCoverAmount"]


def test_exec_inside_a_comment_is_not_a_called_object():
    # Regression: nearly every procedure header carries a usage note like
    # `--exec PRO.DPD_Calculation @timekey=25140` -- without comment
    # stripping, that was read as a real call edge.
    sql = "--exec PRO.DPD_Calculation @timekey=25140\nSELECT 1;"
    info = _analyze(sql)
    assert info.called_objects == []


def test_execute_immediate_is_not_misread_as_a_call_to_immediate():
    sql = "EXECUTE IMMEDIATE 'SELECT 1 FROM DUAL'"
    info = _analyze(sql, dialect=Dialect.ORACLE)
    assert info.called_objects == []
    assert info.has_dynamic_sql
