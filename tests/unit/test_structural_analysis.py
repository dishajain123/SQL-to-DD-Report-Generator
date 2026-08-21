from app.models.core import Dialect
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
    assert "DPD" in info.tables_written
    assert any(cols for cols in info.columns_written_by_table.values())
