from app.models.core import Dialect, ObjectType
from app.parsing.object_splitter import split_objects


def test_single_object_file(dpd_calculation_sql):
    objs = split_objects(dpd_calculation_sql, "PRO_DPD_Calculation_StoredProcedure_2.sql", Dialect.ORACLE)
    assert len(objs) == 1
    assert objs[0].name == "DPD_Calculation"
    assert objs[0].object_type == ObjectType.PROCEDURE


def test_multi_object_file_splits_into_separate_units(multi_object_sql):
    objs = split_objects(multi_object_sql, "multi_object_sample.sql", Dialect.ORACLE)
    assert len(objs) == 2
    names = {o.name for o in objs}
    assert names == {"Reset_Flags", "Get_Risk_Band"}
    types = {o.name: o.object_type for o in objs}
    assert types["Reset_Flags"] == ObjectType.PROCEDURE
    assert types["Get_Risk_Band"] == ObjectType.FUNCTION


def test_each_object_has_unique_id(multi_object_sql):
    objs = split_objects(multi_object_sql, "multi_object_sample.sql", Dialect.ORACLE)
    ids = {o.object_id for o in objs}
    assert len(ids) == len(objs)


def test_no_create_statement_falls_back_to_single_object():
    text = "UPDATE some_table SET x = 1;"
    objs = split_objects(text, "raw.sql", Dialect.ORACLE)
    assert len(objs) == 1
    assert objs[0].raw_sql == text
    assert objs[0].object_type == ObjectType.UNKNOWN


def test_sqlserver_bracketed_object_name_is_split_correctly(sma_marking_sql):
    objs = split_objects(sma_marking_sql, "PRO.SMA_MARKING_12122023.StoredProcedure.sql", Dialect.SQLSERVER)
    assert len(objs) == 1
    assert objs[0].name == "SMA_MARKING_12122023"
    assert objs[0].object_type == ObjectType.PROCEDURE
