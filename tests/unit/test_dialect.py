from app.models.core import Dialect
from app.parsing.dialect import detect_dialect


def test_detects_oracle(dpd_calculation_sql):
    assert detect_dialect(dpd_calculation_sql) == Dialect.ORACLE


def test_detects_mysql(mysql_sample_sql):
    assert detect_dialect(mysql_sample_sql) == Dialect.MYSQL


def test_detects_sqlserver(sma_marking_sql):
    assert detect_dialect(sma_marking_sql) == Dialect.SQLSERVER


def test_ambiguous_text_defaults_to_oracle():
    assert detect_dialect("SELECT 1;") == Dialect.ORACLE
