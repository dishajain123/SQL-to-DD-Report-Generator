from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLES_DIR = ROOT / "samples"


class MockLLMClient:
    """Stands in for app.derivation.llm_client.LLMClient in tests — no
    network calls, deterministic output, same public interface."""

    def technical_reasoning(self, sql_snippets: list[str]) -> str:
        return "The code updates staging table columns based on date-difference calculations and conditional thresholds."

    def business_reasoning(self, technical_summary: str) -> str:
        return "This logic determines days-past-due and downstream NPA/SMA classification for each account."

    def generate_formula_expression(
        self,
        technical_summary,
        business_summary,
        source_sql,
        function_reference,
        column_name="",
        entity_name="",
    ) -> str:
        return (
            'IF(ISNOTEMPTY("FCT_NPA_PRODUCT"."OverDueSinceDt"))'
            'THEN(DATEDIFF("FCT_NPA_PRODUCT"."var"."BUSINESS_DATE",'
            '"FCT_NPA_PRODUCT"."OverDueSinceDt","d")+1)ELSE(0)'
        )

    def retry_with_error(self, previous_expression, error, context) -> str:
        return (
            'IF(ISNOTEMPTY("FCT_NPA_PRODUCT"."OverDueSinceDt"))'
            'THEN(DATEDIFF("FCT_NPA_PRODUCT"."var"."BUSINESS_DATE",'
            '"FCT_NPA_PRODUCT"."OverDueSinceDt","d")+1)ELSE(0)'
        )


class BrokenLLMClient(MockLLMClient):
    """Always returns an invalid expression — used to test the
    grammar-validation-failure and retry path."""

    def generate_formula_expression(self, *args, **kwargs) -> str:
        return "IF(BOGUSFUNC(x)THEN(1)ELSE(0)"  # malformed on purpose

    def retry_with_error(self, *args, **kwargs) -> str:
        return "IF(BOGUSFUNC(x)THEN(1)ELSE(0)"  # still broken


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def broken_llm_client() -> BrokenLLMClient:
    return BrokenLLMClient()


@pytest.fixture
def samples_dir() -> Path:
    return SAMPLES_DIR


@pytest.fixture
def dpd_calculation_sql(samples_dir: Path) -> str:
    return (samples_dir / "sql" / "PRO_DPD_Calculation_StoredProcedure_2.sql").read_text()


@pytest.fixture
def maxdpd_sql(samples_dir: Path) -> str:
    return (samples_dir / "sql" / "PRO_MaxDPD_ReferencePeriod_Calculation_StoredProcedure.sql").read_text()


@pytest.fixture
def npa_date_sql(samples_dir: Path) -> str:
    return (samples_dir / "sql" / "PRO_NPA_Date_Calculation_StoredProcedure_1.sql").read_text()


@pytest.fixture
def multi_object_sql(samples_dir: Path) -> str:
    return (samples_dir / "sql" / "multi_object_sample.sql").read_text()


@pytest.fixture
def mysql_sample_sql(samples_dir: Path) -> str:
    return (samples_dir / "sql" / "customer_risk_flag_mysql.sql").read_text()


@pytest.fixture
def function_reference(samples_dir: Path) -> str:
    return (samples_dir / "platform_docs" / "4x_functions_operators.md").read_text()


@pytest.fixture
def tmp_db_path(tmp_path) -> str:
    return str(tmp_path / "test.db")