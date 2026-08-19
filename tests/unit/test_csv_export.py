from datetime import date
import csv

from app.models.core import ColumnType, DDRow, DDStatus, DerivationOption
from app.report.dd_export import COLUMNS, export_dd_rows, read_existing_dd_excel


def _sample_row(**overrides) -> DDRow:
    defaults = dict(
        entity_name="FCT_NPA_PRODUCT",
        column_name="DPD_Overdue",
        column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression='IF(ISNOTEMPTY("A"."X"))THEN(1)ELSE(0)',
        effective_start_date=date(2026, 1, 1),
        status=DDStatus.ACTIVE,
        data_type="number",
        source_chain_id="chain-1",
    )
    defaults.update(overrides)
    return DDRow(**defaults)


def _read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_export_writes_correct_headers(tmp_path):
    out = export_dd_rows([_sample_row()], tmp_path / "dd.csv")
    rows = _read_csv(out)
    assert list(rows[0].keys()) == COLUMNS


def test_export_writes_row_data(tmp_path):
    row = _sample_row()
    out = export_dd_rows([row], tmp_path / "dd.csv")
    rows = _read_csv(out)
    assert rows[0]["Entity Name"] == "FCT_NPA_PRODUCT"
    assert rows[0]["Column Name"] == "DPD_Overdue"
    assert rows[0]["Derivation Option"] == "Formula Expression"
    assert rows[0]["Display Derivation Expression"] == row.display_derivation_expression


def test_export_handles_decision_table_rows(tmp_path):
    row = _sample_row(
        derivation_option=DerivationOption.DECISION_TABLE,
        display_derivation_expression="",
        decision_table_json='{"buckets": [{"max_dpd": 0, "label": "Standard"}]}',
    )
    out = export_dd_rows([row], tmp_path / "dd.csv")
    rows = _read_csv(out)
    assert rows[0]["Decision Table Json"] == row.decision_table_json


def test_merge_preserves_unrelated_existing_rows(tmp_path):
    row_a = _sample_row(column_name="DPD_Overdue")
    row_b = _sample_row(column_name="DPD_Renewal")
    first_path = tmp_path / "dd.csv"
    export_dd_rows([row_a, row_b], first_path)

    updated_a = _sample_row(
        column_name="DPD_Overdue",
        display_derivation_expression='IF(ISNOTEMPTY("A"."X"))THEN(1)ELSE(0)',
    )
    second_path = tmp_path / "dd_v2.csv"
    export_dd_rows([updated_a], second_path, existing_dd_path=first_path)

    merged = read_existing_dd_excel(second_path)
    by_column = {r["column_name"]: r for r in merged}

    assert "DPD_Renewal" in by_column
    assert sum(1 for r in merged if r["column_name"] == "DPD_Overdue") == 1
    assert by_column["DPD_Overdue"]["display_derivation_expression"] == updated_a.display_derivation_expression


def test_merge_with_no_existing_file_behaves_like_fresh_export(tmp_path):
    row = _sample_row()
    out = export_dd_rows([row], tmp_path / "dd.csv", existing_dd_path=tmp_path / "does_not_exist.csv")
    assert len(_read_csv(out)) == 1


def test_export_dedupes_equivalent_rows_across_effective_dates(tmp_path):
    row_a = _sample_row(effective_start_date=date(2026, 1, 1))
    row_b = _sample_row(effective_start_date=date(2026, 3, 1))
    out = export_dd_rows([row_a, row_b], tmp_path / "dd.csv")
    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["Effective Start Date"] == "01-01-2026"


def test_export_dedupes_case_variant_logical_columns(tmp_path):
    row_a = _sample_row(
        column_name="REFPERIODMAX",
        display_derivation_expression='IF(ISEMPTY(REFPERIODMAX))THEN(0)ELSE(REFPERIODMAX)',
    )
    row_b = _sample_row(
        column_name="REFPeriodMax",
        display_derivation_expression='IF(ISEMPTY(REFPeriodMax))THEN(0)ELSE(REFPeriodMax)',
    )
    out = export_dd_rows([row_a, row_b], tmp_path / "dd.csv")
    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["Column Name"].upper() == "REFPERIODMAX"


def test_read_existing_dd_excel_returns_empty_for_missing_file(tmp_path):
    assert read_existing_dd_excel(tmp_path / "nope.csv") == []
