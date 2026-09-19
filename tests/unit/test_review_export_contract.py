"""Keep the report exports consumed by Streamlit and the API available."""

from datetime import date

from openpyxl import load_workbook

from app.models.core import ColumnType, DDRow, DDStatus, DerivationOption
from app.report import dd_export
from app.report.dd_export import (
    export_reviewed_dd_rows_for_job_csv,
    export_reviewed_dd_rows_for_job_excel,
)


def test_reviewed_excel_export_is_importable_and_writes_workbook(monkeypatch, tmp_path):
    row = DDRow(
        entity_name="FCT_NPA_PRODUCT",
        column_name="AssetClass",
        column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression='IF(ISNOTEMPTY("FCT_NPA_PRODUCT"."Flag"))THEN("Y")ELSE("N")',
        effective_start_date=date(2026, 9, 18),
        status=DDStatus.ACTIVE,
        data_type="string",
        source_chain_id="chain-1",
    )
    # Simulate a DB row: narrow columns + full row_json payload.
    stored = {
        **row.model_dump(mode="json"),
        "expression": row.display_derivation_expression,
        "chain_id": row.source_chain_id,
        "row_json": row.model_dump_json(),
    }
    monkeypatch.setattr(
        dd_export.db,
        "get_dd_rows_for_job",
        lambda job_id, db_path=None: [stored],
    )

    output = export_reviewed_dd_rows_for_job_excel("job-1", tmp_path / "reviewed.xlsx")
    assert output.exists()
    workbook = load_workbook(output, read_only=True)
    try:
        rows = list(workbook.active.values)
    finally:
        workbook.close()
    assert list(rows[0]) == dd_export.COLUMNS
    assert rows[1][0:2] == ("FCT_NPA_PRODUCT", "AssetClass")
    assert callable(export_reviewed_dd_rows_for_job_csv)
