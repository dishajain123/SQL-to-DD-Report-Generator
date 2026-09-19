"""18-procedure regression corpus — inventory completeness + anomalies.

Expectations come from the independent write-inventory scanner, not from
filename-hardcoded formulas. Parser must account for every scanned write.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.guardrails.source_anomalies import detect_source_anomalies
from app.models.core import (
    ColumnType,
    DDRow,
    DDStatus,
    DerivationOption,
    ReviewState,
)
from app.parsing.coverage_ledger import WriteLedgerEntry, build_coverage_ledger, reconcile_write_inventory
from app.parsing.dialect import detect_dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object
from app.parsing.write_inventory_scan import (
    inventory_to_dict,
    read_sql_file,
    sample_sql_paths,
    scan_expected_writes,
)
from app.report.dd_export import _row_dict_to_dd_row, export_dd_rows_csv, write_qa_coverage_report
from scripts.release_gate_samples_01_18 import evaluate_sample_file, run_gate


SAMPLES = sample_sql_paths()
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "write_inventories"


@pytest.mark.parametrize("path", SAMPLES, ids=[p.name for p in SAMPLES])
def test_fixture_inventory_matches_independent_scanner(path: Path):
    """Committed fixtures are the reviewed oracle; scanner must stay in sync."""
    fixture_path = FIXTURE_DIR / f"{path.stem}.json"
    assert fixture_path.exists(), f"missing fixture {fixture_path.name}"
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))
    actual = inventory_to_dict(scan_expected_writes(read_sql_file(path)))
    assert [(w["operation"], w["target_table"]) for w in actual] == [
        (w["operation"], w["target_table"]) for w in expected
    ]


@pytest.mark.parametrize("path", SAMPLES, ids=[p.name for p in SAMPLES])
def test_independent_scanner_finds_writes(path: Path):
    writes = scan_expected_writes(read_sql_file(path))
    assert writes, f"{path.name}: expected at least one write operation"


@pytest.mark.parametrize("path", SAMPLES, ids=[p.name for p in SAMPLES])
def test_parser_ledger_covers_independent_inventory(path: Path):
    """No source write may disappear because parsing failed."""
    result = evaluate_sample_file(path)
    assert result.inventory_complete, (
        f"{path.name} omitted writes: {result.missing_targets}; notes={result.notes}"
    )


@pytest.mark.parametrize("path", SAMPLES, ids=[p.name for p in SAMPLES])
def test_every_ledger_entry_has_target_or_explicit_failure(path: Path):
    sql = read_sql_file(path)
    obj = split_objects(sql, path.name, detect_dialect(sql))[0]
    ledger = build_coverage_ledger(analyze_object(obj), source_sql=sql)
    for entry in ledger.entries:
        assert entry.target_table or entry.kind.value == "parse_failure", (
            f"{path.name} stmt #{entry.statement_index} has no target"
        )


def test_release_gate_runs_all_18():
    results = run_gate()
    assert len(results) == 18
    incomplete = [r.file_name for r in results if not r.inventory_complete]
    assert not incomplete, f"incomplete inventories: {incomplete}"


def test_inventory_reconciliation_detects_a_missing_repeat_write():
    sql = "UPDATE dbo.Accounts SET Flag='Y'; UPDATE dbo.Accounts SET Flag='N';"
    parsed = [WriteLedgerEntry(0, "UPDATE", "Accounts", columns=["Flag"])]
    assert reconcile_write_inventory(sql, parsed) == ["Missing 1 UPDATE write(s) to ACCOUNTS"]


def test_insert_into_temp_is_one_write_not_select_into():
    sql = "INSERT INTO #Stage (Id) SELECT Id FROM dbo.Source;"
    writes = scan_expected_writes(sql)
    assert [(w.operation, w.target_table) for w in writes] == [("INSERT", "#Stage")]


def test_nested_case_keeps_complete_update_and_resolves_alias():
    path = next(p for p in SAMPLES if p.name.startswith("08_"))
    sql = read_sql_file(path)
    obj = split_objects(sql, path.name, detect_dialect(sql))[0]
    info = analyze_object(obj)
    writes = [s for s in info.statements if s.statement_type == "UPDATE" and "RestructureEligible" in s.columns]
    assert len(writes) == 3
    assert all(s.tables_written == ["LoanAccountCal"] for s in writes)
    assert "FROM PRO.LoanAccountCal A" in writes[0].raw_text
    assert "ELSE 'N'" in writes[0].raw_text


def test_sample_07_08_09_16_anomalies_flagged():
    by_name = {p.name: p for p in SAMPLES}

    a07 = detect_source_anomalies(read_sql_file(by_name["07_DPD_Bucket_Classification.sql"]))
    assert any("NOT_APPLICABLE" in a and "CURRENT" in a for a in a07)

    a08 = detect_source_anomalies(read_sql_file(by_name["08_Loan_Restructuring_Eligibility.sql"]))
    assert any("unreachable" in a.lower() or "never execute" in a.lower() for a in a08)

    a09 = detect_source_anomalies(read_sql_file(by_name["09_Provision_Coverage_Merge.sql"]))
    assert any("always true" in a.lower() or "tautological" in a.lower() for a in a09)

    a16 = detect_source_anomalies(read_sql_file(by_name["16_Guarantee_Cover_Appropriation.sql"]))
    assert any("shared starting value" in a.lower() or "unchanged value" in a.lower() for a in a16)


def test_sample_06_select_into_and_window_classified():
    path = next(p for p in SAMPLES if p.name.startswith("06_"))
    sql = read_sql_file(path)
    expected = scan_expected_writes(sql)
    assert any(w.operation == "SELECT_INTO" for w in expected)
    assert any(w.target_table.upper().endswith("TEMPTABLEAPPGOVGUR") for w in expected)

    obj = split_objects(sql, path.name, detect_dialect(sql))[0]
    info = analyze_object(obj)
    assert any("TEMPTABLEAPPGOVGUR" in t.upper() for t in info.tables_written)
    # ## global temps preserved
    assert any(t.startswith("##") or t.upper() == "ACCOUNTCAL" for t in info.tables_written + info.tables_read)


def test_ready_to_present_blocked_while_unsupported_remain():
    """Job may complete; DD is not ready while unsupported ops exist."""
    results = run_gate()
    # At least complex procedures should list unsupported/manual items.
    with_unsupported = [r for r in results if r.unsupported]
    assert with_unsupported
    assert all(not r.ready_to_present for r in with_unsupported)


def test_export_round_trip_preserves_metadata(tmp_path: Path):
    row = DDRow(
        entity_name="LoanAccountCal",
        column_name="DpdBucket",
        column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.DECISION_TABLE,
        display_derivation_expression='IF(DpdDays==0)THEN("CURRENT")ELSE("X")',
        effective_start_date=date(2024, 1, 1),
        status=DDStatus.PENDING_REVIEW,
        review_state=ReviewState.GENERATED,
        data_type="Text",
        decision_table_json='{"rows":[1]}',
        conditional_json='{"links":[]}',
        source_chain_id="c1",
        source_statement_refs=["07.sql stmt #3"],
        advisory_notes=["staging insert requires workflow"],
        validation_errors=[],
    )
    stored = {
        "expression": 'IF(DpdDays==0)THEN("CURRENT")ELSE("BUCKET_1_30")',
        "status": "ACTIVE",
        "confidence": 0.91,
        "entity_name": row.entity_name,
        "column_name": row.column_name,
        "chain_id": "c1",
        "effective_start_date": "2024-01-01",
        "validation_errors": "[]",
        "row_json": row.model_dump_json(),
    }
    rebuilt = _row_dict_to_dd_row(stored)
    assert rebuilt.data_type == "Text"
    assert rebuilt.decision_table_json == '{"rows":[1]}'
    assert rebuilt.advisory_notes == ["staging insert requires workflow"]
    assert rebuilt.source_statement_refs == ["07.sql stmt #3"]
    assert "BUCKET_1_30" in rebuilt.display_derivation_expression
    assert rebuilt.status == DDStatus.ACTIVE

    csv_path = export_dd_rows_csv([rebuilt], tmp_path / "out.csv")
    text = csv_path.read_text(encoding="utf-8")
    assert "Text" in text
    assert "Decision Table" in text

    qa = write_qa_coverage_report(
        [rebuilt],
        tmp_path / "qa.md",
        blockers=["manual: MERGE DpdBucketHistory"],
    )
    assert "Ready to present: **no**" in qa.read_text(encoding="utf-8")


def test_gate_json_includes_inventories(tmp_path: Path):
    from scripts.release_gate_samples_01_18 import main

    out = tmp_path / "gate.json"
    code = main(["--json", str(out)])
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 18
    assert len(payload["inventories"]) == 18
    # A complete parser inventory is not release approval while unsupported
    # source writes still require a platform workflow.
    assert code == 1
    assert all(not result["ready_to_present"] for result in payload["results"])
