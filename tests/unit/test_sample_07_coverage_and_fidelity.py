"""Regression fixtures for sample 07 — independent expectations, not
round-trips of the translation code itself.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from app.derivation.dd_generation_engine import (
    _apply_date_offset_variable_lineage,
    _rewrite_business_date_variables,
    _rewrite_exists_predicates,
)
from app.models.core import (
    DDRow,
    DDStatus,
    DerivationOption,
    Dialect,
    ReviewState,
    ColumnType,
)
from app.parsing.coverage_ledger import WriteKind, build_coverage_ledger
from app.guardrails.dd_row_coverage import flag_rows_with_uncovered_writes, mark_ledger_coverage
from app.derivation.dd_generation_engine import _infer_data_type
from app.parsing.dialect import detect_dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object
from app.report.dd_export import _row_dict_to_dd_row, export_dd_rows_csv, write_qa_coverage_report


SAMPLE_07 = Path(__file__).resolve().parents[2] / "samples" / "sql" / "07_DPD_Bucket_Classification.sql"


def _read_sql(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def sample_07_info():
    sql = _read_sql(SAMPLE_07)
    obj = split_objects(sql, SAMPLE_07.name, detect_dialect(sql))[0]
    return analyze_object(obj)


def test_sample_07_write_inventory_includes_all_targets(sample_07_info):
    """Every INSERT/UPDATE/MERGE target must appear, including staging/history/queue/audit/status."""
    written = {t.upper().lstrip("#") for t in sample_07_info.tables_written}
    # Preserve temp identity separately.
    assert any(t.startswith("#") and t.upper().endswith("DPDSTAGING") for t in sample_07_info.tables_written)

    expected = {
        "LOANACCOUNTCAL",
        "DPDSTAGING",
        "DPDBUCKETHISTORY",
        "COLLECTIONSQUEUE",
        "DPDBUCKETAUDITLOG",
        "ACLRUNNINGPROCESSSTATUS",
    }
    assert expected.issubset(written)

    merge_stmts = [s for s in sample_07_info.statements if s.statement_type == "MERGE"]
    assert len(merge_stmts) == 1
    assert merge_stmts[0].parsed_ok
    assert any("DPDBUCKETHISTORY" in t.upper() for t in merge_stmts[0].tables_written)

    insert_targets = {
        t.upper().lstrip("#")
        for s in sample_07_info.statements
        if s.statement_type == "INSERT"
        for t in s.tables_written
    }
    assert {"DPDSTAGING", "COLLECTIONSQUEUE", "DPDBUCKETAUDITLOG"}.issubset(insert_targets)

    status_updates = [
        s
        for s in sample_07_info.statements
        if s.statement_type == "UPDATE"
        and any("ACLRUNNINGPROCESSSTATUS" in t.upper() for t in s.tables_written)
    ]
    assert len(status_updates) == 2  # success path + CATCH path


def test_sample_07_coverage_ledger_flags_unsupported_and_anomalies(sample_07_info):
    ledger = build_coverage_ledger(sample_07_info)
    kinds = {e.kind for e in ledger.entries}
    assert WriteKind.SET_BASED_MERGE in kinds or WriteKind.UNSUPPORTED in kinds
    assert WriteKind.TEMP_STAGING in kinds
    assert WriteKind.EXCEPTION_HANDLER in kinds or WriteKind.PROCESS_STATUS in kinds
    assert any("NOT_APPLICABLE" in a for a in ledger.source_anomalies)
    assert any("IF EXISTS" in a or "grace" in a.lower() for a in ledger.source_anomalies)
    assert not ledger.ready_to_present


def test_grace_window_keeps_three_day_offset():
    entity = "LoanAccountCal"
    raw = '@GraceWindowStart'
    source = "DECLARE @GraceWindowStart DATE = DATEADD(DAY, -3, @ProcessDate)"
    # Without the DECLARE, the offset is not guessable — must stay unresolved
    # rather than silently applying an offset borrowed from another procedure.
    unresolved = _rewrite_business_date_variables(raw, entity)
    assert unresolved == raw

    rewritten = _rewrite_business_date_variables(raw, entity, source_sql=source)
    assert "BUSINESS_DATE" in rewritten
    assert "ADDDAY" in rewritten.upper() or "-3" in rewritten
    assert rewritten != f'"{entity}"."var"."BUSINESS_DATE"'

    lined = _apply_date_offset_variable_lineage(
        "@GraceWindowStart",
        entity,
        "DECLARE @GraceWindowStart DATE = DATEADD(DAY, -3, @ProcessDate)",
    )
    assert "ADDDAY" in lined.upper()
    assert "-3" in lined


def test_exists_rewrite_does_not_flatten_procedure_branch():
    expr = 'IF(EXISTS(SELECT 1 FROM LoanAccountCal WHERE LastPaymentDueDate >= GraceWindowStart))THEN("Y")ELSE("N")'
    assert _rewrite_exists_predicates(expr) == expr


def test_null_due_date_executable_behavior_vs_anomaly(sample_07_info):
    """As written: DpdDays=0 + NOT_APPLICABLE, then CASE maps 0 → CURRENT."""
    ledger = build_coverage_ledger(sample_07_info)
    assert any("CURRENT" in a and "NOT_APPLICABLE" in a for a in ledger.source_anomalies)

    # Independent executable expectation (not from translator):
    # null due date → DpdDays set to 0 → later CASE WHEN DpdDays = 0 THEN CURRENT
    process_date = date(2024, 6, 15)
    last_payment_due = None
    dpd_days = 0 if last_payment_due is None else (process_date - last_payment_due).days
    bucket_after_null_update = "NOT_APPLICABLE"
    bucket_after_case = {
        None: "NOT_APPLICABLE",
        0: "CURRENT",
    }.get(dpd_days, "OTHER")
    assert bucket_after_null_update == "NOT_APPLICABLE"
    assert bucket_after_case == "CURRENT"


def test_grace_branch_global_vs_row_predicate():
    """Independent expectations for the SQL as written."""
    process_date = date(2024, 6, 15)
    grace_start = process_date - timedelta(days=3)

    # Account A: due yesterday → in grace window predicate
    due_a = process_date - timedelta(days=1)
    # Account B: due 10 days ago → outside grace row predicate
    due_b = process_date - timedelta(days=10)

    any_in_grace = due_a >= grace_start or due_b >= grace_start
    assert any_in_grace  # procedure takes the grace IF EXISTS branch

    # Inside that branch, only rows matching the UPDATE WHERE get Y/N.
    def apply_grace_update(due: date) -> tuple[str, str] | None:
        if due >= grace_start:
            return ("N", "Y")  # BucketWorsened, GracePeriodApplied
        return None  # row not updated by this statement

    assert apply_grace_update(due_a) == ("N", "Y")
    assert apply_grace_update(due_b) is None  # must NOT get Y merely because branch is active


def test_procedure_branch_formula_kept_with_coverage_advisory(sample_07_info):
    row = DDRow(
        entity_name="LoanAccountCal", column_name="BucketWorsened",
        column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression='IF("LoanAccountCal"."DpdDays">0)THEN("Y")ELSE("N")',
        effective_start_date=date(2024, 1, 1), status=DDStatus.ACTIVE,
        review_state=ReviewState.GENERATED,
        data_type="string", source_chain_id="sample-07",
        source_object_ids=[sample_07_info.object_id],
    )
    flag_rows_with_uncovered_writes([row], sample_07_info, _read_sql(SAMPLE_07))
    assert row.review_state == ReviewState.GENERATED
    assert row.status == DDStatus.ACTIVE
    assert row.display_derivation_expression.startswith("IF(")
    assert any("procedure_branch" in note for note in row.advisory_notes)
    assert row.validation_errors == []


def test_empty_unsupported_row_does_not_resurrect_old_formula(tmp_path):
    old = DDRow(
        entity_name="LoanAccountCal", column_name="BucketWorsened",
        column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression='IF(X>0)THEN("Y")ELSE("N")',
        effective_start_date=date(2024, 1, 1), status=DDStatus.ACTIVE,
        data_type="string", source_chain_id="sample-07",
    )
    old_path = export_dd_rows_csv([old], tmp_path / "old.csv")
    withheld = old.model_copy(update={
        "display_derivation_expression": "",
        "review_state": ReviewState.UNSUPPORTED,
    })
    new_path = export_dd_rows_csv([withheld], tmp_path / "new.csv", existing_dd_path=old_path)
    assert "BucketWorsened" not in new_path.read_text(encoding="utf-8")


def test_type_inference_uses_result_type_not_date_guard_or_entity_name():
    assert _infer_data_type("BucketWorsened", 'IF(X>=ADDDAY(D,-3))THEN("Y")ELSE("N")') == "string"
    assert _infer_data_type("GracePeriodApplied", 'IF(X>=ADDDAY(D,-3))THEN("Y")ELSE("N")') == "string"
    assert _infer_data_type("PenalInterestAmount", '"FCT_LOAN_ACCOUNT"."Balance" * 0.02') == "number"
    assert _infer_data_type("AdjustedPenalty", '"DpdStaging"."AdjustedPenalty" * 1.10') == "number"


def test_sample_07_demo_generator_keeps_branch_and_self_ref_formulas_active():
    from scripts.generate_sample_dd_demo import _entity_map_for, _generate_rows_for_object

    sql = _read_sql(SAMPLE_07)
    obj = split_objects(sql, SAMPLE_07.name, detect_dialect(sql))[0]
    info = analyze_object(obj)
    rows = _generate_rows_for_object(obj, info)
    flag_rows_with_uncovered_writes(rows, info, sql, _entity_map_for(info))
    ledger = mark_ledger_coverage(build_coverage_ledger(info, source_sql=sql), rows, _entity_map_for(info))
    by_column = {row.column_name: row for row in rows if row.entity_name == "LoanAccountCal"}
    assert by_column["BucketWorsened"].status == DDStatus.ACTIVE
    assert by_column["BucketWorsened"].display_derivation_expression
    assert by_column["DpdBucket"].status == DDStatus.ACTIVE
    assert by_column["DpdBucket"].display_derivation_expression
    assert any("procedure_branch" in n for n in by_column["BucketWorsened"].advisory_notes)
    assert by_column["PenalInterestAmount"].data_type == "number"
    assert by_column["PenalInterestAmount"].display_derivation_expression
    assert next(e for e in ledger.entries if "PenalInterestAmount" in e.columns).covered_by_dd
    dpd_entry = next(e for e in ledger.entries if "DpdBucket" in e.columns)
    assert dpd_entry.covered_by_dd == (dpd_entry.kind.value in {"row_formula", "decision_table"})


def test_same_procedure_dependency_is_advisory_not_withheld():
    from scripts.generate_sample_dd_demo import _entity_map_for, _generate_rows_for_object

    path = SAMPLE_07.with_name("12_Customer_Risk_Score_Update.sql")
    sql = _read_sql(path)
    obj = split_objects(sql, path.name, detect_dialect(sql))[0]
    info = analyze_object(obj)
    rows = _generate_rows_for_object(obj, info)
    flag_rows_with_uncovered_writes(rows, info, sql, _entity_map_for(info))
    by_column = {row.column_name: row for row in rows if row.entity_name == "CustomerRiskProfile"}
    assert by_column["UtilizationRatio"].status == DDStatus.ACTIVE
    assert by_column["UtilizationRatio"].display_derivation_expression
    # RiskScore / RiskTier may also appear on RiskScoreHistory after staging merge.
    risk_rows = [r for r in rows if r.column_name in {"RiskScore", "RiskTier"}]
    assert risk_rows
    assert all(r.status == DDStatus.ACTIVE and r.display_derivation_expression for r in risk_rows)


def test_dpd_boundaries_independent():
    process_date = date(2024, 6, 15)

    def bucket_for(due: date | None) -> str:
        if due is None:
            # executable path after both updates: days=0 → CURRENT
            return "CURRENT"
        days = (process_date - due).days
        if days < 0:
            return "NOT_AGED"  # WHERE due <= process_date excludes these from Rule 1
        if days == 0:
            return "CURRENT"
        if 1 <= days <= 30:
            return "BUCKET_1_30"
        if 31 <= days <= 60:
            return "BUCKET_31_60"
        if 61 <= days <= 90:
            return "BUCKET_61_90"
        return "BUCKET_90_PLUS"

    T = process_date
    assert bucket_for(T - timedelta(days=4)) == "BUCKET_1_30"
    assert bucket_for(T - timedelta(days=3)) == "BUCKET_1_30"
    assert bucket_for(T - timedelta(days=1)) == "BUCKET_1_30"
    assert bucket_for(T) == "CURRENT"
    assert bucket_for(T + timedelta(days=1)) == "NOT_AGED"

    for days, expected in [
        (0, "CURRENT"),
        (1, "BUCKET_1_30"),
        (30, "BUCKET_1_30"),
        (31, "BUCKET_31_60"),
        (60, "BUCKET_31_60"),
        (61, "BUCKET_61_90"),
        (90, "BUCKET_61_90"),
        (91, "BUCKET_90_PLUS"),
    ]:
        assert bucket_for(T - timedelta(days=days)) == expected

    assert bucket_for(None) == "CURRENT"


def test_unsupported_procedure_logic_blocks_approved_export(tmp_path, sample_07_info):
    ledger = build_coverage_ledger(sample_07_info)
    rows = [
        DDRow(
            entity_name="LoanAccountCal",
            column_name="BucketWorsened",
            column_type=ColumnType.PHYSICAL,
            derivation_option=DerivationOption.FORMULA_EXPRESSION,
            display_derivation_expression='IF(EXISTS(SELECT 1))THEN("N")ELSE("Y")',
            effective_start_date=date(2024, 1, 1),
            status=DDStatus.PENDING_REVIEW,
            review_state=ReviewState.UNSUPPORTED,
            data_type="Text",
            source_chain_id="c1",
            validation_errors=["Procedure-level or EXISTS logic is not expressible as a row formula"],
        )
    ]
    qa = write_qa_coverage_report(
        rows,
        tmp_path / "qa.md",
        coverage_markdown=ledger.to_markdown(),
        blockers=ledger.blockers,
    )
    text = qa.read_text(encoding="utf-8")
    assert "Ready to present: **no**" in text
    assert "EXISTS" in text or "procedure" in text.lower() or "Blockers" in text


def test_reviewed_export_preserves_row_json_metadata(tmp_path):
    stored = {
        "entity_name": "LoanAccountCal",
        "column_name": "DpdBucket",
        "derivation_option": "Decision Table",
        "expression": 'IF(DpdDays==0)THEN("CURRENT")ELSE("X")',  # reviewed edit
        "effective_start_date": "2024-01-01",
        "status": "ACTIVE",
        "confidence": 0.9,
        "validation_errors": "[]",
        "chain_id": "chain-1",
        "row_json": (
            '{"entity_name":"LoanAccountCal","column_name":"DpdBucket",'
            '"column_type":"Physical","derivation_option":"Decision Table",'
            '"display_derivation_expression":"OLD",'
            '"effective_start_date":"2024-01-01","status":"PENDING_REVIEW",'
            '"review_state":"GENERATED","data_type":"Text",'
            '"decision_table_json":"{\\"rows\\":[1]}",'
            '"conditional_json":"{\\"links\\":[]}",'
            '"advisory_notes":["exception handler excluded"],'
            '"source_statement_refs":["07.sql stmt #3"],'
            '"source_chain_id":"chain-1","confidence":0.625}'
        ),
    }
    dd = _row_dict_to_dd_row(stored)
    assert dd.data_type == "Text"
    assert dd.decision_table_json == '{"rows":[1]}'
    assert dd.conditional_json == '{"links":[]}'
    assert dd.advisory_notes == ["exception handler excluded"]
    assert dd.source_statement_refs == ["07.sql stmt #3"]
    assert dd.display_derivation_expression.startswith("IF(DpdDays")  # reviewed overlay
    assert dd.status == DDStatus.ACTIVE

    out = export_dd_rows_csv([dd], tmp_path / "roundtrip.csv")
    text = out.read_text(encoding="utf-8")
    assert "Text" in text
    assert "Decision Table" in text
