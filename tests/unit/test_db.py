from datetime import date

from app.models.core import ColumnType, DDRow, DDStatus, DerivationOption
from app.utils import db


def test_init_db_creates_tables(tmp_db_path):
    db.init_db(tmp_db_path)
    with db.get_connection(tmp_db_path) as conn:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"jobs", "dd_rows", "review_decisions", "audit_log"} <= tables


def test_record_and_update_job(tmp_db_path):
    db.init_db(tmp_db_path)
    db.record_job("job-1", "Acme", "4X", "Generate DD", "RUNNING", tmp_db_path)
    db.update_job_status("job-1", "COMPLETED", tmp_db_path)

    with db.get_connection(tmp_db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", ("job-1",)).fetchone()
    assert row["status"] == "COMPLETED"


def test_record_dd_row_and_fetch_pending(tmp_db_path):
    db.init_db(tmp_db_path)
    row = DDRow(
        entity_name="FCT_NPA_PRODUCT", column_name="DPD_Overdue", column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression='TODATE("A"."B")',
        effective_start_date=date(2026, 1, 1), status=DDStatus.PENDING_REVIEW,
        data_type="number", source_chain_id="chain-1", confidence=0.4,
        validation_errors=["low confidence"],
    )
    row_id = db.record_dd_row("job-1", "chain-1", 0, row.model_dump(), tmp_db_path)
    assert row_id > 0

    pending = db.get_pending_review_rows(tmp_db_path)
    assert len(pending) == 1
    assert pending[0]["entity_name"] == "FCT_NPA_PRODUCT"


def test_review_decision_updates_status(tmp_db_path):
    db.init_db(tmp_db_path)
    row = DDRow(
        entity_name="FCT_X", column_name="Y", column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression="X", effective_start_date=date(2026, 1, 1),
        status=DDStatus.PENDING_REVIEW, data_type="string", source_chain_id="c1", confidence=0.4,
    )
    row_id = db.record_dd_row("job-1", "c1", 0, row.model_dump(), tmp_db_path)

    db.record_review_decision(row_id, "APPROVE", "alice", db_path=tmp_db_path)

    with db.get_connection(tmp_db_path) as conn:
        updated = conn.execute("SELECT * FROM dd_rows WHERE id = ?", (row_id,)).fetchone()
    assert updated["status"] == "ACTIVE"

    pending_after = db.get_pending_review_rows(tmp_db_path)
    assert len(pending_after) == 0


def test_review_edit_updates_expression(tmp_db_path):
    db.init_db(tmp_db_path)
    row = DDRow(
        entity_name="FCT_X", column_name="Y", column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression="OLD_EXPR", effective_start_date=date(2026, 1, 1),
        status=DDStatus.PENDING_REVIEW, data_type="string", source_chain_id="c1", confidence=0.4,
    )
    row_id = db.record_dd_row("job-1", "c1", 0, row.model_dump(), tmp_db_path)
    db.record_review_decision(row_id, "EDIT", "bob", edited_expression="NEW_EXPR", db_path=tmp_db_path)

    with db.get_connection(tmp_db_path) as conn:
        updated = conn.execute("SELECT * FROM dd_rows WHERE id = ?", (row_id,)).fetchone()
    assert updated["expression"] == "NEW_EXPR"
    assert updated["status"] == "ACTIVE"


def test_log_audit(tmp_db_path):
    db.init_db(tmp_db_path)
    db.log_audit("job-1", "structural_analysis", "3 objects analyzed", tmp_db_path)
    with db.get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT * FROM audit_log WHERE job_id = ?", ("job-1",)).fetchall()
    assert len(rows) == 1
    assert rows[0]["stage"] == "structural_analysis"
