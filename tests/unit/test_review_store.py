from datetime import date

from app.models.core import ColumnType, DDRow, DDStatus, DerivationOption
from app.review import review_store
from app.utils import db


def _add_pending_row(db_path: str) -> int:
    row = DDRow(
        entity_name="FCT_X", column_name="Y", column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression="EXPR", effective_start_date=date(2026, 1, 1),
        status=DDStatus.PENDING_REVIEW, data_type="string", source_chain_id="c1", confidence=0.3,
        validation_errors=["needs review"],
    )
    return db.record_dd_row("job-1", "c1", 0, row.model_dump(), db_path)


def test_list_pending_returns_items(tmp_db_path):
    db.init_db(tmp_db_path)
    _add_pending_row(tmp_db_path)
    pending = review_store.list_pending(tmp_db_path)
    assert len(pending) == 1
    assert pending[0].entity_name == "FCT_X"
    assert pending[0].validation_errors == ["needs review"]


def test_approve_removes_from_pending(tmp_db_path):
    db.init_db(tmp_db_path)
    row_id = _add_pending_row(tmp_db_path)
    review_store.approve(row_id, "alice", "looks good", tmp_db_path)
    assert review_store.list_pending(tmp_db_path) == []


def test_edit_updates_expression_and_removes_from_pending(tmp_db_path):
    db.init_db(tmp_db_path)
    row_id = _add_pending_row(tmp_db_path)
    review_store.edit(row_id, "bob", "NEW_EXPR", db_path=tmp_db_path)
    assert review_store.list_pending(tmp_db_path) == []
    with db.get_connection(tmp_db_path) as conn:
        updated = conn.execute("SELECT * FROM dd_rows WHERE id = ?", (row_id,)).fetchone()
    assert updated["expression"] == "NEW_EXPR"


def test_reject_marks_inactive(tmp_db_path):
    db.init_db(tmp_db_path)
    row_id = _add_pending_row(tmp_db_path)
    review_store.reject(row_id, "alice", db_path=tmp_db_path)
    with db.get_connection(tmp_db_path) as conn:
        updated = conn.execute("SELECT * FROM dd_rows WHERE id = ?", (row_id,)).fetchone()
    assert updated["status"] == "INACTIVE"
