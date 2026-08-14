"""Architecture step 15: Human-in-the-Loop Review — queue management on top
of the SQLite persistence layer."""
from __future__ import annotations

from dataclasses import dataclass

from app.utils import db


@dataclass
class PendingReviewItem:
    id: int
    job_id: str
    chain_id: str
    entity_name: str
    column_name: str
    derivation_option: str
    expression: str
    effective_start_date: str
    confidence: float
    validation_errors: list[str]


def list_pending(db_path: str | None = None) -> list[PendingReviewItem]:
    rows = db.get_pending_review_rows(db_path)
    items = []
    for r in rows:
        import json

        items.append(
            PendingReviewItem(
                id=r["id"],
                job_id=r["job_id"],
                chain_id=r["chain_id"],
                entity_name=r["entity_name"],
                column_name=r["column_name"],
                derivation_option=r["derivation_option"],
                expression=r["expression"] or "",
                effective_start_date=r["effective_start_date"],
                confidence=r["confidence"],
                validation_errors=json.loads(r["validation_errors"] or "[]"),
            )
        )
    return items


def approve(dd_row_id: int, reviewer: str, comment: str = "", db_path: str | None = None) -> None:
    db.record_review_decision(dd_row_id, "APPROVE", reviewer, comment=comment, db_path=db_path)


def reject(dd_row_id: int, reviewer: str, comment: str = "", db_path: str | None = None) -> None:
    db.record_review_decision(dd_row_id, "REJECT", reviewer, comment=comment, db_path=db_path)


def edit(dd_row_id: int, reviewer: str, new_expression: str, comment: str = "", db_path: str | None = None) -> None:
    db.record_review_decision(
        dd_row_id, "EDIT", reviewer, edited_expression=new_expression, comment=comment, db_path=db_path
    )


def override(dd_row_id: int, reviewer: str, comment: str = "", db_path: str | None = None) -> None:
    db.record_review_decision(dd_row_id, "OVERRIDE", reviewer, comment=comment, db_path=db_path)
