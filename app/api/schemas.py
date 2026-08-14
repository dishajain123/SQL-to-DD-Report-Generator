from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.core import Intent


class JobSubmitRequest(BaseModel):
    company: str
    platform: str
    intent: Intent
    include_dd_excel: bool = False
    function_reference: str = ""
    entity_name_map: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str]  # filename -> raw SQL content


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    report_path: Optional[str] = None
    excel_path: Optional[str] = None
    dd_row_count: int = 0
    pending_review_count: int = 0
