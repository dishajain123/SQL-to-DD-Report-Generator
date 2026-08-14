from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.schemas import JobSubmitRequest, JobSubmitResponse
from app.guardrails.input_guardrails import check_input_file, check_job_plan
from app.models.core import JobPlan
from app.orchestration.pipeline import build_pipeline
from app.utils import db

router = APIRouter()


def _execute_job(job_id: str, request_payload: dict[str, Any]) -> None:
    request = JobSubmitRequest.model_validate(request_payload)
    db.update_job_status(job_id, "RUNNING")

    try:
        job_plan = JobPlan(
            job_id=job_id,
            intent=request.intent,
            company=request.company,
            platform=request.platform,
            include_dd_excel=request.include_dd_excel,
        )

        pipeline = build_pipeline()
        result = pipeline.invoke(
            {
                "job_plan": job_plan,
                "uploaded_files": request.files,
                "function_reference": request.function_reference,
                "entity_name_map": request.entity_name_map,
            }
        )
        db.update_job_status(
            job_id,
            "COMPLETED",
            report_path=result.get("report_path"),
            excel_path=result.get("excel_path"),
        )
    except Exception as exc:  # pragma: no cover - defensive background worker guard
        db.update_job_status(job_id, "FAILED", error_message=str(exc))
        db.log_audit(job_id, "job_execution", f"Job failed: {exc}")
        return


@router.post("/jobs", response_model=JobSubmitResponse)
def submit_job(request: JobSubmitRequest, background_tasks: BackgroundTasks) -> JobSubmitResponse:
    plan_check = check_job_plan(request.company, request.platform)
    if not plan_check.passed:
        raise HTTPException(status_code=400, detail=plan_check.errors)

    if not request.files:
        raise HTTPException(status_code=400, detail="At least one SQL file is required")

    for filename, content in request.files.items():
        file_check = check_input_file(filename, content)
        if not file_check.passed:
            raise HTTPException(status_code=400, detail={filename: file_check.errors})

    job_id = f"job-{uuid.uuid4().hex[:10]}"

    db.init_db()
    db.record_job(job_id, request.company, request.platform, request.intent.value, "PENDING")
    background_tasks.add_task(_execute_job, job_id, request.model_dump(mode="json"))

    return JobSubmitResponse(
        job_id=job_id,
        status="QUEUED",
    )


@router.get("/jobs/{job_id}/status")
def get_job_status(job_id: str) -> dict:
    with db.get_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job = dict(row)
    with db.get_connection() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) AS count FROM dd_rows WHERE job_id = ? AND status = 'PENDING_REVIEW'",
            (job_id,),
        ).fetchone()["count"]
        dd_row_count = conn.execute("SELECT COUNT(*) AS count FROM dd_rows WHERE job_id = ?", (job_id,)).fetchone()["count"]

    job["dd_row_count"] = dd_row_count
    job["pending_review_count"] = pending
    return job
