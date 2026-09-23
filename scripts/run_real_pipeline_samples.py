#!/usr/bin/env python3
"""Run the REAL production pipeline (same path as Streamlit -> FastAPI ->
build_pipeline, real LLM calls included) for samples/sql/01-18, one job per
file. Writes each job's output under output/real_pipeline_01_to_18/<job_id>/
via the normal db.get_job_output_dir() mechanism, and prints a small map of
sample file -> job_id -> output dir at the end.

Usage:
  .venv/bin/python -m scripts.run_real_pipeline_samples            # all 18
  .venv/bin/python -m scripts.run_real_pipeline_samples --only 01  # smoke test one
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from app.models.core import Intent, JobPlan
from app.orchestration.pipeline import build_pipeline
from app.parsing.write_inventory_scan import sample_sql_paths
from app.utils import db
from app.utils.text_encoding import decode_text_bytes

ROOT = Path(__file__).resolve().parents[1]
RESULT_MAP_PATH = ROOT / "output" / "real_pipeline_01_to_18" / "job_map.json"


def run_one(path: Path) -> dict:
    job_id = f"job-{uuid.uuid4().hex[:10]}"
    db.init_db()
    db.record_job(job_id, company="Demo Bank", platform="4x", intent=Intent.GENERATE_DD.value, status="PENDING")
    db.update_job_status(job_id, "RUNNING")

    job_plan = JobPlan(job_id=job_id, intent=Intent.GENERATE_DD, company="Demo Bank", platform="4x")
    pipeline = build_pipeline()
    result = pipeline.invoke(
        {
            "job_plan": job_plan,
            "uploaded_files": {path.name: decode_text_bytes(path.read_bytes()).text},
            "function_reference": "",
            "entity_name_map": {},
            "timekey_map": {},
        }
    )
    db.update_job_status(
        job_id,
        "COMPLETED",
        report_path=result.get("report_path"),
        excel_path=result.get("excel_path"),
    )
    out_dir = db.get_job_output_dir(job_id)
    dd_rows = result.get("dd_rows") or []
    active = sum(1 for r in dd_rows if getattr(r, "status", None) and r.status.value == "ACTIVE")
    return {
        "file": path.name,
        "job_id": job_id,
        "output_dir": str(out_dir),
        "rows": len(dd_rows),
        "active": active,
        "pending": len(dd_rows) - active,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="two-digit sample prefix, e.g. 01", default=None)
    args = parser.parse_args()

    paths = sample_sql_paths()
    paths = [p for p in paths if p.stem[:2].isdigit() and 1 <= int(p.stem[:2]) <= 18]
    if args.only:
        paths = [p for p in paths if p.name.startswith(args.only)]
        if not paths:
            print(f"No sample starts with {args.only}", file=sys.stderr)
            sys.exit(1)

    results = []
    for path in sorted(paths):
        print(f"Running real pipeline for {path.name} ...", flush=True)
        try:
            res = run_one(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", file=sys.stderr)
            results.append({"file": path.name, "error": str(exc)})
            continue
        print(f"  job_id={res['job_id']} rows={res['rows']} active={res['active']} pending={res['pending']}")
        results.append(res)

    RESULT_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_MAP_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote job map to {RESULT_MAP_PATH}")


if __name__ == "__main__":
    main()
