"""Regression: structural_errors (from node_structural_analysis) was computed
and stored in pipeline state but never read again anywhere -- an object
could fail structural analysis and the pipeline would still generate and
export DD rows for it with zero trace of the failure in the final report."""
from app.models.core import Intent, JobPlan
from app.orchestration.pipeline import node_report_and_export


def test_structural_errors_surface_as_qa_coverage_blockers(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.orchestration.pipeline.db.get_job_output_dir", lambda job_id: tmp_path
    )
    monkeypatch.setattr(
        "app.orchestration.pipeline.db.update_job_status", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "app.orchestration.pipeline.generate_report",
        lambda *a, **k: tmp_path / "report.md",
    )

    captured: dict = {}

    def fake_write_qa(rows, path, **kwargs):
        captured.update(kwargs)
        return path

    monkeypatch.setattr(
        "app.report.dd_export.write_qa_coverage_report", fake_write_qa
    )

    job_plan = JobPlan(
        job_id="job-structural-errors-1",
        intent=Intent.GENERATE_DD,
        company="Acme Bank",
        platform="4X",
    )
    state = {
        "job_plan": job_plan,
        "objects": {},
        # Empty so the coverage-ledger loop (which needs real StructuralInfo
        # objects) is skipped -- this test targets structural_errors only.
        "structural_infos": {},
        "structural_errors": {
            "obj-broken-1": ["unbalanced BEGIN/END", "unterminated string literal"]
        },
        "canonical_models": [],
        "dd_rows": [],
    }

    node_report_and_export(state)

    blockers = captured.get("blockers") or []
    assert any(
        "obj-broken-1" in b and "structural analysis failed" in b for b in blockers
    )
    assert any("unbalanced BEGIN/END" in b for b in blockers)
