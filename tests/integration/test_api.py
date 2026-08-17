from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.utils import db


@pytest.fixture
def client(tmp_db_path, monkeypatch):
    monkeypatch.setattr(db, "settings", replace(db.settings, sqlite_db_path=tmp_db_path))
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_submit_job_rejects_missing_company(client, dpd_calculation_sql):
    response = client.post(
        "/api/jobs",
        json={
            "company": "",
            "platform": "4X",
            "intent": "Explain",
            "files": {"dpd.sql": dpd_calculation_sql},
        },
    )
    assert response.status_code == 400


def test_submit_job_rejects_bad_file_extension(client):
    response = client.post(
        "/api/jobs",
        json={
            "company": "Acme",
            "platform": "4X",
            "intent": "Explain",
            "files": {"notes.txt": "hello"},
        },
    )
    assert response.status_code == 400


def test_submit_job_rejects_empty_file_list(client):
    response = client.post(
        "/api/jobs",
        json={
            "company": "Acme",
            "platform": "4X",
            "intent": "Explain",
            "files": {},
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "At least one SQL file is required"


def test_submit_job_passes_optional_context_to_pipeline(client, dpd_calculation_sql, monkeypatch):
    captured = {}

    class FakePipeline:
        def invoke(self, state):
            captured.update(state)
            return {"dd_rows": [], "report_path": "output/job-1/report.md"}

    monkeypatch.setattr("app.api.routes.build_pipeline", lambda: FakePipeline())

    response = client.post(
        "/api/jobs",
        json={
            "company": "Acme",
            "platform": "4X",
            "intent": "Generate DD",
            "function_reference": "IF / THEN / ELSE reference",
            "entity_name_map": {"AccountCal_Stg": "FCT_NPA_PRODUCT"},
            "files": {"dpd.sql": dpd_calculation_sql},
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "QUEUED"
    assert captured["function_reference"] == "IF / THEN / ELSE reference"
    assert captured["entity_name_map"] == {"AccountCal_Stg": "FCT_NPA_PRODUCT"}
    assert captured["uploaded_files"] == {"dpd.sql": dpd_calculation_sql}

    status_response = client.get(f"/api/jobs/{response.json()['job_id']}/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "COMPLETED"


def test_get_status_for_unknown_job_returns_404(client):
    response = client.get("/api/jobs/does-not-exist/status")
    assert response.status_code == 404
