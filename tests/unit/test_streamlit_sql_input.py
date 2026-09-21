"""Verify each Streamlit SQL source becomes the API's files payload."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.review.sql_input import (
    bundled_sample_names,
    bundled_sql_file,
    pasted_sql_file,
    uploaded_sql_files,
)
from app.utils import db

# AppTest.from_file() resolves a relative path against the file that calls
# it (this test module's directory), not the process's working directory
# -- an absolute path is required so the test doesn't depend on where
# pytest happens to be invoked from.
STREAMLIT_APP_PATH = str(Path(__file__).resolve().parents[2] / "app" / "review" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def isolated_streamlit_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "settings", replace(db.settings, sqlite_db_path=str(tmp_path / "streamlit.db")))


class _Response:
    status = 200

    def __init__(self, body: bytes):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _capture_submissions(captured):
    def fake_urlopen(req, timeout):
        if req.get_method() == "POST":
            captured.append(json.loads(req.data))
            return _Response(b'{"job_id":"job-ui-test","status":"QUEUED"}')
        if req.full_url.endswith("/status"):
            return _Response(b'{"job_id":"job-ui-test","status":"COMPLETED"}')
        return _Response(b"")

    return fake_urlopen


def test_bundled_sample_preview_and_submission():
    name = "customer_risk_flag_mysql.sql"
    sql = (Path(__file__).resolve().parents[2] / "samples" / "sql" / name).read_bytes().decode("utf-8")
    captured = []
    with patch("urllib.request.urlopen", _capture_submissions(captured)):
        app = AppTest.from_file(STREAMLIT_APP_PATH, default_timeout=15).run()
        assert not app.exception
        app.radio(key="sql-input-mode").set_value("Bundled sample").run()
        assert not app.exception
        assert name in app.selectbox(key="bundled-sql-select").options
        app.selectbox(key="bundled-sql-select").set_value(name).run()
        assert any(sql.strip() == block.value.strip() for block in app.code)
        app.button[0].click().run()

    assert not app.exception
    assert captured[0]["files"] == {name: sql}


def test_pasted_sql_submission():
    sql = "CREATE PROCEDURE pasted_test AS SELECT 1;"
    captured = []
    with patch("urllib.request.urlopen", _capture_submissions(captured)):
        app = AppTest.from_file(STREAMLIT_APP_PATH, default_timeout=15).run()
        app.radio(key="sql-input-mode").set_value("Paste SQL").run()
        app.text_area(key="pasted-sql-text").set_value(sql).run()
        assert any(sql == block.value for block in app.code)
        app.button[0].click().run()

    assert not app.exception
    assert captured[0]["files"] == {"pasted_procedure.sql": sql}


def test_same_job_in_submission_and_review_has_unique_widget_keys(tmp_path, monkeypatch):
    job_id = "job-ui-both-tabs"
    monkeypatch.setattr(
        db, "settings",
        replace(db.settings, output_dir=str(tmp_path / "output")),
    )
    db.init_db()
    db.record_job(job_id, "Acme Bank", "4X", "Generate DD", "COMPLETED")
    report = tmp_path / "report.md"
    report.write_text("# Business Understanding\n\nSample report.\n", encoding="utf-8")
    db.update_job_status(job_id, "COMPLETED", report_path=str(report))
    db.record_dd_row(job_id, "chain-ui", 0, {
        "entity_name": "LoanAccountCal", "column_name": "DpdDays",
        "column_type": "Physical", "derivation_option": "Formula Expression",
        "display_derivation_expression": '"LoanAccountCal"."DaysPastDue"',
        "effective_start_date": "2026-01-01", "status": "ACTIVE",
        "review_state": "GENERATED", "data_type": "number", "confidence": 1.0,
    })
    output_dir = db.get_job_output_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qa_coverage_report.md").write_text(
        "# QA\nReady to present: **no**\n", encoding="utf-8"
    )

    def fake_urlopen(req, timeout):
        if req.get_method() == "POST":
            return _Response(json.dumps({"job_id": job_id, "status": "QUEUED"}).encode())
        if req.full_url.endswith("/status"):
            return _Response(json.dumps({"job_id": job_id, "status": "COMPLETED"}).encode())
        if req.full_url.endswith("/report"):
            return _Response(report.read_bytes())
        if req.full_url.endswith("/excel"):
            return _Response(b"test-excel-bytes")
        return _Response(b"test-csv-bytes")

    with patch("urllib.request.urlopen", fake_urlopen):
        app = AppTest.from_file(STREAMLIT_APP_PATH, default_timeout=30).run()
        app.radio(key="sql-input-mode").set_value("Paste SQL").run()
        app.text_area(key="pasted-sql-text").set_value("CREATE PROCEDURE demo AS SELECT 1;").run()
        app.button[0].click().run()

    assert not app.exception
    textarea_keys = {area.key for area in app.text_area}
    assert f"raw-md-submission-{job_id}" in textarea_keys
    assert f"raw-md-review-{job_id}" in textarea_keys


def test_submit_without_sql_does_not_call_api():
    captured = []
    with patch("urllib.request.urlopen", _capture_submissions(captured)):
        app = AppTest.from_file(STREAMLIT_APP_PATH, default_timeout=15).run()
        app.button[0].click().run()

    assert not app.exception
    assert not captured
    assert any("Upload a .sql file" in error.value for error in app.error)


def test_uploaded_sql_decoding_and_duplicate_name_check():
    class Upload:
        def __init__(self, name, content):
            self.name = name
            self.content = content

        def getvalue(self):
            return self.content

    upload = Upload("procedure.sql", "SELECT 1;".encode("utf-16"))
    assert uploaded_sql_files([upload]) == {"procedure.sql": "SELECT 1;"}
    with pytest.raises(ValueError, match="Duplicate uploaded filename"):
        uploaded_sql_files([upload, upload])


def test_bundled_samples_are_allowlisted_and_paste_requires_content(tmp_path):
    (tmp_path / "a.sql").write_text("SELECT 1;")
    (tmp_path / "ignore.txt").write_text("SELECT 2;")
    assert bundled_sample_names(tmp_path) == ["a.sql"]
    assert bundled_sql_file("a.sql", tmp_path) == {"a.sql": "SELECT 1;"}
    with pytest.raises(ValueError, match="Select a SQL file"):
        bundled_sql_file("../outside.sql", tmp_path)
    with pytest.raises(ValueError, match="File is empty"):
        pasted_sql_file("   ")
