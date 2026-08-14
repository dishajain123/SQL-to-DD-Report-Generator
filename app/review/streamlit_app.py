"""Streamlit app for job intake and human review.

Run with: streamlit run app/review/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.review import review_store
from app.utils import db
from app.utils.config import settings


DEFAULT_API_BASE_URL = os.getenv("DD_AUTOMATION_API_URL", "http://127.0.0.1:8000")

db.init_db()

st.set_page_config(page_title="DD Automation — Intake & Review", layout="wide")
st.title("DD Automation — Intake & Review")
if "ui_logs" not in st.session_state:
    st.session_state["ui_logs"] = []


def _log(message: str) -> None:
    st.session_state["ui_logs"].append(message)


def _render_logs() -> None:
    logs = st.session_state.get("ui_logs", [])
    with st.expander("Run log", expanded=bool(logs)):
        if not logs:
            st.caption("Submission events will appear here.")
        else:
            st.code("\n".join(logs[-100:]), language="text")


def _load_default_function_reference() -> str:
    project_root = Path(__file__).resolve().parents[2]
    path = Path(settings.default_function_reference_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _load_default_entity_name_map() -> dict[str, str]:
    raw = settings.default_entity_name_map_json.strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("DEFAULT_ENTITY_NAME_MAP_JSON must be a JSON object.")

    entity_name_map: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("DEFAULT_ENTITY_NAME_MAP_JSON keys and values must both be strings.")
        clean_key = key.strip()
        clean_value = value.strip()
        if not clean_key or not clean_value:
            raise ValueError("DEFAULT_ENTITY_NAME_MAP_JSON keys and values cannot be blank.")
        entity_name_map[clean_key] = clean_value
    return entity_name_map


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"detail": raw or exc.reason}
        return exc.code, parsed
    except (error.URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", str(exc))
        raise RuntimeError(f"Could not reach the API at {url}: {reason}") from exc


def _get_json(url: str) -> tuple[int, dict]:
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"detail": raw or exc.reason}
        return exc.code, parsed
    except (error.URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", str(exc))
        raise RuntimeError(f"Could not reach the API at {url}: {reason}") from exc


def _wait_for_job(api_base_url: str, job_id: str, max_wait_seconds: int = 1800) -> dict:
    status_url = api_base_url.rstrip("/") + f"/api/jobs/{job_id}/status"
    start = time.monotonic()
    last_status = ""

    while True:
        status_code, response_body = _get_json(status_url)
        if status_code >= 400:
            raise RuntimeError(response_body.get("detail", f"HTTP {status_code} from status endpoint"))

        last_status = str(response_body.get("status", "")).upper()
        _log(f"Job {job_id} status: {last_status or '(unknown)'}")

        if last_status in {"COMPLETED", "FAILED"}:
            return response_body

        if time.monotonic() - start >= max_wait_seconds:
            raise TimeoutError(
                f"Job {job_id} is still {last_status.lower() or 'running'} after {max_wait_seconds} seconds."
            )

        time.sleep(2)


def _render_submission_tab() -> None:
    st.subheader("1. DD Intake")
    st.caption("Upload SQL files and submit.")

    with st.form("job_submission_form", clear_on_submit=False):
        api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL)
        include_dd_excel = st.checkbox("Include DD Excel export", value=False)
        uploaded_files = st.file_uploader(
            "SQL files",
            type=["sql"],
            accept_multiple_files=True,
            help="Upload one or more .sql files.",
        )
        submitted = st.form_submit_button("Submit job")

    if not submitted:
        return

    st.session_state["ui_logs"] = []
    _log("Preparing submission...")

    if not uploaded_files:
        _log("No SQL files were uploaded.")
        _render_logs()
        st.error("Please upload at least one .sql file.")
        return

    files: dict[str, str] = {}
    for uploaded in uploaded_files:
        if not uploaded.name.lower().endswith(".sql"):
            _log(f"Rejected file: {uploaded.name} (unsupported extension)")
            _render_logs()
            st.error(f"Unsupported file type: {uploaded.name}")
            return
        try:
            files[uploaded.name] = uploaded.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            _log(f"Rejected file: {uploaded.name} (invalid UTF-8)")
            _render_logs()
            st.error(f"{uploaded.name} is not valid UTF-8 text.")
            return

    try:
        function_reference = _load_default_function_reference()
    except OSError as exc:
        _log(f"Failed to load function reference: {exc}")
        _render_logs()
        st.error(f"Could not load the default function reference: {exc}")
        return

    try:
        entity_name_map = _load_default_entity_name_map()
    except (json.JSONDecodeError, ValueError) as exc:
        _log(f"Invalid entity name map config: {exc}")
        _render_logs()
        st.error(f"DEFAULT_ENTITY_NAME_MAP_JSON is invalid: {exc}")
        return

    _log(f"Loaded {len(files)} SQL file(s).")
    _log("Submitting job to the API...")

    payload = {
        "company": settings.default_company_name,
        "platform": settings.default_platform_name,
        "intent": settings.default_intent,
        "include_dd_excel": include_dd_excel,
        "function_reference": function_reference,
        "entity_name_map": entity_name_map,
        "files": files,
    }

    with st.spinner("Submitting job..."):
        try:
            status_code, response_body = _post_json(api_base_url.rstrip("/") + "/api/jobs", payload)
        except RuntimeError as exc:
            _log(f"API request failed: {exc}")
            _render_logs()
            st.error(str(exc))
            return

    _log(f"API returned HTTP {status_code}.")
    if status_code >= 400:
        _log("Job submission failed.")
        _render_logs()
        st.error("Job submission failed.")
        st.json(response_body)
        return

    job_id = response_body.get("job_id", "(unknown job id)")
    _log(f"Job submitted successfully: {job_id}")

    try:
        final_status = _wait_for_job(api_base_url, job_id)
    except TimeoutError as exc:
        _log(str(exc))
        _render_logs()
        st.info(f"Job {job_id} was accepted and is still processing. You can check status later.")
        st.session_state["last_submission"] = response_body
        return
    except RuntimeError as exc:
        _log(f"Status polling failed: {exc}")
        _render_logs()
        st.error(str(exc))
        st.session_state["last_submission"] = response_body
        return

    _render_logs()
    if str(final_status.get("status", "")).upper() == "FAILED":
        st.error(f"Job {job_id} failed: {final_status.get('error_message', 'Unknown error')}")
    else:
        st.success(f"Job submitted successfully: {job_id}")

    st.json(final_status)
    st.session_state["last_submission"] = final_status


def _render_review_tab() -> None:
    st.subheader("2. Human Review Queue")

    pending = review_store.list_pending()

    if not pending:
        st.success("No items pending review.")
        return

    st.write(f"{len(pending)} item(s) pending review.")

    for item in pending:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"**{item.entity_name}.{item.column_name}**  "
                    f"(chain `{item.chain_id}`, job `{item.job_id}`)"
                )
                st.code(item.expression or "(no expression — see Decision Table)", language="text")
                if item.validation_errors:
                    st.error("Validation errors:\n" + "\n".join(f"- {e}" for e in item.validation_errors))
            with col2:
                st.metric("Confidence", f"{item.confidence:.2f}")
                st.caption(f"Effective: {item.effective_start_date}")

            edited = st.text_area("Edit expression (optional)", value=item.expression, key=f"edit-{item.id}")
            reviewer = st.text_input("Reviewer", value="reviewer", key=f"reviewer-{item.id}")
            comment = st.text_input("Comment", key=f"comment-{item.id}")

            b1, b2, b3, b4 = st.columns(4)
            if b1.button("Approve", key=f"approve-{item.id}"):
                review_store.approve(item.id, reviewer, comment)
                st.rerun()
            if b2.button("Reject", key=f"reject-{item.id}"):
                review_store.reject(item.id, reviewer, comment)
                st.rerun()
            if b3.button("Save Edit", key=f"save-{item.id}"):
                review_store.edit(item.id, reviewer, edited, comment)
                st.rerun()
            if b4.button("Override", key=f"override-{item.id}"):
                review_store.override(item.id, reviewer, comment)
                st.rerun()


tab_input, tab_review = st.tabs(["Input & Intake", "Human Review"])
with tab_input:
    _render_submission_tab()
    _render_logs()
with tab_review:
    _render_review_tab()
