"""Streamlit app for job intake and human review.

Run with: streamlit run app/review/streamlit_app.py
"""
from __future__ import annotations

import importlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.report import dd_export

# Streamlit can rerun this script while retaining an older imported module.
# Refresh it when the export API was added after the server started.
if not all(
    hasattr(dd_export, name)
    for name in ("export_reviewed_dd_rows_for_job_csv", "export_reviewed_dd_rows_for_job_excel")
):
    importlib.reload(dd_export)

COLUMNS = dd_export.COLUMNS
export_reviewed_dd_rows_for_job_csv = dd_export.export_reviewed_dd_rows_for_job_csv
export_reviewed_dd_rows_for_job_excel = dd_export.export_reviewed_dd_rows_for_job_excel
from app.review import review_store
from app.review.sql_input import bundled_sample_names, bundled_sql_file, pasted_sql_file, uploaded_sql_files
from app.utils import db
from app.utils.config import settings


DEFAULT_API_BASE_URL = os.getenv("DD_AUTOMATION_API_URL", "http://127.0.0.1:8000")
DIALECT_OPTIONS = {
    "Auto-detect": "auto",
    "Oracle SQL / PL-SQL": "oracle",
    "SQL Server T-SQL": "tsql",
    "MySQL": "mysql",
}

db.init_db()

st.set_page_config(
    page_title="DD Automation — Logic & Business Rules",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.6rem; max-width: 1180px; }
        .dda-eyebrow {
            font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase;
            color: #64748B; font-weight: 600; margin-bottom: 0.15rem;
        }
        .dda-badge {
            display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
            background: #EEF2F7; color: #155E75; font-size: 0.75rem; font-weight: 600;
            margin-right: 0.4rem;
        }
        div[data-testid="stMetricValue"] { font-size: 1.35rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="dda-eyebrow">Agentic RAG · Core Banking · 4X Platform</div>', unsafe_allow_html=True)
st.title("🏦 DD Automation — Logic & Business Rules")
st.write(
    "Turn a banking stored procedure into **platform DD conditions** and a structured "
    "**business-focused** Markdown report — what it does and why, not a restatement of the SQL."
)
st.markdown(
    '<span class="dda-badge">SQL → 4X DD</span>'
    '<span class="dda-badge">Business report</span>'
    '<span class="dda-badge">Excel / CSV export</span>'
    '<span class="dda-badge">Human review</span>',
    unsafe_allow_html=True,
)
st.divider()

if "ui_logs" not in st.session_state:
    st.session_state["ui_logs"] = []


def _log(message: str) -> None:
    st.session_state["ui_logs"].append(message)


def _render_copy_button(text: str, key: str = "copy-raw-md") -> None:
    payload = json.dumps(text)
    components.html(
        f"""
        <div style="display:flex; justify-content:flex-end; margin: 0.15rem 0 0.5rem 0;">
            <button id="{key}"
                style="background:#0F766E;color:white;border:none;border-radius:0.5rem;
                       padding:0.55rem 0.9rem;font-size:0.9rem;font-weight:600;cursor:pointer;">
                Copy raw markdown
            </button>
        </div>
        <script>
            const button = document.getElementById("{key}");
            const rawMarkdown = {payload};
            button.addEventListener("click", async () => {{
                try {{ await navigator.clipboard.writeText(rawMarkdown); }}
                catch (err) {{
                    const textarea = document.createElement("textarea");
                    textarea.value = rawMarkdown;
                    document.body.appendChild(textarea);
                    textarea.select();
                    document.execCommand("copy");
                    document.body.removeChild(textarea);
                }}
                const previous = button.textContent;
                button.textContent = "Copied";
                setTimeout(() => {{ button.textContent = previous; }}, 1200);
            }});
        </script>
        """,
        height=58,
    )


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
    from app.utils.entity_name_map import load_configured_entity_overrides

    return load_configured_entity_overrides()


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
    except (error.URLError, TimeoutError, socket.timeout, ConnectionResetError, ConnectionError, OSError) as exc:
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
    except (error.URLError, TimeoutError, socket.timeout, ConnectionResetError, ConnectionError, OSError) as exc:
        reason = getattr(exc, "reason", str(exc))
        raise RuntimeError(f"Could not reach the API at {url}: {reason}") from exc


def _get_bytes(url: str) -> tuple[int, bytes]:
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except error.HTTPError as exc:
        return exc.code, exc.read()
    except (error.URLError, TimeoutError, socket.timeout, ConnectionResetError, ConnectionError, OSError) as exc:
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


def _fetch_report_markdown(api_base_url: str, job_id: str, job_row: dict | None = None) -> str | None:
    report_url = api_base_url.rstrip("/") + f"/api/jobs/{job_id}/report"
    try:
        status, payload = _get_bytes(report_url)
        if status < 400 and payload:
            return payload.decode("utf-8", errors="replace")
    except RuntimeError:
        pass

    report_path = (job_row or {}).get("report_path")
    if report_path and Path(report_path).exists():
        return Path(report_path).read_text(encoding="utf-8", errors="replace")
    return None


def _dd_rows_as_sample_dataframe(dd_rows: list[dict]) -> pd.DataFrame:
    """Map stored DD rows onto the sample Derivations column schema."""
    records = []
    for row in dd_rows:
        records.append(
            {
                "Entity Name": row.get("entity_name") or "",
                "Column Name": row.get("column_name") or "",
                "Column Type": row.get("column_type") or "",
                "Derivation Option": row.get("derivation_option") or "",
                "Display Derivation Expression": row.get("expression")
                or row.get("display_derivation_expression")
                or "",
                "Effective Start Date": row.get("effective_start_date") or "",
                "Status": row.get("status") or "",
                "Data Type": row.get("data_type") or "",
                "Decision Table Json": row.get("decision_table_json") or "",
                "Conditional Json": row.get("conditional_json") or "",
            }
        )
    if not records:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(records, columns=COLUMNS)


def _render_markdown_previews(markdown_text: str, job_id: str, scope: str) -> None:
    widget_key = f"{scope}-{job_id}"
    st.subheader("Business Understanding Report")
    preview_tab, raw_tab = st.tabs(["Rendered preview", "Raw Markdown (copy)"])
    with preview_tab:
        st.markdown(markdown_text, unsafe_allow_html=True)
    with raw_tab:
        _render_copy_button(markdown_text, key=f"copy-md-{widget_key}")
        st.caption("Select all in the box below to copy-paste into Confluence, GitHub, or email.")
        st.text_area(
            "Raw Markdown",
            value=markdown_text,
            height=420,
            key=f"raw-md-{widget_key}",
        )
        st.download_button(
            label="Download report.md",
            data=markdown_text.encode("utf-8"),
            file_name=f"business_understanding_{job_id}.md",
            mime="text/markdown",
            key=f"download-report-md-{widget_key}",
        )


def _render_excel_preview_and_download(
    api_base_url: str, job_id: str, dd_rows: list[dict], scope: str
) -> None:
    widget_key = f"{scope}-{job_id}"

    st.subheader("DD Conditions — Excel preview")
    st.caption(
        "Columns match `samples/derivations/sample_derivations.csv` "
        "(Entity Name, Column Name, Derivation Option, Display Derivation Expression, …). "
        "Platform Status is separate from review_state; generation never auto-approves."
    )
    frame = _dd_rows_as_sample_dataframe(dd_rows)
    # Surface review metadata that the platform CSV omits.
    meta_cols = []
    if dd_rows and any("review_state" in r for r in dd_rows):
        meta_cols.append("review_state")
    if dd_rows and any(r.get("advisory_notes") for r in dd_rows):
        meta_cols.append("advisory_notes")
    if meta_cols:
        meta_frame = pd.DataFrame(
            [
                {
                    "Entity Name": r.get("entity_name") or "",
                    "Column Name": r.get("column_name") or "",
                    "review_state": r.get("review_state") or "",
                    "advisory_notes": "; ".join(r.get("advisory_notes") or [])
                    if isinstance(r.get("advisory_notes"), list)
                    else (r.get("advisory_notes") or ""),
                    "validation_errors": "; ".join(r.get("validation_errors") or [])
                    if isinstance(r.get("validation_errors"), list)
                    else (r.get("validation_errors") or ""),
                }
                for r in dd_rows
            ]
        )
        st.caption("Review / advisory metadata (not in platform CSV columns)")
        st.dataframe(meta_frame, use_container_width=True, hide_index=True, height=220, key=f"review-meta-{widget_key}")

    st.dataframe(frame, use_container_width=True, hide_index=True, height=360, key=f"dd-preview-{widget_key}")

    excel_bytes: bytes | None = None
    excel_url = api_base_url.rstrip("/") + f"/api/jobs/{job_id}/excel"
    try:
        status, payload = _get_bytes(excel_url)
        if status < 400 and payload:
            excel_bytes = payload
    except RuntimeError:
        excel_bytes = None

    if excel_bytes is None:
        fallback = export_reviewed_dd_rows_for_job_excel(
            job_id, db.get_job_output_dir(job_id) / "dd_export_reviewed.xlsx"
        )
        excel_bytes = fallback.read_bytes()

    st.download_button(
        label="Download reviewed DD Excel (.xlsx)",
        data=excel_bytes,
        file_name=f"dd_export_reviewed_{job_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download-excel-{widget_key}",
    )


def _download_reviewed_csv(api_base_url: str, job_id: str) -> None:
    csv_url = api_base_url.rstrip("/") + f"/api/jobs/{job_id}/csv"
    try:
        status, payload = _get_bytes(csv_url)
    except RuntimeError as exc:
        st.warning(f"Could not fetch the reviewed CSV from the API: {exc}. Falling back to a local export.")
        fallback_path = export_reviewed_dd_rows_for_job_csv(job_id, db.get_job_output_dir(job_id) / "dd_export_reviewed.csv")
        payload = fallback_path.read_bytes()
        status = 200

    if status >= 400:
        fallback_path = export_reviewed_dd_rows_for_job_csv(job_id, db.get_job_output_dir(job_id) / "dd_export_reviewed.csv")
        payload = fallback_path.read_bytes()

    st.download_button(
        label="Download Reviewed CSV",
        data=payload,
        file_name=f"dd_export_reviewed_{job_id}.csv",
        mime="text/csv",
        key=f"download-reviewed-csv-{job_id}",
    )


def _list_jobs() -> list[dict]:
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT job_id, company, platform, intent, status, run_number, report_path, excel_path, created_at, updated_at "
            "FROM jobs ORDER BY COALESCE(run_number, 0) DESC, updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def _get_job_dd_rows(job_id: str) -> list[dict]:
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM dd_rows WHERE job_id = ? ORDER BY row_index, id",
            (job_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _render_submission_tab() -> None:
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.caption(f"Provider: `{settings.llm_provider}`")
        st.caption(f"Model: `{settings.llm_model_name or '(from .env)'}`")
        st.caption("API key is read from `.env` and never shown in the UI.")
        st.subheader("SQL Dialect")
        dialect_label = st.selectbox(
            "Preferred dialect hint",
            list(DIALECT_OPTIONS.keys()),
            index=0,
            help="Auto-detect is used by the pipeline from SQL structure. "
            "This hint is stored with the session for review context.",
        )
        st.session_state["preferred_dialect"] = DIALECT_OPTIONS[dialect_label]
        st.divider()
        st.caption("Company / platform defaults come from `.env`.")

    st.subheader("1. Provide a DB object")
    st.caption("Upload, paste, or pick a bundled sample — then run extraction against the API.")

    api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL)
    input_mode = st.radio(
        "Input source",
        ("Upload files", "Bundled sample", "Paste SQL"),
        horizontal=True,
        key="sql-input-mode",
        label_visibility="collapsed",
    )

    files: dict[str, str] = {}
    preview_name: str | None = None
    input_error: str | None = None

    if input_mode == "Upload files":
        uploads = st.file_uploader(
            "SQL files",
            type=["sql", "prc", "pks", "pkb", "txt"],
            accept_multiple_files=True,
            help="Upload one or more SQL procedure files.",
        )
        if uploads:
            try:
                files = uploaded_sql_files(uploads)
                preview_name = st.selectbox("Preview uploaded procedure", list(files))
            except ValueError as exc:
                input_error = str(exc)
    elif input_mode == "Bundled sample":
        sample_names = bundled_sample_names()
        if sample_names:
            selected_sample = st.selectbox(
                "Bundled SQL file",
                sample_names,
                index=None,
                placeholder="Choose a sample procedure",
                key="bundled-sql-select",
            )
            if selected_sample:
                try:
                    files = bundled_sql_file(selected_sample)
                    preview_name = selected_sample
                except ValueError as exc:
                    input_error = str(exc)
        else:
            input_error = "No .sql files were found in samples/sql."
    else:
        pasted_sql = st.text_area(
            "Paste SQL procedure text",
            height=280,
            placeholder="CREATE PROCEDURE ...",
            key="pasted-sql-text",
        )
        if pasted_sql.strip():
            try:
                files = pasted_sql_file(pasted_sql)
                preview_name = next(iter(files))
            except ValueError as exc:
                input_error = str(exc)

    if preview_name is not None:
        st.caption(f"Selected: `{preview_name}` · {len(files[preview_name]):,} characters")
        with st.expander("Preview selected procedure", expanded=False):
            st.code(files[preview_name], language="sql", height=320)

    if input_error:
        st.error(input_error)

    st.subheader("2. Run extraction")
    submitted = st.button("🚀 Run Extraction", type="primary", disabled=not files and not input_error)

    if not submitted:
        return

    st.session_state["review_api_base_url"] = api_base_url
    st.session_state["ui_logs"] = []
    _log("Preparing submission...")

    if input_error:
        _log(f"SQL input error: {input_error}")
        _render_logs()
        return

    if not files:
        _log("No SQL input was selected.")
        _render_logs()
        st.error("Upload a .sql file, choose a bundled sample, or paste SQL text.")
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
        "function_reference": function_reference,
        "entity_name_map": entity_name_map,
        "files": files,
    }

    status_panel = st.container(border=True)
    status_box = status_panel.empty()
    progress = status_panel.progress(5)
    status_box.markdown("### Live Run Status\n\n- **Current step:** Submitting job…")

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
    progress.progress(20)
    status_box.markdown(
        f"### Live Run Status\n\n- **Job:** `{job_id}`\n- **Current step:** Running pipeline…"
    )

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

    progress.progress(100)
    status_box.markdown(
        f"### Live Run Status\n\n- **Job:** `{job_id}`\n"
        f"- **Status:** `{final_status.get('status', '')}`\n"
        "- **Current step:** Complete"
    )
    _render_logs()
    if str(final_status.get("status", "")).upper() == "FAILED":
        st.error(f"Job {job_id} failed: {final_status.get('error_message', 'Unknown error')}")
    else:
        st.success(f"Extraction complete — `{job_id}`")
        markdown = _fetch_report_markdown(api_base_url, job_id, final_status)
        if markdown:
            _render_markdown_previews(markdown, job_id, scope="submission")
        dd_rows = _get_job_dd_rows(job_id)
        if dd_rows:
            m1, m2, m3 = st.columns(3)
            m1.metric("DD rules", len(dd_rows))
            m2.metric(
                "Decision tables",
                sum(1 for r in dd_rows if (r.get("derivation_option") or "") == "Decision Table"),
            )
            m3.metric(
                "With conditions",
                sum(1 for r in dd_rows if (r.get("conditional_json") or r.get("decision_table_json"))),
            )
            _render_excel_preview_and_download(api_base_url, job_id, dd_rows, scope="submission")

    st.session_state["last_submission"] = final_status


def _render_review_tab() -> None:
    st.subheader("2. Human Review Queue")
    jobs = _list_jobs()
    if not jobs:
        st.info("No jobs have been submitted yet.")
        return

    review_api_base_url = st.session_state.get("review_api_base_url", DEFAULT_API_BASE_URL)
    default_job = st.session_state.get("last_submission", {}).get("job_id")
    job_options = [job["job_id"] for job in jobs]
    if default_job not in job_options:
        default_job = job_options[0]

    job_labels = {
        job["job_id"]: f"{job['job_id']} - {job['company']} ({job['status']})"
        for job in jobs
    }

    selected_job = st.selectbox(
        "Select job",
        options=job_options,
        index=job_options.index(default_job),
        format_func=lambda job_id: job_labels.get(job_id, job_id),
        key="review-job-select",
    )

    selected_job_row = next(job for job in jobs if job["job_id"] == selected_job)
    dd_rows = _get_job_dd_rows(selected_job)
    pending = [row for row in dd_rows if row["status"] == "PENDING_REVIEW"]

    m1, m2, m3 = st.columns(3)
    m1.metric("DD rows", len(dd_rows))
    m2.metric("Pending review", len(pending))
    m3.metric("Artifacts ready", "Yes" if selected_job_row.get("report_path") else "No")

    st.caption(
        f"Run #{selected_job_row.get('run_number', '-') or '-'} | Company: {selected_job_row['company']} | "
        f"Platform: {selected_job_row['platform']} | Intent: {selected_job_row['intent']}"
    )

    markdown = _fetch_report_markdown(review_api_base_url, selected_job, selected_job_row)
    if markdown:
        _render_markdown_previews(markdown, selected_job, scope="review")
    else:
        st.info("No Business Understanding report is available for this job yet.")

    if dd_rows:
        _render_excel_preview_and_download(review_api_base_url, selected_job, dd_rows, scope="review")
    _download_reviewed_csv(review_api_base_url, selected_job)

    if not pending:
        st.success("No items pending review for this job.")
    else:
        st.write(f"{len(pending)} item(s) pending review.")

    for row in pending:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{row['entity_name']}.{row['column_name']}**  (chain `{row['chain_id']}`)")
                st.code(row["expression"] or "(no expression — see Decision Table)", language="text")
                validation_errors = json.loads(row["validation_errors"] or "[]")
                if validation_errors:
                    st.error("Validation errors:\n" + "\n".join(f"- {e}" for e in validation_errors))
            with col2:
                st.metric("Confidence", f"{float(row['confidence']):.2f}")
                st.caption(f"Effective: {row['effective_start_date']}")

            edited = st.text_area("Edit expression (optional)", value=row["expression"] or "", key=f"edit-{row['id']}")
            reviewer = st.text_input("Reviewer", value="reviewer", key=f"reviewer-{row['id']}")
            comment = st.text_input("Comment", key=f"comment-{row['id']}")

            b1, b2, b3, b4 = st.columns(4)
            if b1.button("Approve", key=f"approve-{row['id']}"):
                review_store.approve(row["id"], reviewer, comment)
                st.rerun()
            if b2.button("Reject", key=f"reject-{row['id']}"):
                review_store.reject(row["id"], reviewer, comment)
                st.rerun()
            if b3.button("Save Edit", key=f"save-{row['id']}"):
                review_store.edit(row["id"], reviewer, edited, comment)
                st.rerun()
            if b4.button("Override", key=f"override-{row['id']}"):
                review_store.override(row["id"], reviewer, comment)
                st.rerun()

    with st.expander("All DD rows for this job", expanded=False):
        for row in dd_rows:
            st.markdown(
                f"- **{row['entity_name']}.{row['column_name']}** | "
                f"{row['derivation_option']} | {row['status']} | {row['effective_start_date']}"
            )


tab_input, tab_review = st.tabs(["1. Input & Extraction", "2. Human Review"])
with tab_input:
    _render_submission_tab()
with tab_review:
    _render_review_tab()
