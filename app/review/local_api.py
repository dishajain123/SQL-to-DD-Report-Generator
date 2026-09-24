"""Start the local FastAPI server when the UI is pointed at it and nothing is listening.

Streamlit submits jobs over HTTP. If only the UI process is running, the
connect to 127.0.0.1:8000 is refused (Linux errno 111) and no job is created.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_api_process: subprocess.Popen | None = None


def _endpoint(api_base_url: str) -> tuple[str, int] | None:
    parsed = urlparse(api_base_url)
    host = parsed.hostname
    if host not in _LOCAL_HOSTS:
        return None
    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80
    bind_host = "127.0.0.1" if host == "localhost" else host
    return bind_host, port


def api_is_up(api_base_url: str, timeout: float = 1.5) -> bool:
    health = api_base_url.rstrip("/") + "/health"
    try:
        with urlopen(health, timeout=timeout) as resp:
            return resp.status < 500
    except (URLError, TimeoutError, OSError):
        return False


def ensure_local_api(api_base_url: str, *, wait_seconds: float = 30.0) -> str:
    """Return a log line when this call had to start the server, else "".

    A non-local URL is left alone: if it is down, the caller still sees the
    original connection error instead of this process launching a server
    somewhere else.
    """
    if api_is_up(api_base_url):
        return ""

    endpoint = _endpoint(api_base_url)
    if endpoint is None:
        return ""

    host, port = endpoint
    project_root = Path(__file__).resolve().parents[2]
    log_path = project_root / "output" / "api_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    global _api_process
    popen_kwargs: dict = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    # No --reload: a reload mid-job kills the worker and leaves the job RUNNING.
    with log_path.open("ab", buffering=0) as log_file:
        _api_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                host,
                "--port",
                str(port),
            ],
            cwd=str(project_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            **popen_kwargs,
        )

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if api_is_up(api_base_url):
            return f"API was not running. Started it at {api_base_url.rstrip('/')}."
        if _api_process.poll() is not None and not api_is_up(api_base_url):
            raise RuntimeError(
                f"The API process exited before it could accept connections. See {log_path}."
            )
        time.sleep(0.3)

    raise RuntimeError(f"Started the API but {api_base_url}/health did not respond. See {log_path}.")
