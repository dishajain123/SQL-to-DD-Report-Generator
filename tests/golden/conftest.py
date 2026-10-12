"""Pytest hooks for the golden SP snippet suite (slow derivation regressions)."""
from __future__ import annotations

import sys
from pathlib import Path

# With ``--import-mode=importlib`` (pytest 8 default), the golden module can be
# imported before ``tests/conftest.py`` runs. Keep ``app`` importable.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def pytest_configure(config) -> None:
    """Avoid fd-capture temp-file teardown races on Windows / WSL / OneDrive.

    Symptom: ``collected N items`` then ``FileNotFoundError`` on a path under
    ``%TEMP%`` / ``/tmp`` during pytest's capture cleanup. ``tee-sys`` keeps
    output capture without the fragile fd-backed temp file.
    """
    if getattr(config.option, "capture", None) == "fd":
        config.option.capture = "tee-sys"
