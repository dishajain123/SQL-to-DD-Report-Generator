#!/usr/bin/env python3
"""Run layer 1 checks: input intake, guardrails, and API validation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/unit/test_guardrails.py",
    "tests/integration/test_api.py",
]


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "-v", *TESTS]
    print("[layer 1] Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
