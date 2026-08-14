#!/usr/bin/env python3
"""Run layer 2 checks: object splitting, dialect detection, and SQL parsing."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/unit/test_object_splitter.py",
    "tests/unit/test_dialect.py",
    "tests/unit/test_sql_parser.py",
]


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "-v", *TESTS]
    print("[layer 2] Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
