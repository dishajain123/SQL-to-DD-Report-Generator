#!/usr/bin/env python3
"""Run layer 3 checks: structural analysis, smart chunking, and lineage."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/unit/test_structural_analysis.py",
    "tests/unit/test_smart_chunking.py",
    "tests/unit/test_dependency_graph.py",
]


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "-v", *TESTS]
    print("[layer 3] Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
