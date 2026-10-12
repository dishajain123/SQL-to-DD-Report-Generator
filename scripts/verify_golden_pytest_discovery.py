#!/usr/bin/env python3
"""Offline check: golden test module exists and defines pytest functions (no pytest run)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tests" / "golden" / "test_pro_sps_snippets.py"


def main() -> int:
    if not MODULE.is_file():
        print(f"MISSING: {MODULE}", file=sys.stderr)
        return 1
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    tests = [
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]
    print(f"Module: {MODULE.relative_to(ROOT)}")
    print(f"Discoverable test functions: {len(tests)}")
    for name in tests:
        print(f"  - {name}")
    return 0 if tests else 2


if __name__ == "__main__":
    raise SystemExit(main())
