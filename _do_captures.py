"""Run pytest and audit with full stdout/stderr capture to repo root."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
META = ROOT / "_capture_results.json"


def run_capture(args: list[str], out_file: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    with out_file.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(
            [str(PY), *args],
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode


def main() -> int:
    results = []
    jobs = [
        (["-m", "pytest", r"tests\unit\derivation_v2", "-q", "--tb=short"], "_pytest_capture.txt"),
        ([r"scripts\audit_pro_sp_sequenced.py"], "_audit_capture.txt"),
    ]
    for args, name in jobs:
        out = ROOT / name
        code = run_capture(args, out)
        size = out.stat().st_size if out.exists() else 0
        results.append({"file": name, "exit_code": code, "size_bytes": size, "exists": out.exists()})

    import json

    META.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in results:
        print(f"{r['file']}: exists={r['exists']} size={r['size_bytes']} exit={r['exit_code']}")
    return 0 if all(r["exit_code"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
