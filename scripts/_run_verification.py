#!/usr/bin/env python3
"""Run audit + export checks; write output/verification_report.txt"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "verification_report.txt"
KEYS = (
    "01_S00_PRO.InsertDataforAssetClassficationRBL.StoredProcedure.sql",
    "14_S09_PRO.Final_AssetClass_Npadate.StoredProcedure.sql",
    "19_S11_PRO.SMA_MARKING.StoredProcedure.sql",
    "23_S15_PRO.UpdateProvisionKey_AccountWise.StoredProcedure.sql",
    "32_S24_PRO.InsertDataINTOHIST_TABLE_OPT.StoredProcedure.sql",
)


def main() -> int:
    lines: list[str] = []
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)}

    def run(cmd: list[str]) -> int:
        lines.append(f"$ {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.stdout:
            lines.append(proc.stdout.rstrip())
        if proc.stderr:
            lines.append("STDERR: " + proc.stderr.rstrip())
        lines.append(f"exit={proc.returncode}")
        lines.append("")
        return proc.returncode

    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.is_file():
        py = Path(sys.executable)

    run([str(py), "-m", "pytest", "tests/unit/derivation_v2/test_set_based_sql_lowering.py", "tests/unit/test_csv_export.py", "-q", "--tb=line"])
    run([str(py), str(ROOT / "scripts" / "audit_pro_sp_sequenced.py")])

    audit_path = ROOT / "output" / "audit_pro_sp_sequenced.json"
    if audit_path.is_file():
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        lines.append("AUDIT SUMMARY: " + json.dumps(data.get("summary", {})))
        by_name = {f["file"]: f for f in data.get("files", [])}
        lines.append("KEY FILE BUSINESS GAPS:")
        for name in KEYS:
            r = by_name.get(name, {})
            lines.append(
                f"  {name}: empty_business={r.get('empty_expressions_business', '?')} "
                f"issues={len(r.get('issues') or [])}"
            )

    # Export schema smoke
    from app.models.core import ColumnType, DDRow, DDStatus, DerivationOption, ReviewState
    from app.report.dd_export import COLUMNS, export_dd_rows

    row = DDRow(
        entity_name="AccountCal",
        column_name="DPD_IntService",
        column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression='IF(ISNOTEMPTY("A"."X"))THEN(1)ELSE(0)',
        effective_start_date=date(2026, 1, 1),
        status=DDStatus.ACTIVE,
        data_type="Integer",
        source_chain_id="c1",
    )
    xlsx = ROOT / "output" / "_verify_dd_export.xlsx"
    export_dd_rows([row], xlsx)
    lines.append(f"COLUMNS ({len(COLUMNS)}): {COLUMNS}")
    lines.append(f"Rule Explanation index: {COLUMNS.index('Rule Explanation')}")
    lines.append(f"Wrote {xlsx}")

    # S15 ProvisionAlt_Key row probe
    try:
        from scripts.audit_pro_sp_sequenced import audit_file

        s15 = audit_file(ROOT / "samples/sql/PRO_SPs_Sequenced/23_S15_PRO.UpdateProvisionKey_AccountWise.StoredProcedure.sql")
        prov_issues = [i for i in s15.get("issues", []) if "PROVISIONALT" in i.upper()]
        lines.append("S15 PROVISIONALT issues: " + json.dumps(prov_issues[:5]))
    except Exception as exc:
        lines.append(f"S15 probe failed: {exc}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
