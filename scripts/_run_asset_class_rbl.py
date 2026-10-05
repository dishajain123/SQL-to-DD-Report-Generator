"""Generate DD export for PRO.InsertDataforAssetClassficationRBL (01_S00).

Run from repo root (30–90s on a typical laptop):

    python scripts/_run_asset_class_rbl.py

Artifacts:
  output/01_S00_InsertDataforAssetClassficationRBL/<stem>/dd_export.xlsx  (Derivations sheet)
  output/01_S00_InsertDataforAssetClassficationRBL/<stem>/dd_rows.json
  output/01_S00_InsertDataforAssetClassficationRBL/dd_export.xlsx  (copy for convenience)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.generate_sample_dd_demo import generate_sample

SP = Path(
    r"C:\Users\dishaj\Downloads\PRO_SPs_Sequenced\PRO_SPs_Sequenced"
    r"\01_S00_PRO.InsertDataforAssetClassficationRBL.StoredProcedure.sql"
)
OUT = Path(__file__).resolve().parents[1] / "output" / "01_S00_InsertDataforAssetClassficationRBL"


def _acl_sample(rows_json: Path) -> None:
    rows = json.loads(rows_json.read_text(encoding="utf-8"))
    for col in ("COMPLETED", "COUNT"):
        hit = next(
            (
                r
                for r in rows
                if r.get("entity_name") == "ACLRUNNINGPROCESSSTATUS"
                and r.get("column_name") == col
            ),
            None,
        )
        if hit:
            expr = (hit.get("display_derivation_expression") or "")[:200]
            print(f"  ACL.{col}: {expr}...")


if __name__ == "__main__":
    if not SP.is_file():
        raise SystemExit(f"Missing: {SP}")
    summary = generate_sample(SP, OUT)
    job_dir = OUT / SP.stem
    xlsx = job_dir / "dd_export.xlsx"
    dd_json = job_dir / "dd_rows.json"
    if xlsx.is_file():
        shutil.copy2(xlsx, OUT / "dd_export.xlsx")
    print("Job folder:", job_dir)
    print("dd_export.xlsx:", OUT / "dd_export.xlsx" if xlsx.is_file() else xlsx)
    print("rows:", summary["rows"])
    print("writes:", summary["writes"])
    print("inventory_complete:", summary["inventory_complete"])
    print("blockers:", len(summary["blockers"]))
    if dd_json.is_file():
        print("ACL samples:")
        _acl_sample(dd_json)
    for b in summary["blockers"][:12]:
        print(" ", b)
