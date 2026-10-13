#!/usr/bin/env python3
"""Print business-gap counts for key PRO_SPs_Sequenced files (offline)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output" / "audit_pro_sp_sequenced.json"
KEYS = (
    "01_S00_PRO.InsertDataforAssetClassficationRBL.StoredProcedure.sql",
    "14_S09_PRO.Final_AssetClass_Npadate.StoredProcedure.sql",
    "19_S11_PRO.SMA_MARKING.StoredProcedure.sql",
    "23_S15_PRO.UpdateProvisionKey_AccountWise.StoredProcedure.sql",
    "32_S24_PRO.InsertDataINTOHIST_TABLE_OPT.StoredProcedure.sql",
)


def main() -> None:
    if not AUDIT.is_file():
        print("Run scripts/audit_pro_sp_sequenced.py first.")
        return
    data = json.loads(AUDIT.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    print("SUMMARY:", json.dumps(summary, indent=2))
    print("\nKEY FILES:")
    by_name = {f["file"]: f for f in data.get("files", [])}
    for name in KEYS:
        r = by_name.get(name)
        if not r:
            print(f"  {name}: MISSING")
            continue
        print(
            f"  {name}: empty_business={r.get('empty_expressions_business', 0)} "
            f"issues={len(r.get('issues') or [])} with_formula={r.get('with_formula')}"
        )
        for line in (r.get("issues") or [])[:5]:
            print(f"    - {line[:140]}")


if __name__ == "__main__":
    main()
