"""One-off: print TEMPTABLEDPD DPD_NoCredit vs DPD_IntService formula stats."""
from __future__ import annotations

from pathlib import Path

from app.derivation.v2.pipeline import generate_for_sql

_REPO = Path(__file__).resolve().parents[1]
_SQL = _REPO / "samples/sql/PRO_SPs_Sequenced/11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql"


def main() -> None:
    sql = _SQL.read_text(encoding="utf-8", errors="replace")
    for col in ("DPD_IntService", "DPD_NoCredit"):
        row, dbg = generate_for_sql(sql, "TEMPTABLEDPD", col, llm_client=None)
        f = dbg.get("formula") or row.display_derivation_expression or ""
        print(col, "len", len(f), "IF(", f.count("IF("), "IF(IF(", "IF(IF(" in f)
        print(f[:1200])
        print("---")


if __name__ == "__main__":
    main()
