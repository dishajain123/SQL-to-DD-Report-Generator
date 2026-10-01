import traceback
from pathlib import Path

sql_path = Path(r"C:\Users\dishaj\Downloads\PRO_SPs_Sequenced\PRO_SPs_Sequenced\07_S02_PRO.DPD_Calculation.StoredProcedure.sql")
sql = sql_path.read_text(encoding="utf-8", errors="replace")
print("SQL length:", len(sql))

try:
    from app.derivation.v2.pipeline import generate_for_sql

    row, dbg = generate_for_sql(sql, "AccountCal", "ContiExcessDt", llm_client=None)
    print("--- display_derivation_expression ---")
    print(row.display_derivation_expression)
    print("--- validation_errors ---")
    print(row.validation_errors)
except Exception:
    traceback.print_exc()
