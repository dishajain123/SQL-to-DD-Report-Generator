from pathlib import Path
from app.derivation.v2.pipeline import generate_for_sql
from app.derivation.v2.phase2_mutation_folder import fold_column_mutations, MutationSourceIndex
from app.derivation.v2.phase1_lineage import build_lineage_map
from app.report.dd_export import is_exportable_row
from app.derivation.dd_postprocess import should_omit_dd_row_from_presentation

sql = Path("samples/sql/PRO_SPs_Sequenced/11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql").read_text(encoding="utf-8")
out = []
lin = build_lineage_map(sql, None)
muts = fold_column_mutations(sql, "CustomerCal", "SysNPA_Dt", lin, None)
out.append(f"CustomerCal.SysNPA_Dt mutations: {len(muts)}")
for m in muts:
    out.append(f"  ord={m.ordinal} pos={m.source_position} expr={m.assigned_expression[:80]!r}")

row, dbg = generate_for_sql(sql, "CustomerCal", "SysNPA_Dt", llm_client=None)
f_c = dbg.get("formula") or ""
out.append(f"CustomerCal.SysNPA_Dt formula len={len(f_c)} exportable={is_exportable_row(row, should_omit_dd_row_from_presentation)}")
out.append(f"  errs={getattr(row,'validation_errors',None)}")
out.append(f_c[:2500])

row2, dbg2 = generate_for_sql(sql, "AccountCal", "FinalNpaDt", llm_client=None)
f_a = dbg2.get("formula") or ""
out.append("\nAccountCal.FinalNpaDt len=" + str(len(f_a)))
out.append(f_a[:4000])
if "AccountCal" in f_a and "REFPERIODNPA" in f_a:
    bad = '"AccountCal"."REFPERIODNPA"' in f_a.replace(" ", "")
    out.append(f"BAD AccountCal.REFPERIODNPA: {bad}")
    good = "#TEMPTABLENPA" in f_a and "REFPERIODNPA" in f_a
    out.append(f"GOOD #TEMPTABLENPA REFPERIODNPA: {good}")

# branch order heuristics
upper = f_a.upper()
idx_alwys = upper.find("ALWYS_NPA")
idx_sys = upper.find("SYSNPA")
idx_temp = upper.find("TEMPTABLENPA")
idx_pui = upper.find("PUI_CAL")
out.append(f"positions ALWYS={idx_alwys} SYSNPA={idx_sys} TEMPTABLENPA={idx_temp} PUI={idx_pui}")

Path("_s06_check_out.txt").write_text("\n".join(out), encoding="utf-8")
print("wrote _s06_check_out.txt")
