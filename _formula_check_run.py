from pathlib import Path
from app.derivation.v2.pipeline import generate_for_sql
S02 = Path('samples/sql/PRO_SPs_Sequenced/07_S02_PRO.DPD_Calculation.StoredProcedure.sql')
S06 = Path('samples/sql/PRO_SPs_Sequenced/11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql')
s02 = S02.read_text(encoding='utf-8', errors='replace')
s06 = S06.read_text(encoding='utf-8', errors='replace')
cases = [
    (s02, 'PRO.AdvAcRestructureCal', 'DPD_MaxFin'),
    (s06, 'RESTR_NPA', 'FinalNpaDt'),
    (s06, 'RESTR_NPA', 'UcifEntityID'),
    (s06, 'CustomerCal', 'DegReason'),
]
lines = []
for sql, ent, col in cases:
    row, dbg = generate_for_sql(sql, ent, col, llm_client=None)
    f = dbg.get('formula') or row.display_derivation_expression or ''
    errs = row.validation_errors or []
    lines.append(f'=== {ent} {col} ===')
    lines.append(f)
    if errs:
        lines.append('ERRORS: ' + str(errs))
    lines.append('')
Path('output/formula_check.txt').write_text('\n'.join(lines), encoding='utf-8')
print('done')
