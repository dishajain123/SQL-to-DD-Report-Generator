"""Trace DPD derivation failure; writes _repro_trace.txt."""
from __future__ import annotations

import traceback
from pathlib import Path

OUT = Path(__file__).resolve().parent / "_repro_trace.txt"
lines: list[str] = []


def log(msg: str) -> None:
    lines.append(msg)


def main() -> None:
    sql_path = Path(
        r"C:\Users\dishaj\Downloads\PRO_SPs_Sequenced\PRO_SPs_Sequenced"
        r"\07_S02_PRO.DPD_Calculation.StoredProcedure.sql"
    )
    if not sql_path.is_file():
        sql_path = Path(__file__).resolve().parent / "output" / "046_job-aced3dd3f7" / "source" / "0001.sql"
    sql = sql_path.read_text(encoding="utf-8", errors="replace")
    log(f"sql_path={sql_path} len={len(sql)} bom={sql[:1] == chr(0xfeff)}")

    try:
        from app.derivation.v2.phase1_lineage import build_lineage_map

        log("build_lineage_map...")
        lineage = build_lineage_map(sql, None)
        log(f"lineage ok columns={len(lineage.columns)}")
    except Exception:
        log("build_lineage_map FAILED")
        log(traceback.format_exc())
        OUT.write_text("\n".join(lines), encoding="utf-8")
        return

    try:
        from app.derivation.v2.phase2_mutation_folder import MutationSourceIndex, fold_column_mutations

        log("MutationSourceIndex.build...")
        idx = MutationSourceIndex.build(sql)
        log(f"index ok updates={len(idx.updates)}")
        log("fold_column_mutations ContiExcessDt...")
        muts = fold_column_mutations(sql, "AccountCal", "ContiExcessDt", lineage, None, source_index=idx)
        log(f"fold ok mutations={len(muts)}")
        if muts:
            log(f"  first expr={muts[0].assigned_expression[:120]!r}")
            log(f"  effective_condition={muts[0].effective_condition!r}")
    except Exception:
        log("fold FAILED")
        log(traceback.format_exc())
        OUT.write_text("\n".join(lines), encoding="utf-8")
        return

    try:
        from app.derivation.v2.phase3_ast_generator import generate_ast, parse_sql_expression_to_ast

        log("generate_ast...")
        ast = generate_ast(muts, target_entity="AccountCal", target_column="ContiExcessDt")
        log(f"ast type={ast.get('type')}")
        from app.derivation.v2.ast_compiler import compile_ast_to_4x_string

        formula = compile_ast_to_4x_string(ast)
        log(f"formula len={len(formula)} head={formula[:200]!r}")
    except Exception:
        log("generate_ast FAILED")
        log(traceback.format_exc())
        OUT.write_text("\n".join(lines), encoding="utf-8")
        return

    try:
        from app.derivation.v2.pipeline import generate_for_sql

        log("generate_for_sql full...")
        row, _ = generate_for_sql(sql, "AccountCal", "ContiExcessDt", llm_client=None)
        log(f"row expr len={len(row.display_derivation_expression or '')}")
        log(f"errors={row.validation_errors}")
    except Exception:
        log("generate_for_sql FAILED")
        log(traceback.format_exc())

    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
