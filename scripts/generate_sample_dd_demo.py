#!/usr/bin/env python3
"""Generate reviewable outputs for samples 01–18 using the live v2 generator.

No LLM calls and no database writes. No automatic approval or status override.
Run: .venv/bin/python -m scripts.generate_sample_dd_demo
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.derivation.v2.pipeline import generate_dd_rows_for_chains
from app.guardrails.dd_row_coverage import flag_rows_with_uncovered_writes, mark_ledger_coverage
from app.guardrails.structural_guardrails import check_structural_info
from app.models.core import CanonicalModel, Intent, JobPlan, LineageChain
from app.parsing.coverage_ledger import build_coverage_ledger
from app.parsing.dialect import detect_dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object
from app.parsing.write_inventory_scan import read_sql_file, sample_sql_paths
from app.report.dd_export import export_dd_rows_csv, export_dd_rows_excel, write_qa_coverage_report
from app.report.report_generator import generate_report
from app.report.source_workflow import write_source_workflow
from app.utils.entity_name_map import build_entity_name_map_for_tables

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "demo_01_to_18"


def source_fingerprint(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    for module in sorted((ROOT / "app").rglob("*.py")):
        digest.update(str(module.relative_to(ROOT)).encode())
        digest.update(module.read_bytes())
    digest.update((ROOT / "app/grammar/fourx_grammar.lark").read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def generate_sample(path: Path, output_root: Path = OUT) -> dict:
    sql = read_sql_file(path)
    object_list = split_objects(sql, path.name, detect_dialect(sql))
    # Stable IDs make audit artifacts reproducible and source-specific.
    for index, obj in enumerate(object_list):
        obj.object_id = f"{path.stem}:{index}"
    objects = {o.object_id: o for o in object_list}
    infos = {oid: analyze_object(obj) for oid, obj in objects.items()}
    mapping = build_entity_name_map_for_tables([
        t for info in infos.values() for t in info.tables_read + info.tables_written
    ])
    chains = [LineageChain(chain_id=path.stem, object_ids=list(objects), order=list(objects))]
    model = CanonicalModel(chain_id=path.stem, job_id=path.stem, object_ids=list(objects),
                           technical_summary="Offline deterministic generation from source SQL.",
                           business_summary="")
    rows = generate_dd_rows_for_chains(chains, [model], objects, infos, None, entity_name_map=mapping)
    blockers = []; ledgers = []
    for oid, info in infos.items():
        flag_rows_with_uncovered_writes(rows, info, objects[oid].raw_sql, mapping)
        ledger = build_coverage_ledger(info, source_sql=objects[oid].raw_sql)
        mark_ledger_coverage(ledger, rows, mapping)
        ledgers.append(ledger)
        blockers.extend(ledger.blockers)
        blockers.extend(check_structural_info(info).errors)
    for row in rows:
        blockers.extend(f"{row.entity_name}.{row.column_name}: {e}" for e in row.validation_errors)
        blockers.extend(f"{row.entity_name}.{row.column_name}: {e}" for e in row.advisory_notes)
    blockers = list(dict.fromkeys(blockers))
    dest = output_root / path.stem
    dest.mkdir(parents=True, exist_ok=True)
    export_dd_rows_csv(rows, dest / "dd_export.csv")
    export_dd_rows_excel(rows, dest / "dd_export.xlsx")
    (dest / "dd_rows.json").write_text(json.dumps([r.model_dump(mode="json") for r in rows], indent=2))
    plan = JobPlan(job_id=path.stem, company="Sample review", platform="4X", intent=Intent.GENERATE_DD)
    generate_report(plan, [model], rows, dest / "report.md", objects=objects, structural_infos=infos)
    # The report itself must identify the scope of the presentation, not just the QA companion.
    report = dest / "report.md"
    report.write_text("> **Draft for technical review. Not approved for production.** "
                      "Read [QA](qa_coverage_report.md) and [ordered source workflow](source_workflow.md).\n\n"
                      + report.read_text(), encoding="utf-8")
    write_source_workflow(objects, infos, dest)
    write_qa_coverage_report(rows, dest / "qa_coverage_report.md", job_id=path.stem,
                            coverage_markdown="\n\n".join(l.to_markdown() for l in ledgers), blockers=blockers)
    summary = {"file": path.name, "rows": len(rows),
               "pending_rows": sum(r.status.value != "ACTIVE" for r in rows),
               "writes": sum(len(l.entries) for l in ledgers),
               "inventory_complete": not any(l.inventory_errors for l in ledgers),
               "blockers": blockers, "source_anomalies": [a for l in ledgers for a in l.source_anomalies],
               "ready_to_present_as_fully_correct": False,
               "source_fingerprint": source_fingerprint(path)}
    (dest / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [generate_sample(path) for path in sample_sql_paths()]
    (OUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# Samples 01–18: presentation and review pack", "",
             "Suitable for demonstrating extraction, candidate formulas, and explicit gaps. "
             "Not a claim that all 18 procedures are correctly implemented on 4X.", "",
             "Generated offline using the same v2 column-generation entry point as the application. "
             "No LLM or SQL Server execution was used, and no business rules were changed.", "",
             "| Sample | Writes inventoried | DD candidates | Pending candidates | Source findings | Report |",
             "|---|---:|---:|---:|---:|---|"]
    for r in results:
        stem = Path(r['file']).stem
        lines.append(f"| {r['file']} | {r['writes']} | {r['rows']} | {r['pending_rows']} | "
                     f"{len(r['source_anomalies'])} | [Review]({stem}/report.md) |")
        print(f"{r['file']}: {r['rows']} candidates; {r['pending_rows']} pending; {len(r['blockers'])} review items")
    lines += ["", "Every parsed source write is included in the ordered workflow companion. "
              "A candidate formula does not implement INSERT/MERGE/DELETE, exception handling, "
              "join cardinality or set membership by itself. Keep QA and workflow files with each export."]
    (OUT / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
