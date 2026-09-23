"""A complete ordered source-workflow companion to candidate DD formulas.

This is a review specification, not an executable platform workflow.
Every parsed write remains visible, including inserts, deletes and CATCH.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from app.parsing.coverage_ledger import build_coverage_ledger


def write_source_workflow(objects, structural_infos, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    operations = []
    lines = ["# Ordered source workflow", "",
             "Review specification only; this file does not execute on the platform.",
             "Preserve this statement order and the surrounding IF/ELSE and TRY/CATCH context.", ""]
    for oid, obj in objects.items():
        info = structural_infos[oid]
        ledger = build_coverage_ledger(info, source_sql=obj.raw_sql)
        by_index = {}
        for entry in ledger.entries:
            by_index.setdefault(entry.statement_index, []).append(entry)
        lines.extend([f"## {obj.source_file} — {obj.name}", ""])
        for stmt in info.statements:
            entries = by_index.get(stmt.statement_index, [])
            # Include declarations and control flow so a reviewer can interpret guards.
            if not entries and stmt.statement_type not in {"CONTROL_FLOW", "DECLARE", "SET"}:
                continue
            record = {"source_file": obj.source_file, "object_id": oid,
                      "statement_index": stmt.statement_index,
                      "statement_type": stmt.statement_type,
                      "targets": [asdict(e) for e in entries], "source_sql": stmt.raw_text,
                      "implementation_status": "requires_workflow_review" if entries else "source_context"}
            operations.append(record)
            targets = ", ".join(e.target_table for e in entries)
            lines.extend([f"### Statement {stmt.statement_index}: {stmt.statement_type} {targets}", "",
                          "```sql", stmt.raw_text, "```", ""])
    (output_dir / "source_workflow.json").write_text(json.dumps(operations, indent=2), encoding="utf-8")
    (output_dir / "source_workflow.md").write_text("\n".join(lines), encoding="utf-8")
    return operations
