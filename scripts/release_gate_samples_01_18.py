"""Release gate for samples/sql/01_*.sql … 18_*.sql.

A procedure is successful only when every source write is accounted for as
either a validated equivalent output or an explicit unsupported/manual item.
Job completion ≠ ready to present.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.guardrails.function_conformance import analyze_sql_functions
from app.guardrails.source_anomalies import detect_source_anomalies
from app.parsing.coverage_ledger import build_coverage_ledger
from app.parsing.dialect import detect_dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object
from app.parsing.write_inventory_scan import (
    inventory_to_dict,
    read_sql_file,
    sample_sql_paths,
    scan_expected_writes,
)


@dataclass
class FileGateResult:
    file_name: str
    expected_writes: int = 0
    parsed_writes: int = 0
    missing_targets: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    function_blockers: list[str] = field(default_factory=list)
    qa_ready: bool = False
    ready_to_present: bool = False
    inventory_complete: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_sample_file(path: Path) -> FileGateResult:
    sql = read_sql_file(path)
    result = FileGateResult(file_name=path.name)

    expected = scan_expected_writes(sql)
    result.expected_writes = len(expected)

    dialect = detect_dialect(sql)
    objects = split_objects(sql, path.name, dialect)
    all_parsed_targets: list[tuple[str, str]] = []
    unsupported: list[str] = []
    missing: list[str] = []

    for obj in objects:
        info = analyze_object(obj)
        ledger = build_coverage_ledger(info, source_sql=obj.raw_sql)
        missing.extend(ledger.inventory_errors)
        for entry in ledger.entries:
            all_parsed_targets.append((entry.statement_type, entry.target_table))
            if not entry.target_table:
                missing.append(f"stmt #{entry.statement_index} {entry.statement_type}")
            if entry.kind.value in {
                "unsupported",
                "parse_failure",
                "procedure_branch",
                "exception_handler",
                "cross_row",
                "set_based_insert",
                "set_based_merge",
                "set_based_delete",
                "temp_staging",
                "process_status",
            } and not entry.covered_by_dd:
                unsupported.append(
                    f"{entry.statement_type} → {entry.target_table} ({entry.kind.value})"
                )
        result.anomalies.extend(ledger.source_anomalies)

    result.parsed_writes = len(all_parsed_targets)
    result.missing_targets = missing
    result.unsupported = sorted(set(unsupported))
    result.anomalies = sorted(set(result.anomalies + detect_source_anomalies(sql)))
    result.function_blockers = analyze_sql_functions(sql).blockers

    project_root = Path(__file__).resolve().parents[1]
    qa_path = (
        project_root
        / "output" / "demo_01_to_18" / path.stem / "qa_coverage_report.md"
    )
    latest_code_mtime = max(
        (module.stat().st_mtime for module in (project_root / "app").rglob("*.py")),
        default=0,
    )
    result.qa_ready = (
        qa_path.exists()
        and qa_path.stat().st_mtime >= max(path.stat().st_mtime, latest_code_mtime)
        and "Ready to present: **yes**" in qa_path.read_text(encoding="utf-8")
    )

    result.missing_targets = sorted(set(result.missing_targets + missing))
    result.inventory_complete = not result.missing_targets
    # Ready to present requires complete inventory AND no unreviewed blockers.
    # Presence of unsupported items is OK only when they are explicitly listed
    # (fail-closed): ready_to_present stays False until a reviewer accepts them.
    result.ready_to_present = (
        result.inventory_complete
        and not result.anomalies
        and not result.function_blockers
        and not result.unsupported
        and result.qa_ready
    )
    if result.unsupported:
        result.notes.append(
            f"{len(result.unsupported)} operation(s) require manual/platform implementation"
        )
    if result.function_blockers:
        result.notes.append(
            f"{len(result.function_blockers)} function(s) require a reviewed platform mapping"
        )
    if not result.qa_ready:
        result.notes.append("no current approved QA report for generated DD rows")
    return result


def run_gate(root: Path | None = None) -> list[FileGateResult]:
    return [evaluate_sample_file(p) for p in sample_sql_paths(root)]


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    out_json = None
    if "--json" in argv:
        idx = argv.index("--json")
        out_json = Path(argv[idx + 1]) if idx + 1 < len(argv) else Path("output/samples_01_18_gate.json")

    results = run_gate()
    lines = [
        "# Samples 01–18 release gate",
        "",
        "| File | Expected writes | Parsed | Inventory complete | QA approved | Ready to present | Unsupported | Anomalies | Function blockers |",
        "|------|-----------------|--------|--------------------|-------------|------------------|-------------|-----------|-------------------|",
    ]
    for r in results:
        lines.append(
            f"| `{r.file_name}` | {r.expected_writes} | {r.parsed_writes} | "
            f"{'yes' if r.inventory_complete else 'no'} | "
            f"{'yes' if r.qa_ready else 'no'} | "
            f"{'yes' if r.ready_to_present else 'no'} | "
            f"{len(r.unsupported)} | {len(r.anomalies)} | {len(r.function_blockers)} |"
        )
        if r.missing_targets:
            lines.append(f"  - missing: {', '.join(r.missing_targets)}")
        if r.notes:
            for n in r.notes:
                lines.append(f"  - note: {n}")

    complete = sum(1 for r in results if r.inventory_complete)
    ready = sum(1 for r in results if r.ready_to_present)
    lines.extend(
        [
            "",
            f"Inventory complete: **{complete}/{len(results)}**",
            f"Ready to present: **{ready}/{len(results)}**",
            "",
            "Acceptance: every scanned write must match a parsed operation and target. "
            "Ready-to-present requires zero source blockers and a current QA report "
            "that records approved generated DD rows.",
        ]
    )
    report = "\n".join(lines) + "\n"
    print(report)

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "results": [r.to_dict() for r in results],
            "inventories": {
                p.name: inventory_to_dict(scan_expected_writes(read_sql_file(p)))
                for p in sample_sql_paths()
            },
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_json}")

    # A zero exit code is a release decision, not merely a parser smoke test.
    if any(not r.ready_to_present for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
