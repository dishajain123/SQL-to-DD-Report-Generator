#!/usr/bin/env python3
"""Generate DD CSV + Excel + business report.md + QA for samples 01–18.

Deterministic composition (no network). Grammar-valid composed expressions
are exported as ACTIVE / GENERATED. MERGE / procedure-branch / temp-staging /
prior-value caveats stay in advisory notes and the QA ledger — they must not
blank validated formulas. EXISTS / empty / grammar-invalid → UNSUPPORTED or
NEEDS_REVIEW. Process-status tables are omitted from the platform CSV.

Uses the same entity-name map defaults as the live API/Streamlit pipeline so
demo output matches a live run of the same SQL.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

from app.derivation.dd_generation_engine import (
    _assignment_sites,
    _collect_alias_resolution_inventory,
    _compose_simple_assignment_expression,
    _conditional_json_from_formula,
    _decision_table_from_formula_if_categorical,
    _finalize_platform_expression,
    _infer_column_type,
    _infer_data_type,
    _is_non_derivable_expression,
    _should_omit_passthrough_dd_row,
    undeterminable_exception_sites,
)
from app.guardrails.dd_row_coverage import flag_rows_with_uncovered_writes, mark_ledger_coverage
from app.grammar.validator import validate_expression
from app.models.core import (
    CanonicalModel,
    ColumnType,
    DDRow,
    DDStatus,
    DerivationOption,
    Intent,
    JobPlan,
    ReviewState,
)
from app.parsing.coverage_ledger import build_coverage_ledger
from app.parsing.dialect import detect_dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object
from app.report.dd_export import export_dd_rows_csv, export_dd_rows_excel, write_qa_coverage_report
from app.report.report_generator import generate_report
from app.utils.entity_name_map import build_entity_name_map_for_info, load_configured_entity_overrides

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples" / "sql"
OUT_ROOT = ROOT / "output" / "demo_01_to_18"


def _read_sql_file(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


ENTITY_OVERRIDES = load_configured_entity_overrides()

ALIAS_LEAK = re.compile(r'"[A-Za-z]"\s*\.')
SCHEMA_LEAK = re.compile(r'"(?:PRO|dbo|DEMO_MISDB)"\s*\.', re.I)
EXISTS_RE = re.compile(r"(?i)\bEXISTS\s*\(")


def _sample_paths() -> list[Path]:
    paths = []
    for p in sorted(SAMPLES.glob("*.sql")):
        if len(p.name) >= 2 and p.name[:2].isdigit() and 1 <= int(p.name[:2]) <= 18:
            paths.append(p)
    return paths


def _entity_map_for(info) -> dict[str, str]:
    return build_entity_name_map_for_info(info, ENTITY_OVERRIDES)


def _classify_row(expression: str, validation_errors: list[str]) -> tuple[DDStatus, ReviewState]:
    if EXISTS_RE.search(expression or ""):
        return DDStatus.PENDING_REVIEW, ReviewState.UNSUPPORTED
    if validation_errors or not (expression or "").strip():
        return DDStatus.PENDING_REVIEW, ReviewState.NEEDS_REVIEW
    return DDStatus.ACTIVE, ReviewState.GENERATED


def _generate_rows_for_object(obj, info) -> list[DDRow]:
    entity_map = _entity_map_for(info)
    rows: list[DDRow] = []
    for table, columns in info.columns_written_by_table.items():
        bare = table.lstrip("#")
        if re.search(r"(?i)RUNNINGPROCESSSTATUS|PROCESSSTATUS|RUNSTATUS", bare):
            continue
        entity = entity_map.get(table, bare)
        for column in columns:
            sites = _assignment_sites(info, column, target_table=table)
            excluded = undeterminable_exception_sites(sites)
            usable = [site for site in sites if site not in excluded]
            inventory = {
                **_collect_alias_resolution_inventory(obj.raw_sql, obj.dialect),
                **_collect_alias_resolution_inventory(
                    "\n".join(site.raw_sql for site in usable), obj.dialect
                ),
            }

            expression = ""
            derivation_option = DerivationOption.FORMULA_EXPRESSION
            decision_table_json = None
            conditional_json = None
            validation_errors: list[str] = []
            advisory_notes: list[str] = []

            if excluded:
                advisory_notes.append(
                    "Exception-handler write excluded because it shares the same "
                    "row-scoping guard as the normal-flow write; normal-flow value "
                    "is represented below."
                )

            composed = (
                _compose_simple_assignment_expression(
                    usable,
                    entity,
                    column,
                    procedure_sql=obj.raw_sql,
                    entity_name_map=entity_map,
                )
                if usable
                else None
            )
            if not composed:
                continue
            finalized = _finalize_platform_expression(
                composed,
                entity_name=entity,
                entity_name_map=entity_map,
                alias_resolution_inventory=inventory,
                source_sql=obj.raw_sql,
            )
            if _is_non_derivable_expression(finalized):
                continue
            if _should_omit_passthrough_dd_row(
                target_table=table,
                entity_name=entity,
                expression=finalized,
            ):
                continue
            grammar = validate_expression(finalized)
            if grammar.valid:
                expression = finalized
                if ALIAS_LEAK.search(finalized) or SCHEMA_LEAK.search(finalized):
                    advisory_notes.append(
                        "Expression still contains a short SQL alias or schema "
                        "qualifier after rewrite; spot-check entity mapping."
                    )
                dt_payload = _decision_table_from_formula_if_categorical(
                    expression, entity, column
                )
                if dt_payload is not None:
                    derivation_option = DerivationOption.DECISION_TABLE
                    decision_table_json = json.dumps(dt_payload)
                    conditional_json = None
                else:
                    conditional_json = _conditional_json_from_formula(expression, entity)
            else:
                validation_errors.append(
                    grammar.error or "Composed expression failed platform syntax checks"
                )
                expression = finalized  # keep for review visibility

            status, review_state = _classify_row(expression, validation_errors)

            rows.append(
                DDRow(
                    entity_name=entity,
                    column_name=column,
                    column_type=_infer_column_type(column, derivation_option),
                    derivation_option=derivation_option,
                    display_derivation_expression=expression,
                    effective_start_date=date(2026, 7, 20),
                    status=status,
                    review_state=review_state,
                    data_type=_infer_data_type(column, expression),
                    decision_table_json=decision_table_json,
                    conditional_json=conditional_json,
                    source_chain_id=f"demo-{obj.name}",
                    source_object_ids=[obj.object_id],
                    source_statement_refs=[f"{obj.source_file}:{column}"],
                    confidence=1.0 if status == DDStatus.ACTIVE else 0.4,
                    validation_errors=validation_errors,
                    advisory_notes=advisory_notes,
                )
            )
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "# Demo DD generation summary (samples 01–18)",
        "",
        "Platform **Status=ACTIVE** only for representable, grammar-valid expressions.",
        "Unsupported source writes and candidate formulas remain in each QA report.",
        "",
    ]
    totals = Counter()
    review_totals = Counter()
    per_file: list[str] = []

    for path in _sample_paths():
        sql = _read_sql_file(path)
        dialect = detect_dialect(sql)
        objects = split_objects(sql, path.name, dialect)
        all_rows: list[DDRow] = []
        coverage_parts: list[str] = []
        coverage_blockers: list[str] = []
        objects_by_id = {}
        infos_by_id = {}
        for obj in objects:
            info = analyze_object(obj)
            objects_by_id[obj.object_id] = obj
            infos_by_id[obj.object_id] = info
            object_rows = _generate_rows_for_object(obj, info)
            flag_rows_with_uncovered_writes(object_rows, info, obj.raw_sql, _entity_map_for(info))
            all_rows.extend(object_rows)
            ledger = build_coverage_ledger(info, source_sql=obj.raw_sql)
            mark_ledger_coverage(ledger, object_rows, _entity_map_for(info))
            coverage_parts.append(ledger.to_markdown())
            coverage_blockers.extend(ledger.blockers)

        stem = path.stem
        out_dir = OUT_ROOT / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        export_dd_rows_csv(all_rows, out_dir / "dd_export.csv")
        export_dd_rows_excel(all_rows, out_dir / "dd_export.xlsx")
        write_qa_coverage_report(
            all_rows,
            out_dir / "qa_coverage_report.md",
            coverage_markdown="\n\n".join(coverage_parts),
            blockers=sorted(set(coverage_blockers + [
                f"{r.entity_name}.{r.column_name}: {error}"
                for r in all_rows for error in r.validation_errors
            ])),
        )

        # Business report (same generator as the live pipeline's report.md).
        job_plan = JobPlan(
            job_id=f"demo-{stem}",
            intent=Intent.GENERATE_DD,
            company="Demo",
            platform="4X",
        )
        canonical = CanonicalModel(
            chain_id=f"demo-{stem}",
            job_id=job_plan.job_id,
            object_ids=list(objects_by_id),
            technical_summary=f"Deterministic DD derivation from {path.name}.",
            business_summary=(
                f"Business rules derived from {path.name} for platform DD conditions."
            ),
            evidence=sorted(
                {
                    *(o.name for o in objects),
                    *(t for info in infos_by_id.values() for t in (info.tables_written or [])),
                    *(t for info in infos_by_id.values() for t in (getattr(info, "tables_read", None) or [])),
                }
            ),
            confidence=1.0,
        )
        for row in all_rows:
            row.source_chain_id = canonical.chain_id
            if not row.source_object_ids and objects:
                row.source_object_ids = [objects[0].object_id]
        generate_report(
            job_plan,
            [canonical],
            all_rows,
            out_dir / "report.md",
            objects=objects_by_id,
            structural_infos=infos_by_id,
        )

        # Persist full row metadata for analysis (including review_state).
        meta_path = out_dir / "rows_meta.json"
        meta_path.write_text(
            json.dumps(
                [
                    {
                        "entity_name": r.entity_name,
                        "column_name": r.column_name,
                        "status": r.status.value,
                        "review_state": r.review_state.value,
                        "expression": r.display_derivation_expression,
                        "validation_errors": r.validation_errors,
                        "advisory_notes": r.advisory_notes,
                    }
                    for r in all_rows
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        status_counts = Counter(r.status.value for r in all_rows)
        review_counts = Counter(r.review_state.value for r in all_rows)
        totals.update(status_counts)
        review_totals.update(review_counts)

        line = (
            f"- **{path.name}**: {len(all_rows)} rows | "
            f"ACTIVE={status_counts.get('ACTIVE', 0)} "
            f"PENDING_REVIEW={status_counts.get('PENDING_REVIEW', 0)} | "
            f"GENERATED={review_counts.get('GENERATED', 0)} "
            f"NEEDS_REVIEW={review_counts.get('NEEDS_REVIEW', 0)} "
            f"UNSUPPORTED={review_counts.get('UNSUPPORTED', 0)} "
            f"APPROVED={review_counts.get('APPROVED', 0)}"
        )
        per_file.append(line)
        print(
            f"{path.name:55} rows={len(all_rows):3} "
            f"ACTIVE={status_counts.get('ACTIVE', 0):3} "
            f"GEN={review_counts.get('GENERATED', 0):3} "
            f"NEED={review_counts.get('NEEDS_REVIEW', 0):3} "
            f"UNSUP={review_counts.get('UNSUPPORTED', 0):3}"
        )

        # Print root causes for non-ACTIVE rows
        for r in all_rows:
            if r.status != DDStatus.ACTIVE:
                reason = "; ".join(r.validation_errors) or r.review_state.value
                print(f"    ! {r.entity_name}.{r.column_name}: {reason[:120]}")

    summary_lines.extend(per_file)
    summary_lines.extend(
        [
            "",
            "## Totals",
            f"- Platform Status: {dict(totals)}",
            f"- review_state: {dict(review_totals)}",
            "",
            f"Outputs under `{OUT_ROOT.relative_to(ROOT)}/`",
        ]
    )
    (OUT_ROOT / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    print("\nWrote", OUT_ROOT / "SUMMARY.md")
    print("TOTALS status=", dict(totals), "review_state=", dict(review_totals))


if __name__ == "__main__":
    main()
