"""Architecture step 17: Report Generator -> Combined Report.

One document per job: technical + business + derivation summary for every
lineage chain, plus a DD Conditions section (included only when DD
Generation actually ran for that chain).

The generated report is intentionally presentation-focused. Validation
status, review-queue metadata, and other internal processing details stay
in the app/export workflow, not in the user-facing report.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from app.models.core import CanonicalModel, DDRow, JobPlan, SQLObject, StructuralInfo

def _flatten_for_table_cell(text: str) -> str:
    """Markdown table cells cannot contain a literal newline (it would
    prematurely terminate that row and corrupt every row after it) or an
    unescaped pipe character (it would be parsed as a new column
    boundary). This collapses arbitrary text into a single render-safe
    line -- it never changes the underlying stored/exported value (the
    DDRow itself, and the Excel export, are unaffected), only how the text
    is displayed in this Markdown table."""
    flattened = " ".join(text.split())
    return flattened.replace("|", "\\|")


def _format_effective_dates(rows: list[DDRow]) -> str:
    ordered_dates = sorted({row.effective_start_date for row in rows})
    return ", ".join(d.isoformat() for d in ordered_dates)


def _group_dd_rows_for_report(dd_rows: list[DDRow]) -> list[list[DDRow]]:
    grouped: dict[tuple[str, str, str, str, str | None, str | None], list[DDRow]] = defaultdict(list)
    for row in dd_rows:
        expression = row.display_derivation_expression or ""
        key = (
            row.entity_name,
            row.column_name,
            row.derivation_option.value,
            expression,
            row.decision_table_json,
            row.conditional_json,
        )
        grouped[key].append(row)
    return list(grouped.values())


def generate_report(
    job_plan: JobPlan,
    canonical_models: list[CanonicalModel],
    dd_rows: list[DDRow],
    output_path: str | Path,
    objects: dict[str, SQLObject] | None = None,
    structural_infos: dict[str, StructuralInfo] | None = None,
) -> Path:
    objects = objects or {}
    lines: list[str] = []
    lines.append(f"# DD Automation Report — Job {job_plan.job_id}")
    lines.append("")

    dd_by_chain: dict[str, list[DDRow]] = {}
    for row in dd_rows:
        dd_by_chain.setdefault(row.source_chain_id, []).append(row)

    for model in canonical_models:
        object_names = [objects[oid].name if oid in objects else oid for oid in model.object_ids]
        lines.append(f"## {', '.join(object_names)}")
        lines.append("")
        lines.append("### Technical Summary")
        lines.append(model.technical_summary)
        lines.append("")
        lines.append("### Business Summary")
        lines.append(model.business_summary)
        lines.append("")
        if model.derived_rules:
            lines.append("### Derived Rules")
            for rule in model.derived_rules:
                lines.append(f"- {rule}")
            lines.append("")

        chain_dd_rows = dd_by_chain.get(model.chain_id, [])
        if chain_dd_rows:
            grouped_rows = _group_dd_rows_for_report(chain_dd_rows)
            lines.append(
                "_The Business Summary above describes each column's logic in "
                "plain language; the exact, grammar-validated platform formula "
                "for each column is in the DD Conditions table below, so the "
                "two stay consistent with each other and with the DD Excel "
                "export._"
            )
            lines.append("")
            lines.append("### DD Conditions")
            lines.append("")
            lines.append("| Entity | Column | Option | Expression | Effective Start Dates |")
            lines.append("|---|---|---|---|---|")
            for grouped in grouped_rows:
                first = grouped[0]
                expr = first.display_derivation_expression or "(see Decision Table Json)"
                expr = _flatten_for_table_cell(expr)
                lines.append(
                    f"| {first.entity_name} | {first.column_name} | {first.derivation_option.value} "
                    f"| `{expr}` | {_format_effective_dates(grouped)} |"
                )
            lines.append("")
        else:
            lines.append("_DD Generation did not run for this chain "
                         "(Job Plan intent did not require it, or generation was skipped)._")
            lines.append("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
