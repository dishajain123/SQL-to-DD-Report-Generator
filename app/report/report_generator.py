"""Architecture step 17: Report Generator -> Combined Report.

One document per job: technical + business + derivation summary for every
lineage chain, plus a DD Conditions section (included only when DD
Generation actually ran for that chain).
"""
from __future__ import annotations

from pathlib import Path

from app.models.core import CanonicalModel, DDRow, JobPlan, SQLObject


def generate_report(
    job_plan: JobPlan,
    canonical_models: list[CanonicalModel],
    dd_rows: list[DDRow],
    output_path: str | Path,
    objects: dict[str, SQLObject] | None = None,
) -> Path:
    objects = objects or {}
    lines: list[str] = []
    lines.append(f"# DD Automation Report — Job {job_plan.job_id}")
    lines.append("")
    lines.append(f"**Company:** {job_plan.company}  ")
    lines.append(f"**Platform:** {job_plan.platform}  ")
    lines.append(f"**Intent:** {job_plan.intent.value}  ")
    lines.append("")

    dd_by_chain: dict[str, list[DDRow]] = {}
    for row in dd_rows:
        dd_by_chain.setdefault(row.source_chain_id, []).append(row)

    for model in canonical_models:
        object_names = [objects[oid].name if oid in objects else oid for oid in model.object_ids]
        lines.append(f"## Lineage Chain: {model.chain_id}")
        lines.append("")
        lines.append(f"**Objects in this chain:** {', '.join(object_names)}")
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
            lines.append("### DD Conditions")
            lines.append("")
            lines.append("| Entity | Column | Option | Expression | Effective Start Date | Status |")
            lines.append("|---|---|---|---|---|---|")
            for dd in chain_dd_rows:
                expr = dd.display_derivation_expression or "(see Decision Table Json)"
                lines.append(
                    f"| {dd.entity_name} | {dd.column_name} | {dd.derivation_option.value} "
                    f"| `{expr}` | {dd.effective_start_date} | {dd.status.value} |"
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
