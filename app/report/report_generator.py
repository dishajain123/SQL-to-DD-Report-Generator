"""Architecture step 17: Report Generator -> Combined Report.

One document per job: technical + business + derivation summary for every
lineage chain, plus a DD Conditions section (included only when DD
Generation actually ran for that chain).

The report also includes a "Known Limitations / Needs Review" section per
chain, generated directly from the automated analysis (StructuralInfo's
unsupported_constructs and the chain's overall confidence) rather than from
the AI narrative — this surfaces real, system-detected gaps (e.g. SQL that
could not be fully parsed) without any hallucination risk, so a
non-technical reader knows exactly what to have double-checked before
finalizing a DD built from this report.
"""
from __future__ import annotations

from pathlib import Path

from app.models.core import CanonicalModel, DDRow, JobPlan, SQLObject, StructuralInfo

# Below this confidence, the report calls out that a human should review the
# chain before the DD is finalized.
_CONFIDENCE_REVIEW_THRESHOLD = 0.7

# Individual unsupported-construct notes can be long fragments of raw SQL;
# truncate them so the report stays readable for a business user.
_MAX_NOTE_LENGTH = 200


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

        lines.extend(_known_limitations_lines(model, structural_infos))

        chain_dd_rows = dd_by_chain.get(model.chain_id, [])
        if chain_dd_rows:
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


def _collect_unsupported_constructs(
    model: CanonicalModel, structural_infos: dict[str, StructuralInfo]
) -> list[str]:
    """Pull real parse-time gaps for every object in this chain, deduplicated
    and lightly cleaned up for readability. This reports what the automated
    analysis actually could not handle — it never guesses."""
    notes: list[str] = []
    seen: set[str] = set()
    for oid in model.object_ids:
        info = structural_infos.get(oid)
        if not info:
            continue
        for construct in info.unsupported_constructs:
            cleaned = " ".join(construct.split())
            if len(cleaned) > _MAX_NOTE_LENGTH:
                cleaned = cleaned[:_MAX_NOTE_LENGTH].rstrip() + "..."
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                notes.append(cleaned)
    return notes


def _known_limitations_lines(
    model: CanonicalModel, structural_infos: dict[str, StructuralInfo] | None
) -> list[str]:
    if not structural_infos:
        return []

    unsupported = _collect_unsupported_constructs(model, structural_infos)
    low_confidence = model.confidence < _CONFIDENCE_REVIEW_THRESHOLD

    if not unsupported and not low_confidence:
        return []

    lines = ["### Known Limitations / Needs Review", ""]
    lines.append(
        "_This section is generated directly from the automated analysis "
        "(not from the narrative above), so it only reports things the "
        "system actually detected — nothing here is guessed._"
    )
    lines.append("")

    if low_confidence:
        lines.append(
            f"- Overall confidence in this chain's analysis is "
            f"{model.confidence:.0%}. Have someone familiar with the source "
            "procedures review this chain before finalizing the DD."
        )

    if unsupported:
        lines.append(
            "- The following parts of the source logic could not be fully "
            "analyzed automatically and should be checked manually before "
            "the DD is finalized:"
        )
        for note in unsupported:
            lines.append(f"  - {note}")

    lines.append("")
    return lines