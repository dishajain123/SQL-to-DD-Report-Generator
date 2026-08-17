"""Architecture step 17: Report Generator -> Combined Report.

The report is presentation-focused and intentionally avoids embedding the
full source SQL. It organizes DD output into a fixed narrative structure
so reviewers can understand the process, the platform syntax, the business
logic, and the column-level derivations without losing the exact platform
formula text or the traceability metadata needed for review.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

from app.models.core import CanonicalModel, DDRow, JobPlan, SQLObject, StructuralInfo
from app.utils.identity import canonical_expression_key, canonical_logical_name
from app.grammar.validator import KNOWN_FUNCTIONS


_KEYWORDS = {
    "AND",
    "OR",
    "NOT",
    "BETWEEN",
    "IN",
    "THEN",
    "ELSE",
    "ELSEIF",
    "IF",
    "ISEMPTY",
    "ISNOTEMPTY",
    "COALESCE",
    "NULL",
    "TRUE",
    "FALSE",
    "TODATE",
    "DATEDIFF",
    "DATEPART",
    "MAX",
    "MIN",
    "ABS",
    "ROUND",
    "FLOOR",
    "CEIL",
    "CONCAT",
    "TRIM",
    "REPLACE",
    "SUBSTR",
    "LOWER",
    "UPPER",
    "LEN",
    "REGEX",
    "SOM",
    "EOM",
    "SOY",
    "EOY",
    "SOFY",
    "EOFY",
    "PERIOD",
    "SOQ",
    "EOQ",
    "DATE",
}


def _flatten_for_table_cell(text: str) -> str:
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
            canonical_logical_name(row.entity_name),
            canonical_logical_name(row.column_name),
            row.derivation_option.value,
            canonical_expression_key(expression),
            row.decision_table_json,
            row.conditional_json,
        )
        grouped[key].append(row)
    return list(grouped.values())


def _normalize_display_name(value: str) -> str:
    return value.strip().strip('"')


def _process_name(job_plan: JobPlan, canonical_models: list[CanonicalModel], objects: dict[str, SQLObject]) -> str:
    object_names: list[str] = []
    seen: set[str] = set()
    for model in canonical_models:
        for oid in model.object_ids:
            name = objects[oid].name if oid in objects else oid
            canonical = canonical_logical_name(name)
            if canonical in seen:
                continue
            seen.add(canonical)
            object_names.append(name)
    if object_names:
        return ", ".join(object_names)
    return f"{job_plan.company} {job_plan.platform}"


def _extract_dependencies(expression: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    i = 0
    n = len(expression)
    in_double = False

    def add_ref(value: str) -> None:
        canonical = value.strip()
        if not canonical:
            return
        upper = canonical.upper()
        if upper in _KEYWORDS or upper in KNOWN_FUNCTIONS:
            return
        if upper not in seen:
            seen.add(upper)
            refs.append(canonical)

    while i < n:
        ch = expression[i]
        if ch == '"':
            start = i + 1
            j = start
            while j < n and expression[j] != '"':
                j += 1
            segment = expression[start:j]
            parts = [segment]
            k = j + 1
            while True:
                while k < n and expression[k].isspace():
                    k += 1
                if k >= n or expression[k] != ".":
                    break
                k += 1
                while k < n and expression[k].isspace():
                    k += 1
                if k >= n:
                    break
                if expression[k] == '"':
                    start = k + 1
                    j = start
                    while j < n and expression[j] != '"':
                        j += 1
                    parts.append(expression[start:j])
                    k = j + 1
                    continue
                m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", expression[k:])
                if not m:
                    break
                parts.append(m.group(0))
                k += len(m.group(0))
            if len(parts) > 1:
                add_ref(".".join(_normalize_display_name(part) for part in parts))
            else:
                add_ref(_normalize_display_name(parts[0]))
            i = max(j + 1, i + 1)
            continue

        if ch.isalpha() or ch == "_":
            start = i
            j = i + 1
            while j < n and (expression[j].isalnum() or expression[j] == "_"):
                j += 1
            token = expression[start:j]
            tail = j
            while tail < n and expression[tail].isspace():
                tail += 1
            if tail < n and expression[tail] == "(":
                i = j
                continue
            add_ref(token)
            i = j
            continue

        i += 1

    return refs


def _business_meaning_from_formula(column_name: str, expression: str) -> str:
    expr = expression.upper()
    column = column_name.strip()

    if "MAX(" in expr:
        return f"Chooses the maximum contributing value for {column} from the candidate period/rule drivers."
    if "MIN(" in expr:
        return f"Chooses the minimum contributing value for {column} from the candidate drivers."
    if "DATEDIFF(" in expr:
        return f"Calculates an elapsed-day value for {column} using the business date and the source date."
    if "COALESCE(" in expr or "ISEMPTY(" in expr or "ISNOTEMPTY(" in expr:
        return f"Uses null-handling and conditional fallbacks to compute {column} from the listed source fields."
    if "THEN(" in expr and "ELSEIF(" in expr:
        return f"Applies branch-based business rules to determine {column} from the listed source conditions."
    if "THEN(" in expr:
        return f"Applies a conditional rule to derive {column} from the listed source conditions."
    return f"SQL-derived logic for {column} based on the listed dependencies."


def _rule_summary_from_formula(expression: str) -> str:
    expr = _flatten_for_table_cell(expression)
    if len(expr) <= 180:
        return expr
    return expr[:177] + "..."


def _summarize_process_overview(job_plan: JobPlan, canonical_models: list[CanonicalModel], objects: dict[str, SQLObject]) -> tuple[list[str], list[str]]:
    process_name = _process_name(job_plan, canonical_models, objects)
    source_procedures = [objects[oid].name if oid in objects else oid for model in canonical_models for oid in model.object_ids]
    unique_source_procedures = list(dict.fromkeys(source_procedures))
    inputs = [
        f"Company: {job_plan.company}",
        f"Platform: {job_plan.platform}",
        f"Intent: {job_plan.intent.value}",
    ]
    if unique_source_procedures:
        inputs.append("Source files: " + ", ".join(unique_source_procedures))
    return [process_name, *inputs], unique_source_procedures


@dataclass(frozen=True)
class _RuleGroup:
    rule_id: str
    entity_name: str
    column_name: str
    rows: list[DDRow]
    business_meaning: str
    depends_on: list[str]
    formula: str
    effective_dates: str
    status: str
    validation_notes: str
    advisory_notes: str


def _build_rule_groups(dd_rows: list[DDRow]) -> list[_RuleGroup]:
    grouped_rows = _group_dd_rows_for_report(dd_rows)
    rule_groups: list[_RuleGroup] = []
    counter = 1
    for rows in grouped_rows:
        first = rows[0]
        rule_id = f"BR-{counter:03d}"
        counter += 1
        formula = first.display_derivation_expression or ""
        rule_groups.append(
            _RuleGroup(
                rule_id=rule_id,
                entity_name=first.entity_name,
                column_name=first.column_name,
                rows=rows,
                business_meaning=_business_meaning_from_formula(first.column_name, formula),
                depends_on=_extract_dependencies(formula),
                formula=formula,
                effective_dates=_format_effective_dates(rows),
                status=first.status.value,
                validation_notes="; ".join(first.validation_errors) if first.validation_errors else "",
                advisory_notes="; ".join(first.advisory_notes) if first.advisory_notes else "",
            )
        )
    return rule_groups


def _rules_by_entity(rule_groups: list[_RuleGroup]) -> list[tuple[str, list[_RuleGroup]]]:
    grouped: dict[str, list[_RuleGroup]] = defaultdict(list)
    order: list[str] = []
    for rule in rule_groups:
        if rule.entity_name not in grouped:
            order.append(rule.entity_name)
        grouped[rule.entity_name].append(rule)
    return [(entity, grouped[entity]) for entity in order]


def _business_flow_lines(rule_groups: list[_RuleGroup], canonical_models: list[CanonicalModel]) -> list[str]:
    lines: list[str] = []
    for model in canonical_models:
        if model.technical_summary:
            lines.append(f"- {model.technical_summary}")
        if model.business_summary:
            lines.append(f"- {model.business_summary}")
    if not lines:
        lines.append("- No business summary was generated for this job.")
    for rule in rule_groups:
        lines.append(
            f"- {rule.rule_id} ({rule.entity_name}.{rule.column_name}): {rule.business_meaning}"
        )
    return lines


def _special_case_lines(rule_groups: list[_RuleGroup]) -> list[str]:
    lines: list[str] = []
    for rule in rule_groups:
        expr = rule.formula.upper()
        if "ELSEIF(" in expr or "CASE" in expr:
            lines.append(f"- {rule.rule_id}: Preserves branch logic for {rule.entity_name}.{rule.column_name}.")
        if "ISNOTEMPTY(" in expr or "ISEMPTY(" in expr:
            lines.append(f"- {rule.rule_id}: Includes explicit null-handling for {rule.entity_name}.{rule.column_name}.")
        if "ELSE(NULL)" in expr or "ELSE(0)" in expr:
            lines.append(f"- {rule.rule_id}: Uses an explicit fallback when the source condition does not match.")
        if "BETWEEN" in expr or ">" in expr or "<" in expr:
            lines.append(f"- {rule.rule_id}: Keeps source thresholds and comparisons intact.")
    if not lines:
        lines.append("- No special-case branching was detected.")
    return lines


def _aggregation_lines(rule_groups: list[_RuleGroup]) -> list[str]:
    lines: list[str] = []
    for rule in rule_groups:
        expr = rule.formula.upper()
        if any(token in expr for token in ("MAX(", "MIN(", "COALESCE(", "DATEDIFF(")):
            lines.append(f"- {rule.rule_id}: {rule.entity_name}.{rule.column_name} uses aggregation / fallback logic.")
    if not lines:
        lines.append("- No aggregation / max-style logic was detected.")
    return lines


def _business_rule_explanation_lines(rule: _RuleGroup) -> list[str]:
    expr = rule.formula.upper()
    lines: list[str] = []

    if "TODATE(" in expr and "1900-01-01" in expr:
        lines.append(
            "This rule clears placeholder dates so downstream calculations do not treat 1 Jan 1900 as a real business date."
        )
    if "P_TIMEKEY" in expr and "26267" in expr:
        lines.append(
            "This rule changes behavior after a period cutoff, which means the business logic was revised for later processing dates."
        )
    if "AQUA_SCHEME" in expr or "SCHEMETYPE" in expr:
        lines.append(
            "This rule gives special treatment to Aqua/ODA accounts so those products follow their own processing rules instead of the normal one."
        )
    if "SOURCEALT_KEY" in expr and "==6" in expr:
        lines.append(
            "This rule keeps a special zero-day adjustment for SourceAlt_Key 6, so that branch follows the documented exception."
        )
    if "MAX(" in expr:
        lines.append(
            "This rule compares the eligible drivers and keeps the highest one, because the business outcome depends on the most severe applicable value."
        )
    if "COALESCE(" in expr:
        lines.append(
            "This rule uses fallback values when a field is blank, so missing data does not break the calculation."
        )
    if "ELSEIF(" in expr or "CASE" in expr:
        lines.append(
            "This rule is branch-based, so the final value depends on which source condition is true first."
        )
    if "SP_EXPIRYDATE" in expr or "RESTRUCTUREDT" in expr or "PRERESTRUCTURENPA_DATE" in expr:
        lines.append(
            "This rule checks whether a restructure or expiry is still active, then chooses the appropriate NPA-related date."
        )
    if "REFPERIOD" in expr:
        lines.append(
            "This rule preserves the period-specific reference amount tied to the chosen DPD driver, then carries that value forward into later calculations."
        )
    if "NULL" in expr and "ELSE(NULL)" in expr:
        lines.append(
            "This rule intentionally returns no value when none of the business conditions match, which prevents a false positive result."
        )
    if not lines:
        lines.append(
            "This rule follows the source procedure exactly and maps the source conditions into a business decision for the target column."
        )

    if len(rule.rows) > 1:
        lines.append(
            f"The same rule appears on multiple effective dates ({rule.effective_dates}), which means the business logic was revised over time and both versions are preserved."
        )

    return lines


def _business_rules_section(rule_groups: list[_RuleGroup]) -> list[str]:
    lines: list[str] = []
    lines.append("## 6. Business Rules / Logic Explanation")
    lines.append("")
    if not rule_groups:
        lines.append("- No generated rules were available for explanation.")
        return lines

    for entity_name, rules in _rules_by_entity(rule_groups):
        lines.append(f"### {entity_name}")
        lines.append("")
        lines.append("| Rule ID | Column | Business Logic | Special Cases | Effective Dates |")
        lines.append("|---|---|---|---|---|")
        for rule in rules:
            details = _business_rule_explanation_lines(rule)
            summary = rule.business_meaning
            special_cases = "; ".join(details)
            if rule.status == "PENDING_REVIEW":
                summary = f"PENDING_REVIEW: {summary}"
            elif rule.advisory_notes:
                summary = f"{summary} (advisory)"
            lines.append(
                f"| {rule.rule_id} | {rule.column_name} | {summary} | {_flatten_for_table_cell(special_cases)} | {rule.effective_dates} |"
            )
        lines.append("")
    return lines


def _period_rule_rows(rule_groups: list[_RuleGroup]) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for rule in rule_groups:
        if len(rule.rows) <= 1:
            continue
        rows.append(
            (
                rule.rule_id,
                rule.entity_name,
                rule.column_name,
                rule.effective_dates,
                rule.business_meaning,
            )
        )
    return rows


def _traceability_lines(job_plan: JobPlan, canonical_models: list[CanonicalModel], dd_rows: list[DDRow]) -> tuple[list[str], list[str]]:
    pending = [row for row in dd_rows if row.status.value == "PENDING_REVIEW"]
    advisory = [row for row in dd_rows if row.advisory_notes and row.status.value != "PENDING_REVIEW"]
    source_refs: list[str] = []
    for model in canonical_models:
        if model.object_ids:
            source_refs.append(f"- Chain {model.chain_id}: {', '.join(model.object_ids)}")
        if model.evidence:
            source_refs.append(f"- Evidence: {', '.join(model.evidence)}")

    notes: list[str] = []
    if pending:
        notes.append("- Some rows are still marked PENDING_REVIEW because validation or semantic checks were not fully satisfied.")
    else:
        notes.append("- No rows are marked PENDING_REVIEW in this report.")
    if advisory:
        notes.append(f"- {len(advisory)} rows carry advisory notes only and remain auto-validated.")

    notes.append("- The reviewed DD CSV export can be regenerated from the current saved rows.")

    return source_refs or ["- No source reference metadata was available."], notes


def generate_report(
    job_plan: JobPlan,
    canonical_models: list[CanonicalModel],
    dd_rows: list[DDRow],
    output_path: str | Path,
    objects: dict[str, SQLObject] | None = None,
    structural_infos: dict[str, StructuralInfo] | None = None,
) -> Path:
    objects = objects or {}
    rule_groups = _build_rule_groups(dd_rows)
    entity_groups = _rules_by_entity(rule_groups)
    process_overview_lines, source_procedures = _summarize_process_overview(job_plan, canonical_models, objects)
    source_refs, traceability_notes = _traceability_lines(job_plan, canonical_models, dd_rows)

    lines: list[str] = []
    lines.append(f"# DD Automation Report — {process_overview_lines[0]}")
    lines.append("")

    lines.append("## 1. Process Overview")
    lines.append(f"- Purpose: Generate DD conditions and review-ready derivations for {job_plan.company} on {job_plan.platform}.")
    lines.append(f"- Inputs / Parameters: {', '.join(process_overview_lines[1:])}")
    lines.append(f"- Source Procedure(s): {', '.join(source_procedures) if source_procedures else 'None identified'}")
    lines.append("")

    lines.append("## 2. How to Read a DD Condition")
    lines.append("- Platform syntax explanation: DD formulas use the 4X expression form, including `IF(condition)THEN(true_expr)ELSE(false_expr)` and documented functions such as `ISNOTEMPTY`, `ISEMPTY`, `COALESCE`, `DATEDIFF`, and `TODATE`.")
    lines.append("- Field reference explanation: Quoted entity/field references preserve source-table lineage, while bare parameters such as `p_TIMEKEY` represent procedure inputs.")
    lines.append("- Effective-date explanation: Rows with multiple effective dates represent the same logical rule across versioned thresholds; the same Rule ID stays attached to every period-specific instance of that rule.")
    lines.append("")

    lines.append("## 3. Business Logic")
    lines.append("### Business Flow")
    lines.extend(_business_flow_lines(rule_groups, canonical_models))
    lines.append("")
    lines.append("### Special Cases")
    lines.extend(_special_case_lines(rule_groups))
    lines.append("")
    lines.append("### Period-Specific Rules")
    period_rows = _period_rule_rows(rule_groups)
    if period_rows:
        lines.append("| Rule ID | Entity | Column | Effective Dates | Period Logic |")
        lines.append("|---|---|---|---|---|")
        for rule_id, entity, column, effective_dates, logic in period_rows:
            lines.append(
                f"| {rule_id} | {entity} | {column} | {effective_dates} | {_flatten_for_table_cell(logic)} |"
            )
    else:
        lines.append("- No period-specific rule changes were detected.")
    lines.append("")
    lines.append("### Aggregation / Max Logic")
    lines.extend(_aggregation_lines(rule_groups))
    lines.append("")

    lines.append("## 4. Column-Level Derivations & DD Conditions")
    if entity_groups:
        for entity_name, rules in entity_groups:
            lines.append("")
            lines.append(f"### {entity_name}")
            lines.append("")
            lines.append("| Column | Business Meaning | Depends On | Rule ID | Platform Formula | Effective Dates |")
            lines.append("|---|---|---|---|---|---|")
            for rule in rules:
                depends_on = ", ".join(rule.depends_on) if rule.depends_on else "(not confidently extracted)"
                formula = rule.formula or "(pending review)"
                business_meaning = rule.business_meaning
                if rule.status == "PENDING_REVIEW" and rule.validation_notes:
                    business_meaning = f"PENDING_REVIEW: {business_meaning}"
                elif rule.advisory_notes:
                    business_meaning = f"{business_meaning} (advisory)"
                lines.append(
                    f"| {rule.column_name} | {business_meaning} | {depends_on} | {rule.rule_id} | `{_flatten_for_table_cell(formula)}` | {rule.effective_dates} |"
                )
    else:
        lines.append("- No DD rows were generated for this job.")
    lines.append("")

    lines.append("## 5. Process Control & Traceability")
    lines.append("### Success / Error Handling")
    if traceability_notes:
        lines.extend(traceability_notes)
    else:
        lines.append("- No traceability notes were available.")
    lines.append("")
    lines.append("### Source Reference")
    lines.extend(source_refs)
    lines.append("")
    lines.append("### Conversion Notes / Caveats")
    if dd_rows:
        caveats = []
        for row in dd_rows:
            if row.status.value == "PENDING_REVIEW" or row.validation_errors:
                caveats.append(
                    f"- {row.entity_name}.{row.column_name} ({row.effective_start_date}): "
                    + ("; ".join(row.validation_errors) if row.validation_errors else "Marked PENDING_REVIEW.")
                )
            elif row.advisory_notes:
                caveats.append(
                    f"- {row.entity_name}.{row.column_name} ({row.effective_start_date}): "
                    + "; ".join(row.advisory_notes)
                )
        if caveats:
            lines.extend(caveats)
        else:
            lines.append("- No conversion caveats were recorded.")
    else:
        lines.append("- No DD rows were generated.")
    lines.append("")
    lines.append("### Validation / PENDING_REVIEW Items")
    pending_items = [row for row in dd_rows if row.status.value == "PENDING_REVIEW" or row.validation_errors]
    if pending_items:
        for row in pending_items:
            notes = "; ".join(row.validation_errors) if row.validation_errors else "Marked PENDING_REVIEW."
            lines.append(f"- {row.entity_name}.{row.column_name} [{row.effective_start_date}]: {notes}")
    else:
        lines.append("- None.")

    lines.append("")
    lines.extend(_business_rules_section(rule_groups))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
