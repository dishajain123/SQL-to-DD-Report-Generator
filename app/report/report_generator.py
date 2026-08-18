"""Architecture step 17: Report Generator -> Combined Report.

The report is presentation-focused and intentionally avoids embedding the
full source SQL. It organizes DD output into a fixed narrative structure
so reviewers can understand the process, the platform syntax, the business
logic, and the column-level derivations without losing the exact platform
formula text or the traceability metadata needed for review.

Structure:
  1. A one-line plain-English summary + a "How to Read a Condition" legend
     + an optional Glossary -- all aimed at a non-technical reader before
     anything technical appears.
  2. Process Overview -- process/company/platform/intent, the canonical
     model's own technical/business narrative, and a deterministic Tables
     Read / Tables Written breakdown built from StructuralInfo (not the
     LLM), so this is always available even if the LLM summary is thin.
  3. Rule Summary -- a compact, scannable table (no formulas) with anchor
     links into the detail cards below, so a reviewer can jump straight to
     the one rule they care about instead of scrolling a giant table.
  4. Detailed Business Rules & DD Conditions -- one card per rule, grouped
     by target table, each with a deterministically pretty-printed
     decision-chain rendering of the formula (see
     app/report/formula_pretty_printer.py -- reformats, never rewrites,
     the accepted expression) alongside the exact platform formula text a
     reviewer would paste into the platform.

There is no separate "Business Rules / Logic Explanation" section
duplicating the same explanation already shown per rule in the cards, and
no "Process Control & Traceability" section -- pending-review reasons are
inlined directly on the row/card they belong to instead.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

from app.models.core import CanonicalModel, DDRow, JobPlan, SQLObject, StructuralInfo
from app.report.formula_pretty_printer import pretty_print_expression
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

_HOW_TO_READ_A_CONDITION = """## How to Read a Condition

Every rule below is written as a simple decision chain:

```
IF (condition A) THEN (use this value)
ELSEIF (condition B) THEN (use this value instead)
ELSE (use this fallback value)
```

Read it top to bottom, like a flowchart: check the first condition - if it's true, that's the answer.
If not, move to the next condition. If nothing matches, use the final `ELSE` value.

`"TableName"."ColumnName"` just means a specific field in a specific table."""


def _flatten_for_table_cell(text: str) -> str:
    flattened = " ".join(text.split())
    return flattened.replace("|", "\\|")


def _first_sentence(text: str) -> str:
    stripped = " ".join(text.split()).strip()
    if not stripped:
        return ""
    match = re.search(r"^(.+?[.!?])(?:\s|$)", stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _slugify(*parts: str) -> str:
    """A stable, renderer-independent anchor id -- rendered as an explicit
    `<a id="...">` next to each card heading rather than relying on any
    particular Markdown renderer's own header-to-anchor slug algorithm
    (which varies enough between renderers, e.g. handling of em dashes,
    that a Rule Summary link built against one renderer's rules can
    silently fail to jump anywhere in another)."""
    raw = "-".join(parts).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return slug or "rule"


def _render_glossary_lines(canonical_models: list[CanonicalModel]) -> list[str]:
    glossary: dict[str, tuple[str, str]] = {}
    order: list[str] = []
    for model in canonical_models:
        for term in getattr(model, "glossary_terms", []):
            name = _normalize_display_name(getattr(term, "term", "")).strip()
            definition = " ".join(getattr(term, "definition", "").split()).strip()
            if not name or not definition:
                continue
            key = canonical_logical_name(name)
            if key not in glossary:
                order.append(key)
            glossary[key] = (name, definition)

    if not order:
        return []

    lines = ["## Glossary", ""]
    lines.append("| Term | Plain-English meaning |")
    lines.append("|---|---|")
    for key in order:
        term, definition = glossary[key]
        lines.append(f"| {term} | {definition} |")
    lines.append("")
    return lines


def _is_operational_entity(entity_name: str, rules: list["_RuleGroup"]) -> bool:
    entity = canonical_logical_name(entity_name).upper()
    if any(token in entity for token in ("RUNNINGPROCESS", "AUDIT", "PROCESSLOG", "BATCHLOG")):
        return True

    rule_text = " ".join(
        " ".join(
            [
                rule.column_name,
                rule.business_meaning,
                rule.formula,
                " ".join(rule.validation_notes.split()),
            ]
        )
        for rule in rules
    ).upper()
    if "JOB" in entity and any(token in rule_text for token in ("COMPLETED", "ERROR", "STATUS", "MONITOR", "RUN")):
        return True
    if "STATUS" in entity and all(
        token in rule_text for token in ("ERROR", "STATUS", "COUNT", "COMPLETED")
    ):
        return True
    return False


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


def _is_fallback_business_meaning(rule: "_RuleGroup") -> bool:
    fallback = _business_meaning_from_formula(rule.column_name, rule.formula).strip()
    return " ".join(rule.business_meaning.split()).strip() == " ".join(fallback.split()).strip()


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
    source_statement_refs: list[str]


def _build_rule_groups(dd_rows: list[DDRow]) -> list[_RuleGroup]:
    grouped_rows = _group_dd_rows_for_report(dd_rows)
    rule_groups: list[_RuleGroup] = []
    counter = 1
    for rows in grouped_rows:
        first = rows[0]
        rule_id = f"BR-{counter:03d}"
        counter += 1
        formula = first.display_derivation_expression or ""
        business_meaning = first.business_meaning.strip() if getattr(first, "business_meaning", "").strip() else ""
        if not business_meaning:
            business_meaning = _business_meaning_from_formula(first.column_name, formula)
        rule_groups.append(
            _RuleGroup(
                rule_id=rule_id,
                entity_name=first.entity_name,
                column_name=first.column_name,
                rows=rows,
                business_meaning=business_meaning,
                depends_on=_extract_dependencies(formula),
                formula=formula,
                effective_dates=_format_effective_dates(rows),
                status=first.status.value,
                validation_notes="; ".join(first.validation_errors) if first.validation_errors else "",
                advisory_notes="; ".join(first.advisory_notes) if first.advisory_notes else "",
                source_statement_refs=list(getattr(first, "source_statement_refs", []) or []),
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


def _business_rule_explanation_lines(rule: _RuleGroup) -> list[str]:
    """Deterministic, pattern-based supporting detail -- shown only when
    `rule.business_meaning` is itself the deterministic fallback sentence
    (i.e. no real per-rule explanation was available from the LLM), so a
    card is never left with just a generic one-liner and nothing else. If
    a genuine LLM-authored explanation exists, this returns nothing and
    the card relies on that instead of also showing these pattern notes."""
    expr = rule.formula.upper()
    if not _is_fallback_business_meaning(rule):
        return []
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


def _tables_read_written_lines(
    canonical_models: list[CanonicalModel],
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo] | None,
) -> list[str]:
    """Deterministic Tables Read / Tables Written tables built directly
    from StructuralInfo -- not the LLM -- so this is always available
    (and always accurate to what was actually parsed) regardless of how
    detailed the canonical model's own narrative summary happens to be."""
    if not structural_infos:
        return []

    object_ids: list[str] = []
    seen: set[str] = set()
    for model in canonical_models:
        for oid in model.object_ids:
            if oid not in seen:
                seen.add(oid)
                object_ids.append(oid)

    read_by_table: dict[str, set[str]] = defaultdict(set)
    written_by_table: dict[str, set[str]] = defaultdict(set)
    for oid in object_ids:
        info = structural_infos.get(oid)
        if info is None:
            continue
        object_name = objects[oid].name if oid in objects else oid
        for table in info.tables_read:
            read_by_table[table].add(object_name)
        for table, columns in info.columns_written_by_table.items():
            written_by_table[table].update(columns)

    lines: list[str] = []
    if read_by_table:
        lines.append("### Tables Read")
        lines.append("")
        lines.append("| Table | Read By |")
        lines.append("|---|---|")
        for table in sorted(read_by_table):
            lines.append(f"| {table} | {', '.join(sorted(read_by_table[table]))} |")
        lines.append("")

    if written_by_table:
        lines.append("### Tables Written")
        lines.append("")
        lines.append("| Table | Columns Set |")
        lines.append("|---|---|")
        for table in sorted(written_by_table):
            lines.append(f"| {table} | {', '.join(sorted(written_by_table[table]))} |")
        lines.append("")

    return lines


def _process_overview_lines(
    job_plan: JobPlan,
    canonical_models: list[CanonicalModel],
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo] | None,
) -> list[str]:
    process_name = _process_name(job_plan, canonical_models, objects)
    source_objects = [objects[oid].name if oid in objects else oid for model in canonical_models for oid in model.object_ids]
    unique_source_objects = list(dict.fromkeys(source_objects))

    lines: list[str] = ["## 1. Process Overview", ""]
    lines.append(f"**Process:** {process_name}  ")
    lines.append(f"**Company:** {job_plan.company}  ")
    lines.append(f"**Platform:** {job_plan.platform}  ")
    lines.append(f"**Intent:** {job_plan.intent.value}  ")
    if unique_source_objects:
        lines.append(f"**Source objects:** {', '.join(unique_source_objects)}")
    lines.append("")

    technical_summaries = [model.technical_summary.strip() for model in canonical_models if model.technical_summary.strip()]
    business_summaries = [model.business_summary.strip() for model in canonical_models if model.business_summary.strip()]

    if technical_summaries:
        lines.append("### What the Source SQL Does")
        lines.append("")
        for summary in technical_summaries:
            lines.append(summary)
            lines.append("")

    if business_summaries:
        lines.append("### What It Means for the Business")
        lines.append("")
        for summary in business_summaries:
            lines.append(summary)
            lines.append("")

    lines.extend(_tables_read_written_lines(canonical_models, objects, structural_infos))
    return lines


def _rule_summary_table_lines(business_groups: list[_RuleGroup], technical_groups: list[_RuleGroup]) -> list[str]:
    lines: list[str] = ["## 2. Rule Summary", ""]
    all_rules = business_groups + technical_groups
    if not all_rules:
        lines.append("- No DD rows were generated for this job.")
        lines.append("")
        return lines

    lines.append("| Rule ID | Table | Column | Business Meaning | Effective Date(s) |")
    lines.append("|---|---|---|---|---|")
    technical_ids = {rule.rule_id for rule in technical_groups}
    for rule in all_rules:
        anchor = _slugify(rule.rule_id, rule.column_name)
        label = f"{rule.rule_id} (technical)" if rule.rule_id in technical_ids else rule.rule_id
        meaning = rule.business_meaning
        if rule.status == "PENDING_REVIEW":
            meaning = f"PENDING_REVIEW: {meaning}"
        lines.append(
            f"| [{label}](#{anchor}) | {rule.entity_name} | {rule.column_name} | {_flatten_for_table_cell(meaning)} | {rule.effective_dates} |"
        )
    lines.append("")
    if technical_groups:
        lines.append(
            "*Rows marked (technical) are operational monitoring fields, not business rules -- see the "
            "technical-housekeeping subsection below.*"
        )
        lines.append("")
    return lines


def _rule_card_lines(rule: _RuleGroup) -> list[str]:
    anchor = _slugify(rule.rule_id, rule.column_name)
    lines: list[str] = []
    lines.append(f'<a id="{anchor}"></a>')
    lines.append(f"#### {rule.rule_id} \u2014 {rule.column_name}")
    lines.append("")
    lines.append(f"**Table:** `{rule.entity_name}`  ")

    lines.append(f"**Purpose:** {rule.business_meaning}")
    for detail in _business_rule_explanation_lines(rule):
        lines.append(f"- {detail}")
    lines.append("")

    lines.append(f"**Effective Date(s):** {rule.effective_dates}  ")
    if rule.status == "PENDING_REVIEW":
        notes = rule.validation_notes or "Validation did not fully pass; see the formula below before approving."
        lines.append(f"**Status:** PENDING_REVIEW \u2014 {notes}")
    elif rule.advisory_notes:
        lines.append(f"**Status:** ACTIVE (advisory) \u2014 {rule.advisory_notes}")
    else:
        lines.append("**Status:** ACTIVE")
    lines.append("")

    formula = rule.formula or "(pending review \u2014 no formula was accepted)"
    pretty = pretty_print_expression(formula) if rule.formula else None

    if pretty:
        lines.append("**Decision Logic**")
        lines.append("")
        lines.append("```text")
        lines.append(pretty)
        lines.append("```")
        lines.append("")

    lines.append("**Platform Formula**")
    lines.append("")
    lines.append("```text")
    lines.append(formula)
    lines.append("```")
    lines.append("")

    if rule.depends_on:
        lines.append("**Depends On**")
        for dep in rule.depends_on:
            lines.append(f"- {dep}")
        lines.append("")

    if rule.source_statement_refs:
        lines.append("**Source Statements**")
        for ref in rule.source_statement_refs:
            lines.append(f"- {ref}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def _detailed_rules_lines(business_groups: list[_RuleGroup], technical_groups: list[_RuleGroup]) -> list[str]:
    lines: list[str] = ["## 3. Detailed Business Rules & DD Conditions", ""]
    if not business_groups and not technical_groups:
        lines.append("- No DD rows were generated for this job.")
        return lines

    for entity_name, rules in _rules_by_entity(business_groups):
        lines.append(f"### {entity_name}")
        lines.append("")
        for rule in rules:
            lines.extend(_rule_card_lines(rule))

    if technical_groups:
        for entity_name, rules in _rules_by_entity(technical_groups):
            lines.append(f"### {entity_name} \u2014 technical housekeeping, not business logic")
            lines.append("")
            lines.append(
                "> These rows are operational monitoring fields and are excluded from the business-rules count above."
            )
            lines.append("")
            for rule in rules:
                lines.extend(_rule_card_lines(rule))

    return lines


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
    business_groups: list[_RuleGroup] = []
    technical_groups: list[_RuleGroup] = []
    for entity_name, rules in entity_groups:
        if _is_operational_entity(entity_name, rules):
            technical_groups.extend(rules)
        else:
            business_groups.extend(rules)

    process_name = _process_name(job_plan, canonical_models, objects)
    top_summary = _first_sentence(next((model.business_summary for model in canonical_models if model.business_summary.strip()), ""))
    glossary_lines = _render_glossary_lines(canonical_models)

    lines: list[str] = []
    lines.append(f"# DD Automation Report \u2014 {process_name}")
    lines.append("")
    if top_summary:
        lines.append(f"> **What this process does, in one line:** {top_summary}")
        lines.append("")

    lines.extend(_HOW_TO_READ_A_CONDITION.splitlines())
    lines.append("")
    if glossary_lines:
        lines.extend(glossary_lines)

    lines.extend(_process_overview_lines(job_plan, canonical_models, objects, structural_infos))
    lines.append("")

    lines.extend(_rule_summary_table_lines(business_groups, technical_groups))

    lines.extend(_detailed_rules_lines(business_groups, technical_groups))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path