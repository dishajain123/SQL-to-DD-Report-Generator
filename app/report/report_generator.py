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
     by target table, each showing the exact platform condition
     alongside a deterministic plain-English explanation derived from
     the same parsed logic.

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

from lark import Lark, Tree, Token

from app.models.core import CanonicalModel, DDRow, Dialect, JobPlan, SQLObject, StructuralInfo
from app.report.condition_explainer import explain_expression
from app.utils.identity import canonical_expression_key, canonical_logical_name
from app.utils.sql_aliases import (
    collect_known_reference_names,
    collect_table_aliases,
    resolve_aliases_in_expression,
)
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

_DEPENDENCY_LITERAL_VALUES = {
    "Y",
    "N",
    "YES",
    "NO",
    "TRUE",
    "FALSE",
    "NULL",
    "ACTIVE",
    "INACTIVE",
    "PENDING",
    "APPROVED",
    "REJECTED",
    "OPEN",
    "CLOSED",
    "ENABLED",
    "DISABLED",
    "SUCCESS",
    "FAILED",
    "HIGH",
    "LOW",
    "ON",
    "OFF",
}
_DEPENDENCY_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_DEPENDENCY_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DEPENDENCY_GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammar" / "fourx_grammar.lark"
_DEPENDENCY_PARSER = Lark(_DEPENDENCY_GRAMMAR_PATH.read_text(), parser="earley", start="start")

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


def _extract_dependencies(expression: str, known_names: frozenset[str] = frozenset()) -> list[str]:
    try:
        tree = _DEPENDENCY_PARSER.parse(expression)
    except Exception:
        return _extract_dependencies_from_text(expression, known_names)

    refs: list[str] = []
    seen: set[str] = set()
    
    def walk(node):
        if isinstance(node, Tree):
            if node.data == "column_ref":
                yield node
            for child in node.children:
                yield from walk(child)

    for node in walk(tree):
        ref = _render_dependency_ref(node)
        if not ref:
            continue
        upper = ref.upper()
        if upper in _KEYWORDS or upper in KNOWN_FUNCTIONS or _is_dependency_literal(ref, known_names):
            continue
        if upper not in seen:
            seen.add(upper)
            refs.append(ref)

    return refs


def _extract_dependencies_from_text(expression: str, known_names: frozenset[str] = frozenset()) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    i = 0
    n = len(expression)

    def add_ref(value: str) -> None:
        canonical = value.strip()
        if not canonical:
            return
        upper = canonical.upper()
        if upper in _KEYWORDS or upper in KNOWN_FUNCTIONS or _is_dependency_literal(canonical, known_names):
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
                i = max(j + 1, i + 1)
                continue
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


def _render_dependency_ref(node: Tree) -> str | None:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, Tree) and child.data == "path_part" and child.children:
            token = child.children[0]
            if isinstance(token, Token):
                if token.type == "STRING":
                    value = token.value[1:-1] if len(token.value) >= 2 else str(token).strip('"')
                    parts.append(_normalize_display_name(value))
                else:
                    parts.append(_normalize_display_name(str(token)))
            else:
                parts.append(_normalize_display_name(str(token)))
        elif isinstance(child, Token):
            parts.append(_normalize_display_name(str(child)))
        elif isinstance(child, Tree):
            parts.append(_normalize_display_name(str(child)))

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return ".".join(parts)


def _is_dependency_literal(value: str, known_names: frozenset[str] = frozenset()) -> bool:
    """Decide whether a candidate reference is a real source column/parameter
    or a literal/constant.

    The 4X DSL quotes string literals and identifiers identically
    ("ALWYS_STD" vs "ACCOUNTCAL"), so the grammar alone cannot disambiguate
    a bare, single-part quoted token -- and neither can a hardcoded list of
    known business values, since that list can never be complete (e.g.
    "ALWYS_STD" is a real value that was missing from any such list).

    The generic, source-grounded rule:
      - A multi-part dotted reference (table.column, or alias.column after
        alias resolution) is always a real reference -- the DSL only forms
        these from genuine qualified column access.
      - A single, bare token is a real reference only if it was actually
        observed as a column or parameter name in the object's parsed
        source SQL (`known_names`, from `collect_known_reference_names`).
        Anything else bare is a literal/constant.
      - When no source SQL was available to build `known_names` (e.g. the
        row's source objects couldn't be resolved), fall back to the small
        curated list of common boolean/status literals as a safety net
        rather than either accepting or rejecting every bare token.
    """
    token = value.strip().strip('"')
    if not token:
        return True
    if _DEPENDENCY_NUMERIC_RE.fullmatch(token):
        return True
    if _DEPENDENCY_DATE_RE.fullmatch(token):
        return True
    if "." in token:
        return False
    upper = token.upper()
    if known_names:
        return upper not in known_names
    return upper in _DEPENDENCY_LITERAL_VALUES


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


def _row_source_texts(row: DDRow, objects: dict[str, SQLObject]) -> list[tuple[str, "Dialect"]]:
    """The (text, dialect) pairs to resolve this row's aliases/references
    against.

    Prefers the row's own `source_statement_sql` -- the exact statement(s)
    its formula was actually derived from -- because a short alias like
    "A" is routinely reused for a different table in a different statement
    elsewhere in the same object; scoping to just this row's statements
    keeps that unambiguous instead of collapsing it across the whole
    object. Falls back to the whole object's raw SQL for older rows that
    don't carry source_statement_sql (or when a source object can't be
    resolved), which is safe but more conservative -- a genuinely
    cross-statement-ambiguous alias will still be correctly dropped rather
    than guessed.
    """
    dialects: list["Dialect"] = []
    for object_id in row.source_object_ids or []:
        obj = objects.get(object_id)
        if obj is not None:
            dialects.append(obj.dialect)

    if row.source_statement_sql and dialects:
        # All of a row's source objects share one dialect in practice (a
        # DD row is generated per-object); use the first resolved one.
        return [(text, dialects[0]) for text in row.source_statement_sql if text.strip()]

    texts: list[tuple[str, "Dialect"]] = []
    for object_id in row.source_object_ids or []:
        obj = objects.get(object_id)
        if obj is not None and obj.raw_sql.strip():
            texts.append((obj.raw_sql, obj.dialect))
    return texts


def _row_alias_map(row: DDRow, objects: dict[str, SQLObject]) -> dict[str, tuple[str, ...]]:
    alias_map: dict[str, tuple[str, ...]] = {}
    for text, dialect in _row_source_texts(row, objects):
        for alias, parts in collect_table_aliases(text, dialect).items():
            existing = alias_map.get(alias)
            if existing is None:
                alias_map[alias] = parts
            elif existing != parts:
                alias_map.pop(alias, None)
    return alias_map


def _row_known_reference_names(row: DDRow, objects: dict[str, SQLObject]) -> frozenset[str]:
    """Union of real column/parameter/table names actually parsed out of
    this row's source SQL -- the ground truth used to tell a genuine source
    reference apart from a literal/constant in the generated formula."""
    names: set[str] = set()
    for text, dialect in _row_source_texts(row, objects):
        names |= collect_known_reference_names(text, dialect)
    return frozenset(names)


def _build_rule_groups(dd_rows: list[DDRow], objects: dict[str, SQLObject]) -> list[_RuleGroup]:
    grouped_rows = _group_dd_rows_for_report(dd_rows)
    rule_groups: list[_RuleGroup] = []
    counter = 1
    for rows in grouped_rows:
        first = rows[0]
        rule_id = f"BR-{counter:03d}"
        counter += 1
        formula = first.display_derivation_expression or ""
        alias_map = _row_alias_map(first, objects)
        if alias_map:
            formula = resolve_aliases_in_expression(formula, alias_map, quote_replacements=True)
        known_names = _row_known_reference_names(first, objects)
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
                depends_on=_extract_dependencies(formula, known_names),
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


def _human_readable_explanation(rule: _RuleGroup) -> str:
    formula = rule.formula or ""
    explanation = explain_expression(formula)
    if explanation:
        return explanation
    return "This platform condition could not be rendered safely in plain English, but the exact machine-readable condition is preserved above."


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
    lines: list[str] = ["## 1. Process Overview", ""]

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
    explanation = _human_readable_explanation(rule)

    lines.append("**Platform Condition:**")
    lines.append("")
    lines.append("```text")
    lines.append(formula)
    lines.append("```")
    lines.append("")

    lines.append("**Human-Readable Explanation:**")
    lines.append("")
    lines.extend(explanation.splitlines() or [explanation])
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
    rule_groups = _build_rule_groups(dd_rows, objects)
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