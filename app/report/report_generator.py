"""Architecture step 17: Report Generator -> Combined Report.

Presentation-focused business report. Structure:

  1. At a Glance metadata (procedure, dialect, inputs, table counts)
  2. One-line summary + How to Read a Condition + optional Glossary
  3. Process Overview — What This Does, Process Flow, and one combined
     Tables Involved table (read / written / both)
  4. Business Rule Summary — Rule | Affected Field | Business Purpose
  5. Detailed Business Rules & DD Conditions — per rule: table, column,
     platform condition, plain-English explanation (no status noise)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

from lark import Lark, Tree, Token

from app.models.core import CanonicalModel, DDRow, Dialect, JobPlan, SQLObject, StructuralInfo
from app.report.condition_explainer import explain_expression
from app.report.process_metadata import build_at_a_glance_lines
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
    from app.derivation.dd_postprocess import should_omit_dd_row_from_presentation

    grouped_rows = _group_dd_rows_for_report(dd_rows)
    rule_groups: list[_RuleGroup] = []
    counter = 1
    for rows in grouped_rows:
        first = rows[0]
        formula = first.display_derivation_expression or ""
        if should_omit_dd_row_from_presentation(formula):
            continue
        rule_id = f"BR-{counter:03d}"
        counter += 1
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


def _tables_involved_lines(
    canonical_models: list[CanonicalModel],
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo] | None,
) -> list[str]:
    """One combined table of relations touched by the process."""
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
    insert_column_names: set[str] = set()
    for oid in object_ids:
        info = structural_infos.get(oid)
        if info is None:
            continue
        object_name = objects[oid].name if oid in objects else oid
        for table in info.tables_read:
            read_by_table[table].add(object_name)
        for table, columns in info.columns_written_by_table.items():
            written_by_table[table].update(columns)
        for stmt in info.statements or []:
            raw = stmt.raw_text or ""
            for match in re.finditer(
                r"(?is)\bINSERT\s*(?:\s+INTO\s+[A-Za-z0-9_#\.\"]+)?\s*\(([^)]+)\)",
                raw,
            ):
                for part in match.group(1).split(","):
                    token = part.strip().strip('"').split(".")[-1]
                    if token and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
                        insert_column_names.add(token)
            for match in re.finditer(
                r"(?is)\b(?:INSERT\s+INTO|MERGE\s+INTO|MERGE|DELETE\s+FROM)\s+([A-Za-z0-9_#\.\"]+)",
                raw,
            ):
                token = match.group(1).strip().strip('"')
                bare = token.split(".")[-1]
                if not bare or bare.upper() in {"SET", "SELECT", "VALUES", "INTO"}:
                    continue
                # Skip single-letter aliases (UPDATE A / USING S).
                if len(bare) == 1:
                    continue
                written_by_table.setdefault(bare, set())
                if stmt.set_columns_by_table:
                    written_by_table[bare].update(stmt.set_columns_by_table.get(bare, []))

    written_column_names = {
        col.split(".")[-1]
        for cols in written_by_table.values()
        for col in cols
        if col
    } | insert_column_names
    all_tables = sorted(set(read_by_table) | set(written_by_table))
    # Drop SELECT-list / INSERT-list column tokens that structural analysis
    # sometimes mislabels as tables (e.g. LateFee, AccountId).
    filtered_tables: list[str] = []
    for table in all_tables:
        bare = table.split(".")[-1]
        is_written = table in written_by_table
        looks_like_column = (
            bare in written_column_names
            and "." not in table
            and not bare.startswith("#")
            and not is_written
        )
        if looks_like_column:
            continue
        filtered_tables.append(table)
    if not filtered_tables:
        return []

    lines: list[str] = ["### Tables Involved", ""]
    lines.append("| Table | Role | Columns Set | Used By |")
    lines.append("|---|---|---|---|")
    for table in filtered_tables:
        is_read = table in read_by_table
        is_written = table in written_by_table
        if is_read and is_written:
            role = "Read & Written"
        elif is_written:
            role = "Written"
        else:
            role = "Read"
        columns = ", ".join(sorted(written_by_table.get(table, []))) or "—"
        used_by_names = set(read_by_table.get(table, set()))
        for oid in object_ids:
            info = structural_infos.get(oid)
            if info is None:
                continue
            if table in (info.columns_written_by_table or {}) or table in (info.tables_written or []):
                used_by_names.add(objects[oid].name if oid in objects else oid)
        used_by = ", ".join(sorted(used_by_names)) or "—"
        lines.append(f"| {table} | {role} | {columns} | {used_by} |")
    lines.append("")
    return lines


def _statement_target_label(stmt: "StatementInfo") -> str:
    written = sorted({t for t in (stmt.tables_written or []) if t and not t.startswith("@")})
    if written:
        return ", ".join(f"`{t}`" for t in written)
    raw = stmt.raw_text or ""
    match = re.search(
        r"(?is)\b(?:INSERT\s+INTO|MERGE\s+INTO|MERGE|DELETE\s+FROM|UPDATE)\s+([A-Za-z0-9_#\.\"]+)",
        raw,
    )
    if match:
        token = match.group(1).strip().strip('"').split(".")[-1]
        if token and token.upper() not in {"SET", "SELECT", "VALUES", "INTO"} and len(token) > 1:
            return f"`{token}`"
    return "the target table"


def _is_merge_branch_fragment(stmt: "StatementInfo") -> bool:
    """True for WHEN MATCHED / WHEN NOT MATCHED fragments split out of MERGE."""
    raw = (stmt.raw_text or "").lstrip()
    if re.match(r"(?is)^UPDATE\s+SET\b", raw):
        return True
    if re.match(r"(?is)^INSERT\s*\(", raw):
        return True
    return False


def _process_flow_steps(
    canonical_models: list[CanonicalModel],
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo] | None,
) -> list[str]:
    """Numbered process-flow steps derived from statement structure."""
    if not structural_infos:
        return []

    object_ids: list[str] = []
    seen: set[str] = set()
    for model in canonical_models:
        for oid in model.object_ids:
            if oid not in seen:
                seen.add(oid)
                object_ids.append(oid)

    steps: list[str] = []
    for oid in object_ids:
        info = structural_infos.get(oid)
        if info is None:
            continue
        for stmt in info.statements:
            kind = (stmt.statement_type or "").upper()
            if kind in {"", "UNKNOWN", "DECLARE", "SET_VAR", "BEGIN", "END", "COMMIT", "ROLLBACK", "PRINT"}:
                continue
            if _is_merge_branch_fragment(stmt):
                continue

            written_cols = {
                c.split(".")[-1]
                for column_list in (stmt.set_columns_by_table or {}).values()
                for c in column_list
                if c
            }
            raw = stmt.raw_text or ""
            for match in re.finditer(
                r"(?is)\bINSERT\s*(?:\s+INTO\s+[A-Za-z0-9_#\.\"]+)?\s*\(([^)]+)\)",
                raw,
            ):
                for part in match.group(1).split(","):
                    token = part.strip().strip('"').split(".")[-1]
                    if token and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
                        written_cols.add(token)
            read = sorted(
                {
                    t
                    for t in (stmt.tables_read or [])
                    if t
                    and not (
                        "." not in t
                        and not t.startswith("#")
                        and t.split(".")[-1] in written_cols
                    )
                }
            )
            cols: list[str] = []
            for column_list in (stmt.set_columns_by_table or {}).values():
                cols.extend(column_list)
            cols = sorted({c for c in cols if c})
            target = _statement_target_label(stmt)
            target_bare = re.sub(r"[`\"]", "", target).split(",")[0].strip()
            if target_bare.lower() != "the target table":
                read = [t for t in read if t.split(".")[-1] != target_bare]
            col_text = f" ({', '.join(cols)})" if cols else ""
            source = f" from {', '.join(read)}" if read else ""

            if kind == "UPDATE":
                steps.append(f"Update {target}{col_text}.")
            elif kind == "INSERT":
                steps.append(f"Insert into {target}{col_text}{source}.")
            elif kind == "MERGE":
                steps.append(f"Merge into {target}{col_text}{source}.")
            elif kind == "DELETE":
                steps.append(f"Delete from {target}.")
            elif kind == "SELECT":
                if "INTO" in (stmt.raw_text or "").upper() or cols:
                    steps.append(f"Load {target}{source}.")
                elif read:
                    steps.append(f"Read from {', '.join(read)}.")
            elif kind == "CONTROL_FLOW":
                if cols:
                    steps.append(f"Apply conditional updates on {target}{col_text}.")
            elif cols:
                steps.append(f"Process {target}{col_text}.")

    deduped: list[str] = []
    for step in steps:
        if not deduped or deduped[-1] != step:
            deduped.append(step)
    return deduped[:25]


def _what_this_does_lines(
    canonical_models: list[CanonicalModel],
    structural_infos: dict[str, StructuralInfo] | None,
) -> list[str]:
    business_summaries = [
        model.business_summary.strip()
        for model in canonical_models
        if model.business_summary.strip()
    ]
    written_tables: list[str] = []
    if structural_infos:
        seen: set[str] = set()
        for model in canonical_models:
            for oid in model.object_ids:
                info = structural_infos.get(oid)
                if info is None:
                    continue
                candidates = list(info.tables_written or [])
                for stmt in info.statements or []:
                    raw = stmt.raw_text or ""
                    for match in re.finditer(
                        r"(?is)\b(?:INSERT\s+INTO|MERGE\s+INTO|MERGE|DELETE\s+FROM)\s+([A-Za-z0-9_#\.\"]+)",
                        raw,
                    ):
                        token = match.group(1).strip().strip('"').split(".")[-1]
                        if token and len(token) > 1 and token.upper() not in {"SET", "SELECT", "VALUES", "INTO"}:
                            candidates.append(token)
                for table in candidates:
                    key = canonical_logical_name(table)
                    if key in seen:
                        continue
                    seen.add(key)
                    written_tables.append(table)

    lines: list[str] = ["### What This Does", ""]
    if business_summaries:
        body = " ".join(business_summaries)
        if written_tables:
            body = body.rstrip(".")
            body = f"{body}. This procedure also writes to: {', '.join(written_tables)}."
        lines.append(body)
    elif written_tables:
        lines.append(
            "This procedure derives business fields and writes to: "
            + ", ".join(written_tables)
            + "."
        )
    else:
        lines.append("This procedure derives business fields from the uploaded source SQL.")
    lines.append("")
    return lines


def _process_overview_lines(
    job_plan: JobPlan,
    canonical_models: list[CanonicalModel],
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo] | None,
) -> list[str]:
    lines: list[str] = ["## 1. Process Overview", ""]
    lines.extend(_what_this_does_lines(canonical_models, structural_infos))

    flow_steps = _process_flow_steps(canonical_models, objects, structural_infos)
    lines.append("### Process Flow")
    lines.append("")
    if flow_steps:
        for idx, step in enumerate(flow_steps, start=1):
            lines.append(f"{idx}. {step}")
    else:
        # Fall back to technical summary sentences when structure is thin.
        technical = [
            model.technical_summary.strip()
            for model in canonical_models
            if model.technical_summary.strip()
        ]
        if technical:
            sentences = re.split(r"(?<=[.!?])\s+", " ".join(technical))
            for idx, sentence in enumerate([s.strip() for s in sentences if s.strip()][:12], start=1):
                lines.append(f"{idx}. {sentence}")
        else:
            lines.append("1. Execute the uploaded procedure and apply the derived platform conditions.")
    lines.append("")

    lines.extend(_tables_involved_lines(canonical_models, objects, structural_infos))
    return lines


def _rule_display_name(rule: "_RuleGroup") -> str:
    return f"Determine {rule.column_name} ({rule.entity_name})"


def _rule_summary_table_lines(business_groups: list[_RuleGroup], technical_groups: list[_RuleGroup]) -> list[str]:
    lines: list[str] = ["## Business Rule Summary", ""]
    all_rules = business_groups + technical_groups
    if not all_rules:
        lines.append("- No DD rows were generated for this job.")
        lines.append("")
        return lines

    lines.append("| Rule | Affected Field | Business Purpose |")
    lines.append("|---|---|---|")
    for rule in all_rules:
        anchor = _slugify(rule.rule_id, rule.column_name)
        label = _rule_display_name(rule)
        purpose = rule.business_meaning or "Not specified"
        affected = f"{rule.entity_name}.{rule.column_name}" if rule.entity_name else rule.column_name
        lines.append(
            f"| [{label}](#{anchor}) | `{affected}` | {_flatten_for_table_cell(purpose)} |"
        )
    lines.append("")
    return lines


def _rule_card_lines(rule: _RuleGroup) -> list[str]:
    anchor = _slugify(rule.rule_id, rule.column_name)
    lines: list[str] = []
    lines.append(f'<a id="{anchor}"></a>')
    lines.append(f"#### {_rule_display_name(rule)}")
    lines.append("")
    lines.append(f"**Table:** `{rule.entity_name}`  ")
    lines.append(f"**Column:** `{rule.column_name}`  ")
    if rule.effective_dates and rule.effective_dates != "—":
        lines.append(f"**Effective Date(s):** {rule.effective_dates}  ")
    lines.append("")

    formula = rule.formula or "(no formula was accepted for this rule)"
    explanation = _human_readable_explanation(rule)

    lines.append("**Platform Condition:**")
    lines.append("")
    lines.append("```text")
    lines.append(formula)
    lines.append("```")
    lines.append("")

    lines.append("**What this rule does:**")
    lines.append("")
    lines.extend(explanation.splitlines() or [explanation])
    lines.append("")

    if rule.depends_on:
        lines.append("**Depends On**")
        for dep in rule.depends_on:
            lines.append(f"- {dep}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def _detailed_rules_lines(business_groups: list[_RuleGroup], technical_groups: list[_RuleGroup]) -> list[str]:
    lines: list[str] = ["## Detailed Business Rules & DD Conditions", ""]
    if not business_groups and not technical_groups:
        lines.append("- No DD rows were generated for this job.")
        return lines

    for entity_name, rules in _rules_by_entity(business_groups):
        lines.append(f"### {entity_name}")
        lines.append("")
        for rule in rules:
            lines.extend(_rule_card_lines(rule))

    if technical_groups:
        lines.append("### Operational / housekeeping fields")
        lines.append("")
        for entity_name, rules in _rules_by_entity(technical_groups):
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

    lines.extend(
        build_at_a_glance_lines(
            canonical_models,
            objects,
            structural_infos,
            business_rule_count=len(business_groups) + len(technical_groups),
        )
    )

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