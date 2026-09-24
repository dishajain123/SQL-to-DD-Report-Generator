"""Phase 2 — chronological UPDATE folder for a target entity.column.

Collects every ``UPDATE`` that mutates ``TargetEntity.TargetColumn``,
extracts the assigned expression / WHERE / JOIN context, and resolves
``#`` temps to root entities via the Phase-1 lineage map.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.derivation.v2.phase1_lineage import LineageMap
from app.derivation.v2.sql_text import (
    bare_ident,
    exists_condition_to_row_predicate,
    extract_if_else_chains,
    extract_insert_select,
    extract_merge_matched_updates,
    extract_select_into,
    extract_subquery_dependency_refs,
    extract_update_statements,
    normalize_table_name,
    parse_from_join_clause,
    parse_select_list,
    split_csv_respecting_parens,
    strip_sql_comments,
)
from app.utils.entity_name_map import resolve_entity_name
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class JoinInfo:
    table: str
    alias: str | None
    on_clause: str | None
    resolved_entity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "alias": self.alias,
            "on_clause": self.on_clause,
            "resolved_entity": self.resolved_entity,
        }


@dataclass
class MutationPass:
    """One chronological UPDATE assignment against the target column."""

    ordinal: int
    target_entity: str
    target_column: str
    assigned_expression: str
    where_clause: str | None
    joins: list[JoinInfo] = field(default_factory=list)
    alias_map: dict[str, str] = field(default_factory=dict)
    raw_sql: str = ""
    statement_index: int = 0
    guarded: bool = True
    source_position: int = 0
    operation: str = "UPDATE"
    # Procedural IF / ELSE IF / ELSE metadata (mutually exclusive siblings).
    control_branch_group: str | None = None
    control_branch_index: int | None = None
    control_branch_kind: str | None = None  # IF | ELSEIF | ELSE
    outer_condition: str | None = None
    # Column refs harvested from EXISTS / IN (SELECT …) for lineage.
    dependency_refs: list[str] = field(default_factory=list)

    @property
    def effective_condition(self) -> str | None:
        """Row-level guard used for AST folding.

        Combines the UPDATE's own WHERE clause with the outer procedural
        ``IF/ELSEIF EXISTS(...)`` predicate (projected to a row predicate)
        when BOTH are present and genuinely independent — dropping either
        one silently would lose a guard the assigned value depends on. When
        the WHERE clause already restates the outer condition (the common
        case: an ``EXISTS(... WHERE X)`` guard whose UPDATE repeats ``X`` as
        one of several ANDed WHERE terms), the outer condition is redundant
        and the WHERE clause alone is used unchanged.
        """
        where = self.where_clause.strip() if self.where_clause and self.where_clause.strip() else None
        outer = (
            self.outer_condition.strip()
            if self.outer_condition and self.outer_condition.strip()
            else None
        )
        if where and outer:
            if outer.upper() in where.upper():
                return where
            return f"({outer}) AND ({where})"
        return where or outer

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "source_position": self.source_position,
            "operation": self.operation,
            "target_entity": self.target_entity,
            "target_column": self.target_column,
            "assigned_expression": self.assigned_expression,
            "where_clause": self.where_clause,
            "joins": [j.as_dict() for j in self.joins],
            "alias_map": dict(self.alias_map),
            "raw_sql": self.raw_sql,
            "statement_index": self.statement_index,
            "guarded": self.guarded,
            "control_branch_group": self.control_branch_group,
            "control_branch_index": self.control_branch_index,
            "control_branch_kind": self.control_branch_kind,
            "outer_condition": self.outer_condition,
            "effective_condition": self.effective_condition,
            "dependency_refs": list(self.dependency_refs),
        }


def fold_column_mutations(
    sql_text: str,
    target_entity: str,
    target_column: str,
    lineage: LineageMap,
    entity_map: dict[str, str] | None = None,
) -> list[MutationPass]:
    """Chronologically collect UPDATE passes that mutate target_entity.column."""
    target_entity_norm = _normalize_entity(target_entity, entity_map)
    target_col = bare_ident(target_column)
    mutations: list[MutationPass] = []
    ordinal = 0

    # Map UPDATE source offsets → IF/ELSE branch (comment-stripped coordinates).
    stripped = strip_sql_comments(sql_text or "")
    branch_spans = extract_if_else_chains(stripped)

    def _branch_for_offset(pos: int):
        for span in branch_spans:
            if span.body_start <= pos < span.body_end:
                return span
        return None

    for stmt_index, stmt in enumerate(extract_update_statements(sql_text)):
        head = (stmt.get("head") or "").strip()
        set_clause = (stmt.get("set_clause") or "").strip()
        from_clause = (stmt.get("from_clause") or "").strip()
        where_clause = (stmt.get("where_clause") or "").strip() or None
        raw_sql = (stmt.get("raw_sql") or "").strip()
        stmt_start = int(stmt.get("start") or 0)

        alias_map, joins = _parse_update_sources(head, from_clause, lineage, entity_map)
        written_tables = _resolve_update_target_tables(head, alias_map)
        branch = _branch_for_offset(stmt_start)

        for assign in _iter_set_assignments(set_clause):
            if assign["column"].upper() != target_col.upper():
                continue

            tables_for_assign = list(written_tables)
            if assign["alias"]:
                resolved = alias_map.get(assign["alias"].upper())
                if resolved:
                    tables_for_assign = [resolved]

            if not _targets_entity(tables_for_assign, target_entity_norm, entity_map, lineage):
                # Also accept writes on local temps whose primary root is the target.
                if not _targets_via_lineage(tables_for_assign, target_entity_norm, lineage):
                    continue

            ordinal += 1
            resolved_expr = _resolve_expression_tables(
                assign["expr"],
                alias_map,
                lineage,
                entity_map,
                target_entity_norm,
                use_derived_formula=False,
            )
            resolved_where = None
            if where_clause:
                resolved_where = _resolve_expression_tables(
                    where_clause,
                    alias_map,
                    lineage,
                    entity_map,
                    target_entity_norm,
                    use_derived_formula=False,
                )

            outer_cond = None
            dep_refs: list[str] = []
            if branch and branch.condition:
                dep_refs.extend(extract_subquery_dependency_refs(branch.condition))
                row_pred = branch.condition
                if row_pred:
                    outer_cond = _resolve_expression_tables(
                        row_pred,
                        alias_map,
                        lineage,
                        entity_map,
                        target_entity_norm,
                        use_derived_formula=False,
                    )
            if where_clause:
                dep_refs.extend(extract_subquery_dependency_refs(where_clause))

            # Inside an IF/ELSE chain, even an UPDATE without WHERE is "guarded"
            # by mutual exclusion — do not treat ELSE as a global unguarded reset.
            in_control = branch is not None
            guarded = bool(where_clause) or (
                in_control and branch.kind in {"IF", "ELSEIF"}
            )

            mutations.append(
                MutationPass(
                    ordinal=ordinal,
                    target_entity=target_entity_norm,
                    target_column=target_col,
                    assigned_expression=resolved_expr,
                    where_clause=resolved_where,
                    joins=joins,
                    alias_map=alias_map,
                    raw_sql=raw_sql,
                    statement_index=stmt_index,
                    source_position=stmt_start,
                    guarded=guarded,
                    control_branch_group=branch.group_id if branch else None,
                    control_branch_index=branch.index if branch else None,
                    control_branch_kind=branch.kind if branch else None,
                    outer_condition=outer_cond,
                    dependency_refs=_dedupe_refs(dep_refs),
                )
            )

    # INSERT … SELECT writes (permanent + temp) — same chronological fold.
    update_stmt_count = len(list(extract_update_statements(sql_text)))
    for ins_index, ins in enumerate(extract_insert_select(sql_text, temps_only=False)):
        target_table = normalize_table_name(ins.get("target") or "")
        if not _targets_entity([target_table], target_entity_norm, entity_map, lineage):
            if not _targets_via_lineage([target_table], target_entity_norm, lineage):
                continue

        col_names = [bare_ident(c) for c in (ins.get("cols") or "").split(",") if bare_ident(c)]
        select_parts = split_csv_respecting_parens(ins.get("select_list") or "")
        if not col_names:
            # No explicit column list — fall back to the target table's own
            # schema ordinal position when it's a local temp whose CREATE
            # TABLE column order Phase 1 already captured. Without any known
            # schema (e.g. an untracked permanent table) we still cannot map
            # projection -> target column and must skip.
            schema_cols = lineage.temp_table_columns.get(target_table)
            if schema_cols and len(schema_cols) == len(select_parts):
                col_names = schema_cols
            else:
                continue
        if len(col_names) != len(select_parts):
            continue

        try:
            col_idx = next(
                i for i, c in enumerate(col_names) if c.upper() == target_col.upper()
            )
        except StopIteration:
            continue

        from_body = ins.get("from_body") or ""
        alias_map, joins = _parse_update_sources("", from_body, lineage, entity_map)
        where_clause = (ins.get("where_clause") or "").strip() or None
        expr = _strip_trailing_select_alias(select_parts[col_idx].strip())

        resolved_expr = _resolve_expression_tables(
            expr, alias_map, lineage, entity_map, target_entity_norm
        )
        resolved_where = None
        if where_clause:
            resolved_where = _resolve_expression_tables(
                where_clause, alias_map, lineage, entity_map, target_entity_norm
            )

        stmt_start = int(ins.get("start") or 0)
        branch = _branch_for_offset(stmt_start)
        outer_cond = None
        dep_refs: list[str] = []
        if branch and branch.condition:
            dep_refs.extend(extract_subquery_dependency_refs(branch.condition))
            row_pred = branch.condition
            if row_pred:
                outer_cond = _resolve_expression_tables(
                    row_pred, alias_map, lineage, entity_map, target_entity_norm
                )
        if where_clause:
            dep_refs.extend(extract_subquery_dependency_refs(where_clause))
        in_control = branch is not None
        guarded = bool(where_clause) or (
            in_control and branch.kind in {"IF", "ELSEIF"}
        )

        ordinal += 1
        mutations.append(
            MutationPass(
                ordinal=ordinal,
                target_entity=target_entity_norm,
                target_column=target_col,
                assigned_expression=resolved_expr,
                where_clause=resolved_where,
                joins=joins,
                alias_map=alias_map,
                raw_sql=(ins.get("raw_sql") or "").strip(),
                statement_index=update_stmt_count + ins_index,
                source_position=stmt_start,
                operation="INSERT",
                guarded=guarded,
                control_branch_group=branch.group_id if branch else None,
                control_branch_index=branch.index if branch else None,
                control_branch_kind=branch.kind if branch else None,
                outer_condition=outer_cond,
                dependency_refs=_dedupe_refs(dep_refs),
            )
        )

    # SELECT ... INTO #temp FROM ... [WHERE ...] — same chronological fold as
    # INSERT ... SELECT, but the destination column list comes from the
    # projection itself (aliased name, or the bare source column name when
    # unaliased) instead of an explicit ``INSERT INTO target (cols)`` list.
    insert_select_count = len(list(extract_insert_select(sql_text, temps_only=False)))
    for si_index, si in enumerate(extract_select_into(sql_text)):
        target_table = normalize_table_name(si.get("target") or "")
        if not _targets_entity([target_table], target_entity_norm, entity_map, lineage):
            if not _targets_via_lineage([target_table], target_entity_norm, lineage):
                continue

        projections = parse_select_list(si.get("select_list") or "")
        try:
            col_idx = next(
                i
                for i, (_, src_col, dest_alias, _raw) in enumerate(projections)
                if (dest_alias or src_col or "").upper() == target_col.upper()
            )
        except StopIteration:
            continue

        from_body = si.get("from_body") or ""
        alias_map, joins = _parse_update_sources("", from_body, lineage, entity_map)
        where_clause = (si.get("where_clause") or "").strip() or None
        expr = _strip_trailing_select_alias(projections[col_idx][3].strip())
        expr = _attach_groupby_to_bare_aggregate(expr, si.get("group_by") or "")

        # Only the projected VALUE (not the WHERE predicate, which in
        # practice is always alias-qualified already) needs the bare-column
        # fallback -- an unqualified projection column implicitly means
        # "this statement's [single] source table's column".
        resolved_expr = _resolve_expression_tables(
            expr, alias_map, lineage, entity_map, target_entity_norm,
            default_source_table=_pick_primary_source_table(alias_map),
        )
        resolved_where = None
        if where_clause:
            resolved_where = _resolve_expression_tables(
                where_clause, alias_map, lineage, entity_map, target_entity_norm
            )

        stmt_start = int(si.get("start") or 0)
        branch = _branch_for_offset(stmt_start)
        outer_cond = None
        dep_refs: list[str] = []
        if branch and branch.condition:
            dep_refs.extend(extract_subquery_dependency_refs(branch.condition))
            row_pred = branch.condition
            if row_pred:
                outer_cond = _resolve_expression_tables(
                    row_pred, alias_map, lineage, entity_map, target_entity_norm
                )
        if where_clause:
            dep_refs.extend(extract_subquery_dependency_refs(where_clause))
        in_control = branch is not None
        guarded = bool(where_clause) or (
            in_control and branch.kind in {"IF", "ELSEIF"}
        )

        ordinal += 1
        mutations.append(
            MutationPass(
                ordinal=ordinal,
                target_entity=target_entity_norm,
                target_column=target_col,
                assigned_expression=resolved_expr,
                where_clause=resolved_where,
                joins=joins,
                alias_map=alias_map,
                raw_sql=(si.get("raw_sql") or "").strip(),
                statement_index=update_stmt_count + insert_select_count + si_index,
                source_position=stmt_start,
                operation="SELECT_INTO",
                guarded=guarded,
                control_branch_group=branch.group_id if branch else None,
                control_branch_index=branch.index if branch else None,
                control_branch_kind=branch.kind if branch else None,
                outer_condition=outer_cond,
                dependency_refs=_dedupe_refs(dep_refs),
            )
        )

    # MERGE … WHEN MATCHED THEN UPDATE SET … — chronological with UPDATEs/INSERTs.
    prior_stmt_count = update_stmt_count + insert_select_count + len(
        list(extract_select_into(sql_text))
    )
    for merge_index, merge in enumerate(extract_merge_matched_updates(sql_text)):
        target_table = normalize_table_name(merge.get("target") or "")
        if not _targets_entity([target_table], target_entity_norm, entity_map, lineage):
            if not _targets_via_lineage([target_table], target_entity_norm, lineage):
                continue

        alias_map: dict[str, str] = {}
        joins: list[JoinInfo] = []
        target_alias = (merge.get("target_alias") or "").strip()
        source_table = normalize_table_name(merge.get("source_table") or "")
        source_alias = (merge.get("source_alias") or "").strip()
        alias_map[target_table.upper()] = target_table
        if target_alias:
            alias_map[target_alias.upper()] = target_table
        if source_table:
            alias_map[source_table.upper()] = source_table
            if source_alias:
                alias_map[source_alias.upper()] = source_table
            joins.append(
                JoinInfo(
                    table=source_table,
                    alias=source_alias or None,
                    on_clause=merge.get("on_clause"),
                    resolved_entity=_resolve_table_entity(source_table, lineage, entity_map),
                )
            )
        # USING subquery aliases (SRC / Source) with no physical table —
        # still map the alias so Source.col rewrites don't crash.
        if source_alias and source_alias.upper() not in alias_map:
            # Treat as pointing at target lineage root for column resolution.
            alias_map[source_alias.upper()] = source_table or target_table

        # Also parse USING body for nested FROM tables when it's a subquery.
        using_body = merge.get("using_body") or ""
        if using_body.startswith("(") or re.search(r"(?is)\bSELECT\b", using_body):
            for table, alias, on_clause in parse_from_join_clause(using_body):
                if alias:
                    alias_map[alias.upper()] = table
                alias_map[bare_ident(table).upper()] = table
                joins.append(
                    JoinInfo(
                        table=table,
                        alias=alias,
                        on_clause=on_clause,
                        resolved_entity=_resolve_table_entity(table, lineage, entity_map),
                    )
                )

        set_clause = merge.get("set_clause") or ""
        on_clause = (merge.get("on_clause") or "").strip() or None
        resolved_on = None
        dep_refs: list[str] = []
        if on_clause:
            dep_refs.extend(extract_subquery_dependency_refs(on_clause))
            resolved_on = _resolve_expression_tables(
                on_clause, alias_map, lineage, entity_map, target_entity_norm
            )

        for assign in _iter_set_assignments(set_clause):
            col = assign["column"]
            if col.upper() != target_col.upper():
                continue
            # Skip if assignment is explicitly on a different alias than target.
            if assign["alias"] and target_alias and assign["alias"].upper() not in {
                target_alias.upper(),
                target_table.upper(),
            }:
                # Still accept Target.col / table.col
                resolved_alias_table = alias_map.get(assign["alias"].upper())
                if resolved_alias_table and normalize_table_name(resolved_alias_table).upper() != target_table.upper():
                    continue

            resolved_expr = _resolve_expression_tables(
                assign["expr"], alias_map, lineage, entity_map, target_entity_norm
            )
            ordinal += 1
            mutations.append(
                MutationPass(
                    ordinal=ordinal,
                    target_entity=target_entity_norm,
                    target_column=target_col,
                    assigned_expression=resolved_expr,
                    where_clause=resolved_on,
                    joins=joins,
                    alias_map=alias_map,
                    raw_sql=(merge.get("raw_sql") or "").strip(),
                    statement_index=prior_stmt_count + merge_index,
                    source_position=int(merge.get("start") or 0),
                    operation="MERGE",
                    guarded=bool(resolved_on),
                    dependency_refs=_dedupe_refs(dep_refs),
                )
            )

    logger.debug(
        "phase2 mutations for %s.%s: %d pass(es)",
        target_entity_norm,
        target_col,
        len(mutations),
    )
    mutations.sort(key=lambda m: m.source_position)
    for ordinal, mutation in enumerate(mutations, 1):
        mutation.ordinal = ordinal
    return _prune_redundant_mutations(mutations)


def _strip_trailing_select_alias(expr: str) -> str:
    """Strip a trailing projection alias: ``expr AS Alias`` / ``expr Alias``."""
    alias_strip = re.match(
        r"(?is)^(.+?)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*$",
        expr,
    )
    if not alias_strip:
        # Some source SQL has zero whitespace between a function call's
        # closing paren and its trailing alias (e.g. "MIN(x)Alias"). Only
        # safe to split here because the left side ends in ")" with a
        # balanced paren count -- a complete sub-expression, not a column
        # name that merely happens to end near an identifier boundary.
        glued = re.match(r"(?is)^(.+\))([A-Za-z_][A-Za-z0-9_]*)\s*$", expr)
        if glued and glued.group(1).count("(") == glued.group(1).count(")"):
            alias_strip = glued
    if not alias_strip:
        return expr
    if re.search(r"(?is)\b(FROM|WHERE|JOIN|SELECT)\b", alias_strip.group(1)):
        return expr
    # Only strip when the right token looks like an alias, not a function arg.
    right = alias_strip.group(2)
    if right.upper() in {
        "DAY", "DAYS", "DD", "MONTH", "YEAR", "HOUR", "MINUTE", "SECOND",
        "END", "NULL", "TRUE", "FALSE",
    }:
        return expr
    return alias_strip.group(1).strip()


_BARE_MIN_MAX_RE = re.compile(
    r"(?is)^(?P<fn>MIN|MAX)\s*\(\s*(?P<col>[#A-Za-z_][A-Za-z0-9_.]*)\s*\)$"
)


def _attach_groupby_to_bare_aggregate(expr: str, group_by_clause: str) -> str:
    """Rewrite a bare ``MIN(col)``/``MAX(col)`` SELECT-INTO projection into
    the platform's documented ``MIN(<Col>, [<GroupbyColumns>])`` form when
    the statement has a GROUP BY (app/grammar/fourx_grammar.lark's
    ``list_literal``; see samples/platform_docs/4x_functions_operators.md).
    Only a bare single-column aggregate is rewritten -- anything already
    multi-argument or wrapped in other logic is left untouched.
    """
    if not group_by_clause:
        return expr
    match = _BARE_MIN_MAX_RE.match(expr.strip())
    if not match:
        return expr
    groupby_cols = []
    for raw in split_csv_respecting_parens(group_by_clause):
        cleaned = raw.strip().strip('"').strip("'").rstrip(";").strip().strip('"').strip("'")
        name = bare_ident(cleaned.split(".")[-1] if cleaned else "")
        name = name.rstrip(";").strip()
        if name:
            groupby_cols.append(name)
    if not groupby_cols:
        return expr
    quoted = ", ".join(f'"{name}"' for name in groupby_cols)
    return f"{match.group('fn').upper()}({match.group('col')}, [{quoted}])"


def _prune_redundant_mutations(mutations: list[MutationPass]) -> list[MutationPass]:
    """Drop duplicate ELSEIF arms produced by folding the same UPDATE twice."""
    kept: list[MutationPass] = []
    seen: set[tuple] = set()
    for mutation in mutations:
        if (mutation.control_branch_kind or "").upper() == "ELSEIF":
            key = (
                (mutation.control_branch_group or "").upper(),
                (mutation.outer_condition or "").strip().upper(),
                (mutation.assigned_expression or "").strip().upper(),
                (mutation.where_clause or "").strip().upper(),
            )
            if key in seen:
                continue
            seen.add(key)
        kept.append(mutation)
    return kept


def prune_redundant_ast(node: dict[str, Any] | None) -> dict[str, Any] | None:
    """Remove dead IF arms that cannot change the column.

    - Inside ``IF(ISNOTEMPTY(col))``, an immediate ``IF(ISEMPTY(col))`` in
      THEN is unreachable.
    - Identical nested ELSEIF arms are collapsed.
    - ``ELSE(col)`` under ``IF(ISNOTEMPTY(col))`` is a self-assignment on the
      null path and is dropped.
    """
    if not isinstance(node, dict):
        return node
    cleaned = dict(node)
    for key, value in list(cleaned.items()):
        if isinstance(value, dict) and "type" in value:
            cleaned[key] = prune_redundant_ast(value)
        elif isinstance(value, list):
            cleaned[key] = [
                prune_redundant_ast(item) if isinstance(item, dict) else item for item in value
            ]
    if cleaned.get("type") != "IF_THEN_ELSE":
        return cleaned

    outer = _predicate_column(cleaned.get("condition"), "ISNOTEMPTY")
    then_branch = cleaned.get("then_branch")
    if outer and isinstance(then_branch, dict) and then_branch.get("type") == "IF_THEN_ELSE":
        inner = _predicate_column(then_branch.get("condition"), "ISEMPTY")
        if inner and inner == outer:
            replacement = then_branch.get("else_branch")
            cleaned["then_branch"] = prune_redundant_ast(replacement) if isinstance(replacement, dict) else replacement

    else_branch = cleaned.get("else_branch")
    if (
        isinstance(else_branch, dict)
        and else_branch.get("type") == "IF_THEN_ELSE"
        and else_branch.get("condition") == cleaned.get("condition")
        and else_branch.get("then_branch") == cleaned.get("then_branch")
    ):
        cleaned["else_branch"] = else_branch.get("else_branch")

    if outer and _is_same_column_ref(cleaned.get("else_branch"), outer):
        cleaned.pop("else_branch", None)
    return cleaned


def _predicate_column(node: dict[str, Any] | None, function_name: str) -> tuple[str, str] | None:
    if not isinstance(node, dict) or node.get("type") != "FUNCTION_CALL":
        return None
    if str(node.get("function_name") or "").upper() != function_name:
        return None
    args = node.get("arguments") or []
    if len(args) != 1 or not isinstance(args[0], dict):
        return None
    ref = args[0]
    if ref.get("type") != "COLUMN_REF":
        return None
    return (str(ref.get("entity") or "").upper(), str(ref.get("column") or "").upper())


def _is_same_column_ref(node: dict[str, Any] | None, identity: tuple[str, str]) -> bool:
    if not isinstance(node, dict) or node.get("type") != "COLUMN_REF":
        return False
    return (str(node.get("entity") or "").upper(), str(node.get("column") or "").upper()) == identity


def _dedupe_refs(refs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in refs:
        key = (r or "").upper()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _parse_update_sources(
    head: str,
    from_clause: str,
    lineage: LineageMap,
    entity_map: dict[str, str] | None,
) -> tuple[dict[str, str], list[JoinInfo]]:
    alias_map: dict[str, str] = {}
    joins: list[JoinInfo] = []

    head_match = re.match(
        r"(?is)^(?P<table>\[?#?#?[A-Za-z0-9_\.]+\]?)"
        r"(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?$",
        head.strip(),
    )
    if head_match:
        table = normalize_table_name(head_match.group("table"))
        alias = head_match.group("alias")
        # Only seed alias_map with real tables here; bare aliases resolved via FROM.
        if table.startswith("#") or "." in (head_match.group("table") or "") or len(table) > 1:
            alias_map[bare_ident(table).upper()] = table
            if alias:
                alias_map[alias.upper()] = table

    for table, alias, on_clause in parse_from_join_clause(from_clause or ""):
        if alias:
            alias_map[alias.upper()] = table
        alias_map[bare_ident(table).upper()] = table
        alias_map[table.upper()] = table
        joins.append(
            JoinInfo(
                table=table,
                alias=alias,
                on_clause=on_clause,
                resolved_entity=_resolve_table_entity(table, lineage, entity_map),
            )
        )

    # UPDATE A ... FROM table A  — resolve bare head alias now that FROM is known.
    if head_match:
        bare_head = normalize_table_name(head_match.group("table"))
        if bare_head.upper() in alias_map:
            pass
        elif len(bare_head) <= 3 and not bare_head.startswith("#") and bare_head.upper() in alias_map:
            pass
        elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", bare_head) and bare_head.upper() in alias_map:
            pass
        elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", bare_head):
            # Still unknown — leave for target resolution via FROM-only tables.
            pass

    return alias_map, joins


def _resolve_update_target_tables(head: str, alias_map: dict[str, str]) -> list[str]:
    head = head.strip()
    if not head:
        return list(dict.fromkeys(alias_map.values()))

    match = re.match(
        r"(?is)^(?P<table>\[?#?#?[A-Za-z0-9_\.]+\]?)"
        r"(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?$",
        head,
    )
    if not match:
        return list(dict.fromkeys(alias_map.values()))

    token = normalize_table_name(match.group("table"))
    alias = match.group("alias")
    if alias and alias.upper() in alias_map:
        return [alias_map[alias.upper()]]
    if token.upper() in alias_map:
        return [alias_map[token.upper()]]
    if token.startswith("#") or "." in (match.group("table") or ""):
        return [token]
    # Bare alias with no FROM resolution yet
    return [token]


def _iter_set_assignments(set_clause: str) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = []
    for part in split_csv_respecting_parens(set_clause):
        match = re.match(
            r"(?is)^\s*(?:(?P<alias>[#A-Za-z_][A-Za-z0-9_]*)\.)?"
            r"(?P<column>\[?[A-Za-z_][A-Za-z0-9_]*\]?)\s*=\s*(?P<expr>.+?)\s*$",
            part.strip(),
        )
        if not match:
            continue
        assignments.append(
            {
                "alias": (match.group("alias") or "").strip(),
                "column": bare_ident(match.group("column")),
                "expr": match.group("expr").strip(),
            }
        )
    return assignments


def _targets_entity(
    written_tables: list[str],
    target_entity: str,
    entity_map: dict[str, str] | None,
    lineage: LineageMap,
) -> bool:
    target_keys = _entity_keys(target_entity)
    for table in written_tables:
        entity = _resolve_table_entity(table, lineage, entity_map)
        candidates = _entity_keys(entity) | _entity_keys(table) | _entity_keys(
            resolve_entity_name(table, entity_map)
        )
        if target_keys & candidates:
            return True
    return False


def _targets_via_lineage(
    written_tables: list[str],
    target_entity: str,
    lineage: LineageMap,
) -> bool:
    target_keys = _entity_keys(target_entity)
    for table in written_tables:
        root = lineage.tables.get(normalize_table_name(table))
        if root and (_entity_keys(root) & target_keys):
            return True
    return False


def _entity_keys(name: str) -> set[str]:
    text = (name or "").strip()
    if not text:
        return set()
    bare = bare_ident(text)
    stripped = bare.lstrip("#")
    if "." in stripped:
        stripped = stripped.split(".")[-1]
    return {text.upper(), bare.upper(), stripped.upper(), f"##{stripped}".upper()}


def _resolve_table_entity(
    table: str,
    lineage: LineageMap,
    entity_map: dict[str, str] | None,
) -> str:
    norm = normalize_table_name(table)
    if norm in lineage.tables:
        return lineage.tables[norm]
    if norm.startswith("##"):
        return resolve_entity_name(norm, entity_map) or norm
    if norm.startswith("#"):
        return lineage.tables.get(norm) or resolve_entity_name(norm, entity_map) or norm
    return resolve_entity_name(norm, entity_map) or norm


_KEYWORD_SKIP = {
    "AND", "OR", "NOT", "NULL", "TRUE", "FALSE", "CASE", "WHEN", "THEN",
    "ELSE", "END", "DISTINCT", "AS", "IS", "IN", "LIKE", "BETWEEN", "TOP",
}


def _qualify_bare_identifiers(text: str, table: str) -> str:
    """Prefix every bare (unqualified) column-like identifier with ``table``.

    A SELECT projection column with no explicit ``alias.`` prefix means
    "this [single, unambiguous] source table's column" -- the same implicit
    scoping SQL itself uses -- but downstream resolution only ever rewrote
    already-qualified ``alias.col`` references, so an unqualified column
    silently fell through unresolved and was later treated as a
    self-reference on the DD row's own target entity.

    Skips: string-literal contents; already-qualified references (preceded
    by ``.``); an identifier that is ITSELF a qualifier -- i.e. immediately
    FOLLOWED by ``.`` (e.g. the ``A`` in ``A.FacilityType``, which is a
    table alias, not a column, even though nothing precedes it in the
    string); T-SQL ``@variables``; function-call names (identifier
    immediately followed by ``(``); and anything inside ``[...]`` (a
    documented ``[<GroupbyColumns>]`` list, already the exact names it
    should render as).
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_single = False
    bracket_depth = 0
    ident_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    while i < n:
        ch = text[i]
        if not in_single and ch in "[]":
            bracket_depth += 1 if ch == "[" else -1
            bracket_depth = max(0, bracket_depth)
            out.append(ch)
            i += 1
            continue
        if in_single:
            out.append(ch)
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    out.append(text[i + 1])
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            out.append(ch)
            i += 1
            continue
        if ch in (".", "@"):
            # Already-qualified column or a T-SQL variable -- copy the
            # whole following identifier untouched.
            out.append(ch)
            i += 1
            m = ident_re.match(text, i)
            if m:
                out.append(m.group(0))
                i = m.end()
            continue
        m = ident_re.match(text, i) if (i == 0 or text[i - 1] not in ".@") else None
        if m:
            ident = m.group(0)
            end = m.end()
            j = end
            while j < n and text[j].isspace():
                j += 1
            is_call = j < n and text[j] == "("
            is_qualifier = j < n and text[j] == "."
            if (
                is_call
                or is_qualifier
                or bracket_depth > 0
                or ident.upper() in _KEYWORD_SKIP
                or ident.isdigit()
            ):
                out.append(ident)
            else:
                out.append(f"{table}.{ident}")
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _pick_primary_source_table(alias_map: dict[str, str]) -> str | None:
    """The one table an unqualified projection column implicitly refers to.

    Mirrors phase1's ``_pick_primary_root`` preference (global temp, then
    physical, then whatever's left) but returns ``None`` instead of
    guessing when more than one local-temp candidate remains ambiguous --
    leaving those bare references unresolved is safer than qualifying them
    against the wrong table.
    """
    tables = list(dict.fromkeys(alias_map.values()))
    if not tables:
        return None
    for t in tables:
        if t.startswith("##"):
            return t
    for t in tables:
        if not t.startswith("#"):
            return t
    return tables[0] if len(tables) == 1 else None


def _resolve_expression_tables(
    expression: str,
    alias_map: dict[str, str],
    lineage: LineageMap,
    entity_map: dict[str, str] | None,
    default_entity: str,
    *,
    use_derived_formula: bool = True,
    default_source_table: str | None = None,
) -> str:
    """Rewrite ``alias.col`` using the statement alias map only.

    Important: do **not** rewrite arbitrary ``schema.table`` tokens (e.g.
    ``PRO.AssetClassMovementHistory``) — those are not column refs and
    corrupting them turns scalar subqueries into invalid 4X strings.

    ``use_derived_formula=False`` must be passed when resolving an UPDATE
    statement's own SET/WHERE text: a temp table's UPDATE can be reached
    here a second time (e.g. because the temp's lineage-mapped root entity
    happens to equal the entity being folded) *after* Phase 1 has already
    recorded that very UPDATE as the column's ``derived_formula`` — splicing
    it back in here would self-reference and corrupt the expression. Only
    genuinely downstream reads (a later INSERT/MERGE reading the temp's
    already-settled value) should inherit the derived formula.
    """
    if default_source_table:
        expression = _qualify_bare_identifiers(expression, default_source_table)

    def repl(match: re.Match[str]) -> str:
        qual = match.group("qual")
        col = bare_ident(match.group("col"))
        # Never entity-qualify T-SQL scalars.
        if qual.startswith("@") or col.startswith("@"):
            return match.group(0)
        # Only rewrite known aliases / temps from this statement.
        table = alias_map.get(qual.upper())
        if table is None:
            # Bare #temp.col without alias entry
            if qual.startswith("#"):
                table = normalize_table_name(qual)
            else:
                return match.group(0)
        ref = lineage.resolve_column(table, col, entity_map)
        if use_derived_formula and getattr(ref, "derived_formula", None):
            # Temp-table mutation checkpoint (Phase 1) — splice the folded
            # expression in verbatim instead of a flat entity/column marker,
            # so this reference inherits e.g. a staging-table markup pass
            # rather than the pre-mutation root value.
            return f"({ref.derived_formula})"
        relationship = ref.relationship
        if (
            relationship is None
            and ref.entity
            and default_entity
            and ref.entity.upper() != default_entity.upper()
            and (ref.entity.startswith("##") or str(table).startswith("##") or
                 resolve_entity_name(str(table), entity_map).upper() != default_entity.upper())
        ):
            # Encode cross-entity join path using the real joined table from
            # the procedure. A ## prefix is only correct when the actual
            # source table has one (a genuine global temp interface entity)
            # -- fabricating "##" on a plain physical/dimension table (e.g.
            # DimProduct) invents a relationship name that never existed in
            # the SQL.
            if str(table).startswith("##"):
                rel = str(table)
            elif ref.source_table.startswith("##"):
                rel = ref.source_table
            else:
                rel = ref.entity
            if resolve_entity_name(str(table), entity_map).upper() != default_entity.upper():
                return f"{default_entity}::{rel}::{ref.column}"
        if relationship:
            return f"{ref.entity}::{relationship}::{ref.column}"
        return f"{ref.entity}::{ref.column}"

    pattern = re.compile(
        r"(?P<qual>[#A-Za-z_][A-Za-z0-9_]*)\.(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?)"
    )
    return pattern.sub(repl, expression)


def _normalize_entity(entity: str, entity_map: dict[str, str] | None) -> str:
    return resolve_entity_name(entity, entity_map) or normalize_table_name(entity)
