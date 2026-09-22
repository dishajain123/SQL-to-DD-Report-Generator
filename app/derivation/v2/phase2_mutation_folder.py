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
    extract_subquery_dependency_refs,
    extract_update_statements,
    normalize_table_name,
    parse_from_join_clause,
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

        Prefer the UPDATE WHERE clause; fall back to the outer IF predicate
        (EXISTS … WHERE projected to a row predicate when possible).
        """
        if self.where_clause and self.where_clause.strip():
            return self.where_clause.strip()
        if self.outer_condition and self.outer_condition.strip():
            return self.outer_condition.strip()
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
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
                assign["expr"], alias_map, lineage, entity_map, target_entity_norm
            )
            resolved_where = None
            if where_clause:
                resolved_where = _resolve_expression_tables(
                    where_clause, alias_map, lineage, entity_map, target_entity_norm
                )

            outer_cond = None
            dep_refs: list[str] = []
            if branch and branch.condition:
                dep_refs.extend(extract_subquery_dependency_refs(branch.condition))
                row_pred = exists_condition_to_row_predicate(branch.condition)
                if row_pred and not re.match(r"(?is)^EXISTS\b", row_pred.strip()):
                    outer_cond = _resolve_expression_tables(
                        row_pred, alias_map, lineage, entity_map, target_entity_norm
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
        if not col_names or len(col_names) != len(select_parts):
            # Without an explicit column list we cannot map projection → target col.
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
        expr = select_parts[col_idx].strip()
        # Strip trailing aliases: ``expr AS Alias`` / ``expr Alias``
        alias_strip = re.match(
            r"(?is)^(.+?)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*$",
            expr,
        )
        if alias_strip and not re.search(
            r"(?is)\b(FROM|WHERE|JOIN|SELECT)\b", alias_strip.group(1)
        ):
            # Only strip when right token looks like an alias, not a function arg.
            right = alias_strip.group(2)
            if right.upper() not in {
                "DAY",
                "DAYS",
                "DD",
                "MONTH",
                "YEAR",
                "HOUR",
                "MINUTE",
                "SECOND",
            }:
                expr = alias_strip.group(1).strip()

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
            row_pred = exists_condition_to_row_predicate(branch.condition)
            if row_pred and not re.match(r"(?is)^EXISTS\b", row_pred.strip()):
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
                guarded=guarded,
                control_branch_group=branch.group_id if branch else None,
                control_branch_index=branch.index if branch else None,
                control_branch_kind=branch.kind if branch else None,
                outer_condition=outer_cond,
                dependency_refs=_dedupe_refs(dep_refs),
            )
        )

    # MERGE … WHEN MATCHED THEN UPDATE SET … — chronological with UPDATEs/INSERTs.
    prior_stmt_count = update_stmt_count + len(
        list(extract_insert_select(sql_text, temps_only=False))
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
    return mutations


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


def _resolve_expression_tables(
    expression: str,
    alias_map: dict[str, str],
    lineage: LineageMap,
    entity_map: dict[str, str] | None,
    default_entity: str,
) -> str:
    """Rewrite ``alias.col`` using the statement alias map only.

    Important: do **not** rewrite arbitrary ``schema.table`` tokens (e.g.
    ``PRO.AssetClassMovementHistory``) — those are not column refs and
    corrupting them turns scalar subqueries into invalid 4X strings.
    """

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
        relationship = ref.relationship
        if (
            relationship is None
            and ref.entity
            and default_entity
            and ref.entity.upper() != default_entity.upper()
            and (ref.entity.startswith("##") or str(table).startswith("##") or
                 resolve_entity_name(str(table), entity_map).upper() != default_entity.upper())
        ):
            # Encode cross-entity join path; prefer ## root interface spelling.
            if str(table).startswith("##"):
                rel = str(table)
            elif ref.source_table.startswith("##"):
                rel = ref.source_table
            else:
                rel = ref.entity if ref.entity.startswith("##") else f"##{ref.entity}"
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
