"""Phase 1 — local `#` temp-table lineage resolver.

Builds a symbol map from ``#LocalTable.Column`` → root entity column
references. Global temps (``##AccountCal``, ``##CUSTOMERCAL``, …) are
treated as root interface entities and are never traced outside the script.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.derivation.v2.sql_text import (
    bare_ident,
    extract_cte_definitions,
    extract_insert_select,
    extract_select_into,
    extract_update_statements,
    iter_set_assignments,
    normalize_table_name,
    parse_from_join_clause,
    resolve_expression_column_refs,
    split_csv_respecting_parens,
    split_select_from,
    strip_sql_comments,
)
from app.utils.entity_name_map import resolve_entity_name
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_CREATE_TEMP_RE = re.compile(
    r"(?is)\bCREATE\s+TABLE\s+(?P<target>#+#?[A-Za-z0-9_]+)\s*\((?P<body>.*?)\)",
)

_COL_AS_RE = re.compile(
    r"(?is)(?:(?P<qual>[#A-Za-z0-9_]+)\.)?(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?)"
    r"(?:\s+AS\s+(?P<alias>\[?[A-Za-z_][A-Za-z0-9_]*\]?))?",
)


@dataclass
class RootColumnRef:
    """Resolved root location for a local temp column."""

    entity: str
    column: str
    relationship: str | None = None
    source_table: str = ""
    # Mutation checkpoint: a resolved 4X-marker-ready expression string when
    # this temp column was re-derived by its own UPDATE after being written
    # (e.g. a staging-table markup pass). Downstream consumers that read this
    # column should inherit this expression instead of the flat entity/column
    # marker, or they lose the derived transformation entirely.
    derived_formula: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "column": self.column,
            "relationship": self.relationship,
            "source_table": self.source_table,
            "derived_formula": self.derived_formula,
        }


@dataclass
class LineageMap:
    """Phase-1 symbol map: ``#Temp.Col`` → root entity column."""

    columns: dict[str, RootColumnRef] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    root_entities: set[str] = field(default_factory=set)
    # Column order per local temp table, captured from its CREATE TABLE
    # definition. Used to fall back to ordinal alignment when a downstream
    # INSERT ... SELECT into that temp omits an explicit column list.
    temp_table_columns: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": {key: ref.as_dict() for key, ref in self.columns.items()},
            "tables": dict(self.tables),
            "root_entities": sorted(self.root_entities),
            "temp_table_columns": dict(self.temp_table_columns),
        }

    def resolve_column(
        self,
        table: str,
        column: str,
        entity_map: dict[str, str] | None = None,
    ) -> RootColumnRef:
        key = _column_key(table, column)
        if key in self.columns:
            return self.columns[key]

        canon_table = normalize_table_name(table)
        if _is_global_temp(canon_table):
            entity = resolve_entity_name(canon_table, entity_map) or canon_table
            self.root_entities.add(entity)
            ref = RootColumnRef(entity=entity, column=bare_ident(column), source_table=canon_table)
            self.columns[key] = ref
            return ref

        if canon_table in self.tables:
            entity = self.tables[canon_table]
            ref = RootColumnRef(entity=entity, column=bare_ident(column), source_table=canon_table)
            self.columns[key] = ref
            return ref

        entity = resolve_entity_name(canon_table, entity_map) or canon_table
        if _is_global_temp(canon_table) or not canon_table.startswith("#"):
            self.root_entities.add(entity)
        return RootColumnRef(entity=entity, column=bare_ident(column), source_table=canon_table)


def build_lineage_map(
    sql_text: str,
    entity_map: dict[str, str] | None = None,
) -> LineageMap:
    """Scan T-SQL and build a local-temp → root-entity column symbol map."""
    lineage = LineageMap()
    text = strip_sql_comments(sql_text or "")

    for match in re.finditer(r"##[A-Za-z_][A-Za-z0-9_]*", text):
        table = match.group(0)
        entity = resolve_entity_name(table, entity_map) or table
        lineage.root_entities.add(entity)
        lineage.tables.setdefault(table, entity)

    for match in _CREATE_TEMP_RE.finditer(text):
        target = normalize_table_name(match.group("target"))
        if _is_global_temp(target):
            entity = resolve_entity_name(target, entity_map) or target
            lineage.root_entities.add(entity)
            lineage.tables[target] = entity
        else:
            lineage.tables.setdefault(target, target)
        col_order = [
            bare_ident(part.strip().split()[0])
            for part in split_csv_respecting_parens(match.group("body"))
            if part.strip()
        ]
        if col_order:
            lineage.temp_table_columns.setdefault(target, col_order)

    for item in extract_select_into(text):
        _ingest_select_projection(
            lineage,
            target=item["target"],
            select_list=item["select_list"],
            from_body=item["from_body"],
            entity_map=entity_map,
        )

    for item in extract_insert_select(text, temps_only=True):
        col_names = [bare_ident(c) for c in item.get("cols", "").split(",") if bare_ident(c)]
        _ingest_select_projection(
            lineage,
            target=item["target"],
            select_list=item["select_list"],
            from_body=item["from_body"],
            entity_map=entity_map,
            explicit_target_columns=col_names or None,
        )

    # Common table expressions: ``;WITH cte(cols) AS (SELECT ...) UPDATE ...
    # FROM cte alias``. Registered exactly like a local temp table's
    # projection lineage — without this, a CTE alias reference resolves
    # through no lineage at all and the engine treats the literal CTE name
    # as if it were a real database table.
    for cte in extract_cte_definitions(text):
        col_names = [bare_ident(c) for c in (cte.get("cols") or "").split(",") if bare_ident(c)]
        select_list, from_body = split_select_from(cte.get("body") or "")
        if not select_list:
            continue
        _ingest_select_projection(
            lineage,
            target=cte["name"],
            select_list=select_list,
            from_body=from_body,
            entity_map=entity_map,
            explicit_target_columns=col_names or None,
        )

    _capture_temp_column_mutations(lineage, text, entity_map)
    _expand_transitive(lineage)
    logger.debug(
        "phase1 lineage: %d column(s), %d root entit(y/ies)",
        len(lineage.columns),
        len(lineage.root_entities),
    )
    return lineage


def _ingest_select_projection(
    lineage: LineageMap,
    *,
    target: str,
    select_list: str,
    from_body: str,
    entity_map: dict[str, str] | None,
    explicit_target_columns: list[str] | None = None,
) -> None:
    if _is_global_temp(target):
        entity = resolve_entity_name(target, entity_map) or target
        lineage.root_entities.add(entity)
        lineage.tables[target] = entity
        return

    alias_to_table = _alias_map(from_body)
    primary_root = _pick_primary_root(alias_to_table, entity_map)
    if primary_root:
        lineage.tables[target] = primary_root
        lineage.root_entities.add(primary_root)

    projections = _parse_select_list(select_list)
    if explicit_target_columns and len(explicit_target_columns) == len(projections):
        # Strict zero-indexed ordinal alignment: column Ci pairs ONLY with
        # projection Ei, even when some Ej in between is a CASE/function
        # expression with no direct column ref (that slot is simply skipped,
        # not dropped from the list — dropping it would shift every column
        # after it out of position).
        for dest_col, (src_qual, src_col, _alias, _raw) in zip(
            explicit_target_columns, projections
        ):
            if src_qual is None and src_col is None:
                continue
            _register_projection(
                lineage, target, dest_col, src_qual, src_col, alias_to_table, entity_map
            )
    else:
        for src_qual, src_col, dest_alias, _raw in projections:
            if src_qual is None and src_col is None:
                continue
            dest = dest_alias or src_col
            _register_projection(
                lineage, target, dest, src_qual, src_col, alias_to_table, entity_map
            )


def _register_projection(
    lineage: LineageMap,
    target: str,
    dest_col: str,
    src_qual: str | None,
    src_col: str,
    alias_to_table: dict[str, str],
    entity_map: dict[str, str] | None,
) -> None:
    src_table = None
    if src_qual:
        src_table = alias_to_table.get(bare_ident(src_qual).upper()) or normalize_table_name(src_qual)
    if not src_table and alias_to_table:
        # Prefer first global-temp / physical table.
        for table in alias_to_table.values():
            if _is_global_temp(table) or not table.startswith("#"):
                src_table = table
                break
        if not src_table:
            src_table = next(iter(alias_to_table.values()))

    if not src_table:
        return

    resolved = lineage.resolve_column(src_table, src_col, entity_map)
    relationship = resolved.relationship
    if relationship is None and _is_global_temp(src_table):
        # Preserve join-path hint when source is a different root interface.
        primary = lineage.tables.get(target)
        entity = resolve_entity_name(src_table, entity_map) or src_table
        if primary and entity.upper() != str(primary).upper():
            relationship = entity if entity.startswith("##") else src_table

    key = _column_key(target, dest_col)
    lineage.columns[key] = RootColumnRef(
        entity=resolved.entity,
        column=resolved.column,
        relationship=relationship,
        source_table=src_table,
    )


def _expand_transitive(lineage: LineageMap, max_passes: int = 8) -> None:
    for _ in range(max_passes):
        changed = False
        for key, ref in list(lineage.columns.items()):
            if not ref.source_table.startswith("#") or ref.source_table.startswith("##"):
                continue
            upstream_key = _column_key(ref.source_table, ref.column)
            upstream = lineage.columns.get(upstream_key)
            if upstream is None:
                root = lineage.tables.get(ref.source_table)
                if root and root != ref.entity:
                    lineage.columns[key] = RootColumnRef(
                        entity=root,
                        column=ref.column,
                        relationship=ref.relationship,
                        source_table=ref.source_table,
                        derived_formula=ref.derived_formula,
                    )
                    changed = True
                continue
            if (
                upstream.entity != ref.entity
                or upstream.column != ref.column
                or upstream.derived_formula
            ):
                lineage.columns[key] = RootColumnRef(
                    entity=upstream.entity,
                    column=upstream.column,
                    relationship=upstream.relationship or ref.relationship,
                    source_table=ref.source_table,
                    # A downstream temp's own mutation checkpoint (if any)
                    # takes precedence; otherwise inherit the upstream one so
                    # chained #temp -> #temp2 -> root derivations aren't lost.
                    derived_formula=ref.derived_formula or upstream.derived_formula,
                )
                changed = True
        if not changed:
            break


def _capture_temp_column_mutations(
    lineage: LineageMap,
    text: str,
    entity_map: dict[str, str] | None,
) -> None:
    """Fold UPDATEs against local ``#temp`` columns into mutation checkpoints.

    Phase 1 otherwise only ever sees a temp column's *initial* INSERT/SELECT
    INTO projection and immediately strips the temp table, pointing straight
    at the root entity. When that column is later re-derived by its own
    UPDATE (a classic staging-table markup pass ahead of a downstream MERGE
    or UPDATE), that re-derivation must not be silently discarded — record
    it as ``derived_formula`` so downstream consumers (phase2's expression
    resolver) inherit the folded expression instead of the pre-mutation
    root value.
    """
    for stmt in extract_update_statements(text):
        head = (stmt.get("head") or "").strip()
        set_clause = (stmt.get("set_clause") or "").strip()
        from_clause = (stmt.get("from_clause") or "").strip()
        where_clause = (stmt.get("where_clause") or "").strip() or None
        if not set_clause or not head:
            continue

        alias_map: dict[str, str] = {}
        target_tables: list[str] = []
        head_match = re.match(
            r"(?is)^(?P<table>\[?#+[A-Za-z0-9_\.]+\]?)"
            r"(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?$",
            head,
        )
        if head_match:
            table = normalize_table_name(head_match.group("table"))
            alias_map[bare_ident(table).upper()] = table
            if head_match.group("alias"):
                alias_map[head_match.group("alias").upper()] = table
            target_tables.append(table)

        for from_table, from_alias, _on in parse_from_join_clause(from_clause):
            alias_map[bare_ident(from_table).upper()] = from_table
            if from_alias:
                alias_map[from_alias.upper()] = from_table
            norm_from = normalize_table_name(from_table)
            # ``UPDATE alias SET ... FROM #Temp alias`` — head is a bare alias
            # that resolves to the temp table declared in FROM.
            if (
                not head_match
                and not target_tables
                and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", head)
                and from_alias
                and from_alias.upper() == head.upper()
            ):
                target_tables.append(norm_from)

        for table in target_tables:
            norm = normalize_table_name(table)
            if not norm.startswith("#") or norm.startswith("##"):
                continue
            for assign in iter_set_assignments(set_clause):
                col = assign["column"]
                if assign["alias"]:
                    resolved_alias_tbl = alias_map.get(assign["alias"].upper())
                    if (
                        resolved_alias_tbl
                        and normalize_table_name(resolved_alias_tbl).upper() != norm.upper()
                    ):
                        continue

                key = _column_key(norm, col)
                prior = lineage.columns.get(key)
                resolved_expr = resolve_expression_column_refs(
                    assign["expr"], alias_map, lineage, entity_map, norm
                )
                if where_clause:
                    resolved_where = resolve_expression_column_refs(
                        where_clause, alias_map, lineage, entity_map, norm
                    )
                    if prior and prior.derived_formula:
                        prior_expr = prior.derived_formula
                    elif prior:
                        prior_expr = f"{prior.entity}::{prior.column}"
                    else:
                        prior_expr = "NULL"
                    formula = (
                        f"CASE WHEN {resolved_where} THEN {resolved_expr} "
                        f"ELSE {prior_expr} END"
                    )
                else:
                    formula = resolved_expr

                if prior:
                    prior.derived_formula = formula
                else:
                    entity = lineage.tables.get(norm, norm)
                    lineage.columns[key] = RootColumnRef(
                        entity=entity,
                        column=col,
                        source_table=norm,
                        derived_formula=formula,
                    )


def _alias_map(from_body: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for table, alias, _on in parse_from_join_clause(from_body):
        mapping[bare_ident(table).upper()] = table
        mapping[table.upper()] = table
        if alias:
            mapping[alias.upper()] = table
    return mapping


def _parse_select_list(
    select_list: str,
) -> list[tuple[str | None, str | None, str | None, str]]:
    """Parse a SELECT projection list, preserving one entry per ordinal slot.

    Returns ``(src_qual, src_col, dest_alias, raw_chunk)`` per top-level item.
    Complex expressions (CASE / function calls / subqueries) cannot resolve
    to a single ``src_qual``/``src_col`` and are returned as
    ``(None, None, alias_if_any, raw_chunk)`` — the caller MUST still count
    this slot (not skip it) so positional alignment with an explicit target
    column list stays correct for every projection after it.
    """
    results: list[tuple[str | None, str | None, str | None, str]] = []
    if not select_list or not select_list.strip() or select_list.strip() == "*":
        return results

    # Strip leading SELECT modifiers (DISTINCT / ALL / TOP n) before
    # splitting — otherwise the first chunk's own column regex greedily
    # matches the modifier keyword itself as if it were the column name
    # (``DISTINCT UcifEntityID`` -> column "DISTINCT").
    select_list = select_list.strip()
    for _ in range(3):
        m = re.match(r"(?is)^(?:DISTINCT|ALL)\s+", select_list)
        if m:
            select_list = select_list[m.end():]
            continue
        m = re.match(r"(?is)^TOP\s*\(?\s*\d+\s*\)?\s+", select_list)
        if m:
            select_list = select_list[m.end():]
            continue
        break

    for part in split_csv_respecting_parens(select_list):
        chunk = part.strip()
        if not chunk or chunk == "*":
            results.append((None, None, None, chunk))
            continue
        if "(" in chunk:
            alias = None
            alias_m = re.search(
                r"(?is)\)\s*(?:AS\s+)?(?P<alias>\[?[A-Za-z_][A-Za-z0-9_]*\]?)\s*$",
                chunk,
            )
            if alias_m:
                alias = bare_ident(alias_m.group("alias"))
            results.append((None, None, alias, chunk))
            continue
        match = _COL_AS_RE.search(chunk)
        if not match:
            results.append((None, None, None, chunk))
            continue
        qual = match.group("qual")
        col = bare_ident(match.group("col"))
        alias = bare_ident(match.group("alias")) if match.group("alias") else None
        if col.upper() in {"AS", "FROM", "INTO"}:
            results.append((None, None, None, chunk))
            continue
        results.append((qual, col, alias, chunk))
    return results


def _pick_primary_root(
    alias_to_table: dict[str, str],
    entity_map: dict[str, str] | None,
) -> str | None:
    tables = list(dict.fromkeys(alias_to_table.values()))
    for table in tables:
        if _is_global_temp(table):
            return resolve_entity_name(table, entity_map) or table
    for table in tables:
        if not table.startswith("#"):
            return resolve_entity_name(table, entity_map) or table
    for table in tables:
        return resolve_entity_name(table, entity_map) or table
    return None


def _column_key(table: str, column: str) -> str:
    return f"{normalize_table_name(table)}.{bare_ident(column)}"


def _is_global_temp(table: str) -> bool:
    return (table or "").startswith("##")


# Back-compat aliases used by phase2
_bare = bare_ident
_normalize_table = normalize_table_name
