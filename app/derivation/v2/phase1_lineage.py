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
    extract_insert_select,
    extract_select_into,
    normalize_table_name,
    parse_from_join_clause,
    split_csv_respecting_parens,
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "column": self.column,
            "relationship": self.relationship,
            "source_table": self.source_table,
        }


@dataclass
class LineageMap:
    """Phase-1 symbol map: ``#Temp.Col`` → root entity column."""

    columns: dict[str, RootColumnRef] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    root_entities: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": {key: ref.as_dict() for key, ref in self.columns.items()},
            "tables": dict(self.tables),
            "root_entities": sorted(self.root_entities),
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
        for dest_col, (src_qual, src_col, _) in zip(explicit_target_columns, projections):
            _register_projection(
                lineage, target, dest_col, src_qual, src_col, alias_to_table, entity_map
            )
    else:
        for src_qual, src_col, dest_alias in projections:
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
                    )
                    changed = True
                continue
            if upstream.entity != ref.entity or upstream.column != ref.column:
                lineage.columns[key] = RootColumnRef(
                    entity=upstream.entity,
                    column=upstream.column,
                    relationship=upstream.relationship or ref.relationship,
                    source_table=ref.source_table,
                )
                changed = True
        if not changed:
            break


def _alias_map(from_body: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for table, alias, _on in parse_from_join_clause(from_body):
        mapping[bare_ident(table).upper()] = table
        mapping[table.upper()] = table
        if alias:
            mapping[alias.upper()] = table
    return mapping


def _parse_select_list(select_list: str) -> list[tuple[str | None, str, str | None]]:
    results: list[tuple[str | None, str, str | None]] = []
    if not select_list or not select_list.strip() or select_list.strip() == "*":
        return results

    for part in split_csv_respecting_parens(select_list):
        chunk = part.strip()
        if not chunk or chunk == "*":
            continue
        if "(" in chunk:
            continue
        match = _COL_AS_RE.search(chunk)
        if not match:
            continue
        qual = match.group("qual")
        col = bare_ident(match.group("col"))
        alias = bare_ident(match.group("alias")) if match.group("alias") else None
        if col.upper() in {"AS", "FROM", "INTO"}:
            continue
        results.append((qual, col, alias))
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
