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
    extract_catch_spans,
    extract_if_else_chains,
    extract_insert_select,
    extract_merge_matched_updates,
    extract_merge_using_row_predicate,
    extract_select_into,
    extract_subquery_dependency_refs,
    extract_table_resets,
    extract_update_statements,
    extract_workflow_gates,
    lineage_keeps_target_hop,
    normalize_table_name,
    parse_from_join_clause,
    parse_from_join_clause_with_type,
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
    # "FROM", "JOIN", "INNER JOIN", "LEFT JOIN", ... — see
    # ``_join_filter_terms``, which only treats a non-key ON-clause predicate
    # as a row filter for INNER/plain/CROSS joins, never LEFT/RIGHT/FULL.
    join_type: str = "JOIN"

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "alias": self.alias,
            "on_clause": self.on_clause,
            "resolved_entity": self.resolved_entity,
            "join_type": self.join_type,
        }


_AND_KEYWORD_RE = re.compile(r"(?i)\bAND\b")
_BETWEEN_KEYWORD_RE = re.compile(r"(?i)\bBETWEEN\b")

# DATEADD unit tokens must not be bare-qualified as ``table.YY`` etc.
_DATEADD_UNIT_KEYWORDS = frozenset({
    "DD", "DY", "DAY", "DAYS", "WK", "WW", "WEEK", "WEEKS",
    "MM", "MONTH", "MONTHS", "YY", "YYYY", "YEAR", "YEARS",
    "QQ", "QUARTER", "QUARTERS", "HH", "MI", "SS", "MS",
})


def _split_top_level_and_terms(text: str) -> list[str]:
    """Split ``text`` on top-level ``AND`` (paren- and quote-aware)."""
    if not text:
        return []
    terms: list[str] = []
    depth = 0
    in_quote = False
    quote_char = ""
    between_active = False
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_quote:
            if ch == quote_char:
                in_quote = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = True
            quote_char = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            i += 1
            continue
        if depth == 0:
            between_match = _BETWEEN_KEYWORD_RE.match(text, i)
            if between_match:
                between_active = True
                i = between_match.end()
                continue
            match = _AND_KEYWORD_RE.match(text, i)
            if match:
                if between_active:
                    between_active = False
                    i = match.end()
                    continue
                terms.append(text[start:i])
                i = match.end()
                start = i
                continue
        i += 1
    terms.append(text[start:])
    return [t.strip() for t in terms if t.strip()]


_JOIN_KEY_EQUALITY_RE = re.compile(
    r"(?is)^\s*(?P<l_alias>[A-Za-z_][\w]*)\.(?P<l_col>[A-Za-z_][\w]*)\s*=\s*"
    r"(?P<r_alias>[A-Za-z_][\w]*)\.(?P<r_col>[A-Za-z_][\w]*)\s*$"
)


def _is_inner_style_join(join: "JoinInfo") -> bool:
    """True for joins whose ON clause restricts which target rows are updated."""
    join_type = (join.join_type or "JOIN").upper()
    return not any(kw in join_type for kw in ("LEFT", "RIGHT", "FULL"))


def _join_filter_terms(join: "JoinInfo") -> list[str]:
    """Extra row-filtering predicates in a JOIN's ON clause, beyond the
    structural ``<this>.<col> = <other>.<col>`` linking key(s).

    ``INNER JOIN X a ON a.k = b.k AND a.Status = 'ACTIVE'`` filters rows
    exactly as a WHERE term would — only the plain key-equality piece is
    structural (already captured separately as the join relationship).
    Dropping the rest silently widens the derived row condition to match
    every row regardless of that filter. This is what lets
    ``effective_condition`` fold a JOIN's own predicates in below.

    Only applies to INNER/plain/CROSS joins. A LEFT/RIGHT/FULL JOIN's
    ON-clause predicates decide whether the *joined* side matches, not
    whether the driving row survives — folding those in as a row filter
    would incorrectly drop driving rows the source UPDATE still touches.
    """
    if not join.on_clause:
        return []
    if not _is_inner_style_join(join):
        return []
    alias = (join.alias or join.table or "").strip().strip('"[]').upper()
    if not alias:
        return [join.on_clause]
    kept: list[str] = []
    for term in _split_top_level_and_terms(join.on_clause):
        match = _JOIN_KEY_EQUALITY_RE.match(term)
        if match:
            l_alias = match.group("l_alias").strip('"[]').upper()
            r_alias = match.group("r_alias").strip('"[]').upper()
            if alias in (l_alias, r_alias):
                continue
        kept.append(term)
    return kept


def _resolve_join_filter_condition(
    joins: list["JoinInfo"],
    alias_map: dict[str, str],
    lineage: "LineageMap",
    entity_map: dict[str, str] | None,
    target_entity: str,
) -> str | None:
    """Alias-resolved form of every JOIN's extra ON-clause filter term(s)
    (see ``_join_filter_terms``), ANDed together.

    Runs each raw term through the same ``_resolve_expression_tables`` pass
    that ``where_clause``/``outer_condition`` already get, so a JOIN's own
    filter reads consistently -- e.g. the resolved entity/column form, not
    the bare source alias (``P.ProductGroup``) -- and parses/validates the
    same way the rest of the derived condition does.
    """
    terms: list[str] = []
    for join in joins:
        extra = _join_filter_terms(join)
        if extra:
            terms.extend(extra)
        elif _is_inner_style_join(join) and (join.on_clause or "").strip():
            # Pure key-equality ON clauses are stripped by ``_join_filter_terms``,
            # but an UPDATE … INNER JOIN still only touches matching rows.
            terms.append(join.on_clause.strip())
    if not terms:
        return None
    resolved = [
        _resolve_expression_tables(
            term, alias_map, lineage, entity_map, target_entity, use_derived_formula=False,
        )
        for term in terms
    ]
    return " AND ".join(f"({t})" for t in resolved)


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
    # "CATCH" when the write sits in a BEGIN CATCH … END CATCH handler; None
    # for the normal (TRY / main) execution path.
    exception_scope: str | None = None
    # Procedure-wide IF gate guarding this write ("Gate N") and its original
    # SQL condition — report context only; never emitted into the formula.
    workflow_gate: str | None = None
    workflow_gate_condition: str | None = None
    # Set on the first write after a TRUNCATE / DELETE / DROP of the target
    # table that discarded earlier writes (and the reset's source offset).
    state_reset: str | None = None
    state_reset_position: int | None = None
    # Extra JOIN ON-clause filter predicate(s) (already alias-resolved to
    # full entity/column form, same as ``where_clause``) — see
    # ``_resolve_join_filter_condition``. None when no JOIN in this pass
    # carries a non-key filter.
    join_filter_condition: str | None = None

    @property
    def is_exception_handler(self) -> bool:
        return (self.exception_scope or "").upper() == "CATCH"

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

        Also folds in ``join_filter_condition`` — an alias-resolved extra
        filter predicate from a JOIN's own ON clause (see
        ``_resolve_join_filter_condition``), when the statement's WHERE
        doesn't already state it.
        """
        where = self.where_clause.strip() if self.where_clause and self.where_clause.strip() else None
        outer = (
            self.outer_condition.strip()
            if self.outer_condition and self.outer_condition.strip()
            else None
        )
        if where and outer:
            base = where if outer.upper() in where.upper() else f"({outer}) AND ({where})"
        else:
            base = where or outer

        extra = self.join_filter_condition.strip() if self.join_filter_condition else None
        if not extra:
            return base
        if base and extra.upper() in base.upper():
            return base
        return f"({base}) AND ({extra})" if base else extra

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
            "join_filter_condition": self.join_filter_condition,
            "effective_condition": self.effective_condition,
            "dependency_refs": list(self.dependency_refs),
            "exception_scope": self.exception_scope,
            "workflow_gate": self.workflow_gate,
            "workflow_gate_condition": self.workflow_gate_condition,
            "state_reset": self.state_reset,
        }


@dataclass
class MutationSourceIndex:
    """Object-scoped scan results, prepared once before column workers start.

    UPDATE assignments are indexed by column, preserving statement indexes,
    source order and duplicate assignments. Workers only read this structure;
    each fold still creates its own mutable MutationPass objects.
    """

    sql_text: str
    updates: list
    updates_by_column: dict
    inserts: list
    selects: list
    merges: list
    branches: list
    gates_by_arm: dict
    catch_spans: list
    resets: list

    @classmethod
    def build(cls, sql_text: str) -> "MutationSourceIndex":
        updates = extract_update_statements(sql_text)
        by_column: dict = {}
        for index, stmt in enumerate(updates):
            grouped: dict = {}
            for assignment in _iter_set_assignments(stmt.get("set_clause") or ""):
                grouped.setdefault(assignment["column"].upper(), []).append(assignment)
            for column, assignments in grouped.items():
                by_column.setdefault(column, []).append((index, stmt, assignments))
        stripped = strip_sql_comments(sql_text)
        return cls(
            sql_text, updates, by_column,
            extract_insert_select(sql_text), extract_select_into(sql_text),
            extract_merge_matched_updates(sql_text), extract_if_else_chains(stripped),
            {(g.group_id, g.index): g for g in extract_workflow_gates(stripped)},
            extract_catch_spans(stripped), extract_table_resets(stripped),
        )


_BARE_IDENT_RE = re.compile(r"(?<![\w.@#\[\]:\"])([A-Za-z_]\w*)")


def _written_columns_by_table(
    source: "MutationSourceIndex",
    lineage: LineageMap,
    entity_map: dict[str, str] | None,
) -> dict[str, set[str]]:
    """``{NORMALIZED_TABLE: {COLUMN,…}}`` assigned by any UPDATE in the object.

    File-local evidence of which table owns a column; computed once per source.
    """
    cached = getattr(source, "_written_cols_cache", None)
    if cached is not None:
        return cached
    out: dict[str, set[str]] = {}
    for stmt in source.updates:
        alias_map, _ = _parse_update_sources(
            (stmt.get("head") or "").strip(),
            (stmt.get("from_clause") or "").strip(),
            lineage,
            entity_map,
        )
        written = _resolve_update_target_tables((stmt.get("head") or "").strip(), alias_map)
        for assign in _iter_set_assignments(stmt.get("set_clause") or ""):
            tables = list(written)
            if assign["alias"] and alias_map.get(assign["alias"].upper()):
                tables = [alias_map[assign["alias"].upper()]]
            for table in tables:
                key = normalize_table_name(str(table)).upper()
                out.setdefault(key, set()).add(bare_ident(assign["column"]).upper())
    source._written_cols_cache = out  # type: ignore[attr-defined]
    return out


_QUALIFIED_READ_RE = re.compile(r"(?<![\w.@#\[\]\"])([A-Za-z_]\w*)\.\[?([A-Za-z_]\w*)\]?")


def _dim_read_columns_by_table(
    source: "MutationSourceIndex",
    lineage: LineageMap,
    entity_map: dict[str, str] | None,
) -> dict[str, set[str]]:
    """``{DIM_TABLE: {COLUMN,…}}`` seen qualified as ``alias.col`` in any UPDATE.

    A lookup attribute (``DimParameter.ParameterShortNameEnum``) is read as
    ``D.ParameterShortNameEnum`` in one statement and bare in another; the
    qualified reads are file-local proof of which table owns it. Limited to
    ``Dim*`` lookup tables: fact tables share column names with their own
    target, where a bare name must keep meaning the UPDATE target's column.
    """
    cached = getattr(source, "_dim_read_cols_cache", None)
    if cached is not None:
        return cached
    out: dict[str, set[str]] = {}
    for stmt in source.updates:
        alias_map, _ = _parse_update_sources(
            (stmt.get("head") or "").strip(),
            (stmt.get("from_clause") or "").strip(),
            lineage,
            entity_map,
        )
        text = " ".join(
            str(stmt.get(k) or "") for k in ("set_clause", "from_clause", "where_clause")
        )
        for alias, col in _QUALIFIED_READ_RE.findall(text):
            table = alias_map.get(alias.upper())
            if not table or isinstance(table, DerivedTable):
                continue
            key = normalize_table_name(str(table)).upper()
            if key.lstrip("#").startswith("DIM"):
                out.setdefault(key, set()).add(bare_ident(col).upper())
    source._dim_read_cols_cache = out  # type: ignore[attr-defined]
    return out


def _augment_written_cols_with_join_temp_schemas(
    written_cols: dict[str, set[str]],
    alias_map: dict[str, str],
    lineage: LineageMap,
    dim_read_cols: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    """Add ``SELECT INTO #temp`` column names so bare refs qualify to the join alias."""
    out: dict[str, set[str]] = {k: set(v) for k, v in written_cols.items()}
    for key, cols in (dim_read_cols or {}).items():
        out.setdefault(key, set()).update(cols)
    for _alias, table in alias_map.items():
        key = normalize_table_name(str(table)).upper()
        schema = lineage.temp_table_columns.get(normalize_table_name(str(table))) or []
        if not schema:
            continue
        bucket = out.setdefault(key, set())
        for col in schema:
            bucket.add(bare_ident(col).upper())
    return out


def _qualify_foreign_bare_columns(
    text: str | None,
    alias_map: dict[str, str],
    written_tables: list[str],
    written_cols: dict[str, set[str]],
) -> str | None:
    """Qualify a bare column that only a JOINED table (never the UPDATE target)
    is assigned in this object, e.g. ``UPDATE A SET T = ISNULL(RestructureProvision,0)
    FROM ##AccountCal A JOIN PRO.AdvAcRestructureCal B`` -> ``B.RestructureProvision``.

    T-SQL binds an unqualified name to whichever in-scope table owns it; without
    a schema, the only evidence available is the columns this object itself
    assigns per table. A column is rewritten only when exactly one non-target
    joined table owns it and the target table does not.
    """
    if not text or not alias_map:
        return text
    target_keys = {normalize_table_name(str(t)).upper() for t in written_tables}
    owners: dict[str, str] = {}  # column -> alias, for columns owned by one foreign table
    ambiguous: set[str] = set()
    seen_tables: set[str] = set()
    for alias, table in alias_map.items():
        key = normalize_table_name(str(table)).upper()
        if key in target_keys or key in seen_tables or "." in alias:
            continue
        seen_tables.add(key)
        for col in written_cols.get(key, ()):
            if col in owners:
                ambiguous.add(col)
            owners[col] = alias
    if not owners:
        return text
    target_cols: set[str] = set()
    for key in target_keys:
        target_cols |= written_cols.get(key, set())

    def _sub(segment: str) -> str:
        def repl(match: re.Match[str]) -> str:
            ident = match.group(1)
            col = ident.upper()
            if col not in owners or col in ambiguous or col in target_cols:
                return ident
            tail = segment[match.end():].lstrip()
            if tail[:1] in {"(", ".", ":"}:
                return ident
            return f"{owners[col]}.{ident}"

        return _BARE_IDENT_RE.sub(repl, segment)

    parts = re.split(r"('(?:[^']|'')*')", text)
    return "".join(p if p.startswith("'") else _sub(p) for p in parts)


def fold_column_mutations(
    sql_text: str,
    target_entity: str,
    target_column: str,
    lineage: LineageMap,
    entity_map: dict[str, str] | None = None,
    *,
    source_index: MutationSourceIndex | None = None,
) -> list[MutationPass]:
    """Chronologically collect UPDATE passes that mutate target_entity.column."""
    target_entity_norm = _normalize_entity(target_entity, entity_map)
    target_col = bare_ident(target_column)
    mutations: list[MutationPass] = []
    ordinal = 0

    # ``#X`` (session temp) and ``X`` (permanent table) sharing a bare name are
    # separate entities; ``#X`` targets keep their hash and only see temp writes.
    _raw_target = str(target_entity or "").strip()
    target_is_hash = _raw_target.startswith("#") and not _raw_target.startswith("##")
    target_bare_upper = bare_ident(normalize_table_name(_raw_target).lstrip("#")).upper()
    if target_is_hash:
        target_entity_norm = "#" + bare_ident(normalize_table_name(_raw_target).lstrip("#"))

    def _collision_side_ok(tables: list[str]) -> bool:
        if target_bare_upper not in lineage.hash_collisions:
            return True
        for table in tables:
            norm = normalize_table_name(str(table))
            if bare_ident(norm.lstrip("#")).upper() != target_bare_upper:
                continue
            is_temp = norm.startswith("#") and not norm.startswith("##")
            if is_temp != target_is_hash:
                return False
        return True

    # Map UPDATE source offsets → IF/ELSE branch (comment-stripped coordinates).
    source = source_index or MutationSourceIndex.build(sql_text)
    if source.sql_text != sql_text:
        raise ValueError("Mutation source index does not match the SQL input")
    branch_spans = source.branches

    def _branch_for_offset(pos: int):
        for span in branch_spans:
            if span.body_start <= pos < span.body_end:
                return span
        return None

    gates_by_arm = source.gates_by_arm
    catch_spans = source.catch_spans

    def _scope_for_offset(pos: int) -> str | None:
        return "CATCH" if any(s <= pos < e for s, e in catch_spans) else None

    def _outer_for_branch(branch, resolve) -> tuple[str | None, Any]:
        """Row predicate for a branch arm, plus its procedure-wide gate (if any).

        The exported formula must contain only real columns and parameters,
        so a set-level ``IF EXISTS (… WHERE p)`` still folds as its row
        projection ``p``. The gate is returned separately so reports can state
        the procedural context the formula cannot express.
        """
        if not branch or not branch.condition:
            return None, None
        return resolve(branch.condition), gates_by_arm.get((branch.group_id, branch.index))

    _written_back_cache: dict[str, bool] = {}

    def _temp_written_back(temp_norm: str) -> bool:
        """True when ``temp_norm``'s rows are copied into ``target_col`` by some
        other statement (UPDATE … FROM #temp, INSERT … SELECT FROM #temp, MERGE)."""
        key = temp_norm.upper()
        if key in _written_back_cache:
            return _written_back_cache[key]
        pattern = re.compile(rf"(?i)(?<![\w#]){re.escape(temp_norm)}(?!\w)")
        found = False
        for _idx, wb_stmt, _asg in source.updates_by_column.get(target_col.upper(), []):
            wb_head = (wb_stmt.get("head") or "").strip()
            wb_from = (wb_stmt.get("from_clause") or "").strip()
            if not pattern.search(wb_from):
                continue
            wb_map, _wb_joins = _parse_update_sources(wb_head, wb_from, lineage, entity_map)
            written = _resolve_update_target_tables(wb_head, wb_map)
            if any(normalize_table_name(str(w)).upper() == key for w in written):
                continue  # the temp's own UPDATE, not a copy into another table
            found = True
            break
        if not found:
            for ins in source.inserts:
                cols = [bare_ident(c).upper() for c in (ins.get("cols") or "").split(",")]
                if target_col.upper() in cols and pattern.search(ins.get("from_body") or ""):
                    if normalize_table_name(ins.get("target") or "").upper() != key:
                        found = True
                        break
        if not found:
            for merge in source.merges:
                body = " ".join(
                    str(merge.get(k) or "") for k in ("using_body", "source_table", "set_clause")
                )
                if target_col.upper() in body.upper() and pattern.search(body):
                    found = True
                    break
        _written_back_cache[key] = found
        return found

    def _skip_unwritten_session_temp(tables: list[str]) -> bool:
        """A write into a session ``#temp`` that is not this target's own table, and
        whose rows are never copied back into the target column, is not a write to
        the target at all (``SELECT … INTO #DPD`` / ``UPDATE #DPD`` copies of
        ``##AccountCal`` columns). Folding it only injects stray arms into the
        target's formula."""
        if not tables:
            return False
        for table in tables:
            norm = normalize_table_name(str(table))
            if not norm.startswith("#") or norm.startswith("##"):
                return False
            if _entity_keys(target_entity_norm) & _entity_keys(norm):
                return False
            if _temp_written_back(norm):
                return False
        return True

    for stmt_index, stmt, assignments in source.updates_by_column.get(target_col.upper(), []):
        head = (stmt.get("head") or "").strip()
        set_clause = (stmt.get("set_clause") or "").strip()
        from_clause = (stmt.get("from_clause") or "").strip()
        where_clause = (stmt.get("where_clause") or "").strip() or None
        raw_sql = (stmt.get("raw_sql") or "").strip()
        stmt_start = int(stmt.get("start") or 0)

        alias_map, joins = _parse_update_sources(head, from_clause, lineage, entity_map)
        written_tables = _resolve_update_target_tables(head, alias_map)
        branch = _branch_for_offset(stmt_start)

        for assign in assignments:

            tables_for_assign = list(written_tables)
            if assign["alias"]:
                resolved = alias_map.get(assign["alias"].upper())
                if resolved:
                    tables_for_assign = [resolved]

            if not _collision_side_ok(tables_for_assign):
                continue
            if not _targets_entity(tables_for_assign, target_entity_norm, entity_map, lineage):
                # Also accept writes on local temps whose primary root is the target.
                if not _targets_via_lineage(tables_for_assign, target_entity_norm, lineage):
                    continue
            if _skip_unwritten_session_temp(tables_for_assign):
                continue

            ordinal += 1
            expr_text = assign["expr"]
            where_text = where_clause
            if len(set(map(str, alias_map.values()))) > 1:
                written_cols = _augment_written_cols_with_join_temp_schemas(
                    _written_columns_by_table(source, lineage, entity_map),
                    alias_map,
                    lineage,
                    _dim_read_columns_by_table(source, lineage, entity_map),
                )
                expr_text = _qualify_foreign_bare_columns(
                    expr_text, alias_map, written_tables, written_cols
                )
                where_text = _qualify_foreign_bare_columns(
                    where_text, alias_map, written_tables, written_cols
                )
            resolved_expr = _resolve_expression_tables(
                expr_text,
                alias_map,
                lineage,
                entity_map,
                target_entity_norm,
                use_derived_formula=False,
            )
            resolved_where = None
            if where_clause:
                resolved_where = _resolve_expression_tables(
                    where_text,
                    alias_map,
                    lineage,
                    entity_map,
                    target_entity_norm,
                    use_derived_formula=False,
                )

            dep_refs: list[str] = []
            if branch and branch.condition:
                dep_refs.extend(extract_subquery_dependency_refs(branch.condition))
            outer_cond, gate = _outer_for_branch(
                branch,
                lambda text: _resolve_expression_tables(
                    text,
                    alias_map,
                    lineage,
                    entity_map,
                    target_entity_norm,
                    use_derived_formula=False,
                ),
            )
            if where_clause:
                dep_refs.extend(extract_subquery_dependency_refs(where_clause))
            # UPDATE … FROM … JOIN: the ON keys decide which rows are updated.
            dep_refs.extend(_join_dependency_refs(joins, alias_map))

            # Inside an IF/ELSE chain, even an UPDATE without WHERE is "guarded"
            # by mutual exclusion — do not treat ELSE as a global unguarded reset.
            in_control = branch is not None
            join_filter_condition = _resolve_join_filter_condition(
                joins, alias_map, lineage, entity_map, target_entity_norm
            )
            guarded = bool(where_clause) or bool(join_filter_condition) or (
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
                    join_filter_condition=join_filter_condition,
                    dependency_refs=_dedupe_refs(dep_refs),
                    exception_scope=_scope_for_offset(stmt_start),
                    workflow_gate=gate.name if gate else None,
                    workflow_gate_condition=gate.condition if gate else None,
                )
            )

    # INSERT … SELECT writes (permanent + temp) — same chronological fold.
    update_stmt_count = len(source.updates)
    for ins_index, ins in enumerate(source.inserts):
        target_table = normalize_table_name(ins.get("target") or "")
        if not _collision_side_ok([target_table]):
            continue
        if not _targets_entity([target_table], target_entity_norm, entity_map, lineage):
            if not _targets_via_lineage([target_table], target_entity_norm, lineage):
                continue
        if _skip_unwritten_session_temp([target_table]):
            continue

        col_names = [bare_ident(c) for c in (ins.get("cols") or "").split(",") if bare_ident(c)]
        projections = parse_select_list(ins.get("select_list") or "")
        if not col_names:
            # No explicit column list — fall back to the target table's own
            # schema ordinal position when it's a local temp whose CREATE
            # TABLE column order Phase 1 already captured. Without any known
            # schema (e.g. an untracked permanent table) we still cannot map
            # projection -> target column and must skip.
            schema_cols = lineage.temp_table_columns.get(target_table)
            if schema_cols and len(schema_cols) == len(projections):
                col_names = schema_cols
            else:
                continue
        if len(col_names) != len(projections):
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
        expr = _strip_trailing_select_alias(projections[col_idx][3].strip())
        expr = _attach_groupby_to_bare_aggregate(expr, ins.get("group_by") or "")

        # An unqualified projection column (``INSERT … SELECT UCIF_ID FROM ##AccountCal``)
        # means "the single source table's column", exactly as in SELECT INTO below —
        # otherwise it resolves to the INSERT target itself (a self-reference).
        primary_source = _pick_primary_source_table(
            alias_map, exclude_entities=frozenset({target_table, target_entity_norm})
        )
        resolved_expr = _resolve_expression_tables(
            expr, alias_map, lineage, entity_map, target_entity_norm,
            default_source_table=primary_source,
        )
        resolved_where = None
        if where_clause:
            resolved_where = _resolve_expression_tables(
                where_clause, alias_map, lineage, entity_map, target_entity_norm
            )

        stmt_start = int(ins.get("start") or 0)
        branch = _branch_for_offset(stmt_start)
        dep_refs: list[str] = []
        if branch and branch.condition:
            dep_refs.extend(extract_subquery_dependency_refs(branch.condition))
        outer_cond, gate = _outer_for_branch(
            branch,
            lambda text: _resolve_expression_tables(
                text, alias_map, lineage, entity_map, target_entity_norm
            ),
        )
        if where_clause:
            dep_refs.extend(extract_subquery_dependency_refs(where_clause))
        dep_refs.extend(_join_dependency_refs(joins, alias_map))
        in_control = branch is not None
        join_filter_condition = _resolve_join_filter_condition(
            joins, alias_map, lineage, entity_map, target_entity_norm
        )
        guarded = bool(where_clause) or bool(join_filter_condition) or (
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
                join_filter_condition=join_filter_condition,
                dependency_refs=_dedupe_refs(dep_refs),
                exception_scope=_scope_for_offset(stmt_start),
                workflow_gate=gate.name if gate else None,
                workflow_gate_condition=gate.condition if gate else None,
            )
        )

    # SELECT ... INTO #temp FROM ... [WHERE ...] — same chronological fold as
    # INSERT ... SELECT, but the destination column list comes from the
    # projection itself (aliased name, or the bare source column name when
    # unaliased) instead of an explicit ``INSERT INTO target (cols)`` list.
    insert_select_count = len(source.inserts)
    for si_index, si in enumerate(source.selects):
        target_table = normalize_table_name(si.get("target") or "")
        if not _collision_side_ok([target_table]):
            continue
        if not _targets_entity([target_table], target_entity_norm, entity_map, lineage):
            if not _targets_via_lineage([target_table], target_entity_norm, lineage):
                continue
        if _skip_unwritten_session_temp([target_table]):
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

        # ``SELECT SUM(ISNULL(TotalProvision,0)) TotalProvision INTO #TotalProvCust
        # … GROUP BY CustomerEntityId`` only shares the root's *lineage*; it is a
        # different-grain roll-up into a session temp, not a write to the root
        # entity's own column. Folding it in would append an unguarded pass after
        # every real UPDATE and reset the column to ``COALESCE(Col, 0)``.
        if (
            target_table.startswith("#")
            and not target_table.startswith("##")
            and not (_entity_keys(target_entity_norm) & _entity_keys(target_table))
            and (si.get("group_by") or "").strip()
            and re.search(
                r"(?is)\b(?:SUM|COUNT|AVG|MIN|MAX)\s*\(", projections[col_idx][3] or ""
            )
        ):
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
        dep_refs: list[str] = []
        if branch and branch.condition:
            dep_refs.extend(extract_subquery_dependency_refs(branch.condition))
        outer_cond, gate = _outer_for_branch(
            branch,
            lambda text: _resolve_expression_tables(
                text, alias_map, lineage, entity_map, target_entity_norm
            ),
        )
        if where_clause:
            dep_refs.extend(extract_subquery_dependency_refs(where_clause))
        dep_refs.extend(_join_dependency_refs(joins, alias_map))
        in_control = branch is not None
        join_filter_condition = _resolve_join_filter_condition(
            joins, alias_map, lineage, entity_map, target_entity_norm
        )
        guarded = bool(where_clause) or bool(join_filter_condition) or (
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
                join_filter_condition=join_filter_condition,
                dependency_refs=_dedupe_refs(dep_refs),
                exception_scope=_scope_for_offset(stmt_start),
                workflow_gate=gate.name if gate else None,
                workflow_gate_condition=gate.condition if gate else None,
            )
        )

    # MERGE … WHEN MATCHED THEN UPDATE SET … — chronological with UPDATEs/INSERTs.
    prior_stmt_count = update_stmt_count + insert_select_count + len(source.selects)
    for merge_index, merge in enumerate(source.merges):
        target_table = normalize_table_name(merge.get("target") or "")
        if not _collision_side_ok([target_table]):
            continue
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

        using_row_pred = extract_merge_using_row_predicate(using_body)
        resolved_using_pred = None
        if using_row_pred:
            dep_refs.extend(extract_subquery_dependency_refs(using_row_pred))
            resolved_using_pred = _resolve_expression_tables(
                using_row_pred,
                alias_map,
                lineage,
                entity_map,
                target_entity_norm,
                use_derived_formula=False,
            )
        join_filter = _resolve_join_filter_condition(
            joins, alias_map, lineage, entity_map, target_entity_norm
        )
        merge_row_guards: list[str] = []
        if resolved_using_pred:
            merge_row_guards.append(resolved_using_pred)
        if join_filter and join_filter.upper() not in " AND ".join(merge_row_guards).upper():
            merge_row_guards.append(join_filter)
        if not merge_row_guards and resolved_on:
            merge_row_guards.append(resolved_on)
        merge_where = (
            " AND ".join(f"({g})" for g in merge_row_guards) if merge_row_guards else None
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
                    where_clause=merge_where,
                    joins=joins,
                    alias_map=alias_map,
                    raw_sql=(merge.get("raw_sql") or "").strip(),
                    statement_index=prior_stmt_count + merge_index,
                    source_position=int(merge.get("start") or 0),
                    operation="MERGE",
                    guarded=bool(merge_where),
                    join_filter_condition=None,
                    dependency_refs=_dedupe_refs(dep_refs),
                    exception_scope=_scope_for_offset(int(merge.get("start") or 0)),
                )
            )

    logger.debug(
        "phase2 mutations for %s.%s: %d pass(es)",
        target_entity_norm,
        target_col,
        len(mutations),
    )
    mutations.sort(key=lambda m: (m.source_position, m.statement_index))
    mutations = _apply_table_resets(
        mutations, source.resets, target_entity_norm, entity_map
    )
    for ordinal, mutation in enumerate(mutations, 1):
        mutation.ordinal = ordinal
    return _prune_redundant_mutations(mutations)


def _apply_table_resets(
    mutations: list[MutationPass],
    resets: list[dict[str, Any]],
    target_entity: str,
    entity_map: dict[str, str] | None,
) -> list[MutationPass]:
    """Start a fresh derivation after the last reset of the target table.

    ``TRUNCATE TABLE #T`` / ``DELETE FROM #T`` / ``DROP TABLE #T`` discard every
    row, so writes before the reset cannot feed values after it. The column is
    derived from the writes after the last reset that is followed by a write;
    a trailing cleanup reset (after the final write) is ignored. Matching is by
    the table's own name — never via Phase-1 lineage, which maps a temp table
    to its root and would make ``TRUNCATE #Staging`` look like a reset of the
    physical source table. CATCH-handler writes are left untouched.
    """
    target_keys = _entity_keys(target_entity)
    own = [
        r for r in resets
        if target_keys & (_entity_keys(r["table"]) | _entity_keys(resolve_entity_name(r["table"], entity_map)))
    ]
    main = [m for m in mutations if not m.is_exception_handler]
    if not own or not main:
        return mutations
    last_write = max(m.source_position for m in main)
    effective = [r for r in own if r["start"] < last_write]
    if not effective:
        return mutations
    reset = max(effective, key=lambda r: r["start"])
    dropped = [m for m in main if m.source_position < reset["start"]]
    if not dropped:
        return mutations
    kept = [m for m in mutations if m.is_exception_handler or m.source_position > reset["start"]]
    first = next(m for m in kept if not m.is_exception_handler)
    first.state_reset = (
        f"{reset['kind']} {reset['table']} cleared the table before this write; "
        f"{len(dropped)} earlier write(s) to this column do not carry over"
    )
    first.state_reset_position = reset["start"]
    logger.debug("phase2 reset for %s: dropped %d pre-reset write(s)", target_entity, len(dropped))
    return kept


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


def prune_redundant_ast(node: dict[str, Any] | None, _memo=None) -> dict[str, Any] | None:
    """Remove dead IF arms that cannot change the column.

    - ``IF(A) THEN(IF(A) THEN(x) ELSE(y))`` collapses to ``IF(A) THEN(x)`` —
      the inner ELSE is unreachable once the outer guard already holds (an
      ``IF EXISTS(... WHERE A)`` block wrapping ``UPDATE ... WHERE A``).
    - Inside ``IF(ISNOTEMPTY(col))``, an immediate ``IF(ISEMPTY(col))`` in
      THEN is unreachable.
    - Identical nested ELSEIF arms are collapsed.
    - ``ELSE(col)`` under ``IF(ISNOTEMPTY(col))`` is a self-assignment on the
      null path and is dropped.
    """
    if _memo is None:
        from app.derivation.v2.ast_limits import check_formula_expansion

        _memo = {}
        pruned = _prune_redundant_ast_impl(node, _memo)
        if isinstance(pruned, dict):
            from app.derivation.v2.ast_optimize import (
                collapse_degenerate_if_branches,
                enforce_later_update_precedence,
            )

            pruned = enforce_later_update_precedence(pruned)
            pruned = collapse_degenerate_if_branches(pruned)
            check_formula_expansion(pruned)
        return pruned
    return _prune_redundant_ast_impl(node, _memo)


def _prune_redundant_ast_impl(node: dict[str, Any] | None, _memo) -> dict[str, Any] | None:
    if id(node) in _memo:
        return _memo[id(node)]
    if not isinstance(node, dict):
        return node
    cleaned = dict(node)
    _memo[id(node)] = cleaned
    for key, value in list(cleaned.items()):
        if isinstance(value, dict) and "type" in value:
            cleaned[key] = _prune_redundant_ast_impl(value, _memo)
        elif isinstance(value, list):
            cleaned[key] = [
                _prune_redundant_ast_impl(item, _memo) if isinstance(item, dict) else item
                for item in value
            ]
    if cleaned.get("type") != "IF_THEN_ELSE":
        return cleaned

    # Peel repeated identical guards on the THEN side. When the inner arm is
    # ``IF(G) THEN NULL ELSE <real>``, promote <real> (same rule as ast_optimize)
    # so a widening NULL pass does not shadow ADDDAY/CASE logic under ``G``.
    from app.derivation.v2.ast_optimize import _is_null_literal_ast

    then_branch = cleaned.get("then_branch")
    condition_sig = guard_formula_signature(cleaned.get("condition"))
    while (
        isinstance(then_branch, dict)
        and then_branch.get("type") == "IF_THEN_ELSE"
        and guard_formula_signature(then_branch.get("condition")) == condition_sig
    ):
        inner_then = then_branch.get("then_branch")
        inner_else = then_branch.get("else_branch")
        if _is_null_literal_ast(inner_then) and isinstance(inner_else, dict):
            then_branch = inner_else
            cleaned["then_branch"] = then_branch
            continue
        if not isinstance(inner_then, dict):
            break
        then_branch = inner_then
        cleaned["then_branch"] = then_branch

    outer = _predicate_column(cleaned.get("condition"), "ISNOTEMPTY")
    if outer and isinstance(then_branch, dict) and then_branch.get("type") == "IF_THEN_ELSE":
        inner = _predicate_column(then_branch.get("condition"), "ISEMPTY")
        if inner and inner == outer:
            replacement = then_branch.get("else_branch")
            cleaned["then_branch"] = (
                _prune_redundant_ast_impl(replacement, _memo)
                if isinstance(replacement, dict)
                else replacement
            )

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

    flattened = _try_flatten_join_assignment_zero_default(cleaned)
    if flattened is not None:
        return _prune_redundant_ast_impl(flattened, _memo)
    deduped_clamp = _try_dedupe_coalesce_negative_clamp(cleaned)
    if deduped_clamp is not None:
        return _prune_redundant_ast_impl(deduped_clamp, _memo)
    cond = cleaned.get("condition")
    if isinstance(cond, dict):
        deduped_cond = _dedupe_or_condition_ast(cond)
        if deduped_cond is not None:
            cleaned["condition"] = deduped_cond
    return cleaned


_IDENTIFIER_KEYS = {"entity", "relationship", "column", "function_name", "name", "operator"}


def _flatten_or_conditions(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Left-to-right leaves of an ``OR`` tree."""
    if node.get("type") != "BINARY_OP" or str(node.get("operator") or "").upper() != "OR":
        return [node]
    left = node.get("left")
    right = node.get("right")
    parts: list[dict[str, Any]] = []
    if isinstance(left, dict):
        parts.extend(_flatten_or_conditions(left))
    if isinstance(right, dict):
        parts.extend(_flatten_or_conditions(right))
    return parts or [node]


def _rebuild_or_chain(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        return {"type": "LITERAL", "value_type": "NULL", "value": None}
    out = parts[0]
    for part in parts[1:]:
        out = {"type": "BINARY_OP", "operator": "OR", "left": out, "right": part}
    return out


def _dedupe_or_condition_ast(node: dict[str, Any]) -> dict[str, Any] | None:
    """Drop duplicate disjuncts from a flat ``OR`` guard (same Class-A arm)."""
    if node.get("type") != "BINARY_OP" or str(node.get("operator") or "").upper() != "OR":
        return None
    parts = _flatten_or_conditions(node)
    if len(parts) < 2:
        return None
    seen: set[Any] = set()
    unique: list[dict[str, Any]] = []
    for part in parts:
        sig = _ast_signature(part)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(part)
    if len(unique) == len(parts):
        return None
    return _rebuild_or_chain(unique)


def _is_zero_literal(node: Any) -> bool:
    if not isinstance(node, dict) or node.get("type") != "LITERAL":
        return False
    if str(node.get("value_type") or "").upper() == "NULL":
        return False
    value = node.get("value")
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _and_ast(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"type": "BINARY_OP", "operator": "AND", "left": left, "right": right}


def _condition_is_expr_equals_zero(condition: Any, expr: dict[str, Any]) -> bool:
    """True when ``condition`` is ``(expr == 0)`` (or reversed operands)."""
    if not isinstance(condition, dict) or condition.get("type") != "BINARY_OP":
        return False
    op = str(condition.get("operator") or "").strip()
    if op not in {"==", "="}:
        return False
    left = condition.get("left")
    right = condition.get("right")
    expr_sig = _ast_signature(expr)
    if _is_zero_literal(right) and _ast_signature(left) == expr_sig:
        return True
    if _is_zero_literal(left) and _ast_signature(right) == expr_sig:
        return True
    return False


def _try_reconstruct_zero_comparison_subject(
    node: Any,
    ops: set[str],
) -> dict[str, Any] | None:
    """Undo ``_distribute_if_over_comparisons`` on ``expr < 0`` / ``expr <= 0``.

    ``IF(c) THEN (a<0) ELSE (b<0)`` reconstructs as ``IF(c) THEN a ELSE b``.
    """
    if not isinstance(node, dict):
        return None
    if node.get("type") == "BINARY_OP":
        if str(node.get("operator") or "").strip() not in ops:
            return None
        if not _is_zero_literal(node.get("right")):
            return None
        left = node.get("left")
        return left if isinstance(left, dict) else None
    if node.get("type") != "IF_THEN_ELSE":
        return None
    then_v = _try_reconstruct_zero_comparison_subject(node.get("then_branch"), ops)
    else_v = _try_reconstruct_zero_comparison_subject(node.get("else_branch"), ops)
    if then_v is None or else_v is None:
        return None
    cond = node.get("condition")
    if not isinstance(cond, dict):
        return None
    return {
        "type": "IF_THEN_ELSE",
        "condition": cond,
        "then_branch": then_v,
        "else_branch": else_v,
    }


def _try_dedupe_coalesce_negative_clamp(node: dict[str, Any]) -> dict[str, Any] | None:
    """Collapse ``IF(COALESCE(deriv,0)<0) THEN 0 ELSE deriv`` (or the same with
    ``deriv`` inlined in the guard) into ``MAX(deriv, 0)``.

    Prior-value substitution copies the full derivation into the clamp guard
    while the ELSE arm still carries the same tree — the export then lists every
    ``@TIMEKEY`` / ``DATEDIFF`` arm twice. When guard and ELSE payloads match,
    ``MAX`` is exact for the integer DPD metrics this pattern serves.

    Also matches the 4X-safe distributed form produced by
    ``_distribute_if_over_comparisons``:

        IF(IF(c) THEN(a<0) ELSE(b<0)) THEN 0 ELSE IF(c) THEN a ELSE b
    """
    if node.get("type") != "IF_THEN_ELSE" or not _is_zero_literal(node.get("then_branch")):
        return None
    else_branch = node.get("else_branch")
    if not isinstance(else_branch, dict):
        return None
    cond = node.get("condition")
    if not isinstance(cond, dict):
        return None
    if cond.get("_keep_inline"):
        return None  # explicit ``ELSEIF(total < 0) THEN 0`` floor — keep as written
    inner: dict[str, Any] | None = None
    if cond.get("type") == "BINARY_OP":
        if str(cond.get("operator") or "").strip() not in {"<", "<="}:
            return None
        if not _is_zero_literal(cond.get("right")):
            return None
        left = cond.get("left")
        if not isinstance(left, dict):
            return None
        if left.get("type") == "FUNCTION_CALL" and str(left.get("function_name") or "").upper() == "COALESCE":
            args = left.get("arguments") or []
            if len(args) == 2 and _is_zero_literal(args[1]):
                inner = args[0] if isinstance(args[0], dict) else None
        elif _ast_signature(left) == _ast_signature(else_branch):
            inner = left
    if inner is None:
        reconstructed = _try_reconstruct_zero_comparison_subject(cond, {"<", "<="})
        if reconstructed is not None and _ast_signature(reconstructed) == _ast_signature(else_branch):
            inner = reconstructed
    if inner is None or _ast_signature(inner) != _ast_signature(else_branch):
        return None
    zero_lit = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    return {
        "type": "FUNCTION_CALL",
        "function_name": "MAX",
        "arguments": [else_branch, zero_lit],
    }


def _try_flatten_join_assignment_zero_default(node: dict[str, Any]) -> dict[str, Any] | None:
    """Collapse ``IF(prior==0) THEN default ELSE IF(join) THEN src ELSE 0``.

    Chronological folding for ``JOIN … SET col=src`` followed by
    ``SET col=default WHERE col=0`` substitutes the nested IF into the
    zero-check, producing a duplicated join guard. Semantically:

        IF(join_active AND COALESCE(src,0) != 0) THEN src ELSE default

    is equivalent when an earlier pass reset the column to 0 and ``src`` is
    only written under the join.

    Also matches the 4X-safe distributed form produced by
    ``_distribute_if_over_comparisons``:

        IF(IF(join) THEN(src==0) ELSE(0==0)) THEN default
        ELSE IF(join) THEN src ELSE 0
    """
    if node.get("type") != "IF_THEN_ELSE":
        return None
    outer_else = node.get("else_branch")
    if not isinstance(outer_else, dict) or outer_else.get("type") != "IF_THEN_ELSE":
        return None
    join_cond = outer_else.get("condition")
    source_value = outer_else.get("then_branch")
    if not isinstance(join_cond, dict) or not isinstance(source_value, dict):
        return None
    if not _is_zero_literal(outer_else.get("else_branch")):
        return None
    default_value = node.get("then_branch")
    if not isinstance(default_value, dict) or default_value.get("type") != "LITERAL":
        return None

    cond = node.get("condition")
    direct = _condition_is_expr_equals_zero(cond, outer_else)
    distributed = _condition_is_distributed_join_zero_check(cond, join_cond, source_value)
    if not direct and not distributed:
        return None

    coalesce_source = {
        "type": "FUNCTION_CALL",
        "function_name": "COALESCE",
        "arguments": [
            source_value,
            {"type": "LITERAL", "value_type": "NUMBER", "value": 0},
        ],
    }
    nonzero = {
        "type": "BINARY_OP",
        "operator": "!=",
        "left": coalesce_source,
        "right": {"type": "LITERAL", "value_type": "NUMBER", "value": 0},
    }
    return {
        "type": "IF_THEN_ELSE",
        "condition": _and_ast(join_cond, nonzero),
        "then_branch": source_value,
        "else_branch": default_value,
    }


def _is_zero_equals_zero(node: Any) -> bool:
    if not isinstance(node, dict) or node.get("type") != "BINARY_OP":
        return False
    if str(node.get("operator") or "").strip() not in {"==", "="}:
        return False
    return _is_zero_literal(node.get("left")) and _is_zero_literal(node.get("right"))


def _condition_is_distributed_join_zero_check(
    condition: Any,
    join_cond: dict[str, Any],
    source_value: dict[str, Any],
) -> bool:
    """True for ``IF(join) THEN(src==0) ELSE(0==0)`` (distributed ``prior==0``)."""
    if not isinstance(condition, dict) or condition.get("type") != "IF_THEN_ELSE":
        return False
    if _ast_signature(condition.get("condition")) != _ast_signature(join_cond):
        return False
    if not _condition_is_expr_equals_zero(condition.get("then_branch"), source_value):
        return False
    return _is_zero_equals_zero(condition.get("else_branch"))


def _ast_signature(node: Any) -> Any:
    """Hashable, comparison-only form of an AST subtree.

    Drops ``_``-prefixed metadata (e.g. ``_dependency_refs`` attached when an
    ``EXISTS(...)`` guard is projected to a row predicate, which the UPDATE's
    own WHERE clause never carries) and case-folds identifiers, since SQL
    names are case-insensitive. Literal values keep their case.
    """
    if isinstance(node, dict):
        items = []
        for key, value in node.items():
            if str(key).startswith("_"):
                continue
            if key in _IDENTIFIER_KEYS and isinstance(value, str):
                value = value.upper()
            items.append((key, _ast_signature(value)))
        # Keys are unique within one dict, so order by key only; comparing the
        # signature values would raise on ``None`` vs ``str`` literal leaves.
        return tuple(sorted(items, key=lambda kv: str(kv[0])))
    if isinstance(node, list):
        return tuple(_ast_signature(item) for item in node)
    return node


def _literals_equal_for_guard(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.get("type") != "LITERAL" or b.get("type") != "LITERAL":
        return False
    return a.get("value") == b.get("value") and (
        str(a.get("value_type") or "").upper() == str(b.get("value_type") or "").upper()
    )


def normalize_guard_conjunct(node: Any) -> Any:
    """Canonical form for duplicate-guard detection and narrowing-fold matching.

    Maps ``COALESCE(Col, 'N') == 'N'`` and bare ``Col == 'N'`` to the same
    shape so chronological folds do not treat procedurally identical UPDATE
    guards as distinct conjuncts.
    """
    if not isinstance(node, dict):
        return node
    ntype = node.get("type")
    if ntype == "BINARY_OP" and str(node.get("operator") or "").upper() == "AND":
        left = normalize_guard_conjunct(node.get("left"))
        right = normalize_guard_conjunct(node.get("right"))
        return {"type": "BINARY_OP", "operator": "AND", "left": left, "right": right}
    if ntype == "BINARY_OP" and str(node.get("operator") or "").upper() == "OR":
        left = normalize_guard_conjunct(node.get("left"))
        right = normalize_guard_conjunct(node.get("right"))
        return {"type": "BINARY_OP", "operator": "OR", "left": left, "right": right}
    if ntype == "BINARY_OP" and str(node.get("operator") or "") == "==":
        left = node.get("left")
        right = node.get("right")
        if isinstance(left, dict) and left.get("type") == "FUNCTION_CALL":
            fn = str(left.get("function_name") or "").upper()
            if fn == "COALESCE":
                args = left.get("arguments") or []
                if len(args) >= 2 and isinstance(right, dict) and _literals_equal_for_guard(
                    args[1], right
                ):
                    return {
                        "type": "BINARY_OP",
                        "operator": "==",
                        "left": normalize_guard_conjunct(args[0]),
                        "right": right,
                    }
        return {
            "type": "BINARY_OP",
            "operator": "==",
            "left": normalize_guard_conjunct(left),
            "right": normalize_guard_conjunct(right),
        }
    if ntype == "FUNCTION_CALL":
        return {
            **node,
            "arguments": [
                normalize_guard_conjunct(a) if isinstance(a, dict) else a
                for a in (node.get("arguments") or [])
            ],
        }
    return node


def guard_conjunct_signature(node: Any) -> Any:
    """Signature of a guard conjunct after ``normalize_guard_conjunct``."""
    if not isinstance(node, dict):
        return _ast_signature(node)
    return _ast_signature(normalize_guard_conjunct(node))


def flatten_and_conjuncts(node: dict[str, Any]) -> list[dict[str, Any]]:
    if node.get("type") != "BINARY_OP" or str(node.get("operator") or "").upper() != "AND":
        return [node]
    parts: list[dict[str, Any]] = []
    left = node.get("left")
    right = node.get("right")
    if isinstance(left, dict):
        parts.extend(flatten_and_conjuncts(left))
    if isinstance(right, dict):
        parts.extend(flatten_and_conjuncts(right))
    return parts or [node]


def guard_formula_signature(node: Any) -> Any:
    """Order-independent signature for a full row guard (AND of conjuncts)."""
    if not isinstance(node, dict):
        return _ast_signature(node)
    normed = normalize_guard_conjunct(node)
    parts = flatten_and_conjuncts(normed)
    # ``repr`` gives a total, deterministic order; the signatures themselves can hold
    # ``None`` and ``str`` leaves at the same position, which Python cannot compare.
    return tuple(sorted((guard_conjunct_signature(p) for p in parts), key=repr))


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


class DerivedTable(str):
    """Alias-map value for a ``( SELECT ... ) alias`` derived table.

    Behaves as the subquery's primary source table name (so existing
    ``alias_map`` consumers keep working) and additionally carries the
    projected expressions, so ``alias.col`` can be rewritten to the
    underlying expression (aggregates such as ``MIN(...)`` included) instead
    of leaking an unresolved short alias like ``"C"``.
    """

    projections: dict[str, str]
    inner_alias_map: dict[str, str]

    def __new__(cls, primary: str, projections: dict[str, str], inner_alias_map: dict[str, str]):
        obj = super().__new__(cls, primary)
        obj.projections = projections
        obj.inner_alias_map = inner_alias_map
        return obj

    def __getnewargs__(self):  # keeps copy/deepcopy/pickle working
        return (str(self), self.projections, self.inner_alias_map)


_DERIVED_OPEN_RE = re.compile(r"(?is)\b(?:FROM|JOIN)\s*\(")


def _mask_derived_tables(from_clause: str) -> tuple[str, dict[str, str]]:
    """Replace ``( SELECT ... )`` table sources with ``__DERIVED_n__`` tokens."""
    text = from_clause or ""
    masked: dict[str, str] = {}
    out: list[str] = []
    pos = 0
    while True:
        m = _DERIVED_OPEN_RE.search(text, pos)
        if not m:
            break
        open_idx = m.end() - 1
        depth = 0
        in_single = False
        close_idx = -1
        for i in range(open_idx, len(text)):
            ch = text[i]
            if in_single:
                if ch == "'":
                    in_single = False
                continue
            if ch == "'":
                in_single = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    close_idx = i
                    break
        inner = text[open_idx + 1 : close_idx].strip() if close_idx > 0 else ""
        if close_idx < 0 or not re.match(r"(?is)^SELECT\b", inner):
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        token = f"__DERIVED_{len(masked)}__"
        masked[token] = inner
        out.append(text[pos : open_idx])
        # Pad the token: ``) c`` may reach us as ``)c`` once comments/whitespace
        # are stripped, and the alias must stay a separate token.
        out.append(f" {token} ")
        pos = close_idx + 1
    out.append(text[pos:])
    return "".join(out), masked


def _build_derived_table(
    inner_select: str,
    lineage: LineageMap,
    entity_map: dict[str, str] | None,
    fallback: str,
) -> DerivedTable:
    from app.derivation.v2.sql_text import split_select_from

    select_list, from_body = split_select_from(inner_select)
    inner_alias_map, _ = _parse_update_sources("", from_body, lineage, entity_map)
    primary = next(iter(dict.fromkeys(inner_alias_map.values())), fallback)

    group_by = ""
    gm = re.search(
        r"(?is)\bGROUP\s+BY\s+(?P<g>.+?)(?:\bHAVING\b|\bORDER\s+BY\b|$)", inner_select
    )
    if gm:
        group_by = gm.group("g").strip()

    projections: dict[str, str] = {}
    for _qual, src_col, dest_alias, raw in parse_select_list(select_list):
        expr = _strip_trailing_select_alias(raw.strip())
        name = dest_alias or src_col
        if not name:
            tail = re.match(r"(?is)^(?P<e>.+?)\s+AS\s+\[?(?P<n>[A-Za-z_]\w*)\]?\s*$", raw.strip())
            if not tail:
                continue
            name, expr = tail.group("n"), tail.group("e").strip()
        agg = re.match(r"(?is)^STRING_AGG\s*\((?P<args>.*)\)$", expr.strip())
        if agg:
            # Concatenating aggregation has no row-level 4X form; the row's
            # value is the aggregated source column itself.
            args = split_csv_respecting_parens(agg.group("args"))
            expr = args[0].strip() if args else expr
        expr = _attach_groupby_to_bare_aggregate(expr, group_by)
        projections[bare_ident(name).upper()] = expr
    return DerivedTable(primary, projections, inner_alias_map)


def _parse_update_sources(
    head: str,
    from_clause: str,
    lineage: LineageMap,
    entity_map: dict[str, str] | None,
) -> tuple[dict[str, str], list[JoinInfo]]:
    alias_map: dict[str, str] = {}
    joins: list[JoinInfo] = []
    from_clause, derived_sources = _mask_derived_tables(from_clause or "")

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

    for table, alias, on_clause, join_type in parse_from_join_clause_with_type(from_clause or ""):
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
                join_type=join_type,
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

    if derived_sources:
        built = {
            token: _build_derived_table(sql, lineage, entity_map, token)
            for token, sql in derived_sources.items()
        }
        for key, value in list(alias_map.items()):
            if value in built:
                alias_map[key] = built[value]
        for token in built:
            alias_map.pop(token.upper(), None)
        for join in joins:
            if join.table in built:
                join.table = str(built[join.table])
                join.resolved_entity = _resolve_table_entity(join.table, lineage, entity_map)

    return alias_map, joins


_QUALIFIED_REF_RE = re.compile(r"(?<![\w@#.])([#A-Za-z_][\w]*)\s*\.\s*\[?([A-Za-z_][\w]*)\]?")


def _dealias(text: str, alias_map: dict[str, str]) -> tuple[str, list[str]]:
    """Rewrite ``alias.col`` to ``Table.col``; return the text and its refs."""
    refs: list[str] = []

    def repl(match: re.Match[str]) -> str:
        table = alias_map.get(match.group(1).upper())
        if not table:
            return match.group(0)
        name = normalize_table_name(table)
        refs.append(f"{name}.{match.group(2)}")
        return f"{name}.{match.group(2)}"

    return _QUALIFIED_REF_RE.sub(repl, text or ""), refs


def join_context(joins: list[JoinInfo], alias_map: dict[str, str]) -> list[str]:
    """Readable ``JOIN Table ON a = b`` lines for joins that carry an ON clause.

    An INNER JOIN restricts which target rows an UPDATE touches; the formula
    reaches joined columns through a relationship path, so the join keys are
    kept here (and in dependency refs) rather than dropped.
    """
    lines: list[str] = []
    for join in joins:
        if not join.on_clause:
            continue
        on_text, _ = _dealias(" ".join(join.on_clause.split()), alias_map)
        lines.append(f"JOIN {normalize_table_name(join.table)} ON {on_text}")
    return lines


def _join_dependency_refs(joins: list[JoinInfo], alias_map: dict[str, str]) -> list[str]:
    refs: list[str] = []
    for join in joins:
        if join.on_clause:
            refs.extend(_dealias(join.on_clause, alias_map)[1])
    return refs


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
    keys = {text.upper(), bare.upper(), stripped.upper(), f"##{stripped}".upper()}
    # Permanent staging tables (e.g. PRO.AccountCal_Stg for ##AccountCal) fold
    # under the logical entity name (AccountCal) when no explicit entity map
    # is supplied.
    if stripped.upper().endswith("_STG") and len(stripped) > 4:
        keys |= _entity_keys(stripped[:-4])
    return keys


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
    *_DATEADD_UNIT_KEYWORDS,
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
            is_qualifier = j < n and text[j] in {".", ":"}
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


def _pick_primary_source_table(
    alias_map: dict[str, str],
    *,
    exclude_entities: frozenset[str] | None = None,
) -> str | None:
    """The one table an unqualified projection column implicitly refers to.

    For ``INSERT … SELECT``, bare columns come from the driving source
    tables in the FROM clause — not from the INSERT target (a global
    ``##`` temp) and not from an arbitrary joined ``##`` peer when a
    physical source table is present.
    """
    tables = list(dict.fromkeys(alias_map.values()))
    if not tables:
        return None
    exclude = {
        normalize_table_name(e).upper().lstrip("#")
        for e in (exclude_entities or ())
    }
    filtered: list[str] = []
    for t in tables:
        norm = normalize_table_name(str(t))
        bare = bare_ident(norm.lstrip("#")).upper()
        if bare in exclude or norm.upper() in exclude:
            continue
        filtered.append(t)
    if not filtered:
        filtered = tables
    if exclude_entities:
        for t in filtered:
            if not str(t).startswith("#"):
                return t
        for t in filtered:
            if str(t).startswith("##"):
                return t
        return filtered[0] if len(filtered) == 1 else None
    for t in filtered:
        if t.startswith("##"):
            return t
    for t in filtered:
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
        # Spaced / punctuated identifiers cannot travel as ``Entity::Col``
        # markers (the marker regex is ``[A-Za-z0-9_]+``). Keep T-SQL
        # ``Table.[Account No]`` so phase3's bracket parser can read them.
        # Only rewrite known aliases / temps from this statement.
        table = alias_map.get(qual.upper())
        if table is None:
            # Bare #temp.col without alias entry
            if qual.startswith("#"):
                table = normalize_table_name(qual)
            else:
                return match.group(0)
        if re.search(r"[^\w]", col) and not isinstance(table, DerivedTable):
            physical = normalize_table_name(str(table))
            return f"{physical}.[{col}]"
        table_norm = normalize_table_name(str(table))
        if table_norm.startswith("#") and not table_norm.startswith("##"):
            # Session ``#temp`` joins (Cohort_No_PERC_2, TEMPTABLE, …) are not
            # root entities — encode as a relationship hop on the UPDATE target
            # instead of resolving through lineage (which can mis-route
            # UCIF_ID to DerivativeDetail, or collapse CP.* onto AccountCal.*).
            return _session_temp_column_marker(
                table_norm, col, default_entity, entity_map, lineage.hash_collisions
            )

        if isinstance(table, DerivedTable):
            derived_norm = normalize_table_name(str(table))
            if derived_norm.startswith("#") and not derived_norm.startswith("##"):
                return _session_temp_column_marker(
                    derived_norm, col, default_entity, entity_map, lineage.hash_collisions
                )
            projection = table.projections.get(col.upper())
            if projection is not None:
                if re.search(r"(?is)\b(MIN|MAX|SUM|STRING_AGG)\s*\(", projection):
                    single_agg = re.match(
                        r"(?is)^\s*(?:MIN|MAX)\s*\(\s*(?P<inner>[^()]+)\)\s*$",
                        projection.strip(),
                    )
                    if single_agg:
                        # ``MIN(col, ["GroupCol"])`` (see _attach_groupby_to_bare_aggregate):
                        # only ``col`` is a column reference. The bracketed GROUP BY list
                        # is a list literal and must not be alias-resolved or
                        # bare-qualified (that produced ``X::col, X.["GroupCol"]``).
                        agg_args = re.match(
                            r"(?is)^(?P<col>[^,\[\]]+?)\s*(?:,\s*(?P<grp>\[.*\]))?$",
                            single_agg.group("inner").strip(),
                        )
                        agg_col = (
                            agg_args.group("col").strip()
                            if agg_args
                            else single_agg.group("inner").strip()
                        )
                        agg_group_list = (
                            (agg_args.group("grp") or "").strip() if agg_args else ""
                        )
                        inner = _resolve_expression_tables(
                            agg_col,
                            table.inner_alias_map,
                            lineage,
                            entity_map,
                            default_entity,
                            use_derived_formula=use_derived_formula,
                            default_source_table=_pick_primary_source_table(
                                table.inner_alias_map
                            ),
                        )
                        inner_up = inner.upper()
                        if (
                            "::" in inner
                            and "MIN(" not in inner_up
                            and "MAX(" not in inner_up
                        ):
                            func = single_agg.group(0).strip().split("(", 1)[0].upper()
                            ent_norm = normalize_table_name(default_entity).upper().lstrip("#")
                            if (
                                func == "MIN"
                                and ent_norm.endswith("CUSTOMERCAL")
                                and col.upper() == "FINALNPADT"
                            ):
                                if agg_group_list:
                                    return f"MIN({inner}, {agg_group_list})"
                                return f"MIN({inner})"
                            return inner
                    # Customer/account roll-ups: keep the derived-table hop instead
                    # of re-expanding MIN(CASE…) trees onto the wrong entity level.
                    ent = _normalize_entity(default_entity, entity_map)
                    return f'"{ent}"."{qual}"."{col}"'
                inner = _resolve_expression_tables(
                    projection,
                    table.inner_alias_map,
                    lineage,
                    entity_map,
                    default_entity,
                    use_derived_formula=use_derived_formula,
                    default_source_table=_pick_primary_source_table(table.inner_alias_map),
                )
                if re.fullmatch(r"[\w:.#\"]+|\w+\(.*\)", inner.strip(), flags=re.S):
                    return inner
                return f"({inner})"
        ref = lineage.resolve_column(table, col, entity_map)
        if use_derived_formula and getattr(ref, "derived_formula", None):
            table_norm = normalize_table_name(table)
            default_norm = normalize_table_name(default_entity)
            if table_norm.startswith("#") and default_norm.upper() != table_norm.upper().lstrip("#"):
                ent = _normalize_entity(default_entity, entity_map)
                hop = bare_ident(table_norm.lstrip("#"))
                return f'"{ent}"."{hop}"."{col}"'
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
                return _cross_entity_column_marker(default_entity, rel, ref.column)
        if relationship:
            ent = ref.entity
            rel = relationship
            if (
                default_entity
                and ent
                and ent.upper() == normalize_table_name(default_entity).upper()
                and not lineage_keeps_target_hop(ent, rel)
            ):
                return f"{normalize_table_name(rel)}::{ref.column}"
            return f"{ent}::{rel}::{ref.column}"
        root_entity = normalize_table_name(ref.entity or "")
        if default_entity and root_entity.upper() == normalize_table_name(default_entity).upper():
            physical = resolve_entity_name(ref.source_table or "", entity_map) or normalize_table_name(
                ref.source_table or ""
            )
            physical = normalize_table_name(physical)
            if (
                physical
                and physical.upper() != root_entity.upper()
                and not physical.startswith("#")
                and not lineage_keeps_target_hop(root_entity, physical)
            ):
                return f"{physical}::{ref.column}"
        return f"{ref.entity}::{ref.column}"

    pattern = re.compile(
        r"(?P<qual>[#A-Za-z_][A-Za-z0-9_]*)\.(?P<col>\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)"
    )
    # Never rewrite text inside single-quoted string literals: ``'D.FDSEC'``
    # is data, not an ``alias.column`` reference, and rewriting it would embed
    # an entity prefix in the literal (``"DimProduct.FDSEC"``).
    segments = re.split(r"('(?:[^']|'')*')", expression)
    rewritten = "".join(
        seg if seg.startswith("'") else pattern.sub(repl, seg) for seg in segments
    )
    return _qualify_unambiguous_bare_brackets(rewritten, alias_map, default_entity)


def _qualify_unambiguous_bare_brackets(
    expression: str,
    alias_map: dict[str, str],
    default_entity: str,
) -> str:
    """Qualify leftover ``[Account No]`` when exactly one non-target table is in scope.

    T-SQL binds a bare bracketed name in WHERE to the joined table that owns
    it (``Manual_Upgrade.[Account No]``). Without this, the token is parsed
    as a column of the UPDATE target. Identifiers inside ``IN (SELECT …)``
    subqueries are left untouched so the subquery still projects from its
    own FROM table.
    """
    target = normalize_table_name(default_entity).upper().lstrip("#")
    others: list[str] = []
    for table in dict.fromkeys(alias_map.values()):
        name = normalize_table_name(str(table))
        if name.upper().lstrip("#") == target:
            continue
        others.append(name)
    if len(others) != 1:
        return expression
    table = others[0]

    segments = re.split(r"('(?:[^']|'')*')", expression)
    return "".join(
        seg if seg.startswith("'") else _qualify_bare_brackets_outside_parens(seg, table)
        for seg in segments
    )


def _qualify_bare_brackets_outside_parens(text: str, table: str) -> str:
    out: list[str] = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
            out.append(ch)
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            out.append(ch)
            i += 1
            continue
        if (
            depth == 0
            and ch == "["
            and (i == 0 or text[i - 1] not in ".abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
        ):
            close = text.find("]", i)
            if close > i:
                out.append(f"{table}.[{text[i + 1:close]}]")
                i = close + 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _session_temp_column_marker(
    table_norm: str,
    column: str,
    default_entity: str,
    entity_map: dict[str, str] | None,
    hash_collisions: set[str] | None = None,
) -> str:
    """Encode ``#session_temp``.col without tracing lineage through its UNION body."""
    rel = bare_ident(normalize_table_name(table_norm).lstrip("#"))
    if hash_collisions and rel.upper() in hash_collisions:
        rel = "#" + rel  # keep the temp distinct from the same-named permanent table
    ent = _normalize_entity(default_entity, entity_map)
    if lineage_keeps_target_hop(ent, rel):
        return f'"{ent}"."{rel}"."{column}"'
    return _cross_entity_column_marker(ent, rel, column)


def _cross_entity_column_marker(default_entity: str, rel: str, column: str) -> str:
    """Encode a cross-table column read for phase3 (``Entity::Col`` markers)."""
    target = normalize_table_name(default_entity or "")
    rel_name = normalize_table_name(rel or "")
    if not lineage_keeps_target_hop(target, rel_name):
        return f"{rel_name}::{column}"
    return f"{target}::{rel_name}::{column}"


def _normalize_entity(entity: str, entity_map: dict[str, str] | None) -> str:
    return resolve_entity_name(entity, entity_map) or normalize_table_name(entity)
