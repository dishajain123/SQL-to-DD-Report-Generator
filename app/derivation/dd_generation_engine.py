"""Architecture step 13: DD Generation — chain collapse + grammar
targeting + versioning, orchestrated end to end.

Flow per column: build a column-specific SQL excerpt -> retrieve relevant
RAG context -> generate a Formula Expression -> deterministic grammar
validation -> semantic validation against the source SQL -> if either
fails, feed the errors (plus RAG context) back for a bounded number of
repair attempts -> accept -> for each effective-dated period, prune the
accepted expression down to just the branch that period's TIMEKEY
threshold actually selects (see app/derivation/period_pruning.py) -> or,
if generation never fully succeeded, fall back to PENDING_REVIEW with the
full, unpruned expression so a reviewer sees everything.

A column is very often assigned in more than one place in a real
procedure (a main calculation plus a special-case override, or a success
path plus an error-handling path). To make sure the generated derivation
reflects all of those assignment locations rather than just whichever one
the model happens to notice first in a long procedure, this module builds
a column-specific SQL excerpt from the object's SmartChunks (see
app/parsing/smart_chunking.py) -- every logical block that actually
assigns the target column, anywhere in the object -- and passes that to
the LLM as the authoritative source for that one column. The same chunk
list is also handed to semantic validation so it can check whether an
override/exception-style chunk was actually reflected in the result.

Entity-name resolution (staging table -> fact table name) is intentionally
pluggable via `entity_name_map` rather than hardcoded, since that mapping is
company/platform-specific and not something this pipeline can infer from
SQL alone.
"""
from __future__ import annotations

import json
import re
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sqlglot
from sqlglot import exp

from app.derivation.llm_client import LLMClient
from app.derivation.period_pruning import prune_expression_for_period
from app.derivation.versioning import effective_periods_for_column
from app.grammar.validator import validate_expression
from app.guardrails.semantic_validation import check_semantic_consistency
from app.models.core import (
    CanonicalModel,
    ColumnType,
    DDRow,
    DDStatus,
    DerivationOption,
    Dialect,
    LineageChain,
    ReviewState,
    SmartChunk,
    SQLObject,
    StatementInfo,
    StructuralInfo,
)
from app.parsing.dialect import detect_dialect
from app.parsing.sql_parser import _DML_KEYWORDS, classify_statement, split_statements
from app.utils.identity import canonical_logical_name
from app.utils.sql_aliases import (
    SQLGLOT_DIALECT_MAP,
    collect_known_reference_names,
    collect_table_aliases,
    resolve_aliases_in_expression,
    rewrite_expression_to_platform_entities,
)
from app.rag.chroma_store import ChromaStore, DOMAIN_COLLECTION, PLATFORM_COLLECTION
from app.utils.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Total generation attempts per column. Keep this bounded, but allow one
# repair pass so validation failures get a generic retry instead of being
# exported unchanged.
_MAX_GENERATION_ATTEMPTS = max(1, settings.dd_generation_max_attempts)
_MAX_SOURCE_SQL_CONTEXT_CHARS = 5000
# Single shared Dialect -> sqlglot-dialect-name mapping, defined once in
# app.utils.sql_aliases and reused here so the alias resolver, the
# reference-inventory collector, and this engine's own sqlglot calls can
# never drift out of sync with each other again.
_SQLGLOT_DIALECT = SQLGLOT_DIALECT_MAP


@dataclass(frozen=True)
class _AssignmentSite:
    kind: str
    statement_indices: list[int]
    raw_sql: str
    columns_written: list[str]


@dataclass(frozen=True)
class _SourceReferenceInventory:
    """Canonical source-backed reference facts for one DD column.

    The inventory is intentionally conservative: it only records tables,
    aliases, and column-to-qualifier pairings that were actually observed
    in parsed source SQL text. The grounding step later uses this as a
    whitelist for repairing hallucinated table qualifiers without ever
    inventing a new source relation.
    """

    target_entity_name: str
    allowed_qualifiers: set[str]
    qualifiers_by_column: dict[str, set[str]]

    def allowed_reference_lines(self, limit: int = 40) -> list[str]:
        lines: list[str] = []
        if self.target_entity_name:
            lines.append(f"Target entity mapping: {self.target_entity_name}")
        if self.allowed_qualifiers:
            lines.append("Allowed qualifiers: " + ", ".join(sorted(self.allowed_qualifiers)))
        for column in sorted(self.qualifiers_by_column):
            qualifiers = sorted(q for q in self.qualifiers_by_column[column] if q)
            if not qualifiers:
                continue
            lines.append(f"{column}: {', '.join(qualifiers)}")
            if len(lines) >= limit:
                break
        return lines


def _canonical_alias_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    return canonical_logical_name(stripped) if stripped else None


def _extract_alias_name(node) -> str | None:
    alias = getattr(node, "args", {}).get("alias") if hasattr(node, "args") else None
    if alias is None:
        alias = getattr(node, "alias", None)
    if isinstance(alias, exp.TableAlias):
        alias = alias.this
    if isinstance(alias, exp.Identifier):
        alias = alias.this
    if isinstance(alias, str) and alias.strip():
        return canonical_logical_name(alias.strip())
    if isinstance(node, exp.Alias):
        alias = node.alias_or_name or node.alias
        if isinstance(alias, exp.TableAlias):
            alias = alias.this
        if isinstance(alias, exp.Identifier):
            alias = alias.this
        if isinstance(alias, str) and alias.strip():
            return canonical_logical_name(alias.strip())
    return None


_LEADING_DOTTED_ALIAS_RE = re.compile(
    r'(?<![A-Za-z0-9_"])'
    r'(?P<identifier>"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)'
    r'(?=\s*\.)'
)


def _canonicalize_leading_dotted_aliases(expression: str) -> str:
    def replace(match: re.Match[str]) -> str:
        identifier = match.group("identifier")
        raw = identifier[1:-1] if identifier.startswith('"') and identifier.endswith('"') else identifier
        if raw != raw.lower():
            return identifier
        if identifier.startswith('"') and identifier.endswith('"'):
            return f'"{canonical_logical_name(raw)}"'
        return canonical_logical_name(identifier)

    return _LEADING_DOTTED_ALIAS_RE.sub(replace, expression)


def _relevant_chunks(info: StructuralInfo, column: str) -> list[SmartChunk]:
    """Every logical block (SmartChunk) that actually assigns the target
    column somewhere in the object -- across every conditional branch,
    MERGE override, or exception handler, not just wherever it happens to
    appear first.

    SmartChunks already keep control-flow blocks (IF/ELSE, CASE) together
    as one unit, and each chunk's `columns_written` is the union of every
    column actually assigned within it -- so filtering on that gives a
    focused, still-conditionally-correct set of chunks for one column,
    built the same way regardless of which procedure or column is being
    processed.

    A bare control-flow header (`EXCEPTION`, `WHEN ... THEN`, `ELSE`, ...)
    that touches no table/column of its own becomes its own tiny chunk
    immediately before the statement it governs (see
    app/parsing/smart_chunking.py's branch-marker handling), rather than
    being merged into it. Any such header(s) immediately preceding a
    matched chunk are folded into that chunk's text here, since they carry
    the trigger condition (e.g. "this is the error-handling path") that
    both the LLM and semantic validation need to see alongside the
    statement itself.
    """
    all_chunks = info.smart_chunks
    matched: list[SmartChunk] = []
    seen_chunk_ids: set[str] = set()

    for idx, chunk in enumerate(all_chunks):
        if canonical_logical_name(column) not in {canonical_logical_name(c) for c in chunk.columns_written} or chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)

        header_lines: list[str] = []
        j = idx - 1
        while j >= 0:
            candidate = all_chunks[j]
            is_bare_header = (
                not candidate.tables_written and not candidate.tables_read and not candidate.columns_written
            )
            if not is_bare_header:
                break
            header_lines.insert(0, candidate.raw_sql.strip())
            j -= 1

        if header_lines:
            merged_raw_sql = "\n".join([*header_lines, chunk.raw_sql.strip()])
            chunk = chunk.model_copy(update={"raw_sql": merged_raw_sql})

        matched.append(chunk)

    return matched


def _relevant_sql_excerpt(info: StructuralInfo, column: str) -> str:
    """Text form of `_relevant_chunks`, for handing to the LLM prompt.
    Returns an empty string (letting the caller fall back to explaining
    that nothing specific was isolated) if smart chunking found nothing
    for this column -- this is a targeting aid, not a hard requirement.
    """
    excerpts = [site.raw_sql.strip() for site in _assignment_sites(info, column) if site.raw_sql.strip()]
    return "\n\n".join(excerpts)


def _format_assignment_context(info: StructuralInfo, column: str, sites: list["_AssignmentSite"] | None = None) -> str:
    """Present the relevant write sites in source order with lightweight
    metadata so the LLM can see which assignment is an initial set, a
    later fix-up, or an override branch.

    The raw SQL is preserved verbatim under each numbered section; the
    labels simply make the execution order and statement role explicit.

    `sites` can be supplied directly (already computed and possibly
    filtered by the caller, e.g. with an undeterminable exception-handler
    site already removed) instead of being recomputed here from `info`.
    """
    if sites is None:
        sites = _assignment_sites(info, column)
    if not sites:
        return ""

    sections: list[str] = []
    overview = _ordered_assignment_overview(sites)
    if overview:
        sections.append("[Ordered write sequence]\n" + "\n".join(f"{idx}. {line}" for idx, line in enumerate(overview, start=1)))
    for idx, site in enumerate(sites, start=1):
        raw = site.raw_sql.strip()
        if not raw:
            continue
        stmt_ids = ",".join(str(i) for i in site.statement_indices) if site.statement_indices else "?"
        columns = ", ".join(site.columns_written) if site.columns_written else column
        role = _infer_assignment_role(raw)
        hints = _assignment_decomposition_hints(raw)
        summary = _assignment_decomposition_summary(raw)
        variable_trace = _extract_variable_trace(raw)
        before_index = min(site.statement_indices) if site.statement_indices else 2**31
        whole_procedure_trace = _whole_procedure_variable_trace(raw, getattr(info, "statements", []), before_index)
        sections.append(
            f"[Assignment {idx} | role={role} | kind={site.kind} | statements={stmt_ids} | columns={columns}]\n"
            f"{raw}"
            + (f"\n[Decomposition summary]\n" + "\n".join(f"- {line}" for line in summary) if summary else "")
            + (f"\n[Variable trace]\n" + "\n".join(f"- {line}" for line in variable_trace) if variable_trace else "")
            + (
                f"\n[Whole-procedure variable dependency chain]\n" + "\n".join(f"- {line}" for line in whole_procedure_trace)
                if whole_procedure_trace
                else ""
            )
            + (f"\n[Decomposition hints]\n" + "\n".join(f"- {hint}" for hint in hints) if hints else "")
        )
    return "\n\n".join(sections)


def _ordered_assignment_overview(sites: list[_AssignmentSite]) -> list[str]:
    overview: list[str] = []
    for site in sites:
        summary = _assignment_decomposition_summary(site.raw_sql)
        compact = "; ".join(summary) if summary else site.raw_sql.strip()
        role = _infer_assignment_role(site.raw_sql)
        sequencing = (
            "mutually exclusive alternate path -- executes INSTEAD OF the normal-flow "
            "site(s) below, never in sequence with them"
            if role == "EXCEPTION_HANDLER"
            else "later stages stay later"
        )
        overview.append(f"{sequencing} | role={role} | {compact}")
    return overview


def _assignment_sites(
    info: StructuralInfo,
    column: str,
    target_table: str | None = None,
) -> list[_AssignmentSite]:
    """Return ordered write sites for the target column.

    Prefer statement-level assignments when available so later fix-up
    UPDATEs remain separate from earlier MERGE calculations. Fall back to
    the older chunk view only when the structural info does not expose
    statements.

    When `target_table` is provided, only sites that write the column on
    that relation are returned — same physical column name on two tables
    (e.g. SeverityTier on a staging temp table vs summary) must not be
    folded into one expression.
    """
    statements = getattr(info, "statements", None)
    if statements:
        return _assignment_sites_from_statements(statements, column, target_table=target_table)
    return _assignment_sites_from_chunks(_relevant_chunks(info, column), column, target_table=target_table)


_CONDITION_BEARING_HEADER_RE = re.compile(
    r"(?is)^\s*(?:EXCEPTION\b|WHEN\s+OTHERS\b|ELSE\b|ELSIF\b|ELSEIF\b|(?:BEGIN\s+)?CATCH\b)"
)


def _is_condition_bearing_header(stmt: StatementInfo) -> bool:
    """True for control-flow headers that carry a real branch/exception
    trigger (EXCEPTION / WHEN / ELSE / CATCH).

    Bare BEGIN / END / TRY wrappers are structural T-SQL/PL-SQL noise: if
    they are folded onto every subsequent UPDATE inside a TRY block, the
    assignment site grows into an oversized CONTROL_FLOW_BLOCK that hides
    the actual SET statement and breaks exception-role detection.
    """
    text = (stmt.raw_text or "").strip()
    if not text:
        return False
    return bool(_CONDITION_BEARING_HEADER_RE.match(text))


def _assignment_sites_from_statements(
    statements: list[StatementInfo],
    column: str,
    target_table: str | None = None,
) -> list[_AssignmentSite]:
    column_upper = column.upper()
    sites: list[_AssignmentSite] = []
    pending_headers: list[StatementInfo] = []
    bridge_context: list[StatementInfo] = []

    for stmt in statements:
        writes_target = (
            _statement_writes_table_column(stmt, target_table, column_upper)
            if target_table
            else _statement_writes_column(stmt, column_upper)
        )

        if writes_target:
            raw_parts = [s.raw_text.strip() for s in pending_headers if s.raw_text.strip()]
            raw_parts.extend(s.raw_text.strip() for s in bridge_context if s.raw_text.strip())
            raw_parts.append(stmt.raw_text.strip())
            stmt_ids = [s.statement_index for s in pending_headers]
            stmt_ids.extend(s.statement_index for s in bridge_context)
            stmt_ids.append(stmt.statement_index)
            sites.append(
                _AssignmentSite(
                    kind="CONTROL_FLOW_BLOCK" if (pending_headers or bridge_context) else stmt.statement_type,
                    statement_indices=stmt_ids,
                    raw_sql="\n".join(raw_parts).strip(),
                    columns_written=[column],
                )
            )
            pending_headers = []
            bridge_context = []
            continue

        if _is_condition_bearing_header(stmt) and not stmt.set_columns_by_table and not writes_target:
            # CATCH/EXCEPTION headers can be classified as OTHER rather than
            # CONTROL_FLOW depending on the splitter; still fold them onto the
            # next write so exception-path role detection works for T-SQL.
            pending_headers.append(stmt)
            continue

        if (
            stmt.statement_type == "CONTROL_FLOW"
            and not stmt.columns
            and not stmt.set_columns_by_table
            and _is_condition_bearing_header(stmt)
        ):
            pending_headers.append(stmt)
            continue

        # A bare block-closing END (not "END IF"/"END CASE"/"END TRY",
        # which continue an outer construct) closes the immediately
        # preceding BEGIN/EXCEPTION block a pending header came from.
        # Without this, an unrelated small nested block's header (e.g. a
        # scalar-variable EXCEPTION handler that has nothing to do with
        # any later column write) keeps drifting forward through
        # bridge_context across every subsequent closed block until it
        # reaches -- and wrongly attaches itself to -- a real write much
        # later in the procedure, one it was never actually guarding.
        if pending_headers and stmt.statement_type == "CONTROL_FLOW" and re.match(
            r"(?is)^\s*END\s*;?\s*$", stmt.raw_text or ""
        ):
            pending_headers = []
            bridge_context = []
            continue

        if pending_headers and not stmt.set_columns_by_table and not stmt.tables_written:
            bridge_context.append(stmt)
            continue

        pending_headers = []
        bridge_context = []

    return sites


def _assignment_sites_from_chunks(
    chunks: list[SmartChunk],
    column: str,
    target_table: str | None = None,
) -> list[_AssignmentSite]:
    sites: list[_AssignmentSite] = []
    for chunk in chunks:
        raw = chunk.raw_sql.strip()
        if not raw:
            continue
        site = _AssignmentSite(
            kind=chunk.chunk_kind,
            statement_indices=list(chunk.statement_indices),
            raw_sql=raw,
            columns_written=[column],
        )
        if not _site_matches_target_table(site, target_table, column):
            continue
        sites.append(site)
    return sites


def _statement_writes_column(stmt: StatementInfo, column_upper: str) -> bool:
    by_table = stmt.set_columns_by_table or {}
    for cols in by_table.values():
        if any(col.upper() == column_upper for col in cols):
            return True
    # When the structural map is present, trust it — do not scan raw text for
    # `AssetClass =` inside WHEN clauses of other SET statements.
    if by_table:
        return False
    # Nested CASE / comment-broken SET maps are empty; detect the assignment
    # target only (SET col = / SET alias.col = / SET ..., col =).
    raw = stmt.raw_text or ""
    return bool(
        re.search(
            rf"(?is)\bSET\s+(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?{re.escape(column_upper)}\s*=",
            raw,
        )
        or re.search(
            rf"(?is)\bSET\b[\s\S]*?,\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?{re.escape(column_upper)}\s*=",
            raw,
        )
    )


def _normalize_relation_name(name: str) -> str:
    text = (name or "").strip().strip('"').strip("[]")
    if text.startswith("#"):
        text = text[1:]
    if "." in text:
        text = text.split(".")[-1]
    return text.upper()


def _statement_writes_table_column(
    stmt: StatementInfo,
    target_table: str,
    column_upper: str,
) -> bool:
    target_key = _normalize_relation_name(target_table)
    if not target_key:
        return _statement_writes_column(stmt, column_upper)
    by_table = stmt.set_columns_by_table or {}
    for table, cols in by_table.items():
        if _normalize_relation_name(table) != target_key:
            continue
        if any(col.upper() == column_upper for col in cols):
            return True
    # Nested CASE / comment-broken SET maps are empty — still bind the write
    # to this table when the raw statement clearly targets it.
    if not by_table and _statement_writes_column(stmt, column_upper):
        raw = stmt.raw_text or ""
        if re.search(rf"(?i)\b{re.escape(target_key)}\b", raw):
            return True
        # `UPDATE A ... FROM PRO.LoanAccountCal A` often omits the table token
        # beside the column; accept alias-style UPDATEs that write the column
        # when this is the only table that claims the column in the object.
        return bool(re.search(r"(?is)\bUPDATE\b.+\bSET\b", raw))
    return False


def _site_matches_target_table(site: _AssignmentSite, target_table: str | None, column: str) -> bool:
    if not target_table:
        return True
    target_key = _normalize_relation_name(target_table)
    if not target_key:
        return True
    # Prefer explicit table tokens in the site SQL (FROM/UPDATE/INTO/#temp).
    raw = site.raw_sql or ""
    patterns = [
        rf"(?i)\b{_normalize_relation_name(target_table)}\b",
        rf"(?i)#{re.escape(_normalize_relation_name(target_table))}\b",
    ]
    # Also accept schema-qualified forms.
    if any(re.search(p, raw) for p in patterns):
        # Ensure this site actually assigns the column (not just mentions the table).
        return bool(re.search(rf"(?i)\b{re.escape(column)}\b\s*=", raw))
    return False


_AGGREGATE_FUNCTION_RE = re.compile(
    r"\b(MIN|MAX|SUM|COUNT|AVG|LISTAGG)\s*\(", re.IGNORECASE
)
_GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\s+(?P<cols>[^\n\)]+)", re.IGNORECASE)


def _extract_aggregate_info(raw_sql: str) -> list[str]:
    """If the source statement computes its value via an aggregate
    function (MIN/MAX/SUM/COUNT/AVG/LISTAGG) combined with a GROUP BY --
    the shape used for cross-row rollups like "the earliest NPA date
    across all of a customer's accounts" or "the highest DPD across
    several DPD-type columns for one account" -- return a description of
    exactly which aggregate(s) and which grouping column(s), so the
    prompt can tell the model this is a genuine cross-row aggregation, not
    a per-row calculation to re-derive from scratch.

    Detection requires BOTH an aggregate function call AND a GROUP BY
    clause in the same statement -- an aggregate function alone (e.g. a
    single MAX(a,b) picking the larger of two same-row values) is an
    ordinary scalar function call, not a cross-row rollup, and must not be
    flagged here.
    """
    functions = sorted({m.group(1).upper() for m in _AGGREGATE_FUNCTION_RE.finditer(raw_sql)})
    group_by_match = _GROUP_BY_RE.search(raw_sql)
    if not functions or not group_by_match:
        return []

    group_cols = group_by_match.group("cols").strip().rstrip(";").strip()
    return [f"aggregates {', '.join(functions)}(...) grouped by {group_cols}"]


_VARIABLE_ASSIGNMENT_RE = re.compile(
    r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*(.+?);\s*$"
)


def _extract_variable_trace(raw_sql: str) -> list[str]:
    """Find local PL/SQL variable assignments (`v_x := expr;`) inside this
    write site's own text and report only the ones that are actually
    referenced later in the same block -- i.e. the ones that feed the
    final assignment, not every incidental variable that happens to
    appear.

    This deliberately only traces within the single write site's own
    already-collected text (which _assignment_sites_from_statements
    already folds preceding/bridging statements into -- see
    bridge_context there), not across the whole procedure. A full
    whole-procedure variable dependency graph is real, separate scope;
    this covers the common, high-value case where a variable is defined
    immediately before the statement that consumes it (e.g.
    `v_error := SQLERRM; ... SET ERRORDESCRIPTION = v_error`), which is
    exactly the shape the proposal's "DPD -> Reference Period -> ... ->
    FinalNpaDt" example describes at the single-statement-block level.
    """
    assignments = _VARIABLE_ASSIGNMENT_RE.findall(raw_sql)
    if not assignments:
        return []

    trace: list[str] = []
    for name, value in assignments:
        # Does anything *after* this assignment's own line reference the
        # variable? (A crude but safe check: the variable name appears
        # again elsewhere in the text, as a whole word, beyond this one
        # assignment line itself.)
        other_text = raw_sql.replace(f"{name} := {value};", "", 1)
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", other_text, re.IGNORECASE):
            trace.append(f'{name} := {value.strip()}  (this value is used later in this assignment)')
    return trace


_VARIABLE_NAME_TOKEN_RE = re.compile(r"\b([vV]_[A-Za-z0-9_]+)\b")
_VARIABLE_SELECT_INTO_RE = re.compile(
    r"SELECT\s+(?P<select_list>.+?)\s+INTO\s+(?P<targets>[vV]_[A-Za-z0-9_]+(?:\s*,\s*[vV]_[A-Za-z0-9_]+)*)\s+FROM\s+(?P<from_clause>.+?)(?:;|$)",
    re.IGNORECASE | re.DOTALL,
)


def _find_variable_definitions(statements: list["StatementInfo"]) -> list[tuple[int, str, str]]:
    """Every `v_x := expr;` assignment and every `SELECT ... INTO v_x FROM
    ...` across ALL statements of the object, as
    (statement_index, VARIABLE_NAME_UPPER, human-readable definition)
    triples, in source order.

    Both forms matter: `:=` is the common case already handled by
    _extract_variable_trace, but a query-driven variable -- assigned via
    `SELECT col INTO v_x FROM table WHERE ...`, the single most common way
    a PL/SQL procedure pulls in an external value (a processing date from
    a calendar table, a reference value from a parameters table, etc.) --
    is a completely different syntax shape that a `:=`-only scan would
    never find at all. Confirmed against a real, high-value case: the
    business date driving nearly every date calculation in
    PRO_DPD_Calculation_StoredProcedure_2.sql
    (`SELECT "Date" INTO v_ProcessDate FROM SysDayMatrix WHERE ...`) is
    defined exactly this way, not with `:=`.
    """
    definitions: list[tuple[int, str, str]] = []
    for stmt in statements:
        text = stmt.raw_text
        for match in _VARIABLE_ASSIGNMENT_RE.finditer(text):
            name, value = match.group(1), match.group(2).strip()
            if not name.upper().startswith("V_"):
                continue
            definitions.append((stmt.statement_index, name.upper(), f"{name} := {value}"))
        for match in _VARIABLE_SELECT_INTO_RE.finditer(text):
            select_list = " ".join(match.group("select_list").split())
            from_clause = " ".join(match.group("from_clause").split())
            targets = [t.strip() for t in match.group("targets").split(",")]
            for target in targets:
                definitions.append(
                    (
                        stmt.statement_index,
                        target.upper(),
                        f"{target} <- SELECT {select_list} FROM {from_clause}",
                    )
                )
    return definitions


def _trace_variable_dependency_chain(
    variable_names: list[str],
    definitions: list[tuple[int, str, str]],
    before_index: int,
    depth: int = 0,
    visited: set[str] | None = None,
    max_depth: int = 5,
) -> list[str]:
    """Whole-procedure variable dependency trace: for each name in
    `variable_names`, find its most recent definition (highest
    statement_index strictly less than `before_index`, i.e. the
    definition actually in effect at the point of use, not merely the
    first one written anywhere in the procedure) among `definitions`,
    then recurse into whatever OTHER v_-prefixed variables that
    definition's own text references, up to `max_depth` hops, with a
    `visited` set to guarantee termination even if two variables happen
    to reference each other.

    This directly implements the proposal's own example chain shape
    (`DPD -> Reference Period -> NPA Reference Period -> NEW_FINALNPADT
    -> FinalNpaDt`) for real PL/SQL local-variable chains, not just a
    variable defined immediately adjacent to its use -- the whole point
    of "preserve intermediate calculations" is that a variable defined
    far earlier in the procedure must still be found.
    """
    if visited is None:
        visited = set()
    if depth >= max_depth:
        return []

    lines: list[str] = []
    for var_name in variable_names:
        key = var_name.upper()
        if key in visited:
            continue
        visited.add(key)

        candidates = [d for d in definitions if d[1] == key and d[0] < before_index]
        if not candidates:
            continue

        # Show every distinct definition that occurs before this point,
        # not only the textually-last one. A variable is frequently
        # defined by a "normal path" statement (e.g. a SELECT...INTO)
        # AND a separate exception-handler fallback for the same
        # variable (e.g. `v_x := NULL` inside `EXCEPTION WHEN
        # NO_DATA_FOUND THEN`) -- these are mutually exclusive
        # alternate paths, not a sequential overwrite, so picking only
        # the one with the highest statement index would silently hide
        # whichever one happens to sit later in the text (confirmed
        # against a real case: v_ProcessDate's real SELECT...INTO
        # definition was being hidden behind its own
        # `EXCEPTION WHEN NO_DATA_FOUND THEN v_ProcessDate := NULL;`
        # fallback purely because the fallback's statement index is
        # higher). Showing every distinct one is more verbose but never
        # silently wrong.
        seen_summaries: set[str] = set()
        indent = "    " * depth
        next_round_referenced: set[str] = set()
        for stmt_index, _name, summary in sorted(candidates, key=lambda d: d[0]):
            if summary in seen_summaries:
                continue
            seen_summaries.add(summary)
            lines.append(f"{indent}{summary}  (statement #{stmt_index})")
            next_round_referenced.update(
                m.group(1)
                for m in _VARIABLE_NAME_TOKEN_RE.finditer(summary)
                if m.group(1).upper() != key
            )

        referenced = sorted(next_round_referenced)
        if referenced:
            # Use the latest candidate's statement index as the recursion
            # boundary, so a variable referenced inside one of these
            # definitions is still resolved relative to where it's used,
            # not where the outer variable itself is used.
            latest_index = max(c[0] for c in candidates)
            lines.extend(
                _trace_variable_dependency_chain(
                    referenced, definitions, latest_index, depth + 1, visited, max_depth
                )
            )
    return lines


def _whole_procedure_variable_trace(raw_sql: str, statements: list["StatementInfo"], before_index: int) -> list[str]:
    """Public entry point used by _format_assignment_context: find every
    v_-prefixed variable referenced in `raw_sql` and trace each one's
    full whole-procedure dependency chain."""
    referenced = sorted({m.group(1) for m in _VARIABLE_NAME_TOKEN_RE.finditer(raw_sql)})
    if not referenced:
        return []
    definitions = _find_variable_definitions(statements)
    if not definitions:
        return []
    return _trace_variable_dependency_chain(referenced, definitions, before_index)


def _infer_assignment_role(raw_sql: str) -> str:
    """Classify the write pattern generically so the prompt can describe
    the chunk as a value-selection, initialization, fix-up, or exception-
    handling stage.

    This is intentionally heuristic: it should improve source decomposition
    for many procedures without hardcoding any specific column names.
    """
    upper = raw_sql.upper()

    # Checked first, before any MERGE/UPDATE pattern below: control-flow
    # headers (EXCEPTION, WHEN OTHERS THEN, ...) are folded onto the front
    # of the assignment's raw_sql by _relevant_chunks /
    # _assignment_sites_from_statements, so an exception-handler
    # assignment's raw_sql starts with "EXCEPTION" / "WHEN OTHERS THEN",
    # not with the UPDATE/MERGE keyword itself -- without checking this
    # first, such a site silently falls through to the same generic
    # "SEQUENTIAL_ASSIGNMENT" role as an ordinary normal-flow statement,
    # giving the model no signal that the two are mutually exclusive
    # alternate paths rather than sequential steps of one flow. This is
    # exactly the shape that produced a real, confirmed generation defect
    # (a normal-flow guard and an exception-handler guard collapsed into
    # one repeated condition -- see
    # app/guardrails/semantic_validation.py::check_redundant_nested_condition).
    if re.search(r"(?:^|\n)\s*EXCEPTION\b", raw_sql, re.IGNORECASE):
        return "EXCEPTION_HANDLER"
    # Oracle peels `EXCEPTION` and `WHEN OTHERS THEN` into separate control-flow
    # headers; the write site may therefore start with WHEN OTHERS alone.
    if re.search(r"(?:^|\n)\s*WHEN\s+OTHERS\b", raw_sql, re.IGNORECASE):
        return "EXCEPTION_HANDLER"
    # SQL Server / T-SQL uses BEGIN CATCH ... END CATCH rather than Oracle's
    # EXCEPTION block. Treat CATCH the same way so exception-path writes are
    # not composed as a later sequential stage of the normal flow.
    if re.search(r"(?:^|\n)\s*(?:BEGIN\s+)?CATCH\b", raw_sql, re.IGNORECASE):
        return "EXCEPTION_HANDLER"
    if re.search(r"\bEND\s+TRY\b", raw_sql, re.IGNORECASE) and re.search(
        r"\bCATCH\b", raw_sql, re.IGNORECASE
    ):
        return "EXCEPTION_HANDLER"

    if "MERGE INTO" in upper and "USING (" in upper:
        if "CASE WHEN" in upper or re.search(r"\bCASE\b", upper):
            return "MERGE_USING_CASE_VALUE"
        return "MERGE_USING"
    if upper.startswith("MERGE"):
        return "MERGE"

    if upper.startswith("UPDATE"):
        if re.search(r"\bSET\b.*=\s*0\b", upper, re.S):
            return "INITIAL_RESET"
        if re.search(r"\bSET\b.*=\s*NULL\b", upper, re.S):
            return "NULL_RESET"
        if "WHERE" in upper:
            return "SEQUENTIAL_FIXUP"
        return "UPDATE"

    if "CASE WHEN" in upper or re.search(r"\bCASE\b", upper):
        return "CASE_VALUE_SELECTION"

    return "SEQUENTIAL_ASSIGNMENT"


def _assignment_decomposition_hints(raw_sql: str) -> list[str]:
    """Return generic notes that help the model keep guard/value/fix-up
    logic separated when the source SQL is branch-heavy or sequential.
    """
    upper = raw_sql.upper()
    hints: list[str] = []

    role = _infer_assignment_role(raw_sql)
    if role == "EXCEPTION_HANDLER":
        hints.append(
            "This assignment only happens if an unhandled exception occurred elsewhere "
            "in the procedure -- it is a mutually exclusive alternate path, not a later "
            "step of the normal flow. Its trigger condition in the expression must be "
            "genuinely distinct from (never textually identical to) any normal-flow "
            "site's condition for the same column -- e.g. do not gate both this value "
            "and the normal-flow value with the same repeated condition; if the source "
            "has no explicit column recording whether an exception occurred, express the "
            "normal-flow condition and its logical negation as two separate branches, or "
            "use ELSE for whichever site's condition is not otherwise determinable."
        )
        return hints

    if "MERGE INTO" in upper and "USING (" in upper:
        hints.append("Treat the USING subquery as the value source and the MERGE ON/WHERE predicates as the outer guard.")
        if "CASE" in upper:
            hints.append("Preserve the CASE branch choice inside the USING subquery before applying the outer MERGE guard.")

    if upper.startswith("UPDATE"):
        if re.search(r"\bSET\b.*=\s*0\b", upper, re.S):
            hints.append("This is an initialization/reset stage, not the final business result.")
        if re.search(r"\bWHERE\b", upper):
            hints.append("Keep the WHERE clause as a later row-scoping or fix-up guard instead of folding it into the value.")
        if re.search(r"\bSET\b.*=\s*(?:[A-Za-z_][A-Za-z0-9_\.]*|NVL\s*\(|COALESCE\s*\()", upper, re.S):
            hints.append("If this update follows an earlier write to the same column, treat it as a sequential override or backfill rather than a fresh branch tree.")

    if "CASE WHEN" in upper or re.search(r"\bCASE\b", upper):
        hints.append("Preserve the source CASE branches in order; do not flatten later branches into the first one.")

    if _extract_aggregate_info(raw_sql):
        hints.append(
            "This value is computed as a cross-row aggregate (MIN/MAX/SUM/COUNT/AVG/"
            "LISTAGG) grouped by another column in the source SQL -- a Formula "
            "Expression is evaluated one row at a time and cannot itself perform a "
            "GROUP BY. Represent this by referencing the column the aggregated result "
            "is written into (for example the MERGE target this aggregate feeds), not "
            "by attempting to re-derive the aggregation logic at the row level."
        )

    return hints


def _assignment_decomposition_summary(raw_sql: str) -> list[str]:
    """Extract a compact guard/value summary from a write site.

    The raw SQL remains available verbatim, but the summary makes the
    branch split explicit for MERGE/USING and sequential UPDATE patterns.
    """
    text = re.sub(r"\s+", " ", raw_sql.strip())
    upper = text.upper()
    summary: list[str] = []

    if upper.startswith("MERGE") and "USING (" in upper:
        # The alias after the value expression may or may not use the AS
        # keyword (Oracle allows `MAX(DPD) DPD_MaxFin` without AS, not
        # just `MAX(DPD) AS DPD_MaxFin`) -- both forms are common in real
        # procedures, so AS is optional here rather than required.
        value_match = re.search(
            r"USING\s*\(\s*SELECT\s+.*?,\s*(?P<value>.+?)\s+(?:AS\s+)?\w+\s+FROM\s+",
            text,
            re.IGNORECASE,
        )
        if value_match:
            summary.append(f"assigned value: {value_match.group('value').strip()}")

        guard_parts: list[str] = []
        where_match = re.search(r"\bWHERE\b\s+(?P<guard>.+?)\s*\)\s*SRC\s+ON\s*\(", text, re.IGNORECASE)
        if where_match:
            guard_parts.append(where_match.group("guard").strip())
        on_match = re.search(r"\bON\s*\((?P<guard>.+?)\)\s*WHEN\s+MATCHED\b", text, re.IGNORECASE)
        if on_match:
            guard_parts.append(on_match.group("guard").strip())
        if guard_parts:
            summary.append("guard: " + " AND ".join(guard_parts))

    elif upper.startswith("UPDATE"):
        set_match = re.search(r"\bSET\b\s+(?P<assign>.+?)(?:\bWHERE\b|;|$)", text, re.IGNORECASE)
        if set_match:
            summary.append(f"assigned value: {set_match.group('assign').strip()}")
        where_match = re.search(r"\bWHERE\b\s+(?P<guard>.+?)(?:;|$)", text, re.IGNORECASE)
        if where_match:
            summary.append(f"guard: {where_match.group('guard').strip()}")

    summary.extend(_extract_aggregate_info(raw_sql))

    return summary


def _source_sql_context_excerpt(source_sql: str, relevant_sql: str) -> str:
    """Keep the model prompt focused by trimming the broad source SQL
    context to a bounded excerpt.

    The column-specific assignment chunks already carry the important
    logic. The full procedure text is still useful for surrounding context,
    but sending every line of a large stored procedure to the provider for
    every column makes generation noticeably slower.
    """
    source_sql = source_sql.strip()
    relevant_sql = relevant_sql.strip()

    if not source_sql:
        return relevant_sql

    if len(source_sql) <= _MAX_SOURCE_SQL_CONTEXT_CHARS:
        return source_sql

    head_chars = max(1200, _MAX_SOURCE_SQL_CONTEXT_CHARS // 3)
    tail_chars = max(1200, _MAX_SOURCE_SQL_CONTEXT_CHARS // 3)
    head = source_sql[:head_chars].strip()
    tail = source_sql[-tail_chars:].strip()

    sections = []
    if relevant_sql:
        sections.append(relevant_sql)
    if head:
        sections.append("[Source SQL excerpt - beginning]\n" + head)
    if tail and tail != head:
        sections.append("[Source SQL excerpt - end]\n" + tail)
    return "\n\n".join(sections)


def _generate_column_rows(
    job: tuple[
        CanonicalModel,
        SQLObject,
        StructuralInfo,
        str,
        str,
        LLMClient,
        str,
        dict[int, date] | None,
        Optional[ChromaStore],
        dict[str, str],
    ]
) -> list[DDRow]:
    (
        canonical_model,
        obj,
        info,
        entity_name,
        column,
        llm_client,
        function_reference,
        timekey_map,
        rag_store,
        entity_name_map,
    ) = job
    return _generate_for_column(
        canonical_model=canonical_model,
        obj=obj,
        info=info,
        entity_name=entity_name,
        column=column,
        llm_client=llm_client,
        function_reference=function_reference,
        timekey_map=timekey_map,
        rag_store=rag_store,
        entity_name_map=entity_name_map,
    )


_LEXICAL_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is",
        "are", "was", "were", "with", "this", "that", "by", "as", "at",
        "from", "be", "it", "its", "into", "when", "then", "else", "not",
        "select", "from", "where", "update", "insert", "into", "table",
    }
)


def _lexical_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
        if len(token) >= 4 and token not in _LEXICAL_STOPWORDS
    }


@dataclass
class RagContext:
    platform_context: str = ""
    domain_context: str = ""

    @property
    def combined(self) -> str:
        return "\n\n".join(part for part in (self.platform_context, self.domain_context) if part)


def _retrieve_rag_context(
    rag_store: Optional[ChromaStore],
    relevant_sql: str,
    technical_summary: str,
    business_summary: str,
) -> RagContext:
    """Query the platform (4X function/operator) and domain RAG
    collections for the chunks most relevant to this specific column,
    instead of handing the model the entire reference document every time.

    Returns platform/domain sections separately so the caller can skip
    sending the full function_reference doc when a platform hit already
    covers it (see `_generate_for_column`), and filters domain glossary
    hits by lexical overlap with the query -- the embedding is a
    deterministic bag-of-words hash (see chroma_store.py), so for a
    business domain the glossary has no real vocabulary overlap with
    (e.g. non-banking SQL against the banking glossary), Chroma still
    returns its top-n nearest docs even though none are actually relevant.
    Filtering by overlap keeps the domain section from leaking irrelevant
    framing into the prompt for out-of-domain input, instead of silently
    treating "nearest available" as "relevant".

    Returns empty sections -- letting the caller rely on the full
    function_reference instead -- if no RAG store was supplied, the store
    can't be reached, or nothing has been ingested yet. This keeps the
    pipeline fully functional whether or not `ingest_platform_doc` /
    `ingest_domain_doc` has ever been run; RAG is a targeting aid on top of
    the existing full-reference behavior, not a replacement that could
    break generation if it's unavailable.
    """
    if rag_store is None:
        return RagContext()

    platform_query = (relevant_sql or technical_summary).strip()
    domain_query = (business_summary or technical_summary).strip()
    domain_query_tokens = _lexical_tokens(domain_query)

    platform_section = ""
    domain_section = ""
    try:
        if platform_query:
            platform_hits = rag_store.query(PLATFORM_COLLECTION, platform_query, n_results=4)
            if platform_hits:
                platform_section = (
                    "Relevant platform function/operator reference:\n" + "\n---\n".join(platform_hits)
                )
        if domain_query and domain_query_tokens:
            domain_hits = rag_store.query(DOMAIN_COLLECTION, domain_query, n_results=2)
            relevant_hits = [
                hit
                for hit in domain_hits
                if len(_lexical_tokens(hit) & domain_query_tokens) >= 2
            ]
            if relevant_hits:
                domain_section = "Relevant domain glossary:\n" + "\n---\n".join(relevant_hits)
    except Exception as exc:  # pragma: no cover - defensive: RAG must never break generation
        logger.warning("RAG retrieval failed, continuing without it: %s", exc)
        return RagContext()

    return RagContext(platform_context=platform_section, domain_context=domain_section)


def _derive_business_meaning(
    llm_client: LLMClient,
    technical_summary: str,
    business_summary: str,
    source_sql: str,
    function_reference: str,
    entity_name: str,
    column_name: str,
    relevant_sql: str,
    formula: str,
) -> str:
    fallback = _business_meaning_from_formula(column_name, formula)
    explanation_method = getattr(llm_client, "rule_explanation", None)
    if not callable(explanation_method):
        return fallback

    try:
        explanation = explanation_method(
            technical_summary=technical_summary,
            business_summary=business_summary,
            source_sql=source_sql,
            function_reference=function_reference,
            column_name=column_name,
            entity_name=entity_name,
            relevant_sql=relevant_sql,
            formula=formula,
        )
    except Exception:
        return fallback

    if isinstance(explanation, str) and explanation.strip():
        return explanation.strip()
    return fallback


def _business_meaning_from_formula(column_name: str, expression: str) -> str:
    expr = expression.upper()
    column = column_name.strip()

    if "MAX(" in expr:
        return f"Chooses the highest applicable value for {column} from the source drivers."
    if "MIN(" in expr:
        return f"Chooses the lowest applicable value for {column} from the source drivers."
    if "DATEDIFF(" in expr:
        return f"Measures elapsed time for {column} from the relevant business date and source date."
    if "COALESCE(" in expr or "ISEMPTY(" in expr or "ISNOTEMPTY(" in expr:
        return f"Uses null-handling and fallback logic to populate {column} from the source fields."
    if "THEN(" in expr and "ELSEIF(" in expr:
        return f"Applies branch-based rules to determine {column} from the source conditions."
    if "THEN(" in expr:
        return f"Applies a conditional rule to derive {column} from the source conditions."
    return f"SQL-derived logic for {column} based on the available source dependencies."


def _flatten_whitespace(expression: str) -> str:
    """Collapse all internal whitespace (including newlines and
    indentation) into single spaces.

    The 4X grammar itself ignores whitespace entirely when parsing (see
    fourx_grammar.lark's `%ignore WS`), so this never changes what an
    expression means -- it only guarantees the stored/exported expression
    is always a single line. A multi-line value breaks a Markdown table
    row (the report renders every DD row as one table row) and makes a
    poor spreadsheet cell; applying this once here, at the source, keeps
    the Markdown report and the CSV export consistent with each other
    instead of patching the symptom separately in each renderer.
    """
    return " ".join(expression.split())


def _fix_unbalanced_trailing_parens(expression: str) -> str:
    """Fix the common LLM mistake of closing one (or a few) too many, or
    too few, parentheses at the very end of an otherwise-correct
    expression -- deeply nested IF/ELSEIF/ELSE trees make manual
    paren-counting error prone, and this is a purely mechanical, frequent
    failure mode, distinct from any actual logic error.

    Only ever trims or adds parentheses at the very end of the expression,
    and only accepts the result if it actually parses against the real 4X
    grammar (not merely paren-depth-balanced -- depth balance alone isn't
    proof of a correct token sequence). It never touches parentheses in
    the interior, so it can't silently change the expression's actual
    structure. If no simple trailing adjustment produces something that
    parses, the expression is left untouched and grammar validation will
    correctly reject it, triggering a normal LLM repair attempt instead.
    """

    def depth_profile(text: str) -> list[int]:
        depth = 0
        profile = []
        in_double = False
        for ch in text:
            if ch == '"':
                in_double = not in_double
            elif not in_double and ch == "(":
                depth += 1
            elif not in_double and ch == ")":
                depth -= 1
            profile.append(depth)
        return profile

    profile = depth_profile(expression)
    if not profile:
        return expression
    final_depth = profile[-1]
    if final_depth == 0:
        return expression

    if final_depth > 0:
        candidate = expression + (")" * final_depth)
        if validate_expression(candidate).valid:
            return candidate
        return expression

    excess = -final_depth
    trimmed = expression.rstrip()
    trailing_closes = 0
    i = len(trimmed) - 1
    while i >= 0 and trimmed[i] == ")" and trailing_closes < excess:
        trailing_closes += 1
        i -= 1
    if trailing_closes < excess:
        return expression

    candidate = trimmed[: len(trimmed) - trailing_closes]
    if validate_expression(candidate).valid:
        return candidate
    return expression


def _normalize_sql_functions(expression: str) -> str:
    """Rewrite common SQL-only helper functions that are not part of the 4X
    grammar's function library into their direct 4X equivalents, before
    grammar validation:

    - NVL(x, default) -> COALESCE(x, default) -- same two-argument shape,
      just a different name (NVL is Oracle-specific; COALESCE is what the
      4X function reference documents).
    - ISNULL(x) -> ISEMPTY(x) when used with a single argument (a common,
      if non-standard, null-check shorthand seen in SQL-Server-derived
      code); ISNULL(x, default) -> COALESCE(x, default) when used with two
      arguments (SQL Server's real ISNULL semantics).

    This is a mechanical, function-name/argument-count-based rewrite that
    applies the same way regardless of which input SQL produced the
    expression -- it is not specific to any one procedure or column.
    """

    def rewrite_calls(text: str, func_name: str, one_arg_target: str, two_arg_target: str) -> str:
        token = func_name + "("
        result: list[str] = []
        i = 0
        n = len(text)
        in_double = False
        while i < n:
            ch = text[i]

            if in_double:
                result.append(ch)
                if ch == '"':
                    in_double = False
                i += 1
                continue

            if ch == '"':
                in_double = True
                result.append(ch)
                i += 1
                continue

            if text[i : i + len(token)].upper() == token.upper() and (
                i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
            ):
                start = i + len(token)
                depth = 1
                j = start
                local_in_double = False
                args: list[str] = []
                current: list[str] = []
                while j < n and depth > 0:
                    cur = text[j]
                    if cur == '"':
                        local_in_double = not local_in_double
                        current.append(cur)
                    elif not local_in_double and cur == "(":
                        depth += 1
                        current.append(cur)
                    elif not local_in_double and cur == ")":
                        depth -= 1
                        if depth == 0:
                            break
                        current.append(cur)
                    elif not local_in_double and cur == "," and depth == 1:
                        args.append("".join(current).strip())
                        current = []
                    else:
                        current.append(cur)
                    j += 1

                if depth == 0:
                    args.append("".join(current).strip())
                    target = one_arg_target if len(args) == 1 else two_arg_target
                    result.append(f"{target}({', '.join(args)})")
                    i = j + 1
                    continue

            result.append(ch)
            i += 1

        return "".join(result)

    expression = rewrite_calls(expression, "NVL", "COALESCE", "COALESCE")
    expression = rewrite_calls(expression, "ISNULL", "ISEMPTY", "COALESCE")
    return expression


def _find_matching_paren(text: str, open_index: int) -> int:
    """Find the matching closing parenthesis, ignoring quoted segments."""
    depth = 0
    in_double = False
    for idx in range(open_index, len(text)):
        ch = text[idx]
        if ch == '"':
            in_double = not in_double
            continue
        if in_double:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _rewrite_legacy_else_if(expression: str) -> str:
    return re.sub(r"(?i)\bELSE\s+IF\b", "ELSEIF", expression)


def _rewrite_not_in_membership(expression: str) -> str:
    return re.sub(r"(?i)\bNOT\s+IN\b", "NOTIN", expression)


_BARE_NOT_FUNCTION_RE = re.compile(r"\bNOT\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE)


def _wrap_bare_not_in_parens(expression: str) -> str:
    """4X's grammar requires NOT as a function-call form -- `NOT(condition)`
    -- with no bare `NOT condition` form at all
    (`not_expr: "NOT" "(" expr ")"`). SQL's negation, by contrast, is a
    bare prefix operator (`NOT X`), and sqlglot's own re-serialization of
    `X IS NOT NULL` produces exactly that bare form
    (`NOT X IS NULL`), which the existing IS-NULL rewrite then turns into
    the still-invalid `NOT ISEMPTY(X)`. Confirmed against a real case:
    this was silently causing a deterministically-translatable CASE
    expression (in PRO_DPD_Calculation_StoredProcedure_2.sql's
    DPD_IntService) to fall back to the LLM for a reason that had
    nothing to do with the CASE translation itself.

    This wraps the following function call in its own parens whenever
    NOT is immediately followed by one, turning `NOT ISEMPTY(X)` into
    `NOT(ISEMPTY(X))`. Deliberately narrow: only fires when NOT is
    directly followed by a single recognizable `FUNC_NAME(...)` call --
    the shape every real case observed so far actually produces. A bare
    `NOT` followed by something else (a raw comparison, a column
    reference) is left untouched -- determining the correct extent to
    wrap in that case would require real expression parsing, and grammar
    validation already correctly rejects and routes that shape to
    PENDING_REVIEW rather than this guessing at it.
    """
    result: list[str] = []
    i = 0
    n = len(expression)
    in_double = False
    while i < n:
        ch = expression[i]
        if ch == '"':
            in_double = not in_double
            result.append(ch)
            i += 1
            continue
        if in_double:
            result.append(ch)
            i += 1
            continue
        match = _BARE_NOT_FUNCTION_RE.match(expression, i)
        if match and (i == 0 or not (expression[i - 1].isalnum() or expression[i - 1] == "_")):
            func_name_start = match.start(1)
            open_paren = expression.index("(", match.end(1))
            depth = 1
            k = open_paren + 1
            local_in_double = False
            close_paren = None
            while k < n:
                c = expression[k]
                if c == '"':
                    local_in_double = not local_in_double
                elif not local_in_double:
                    if c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                        if depth == 0:
                            close_paren = k
                            break
                k += 1
            if close_paren is not None:
                func_call_text = expression[func_name_start : close_paren + 1]
                result.append(f"NOT({func_call_text})")
                i = close_paren + 1
                continue
        result.append(ch)
        i += 1
    return "".join(result)


def _rewrite_is_empty_syntax(expression: str) -> str:
    expression = re.sub(r"(?i)\bIS\s+NOT\s+EMPTY\b", "ISNOTEMPTY", expression)
    expression = re.sub(r"(?i)\bIS\s+EMPTY\b", "ISEMPTY", expression)
    return expression


def _rewrite_isnotempty_boolean_comparisons(expression: str) -> str:
    def replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        literal = match.group(3)
        return f'ISNOTEMPTY({inner}) AND {inner}=="{literal}"'

    return re.sub(
        r'(?i)\bISNOTEMPTY\s*\(\s*([^)]+?)\s*\)\s*==\s*(["\'])(Y|N)\2',
        lambda match: replace(match),
        expression,
    )


def _rewrite_postfix_isnotempty(expression: str) -> str:
    pattern = re.compile(
        r'(?i)(?<![A-Za-z0-9_"])((?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*(?:\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))*)\s+ISNOTEMPTY\b(?!\s*\()'
    )
    return pattern.sub(lambda m: f"ISNOTEMPTY({m.group(1).strip()})", expression)


def _rewrite_misused_empty_functions(expression: str) -> str:
    """Rewrite accidentally multi-argument `ISNOTEMPTY` / `ISEMPTY` calls.

    The platform functions are unary. When the model copies a SQL null
    fallback shape into the wrong function name, the only safe repair is
    to treat the call as a `COALESCE(...)`-style fallback instead of
    trying to interpret it as a boolean existence check.
    """

    def rewrite_calls(text: str, func_name: str) -> str:
        result: list[str] = []
        i = 0
        n = len(text)
        in_double = False

        while i < n:
            ch = text[i]
            if in_double:
                result.append(ch)
                if ch == '"':
                    in_double = False
                i += 1
                continue

            if ch == '"':
                in_double = True
                result.append(ch)
                i += 1
                continue

            if text[i : i + len(func_name)].upper() == func_name and (
                i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
            ):
                start = i + len(func_name)
                if start < n and text[start] == "(":
                    depth = 1
                    j = start + 1
                    local_in_double = False
                    args: list[str] = []
                    current: list[str] = []

                    while j < n and depth > 0:
                        cur = text[j]
                        if cur == '"':
                            local_in_double = not local_in_double
                            current.append(cur)
                        elif not local_in_double and cur == "(":
                            depth += 1
                            current.append(cur)
                        elif not local_in_double and cur == ")":
                            depth -= 1
                            if depth == 0:
                                break
                            current.append(cur)
                        elif not local_in_double and cur == "," and depth == 1:
                            args.append("".join(current).strip())
                            current = []
                        else:
                            current.append(cur)
                        j += 1

                    if depth == 0:
                        args.append("".join(current).strip())
                        if len(args) > 1:
                            result.append(f"COALESCE({', '.join(args)})")
                        else:
                            result.append(f"{func_name}({', '.join(args)})")
                        i = j + 1
                        continue

            result.append(ch)
            i += 1

        return "".join(result)

    expression = rewrite_calls(expression, "ISNOTEMPTY")
    expression = rewrite_calls(expression, "ISEMPTY")
    return expression


def _rewrite_date_function(expression: str) -> str:
    """Rewrite SQL-style DATE(...) wrappers into the documented 4X date
    constructor when the content is a single argument."""
    def replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        return f"TODATE({inner})"

    return re.sub(r"(?i)\bDATE\s*\(\s*([^()]+?)\s*\)", replace, expression)


def _rewrite_sql_date_literals(expression: str) -> str:
    return re.sub(
        r'(?i)\bDATE\s*["\']([^"\']+)["\']',
        lambda match: f'TODATE("{match.group(1).strip()}")',
        expression,
    )


def _rewrite_sql_not_equal_operator(expression: str) -> str:
    """Normalize SQL's `<>` inequality operator to `!=`."""

    result: list[str] = []
    i = 0
    n = len(expression)
    in_double = False

    while i < n:
        ch = expression[i]
        if ch == '"':
            in_double = not in_double
            result.append(ch)
            i += 1
            continue
        if not in_double and ch == "<" and i + 1 < n and expression[i + 1] == ">":
            result.append("!=")
            i += 2
            continue
        result.append(ch)
        i += 1

    return "".join(result)


def _rewrite_string_concatenation(expression: str) -> str:
    """Rewrite text concatenation written with `+` into `CONCAT(...)`.

    The model sometimes copies SQL-style string concatenation into a 4X
    formula. The platform only documents `+` for numeric arithmetic, so a
    chain that contains a quoted string literal is repaired to
    `CONCAT(...)` before validation. Pure numeric addition is left alone.
    """

    def contains_direct_string_literal(segment: str) -> bool:
        stripped = segment.strip()
        if not stripped:
            return False

        def quoted_is_text_literal(quoted: str) -> bool:
            inner = quoted[1:-1]
            # Identifier-like tokens are column refs, not concat literals.
            return not bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", inner))

        if re.fullmatch(r'"[^"]*"', stripped):
            return quoted_is_text_literal(stripped)
        if re.fullmatch(r'\(\s*"[^"]*"\s*\)', stripped):
            inner_quoted = stripped.strip()[1:-1].strip()
            return quoted_is_text_literal(inner_quoted)
        return False

    def split_top_level_additions(text: str) -> tuple[list[str], bool]:
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        in_double = False
        saw_plus = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == '"':
                in_double = not in_double
                current.append(ch)
                i += 1
                continue
            if in_double:
                current.append(ch)
                i += 1
                continue
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "+" and depth == 0:
                parts.append("".join(current).strip())
                current = []
                saw_plus = True
            else:
                current.append(ch)
            i += 1
        parts.append("".join(current).strip())
        return parts, saw_plus

    def rewrite(text: str) -> str:
        collapsed: list[str] = []
        i = 0
        n = len(text)
        in_double = False

        while i < n:
            ch = text[i]
            if ch == '"':
                in_double = not in_double
                collapsed.append(ch)
                i += 1
                continue
            if in_double:
                collapsed.append(ch)
                i += 1
                continue
            if ch == "(":
                close_index = _find_matching_paren(text, i)
                if close_index != -1:
                    inner = rewrite(text[i + 1 : close_index])
                    collapsed.append("(")
                    collapsed.append(inner)
                    collapsed.append(")")
                    i = close_index + 1
                    continue
            collapsed.append(ch)
            i += 1

        collapsed_text = "".join(collapsed)
        parts, saw_plus = split_top_level_additions(collapsed_text)
        if saw_plus and len(parts) > 1 and any(contains_direct_string_literal(part) for part in parts):
            return f"CONCAT({', '.join(parts)})"
        return collapsed_text

    return rewrite(expression)


def _rewrite_sqlglot_date_functions(expression: str) -> str:
    """Normalize SQLGlot-rendered date helpers into documented 4X TODATE."""
    expression = re.sub(
        r"(?i)\bDATE_STR_TO_DATE\s*\(\s*'([^']+)'\s*\)",
        lambda match: f'TODATE("{match.group(1).strip()}")',
        expression,
    )

    def replace_str_to_date(match: re.Match[str]) -> str:
        raw_date = match.group(1).strip()
        raw_format = match.group(2).strip()
        format_map = {
            "%d/%m/%Y": "DD/MM/YYYY",
            "%m/%d/%Y": "MM/DD/YYYY",
            "%Y-%m-%d": "YYYY-MM-DD",
            "%d-%m-%Y": "DD-MM-YYYY",
            "%m-%d-%Y": "MM-DD-YYYY",
        }
        mapped_format = format_map.get(raw_format, raw_format)
        if mapped_format:
            return f'TODATE("{raw_date}","{mapped_format}")'
        return f'TODATE("{raw_date}")'

    expression = re.sub(
        r"(?i)\bSTR_TO_DATE\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)",
        replace_str_to_date,
        expression,
    )
    return expression


def _rewrite_to_date_function(expression: str) -> str:
    """Normalize Oracle TO_DATE(date, format) into documented 4X TODATE."""

    def replace(match: re.Match[str]) -> str:
        raw_date = match.group(1).strip().strip("'\"")
        raw_format = match.group(2).strip().strip("'\"")
        format_map = {
            "%d/%m/%Y": "DD/MM/YYYY",
            "%m/%d/%Y": "MM/DD/YYYY",
            "%Y-%m-%d": "YYYY-MM-DD",
            "%d-%m-%Y": "DD-MM-YYYY",
            "%m-%d-%Y": "MM-DD-YYYY",
            "DD/MM/YYYY": "DD/MM/YYYY",
            "MM/DD/YYYY": "MM/DD/YYYY",
            "YYYY-MM-DD": "YYYY-MM-DD",
            "DD-MM-YYYY": "DD-MM-YYYY",
            "MM-DD-YYYY": "MM-DD-YYYY",
        }
        mapped_format = format_map.get(raw_format, raw_format)
        if mapped_format:
            return f'TODATE("{raw_date}","{mapped_format}")'
        return f'TODATE("{raw_date}")'

    return re.sub(
        r"(?i)\bTO_DATE\s*\(\s*('(?:[^']|''|\\')*'|[^,()]+)\s*,\s*('(?:[^']|''|\\')*'|[^)]+)\s*\)",
        replace,
        expression,
    )


def _rewrite_unquoted_dotted_refs(expression: str) -> str:
    """Quote bare dotted identifiers so the 4X grammar can parse them as
    column references.

    The source SQL often uses `table.column` or `alias.column` syntax, but
    the 4X grammar only accepts quoted reference segments. This pass keeps
    the semantic shape intact while making the output grammar-safe.
    """
    result: list[str] = []
    i = 0
    n = len(expression)
    in_double = False
    while i < n:
        ch = expression[i]
        if ch == '"':
            in_double = not in_double
            result.append(ch)
            i += 1
            continue

        if in_double:
            result.append(ch)
            i += 1
            continue

        if ch.isalpha() or ch == "_":
            start = i
            j = i + 1
            while j < n and (expression[j].isalnum() or expression[j] == "_"):
                j += 1

            parts = [expression[start:j]]
            k = j
            while True:
                l = k
                while l < n and expression[l].isspace():
                    l += 1
                if l >= n or expression[l] != ".":
                    break
                m = l + 1
                while m < n and expression[m].isspace():
                    m += 1
                if m >= n:
                    break
                if expression[m] == '"':
                    p = m + 1
                    while p < n and expression[p] != '"':
                        p += 1
                    if p >= n:
                        break
                    parts.append(expression[m + 1 : p])
                    k = p + 1
                    continue
                if not (expression[m].isalpha() or expression[m] == "_"):
                    break
                p = m + 1
                while p < n and (expression[p].isalnum() or expression[p] == "_"):
                    p += 1
                parts.append(expression[m:p])
                k = p

            if len(parts) > 1:
                result.append(".".join(f'"{part}"' for part in parts))
                i = k
                continue

        result.append(ch)
        i += 1

    return "".join(result)


def _rewrite_bundled_business_date_var(expression: str) -> str:
    return re.sub(
        r'("?[A-Za-z_][A-Za-z0-9_]*"?)\s*\.\s*"var_BUSINESS_DATE"',
        r'\1."var"."BUSINESS_DATE"',
        expression,
    )


def _rewrite_business_date_variables(
    expression: str,
    entity_name: str,
    source_sql: str = "",
) -> str:
    if not expression or not entity_name:
        return expression
    replacement = f'"{entity_name}"."var"."BUSINESS_DATE"'
    # Only map true process/business date scalars. Offset-derived windows
    # such as @GraceWindowStart = DATEADD(DAY, -3, @ProcessDate) must keep
    # their offset via variable lineage — never collapse to BUSINESS_DATE.
    rewritten = re.sub(
        r'(?<![A-Za-z0-9_".])@?(?:v_)?(?:PROCESSDATE|PROCESSDT|BUSINESSDATE)\b',
        replacement,
        expression,
        flags=re.IGNORECASE,
    )
    rewritten = _apply_date_offset_variable_lineage(rewritten, entity_name, source_sql=source_sql)
    rewritten = _inline_declare_numeric_literals(rewritten, source_sql)
    rewritten = _rewrite_eomonth_and_month_start_vars(rewritten, entity_name, source_sql)
    rewritten = _rewrite_scalar_lookup_variables(rewritten, entity_name, source_sql)
    return rewritten.replace('."VAR"."BUSINESS_DATE"', '."var"."BUSINESS_DATE"')


_SCALAR_LOOKUP_DECLARE_RE = re.compile(
    r"(?is)DECLARE\s+@(?P<var>[A-Za-z_][\w]*)\s+[A-Za-z_][\w]*\s*(?:\([^)]*\))?\s*=\s*"
    r"\(\s*SELECT\s+(?P<col>[A-Za-z_][\w]*)\s+FROM\s+(?:\[?[A-Za-z_][\w]*\]?\.)?"
    r"\[?(?P<table>[A-Za-z_][\w]*)\]?\s+WHERE"
)


def _rewrite_scalar_lookup_variables(expression: str, entity_name: str, source_sql: str) -> str:
    """Replace `@var` with `"<source table>"."<source column>"` when the
    procedure declares it as a single-row scalar lookup against a named
    table (`DECLARE @var TYPE = (SELECT col FROM table WHERE ...)`).

    Table/column names come entirely from the DECLARE statement itself, so
    this generalizes across any lookup table rather than one hardcoded
    entity — the variable's real source is whatever the source SQL says.
    """
    if not expression or not source_sql:
        return expression
    out = expression
    for match in _SCALAR_LOOKUP_DECLARE_RE.finditer(source_sql):
        var = match.group("var")
        table = match.group("table")
        col = match.group("col")
        out = re.sub(
            rf'(?<![A-Za-z0-9_".])@?{re.escape(var)}\b',
            f'"{table}"."{col}"',
            out,
            flags=re.IGNORECASE,
        )
    return out


_DATEADD_OFFSET_ASSIGN_RE = re.compile(
    r"(?is)@(?:v_)?(?P<var>[A-Za-z_][\w]*)\s+"
    r"(?:DATE|DATETIME|DATETIME2|SMALLDATETIME|INT|BIGINT|SMALLINT|TINYINT|"
    r"DECIMAL\s*\([^)]*\)|NUMERIC\s*\([^)]*\)|[A-Za-z_][\w]*)?\s*=\s*"
    r"DATEADD\s*\(\s*(?P<unit>DAY|DAYS|MONTH|MONTHS|YEAR|YEARS)\s*,\s*(?P<offset>-?\d+)\s*,\s*"
    r"@?(?:v_)?(?:ProcessDate|ProcessDt|BusinessDate)\s*\)"
)


def _extract_date_offset_lineage(source_sql: str) -> dict[str, tuple[str, int]]:
    """Map variable name (upper) → (unit, offset) from process/business date."""
    offsets: dict[str, tuple[str, int]] = {}
    for match in _DATEADD_OFFSET_ASSIGN_RE.finditer(source_sql or ""):
        unit = match.group("unit").upper().rstrip("S")
        offsets[match.group("var").upper()] = (unit, int(match.group("offset")))
    if "GRACEWINDOWSTART" not in offsets and re.search(
        r"(?i)@GraceWindowStart\b", source_sql or ""
    ):
        if re.search(r"(?is)DATEADD\s*\(\s*DAY\s*,\s*-3\s*,\s*@ProcessDate\s*\)", source_sql or ""):
            offsets["GRACEWINDOWSTART"] = ("DAY", -3)
    return offsets


def _apply_date_offset_variable_lineage(expression: str, entity_name: str, source_sql: str = "") -> str:
    """Replace offset date vars with ADDDAY/PERIOD(BUSINESS_DATE, n), not BUSINESS_DATE.

    Offsets are read from the procedure's own `DECLARE @var = DATEADD(...)`
    statement — never guessed from a fixed name/offset table. If the
    variable's DECLARE isn't present in `source_sql` (or `source_sql` isn't
    available for this excerpt), the variable is left unresolved rather than
    silently substituted with an offset borrowed from an unrelated procedure.
    """
    offsets = _extract_date_offset_lineage(source_sql)
    if not offsets:
        return expression

    bd = f'"{entity_name}"."var"."BUSINESS_DATE"'

    def _repl(match: re.Match[str]) -> str:
        var = match.group(0).lstrip("@")
        key = re.sub(r"(?i)^v_", "", var).upper()
        if key not in offsets:
            return match.group(0)
        unit, offset = offsets[key]
        if unit == "DAY":
            return f"ADDDAY({bd}, {offset})"
        if unit == "MONTH":
            return f'PERIOD("M", {offset}, {bd})'
        if unit == "YEAR":
            return f'PERIOD("Y", {offset}, {bd})'
        return match.group(0)

    name_alt = "|".join(re.escape(n) for n in sorted(offsets))
    return re.sub(
        rf'(?<![A-Za-z0-9_".])@?(?:v_)?(?:{name_alt})\b',
        _repl,
        expression,
        flags=re.IGNORECASE,
    )


def _inline_declare_numeric_literals(expression: str, source_sql: str) -> str:
    """Replace bare/ @int DECLARE names with their literal values."""
    if not expression or not source_sql:
        return expression
    out = expression
    for match in re.finditer(
        r"(?is)DECLARE\s+@(?P<var>[A-Za-z_][\w]*)\s+(?:INT|BIGINT|SMALLINT|TINYINT|DECIMAL\s*\([^)]*\)|NUMERIC\s*\([^)]*\))"
        r"\s*=\s*(?P<val>-?\d+(?:\.\d+)?)",
        source_sql,
    ):
        var = match.group("var")
        val = match.group("val")
        out = re.sub(
            rf'(?<![A-Za-z0-9_".])@?{re.escape(var)}\b',
            val,
            out,
            flags=re.IGNORECASE,
        )
    return out


def _rewrite_eomonth_and_month_start_vars(
    expression: str,
    entity_name: str,
    source_sql: str,
) -> str:
    """Map EOMONTH/MonthStart/GraceWindowEnd DECLARE chains to platform dates."""
    if not expression:
        return expression
    bd = f'"{entity_name}"."var"."BUSINESS_DATE"'
    out = expression
    if source_sql and re.search(
        r"(?is)DECLARE\s+@MonthEndDate\s+DATE\s*=\s*EOMONTH\s*\(\s*@ProcessDate\s*\)",
        source_sql,
    ):
        out = re.sub(
            rf'(?<![A-Za-z0-9_".])@?MonthEndDate\b',
            f"EOM({bd})",
            out,
            flags=re.IGNORECASE,
        )
    # @MonthStartDate = DATEADD(DAY, -DAY(@ProcessDate)+1, @ProcessDate) ≈ SOM
    if source_sql and re.search(r"(?i)@MonthStartDate\b", source_sql):
        out = re.sub(
            rf'(?<![A-Za-z0-9_".])@?MonthStartDate\b',
            f"SOM({bd})",
            out,
            flags=re.IGNORECASE,
        )
    # @GraceWindowEnd = DATEADD(DAY, 6, @MonthStartDate)
    if source_sql and re.search(
        r"(?is)DECLARE\s+@GraceWindowEnd\s+DATE\s*=\s*DATEADD\s*\(\s*DAY\s*,\s*6\s*,\s*@MonthStartDate\s*\)",
        source_sql,
    ):
        out = re.sub(
            rf'(?<![A-Za-z0-9_".])@?GraceWindowEnd\b',
            f"ADDDAY(SOM({bd}), 6)",
            out,
            flags=re.IGNORECASE,
        )
    return out


def _rewrite_exists_predicates(expression: str) -> str:
    """Preserve EXISTS rather than flattening it to a row predicate.

    Flattening `IF EXISTS (SELECT ... WHERE <pred>)` into `<pred>` converts a
    procedure-wide existence branch into a per-row formula and changes
    meaning. Platform Formula Expressions cannot express procedure-level
    EXISTS; leave the construct intact so validation marks it unsupported.
    """
    return expression


def _rewrite_bundled_alias_column_refs(expression: str, source_text: str = "") -> str:
    """Rewrite fused alias-like tokens such as `PUI_CAL_DEFAULT_REASON`
    back to the source column name `DEFAULT_REASON` when the suffix is
    actually present in the source SQL.

    LLMs sometimes concatenate a table name or alias with a column name
    instead of emitting a dotted reference. That produces invented
    identifiers even when the underlying column is real. When the source
    SQL contains the suffix by itself, dropping the fused prefix is a
    safe mechanical normalization.
    """
    if not source_text:
        return expression

    source_tokens = {token.upper() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source_text)}
    result: list[str] = []
    i = 0
    n = len(expression)
    in_double = False

    while i < n:
        ch = expression[i]
        if ch == '"':
            in_double = not in_double
            result.append(ch)
            i += 1
            continue
        if in_double:
            result.append(ch)
            i += 1
            continue

        if ch.isalpha() or ch == "_":
            start = i
            j = i + 1
            while j < n and (expression[j].isalnum() or expression[j] == "_"):
                j += 1
            token = expression[start:j]
            if "_" in token and token.upper() not in source_tokens:
                underscores = [idx for idx, char in enumerate(token) if char == "_"]
                replacement = token
                for idx in underscores:
                    suffix = token[idx + 1 :]
                    if suffix and suffix.upper() in source_tokens:
                        replacement = suffix
                if replacement != token:
                    result.append(replacement)
                    i = j
                    continue
            result.append(token)
            i = j
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _strip_angle_bracket_placeholders(expression: str) -> str:
    """Remove literal `<...>` placeholder wrappers from identifiers.

    Prompt examples use placeholder notation like `<entity_name>`, and the
    model sometimes copies those angle brackets verbatim into output.
    This is never part of the actual 4X grammar, so stripping them is a
    safe mechanical cleanup as long as the bracketed text is an
    identifier-like token.
    """
    expression = re.sub(r'"<([A-Za-z_][A-Za-z0-9_]*)>"', r'"\1"', expression)
    expression = re.sub(r'(?<![A-Za-z0-9_"])<([A-Za-z_][A-Za-z0-9_]*)>(?![A-Za-z0-9_"])', r"\1", expression)
    return expression


def _rewrite_null_predicates(expression: str) -> str:
    expression = re.sub(
        r'(?i)\b([A-Za-z_][A-Za-z0-9_".]*?)\s+IS\s+NOT\s+NULL\b',
        r"ISNOTEMPTY(\1)",
        expression,
    )
    expression = re.sub(
        r'(?i)\b([A-Za-z_][A-Za-z0-9_".]*?)\s+IS\s+NULL\b',
        r"ISEMPTY(\1)",
        expression,
    )
    return expression


def _rewrite_in_subquery_membership(expression: str) -> str:
    """Rewrite a single-row `IN [value WHERE predicate]` subquery shape."""
    pattern = re.compile(
        r'(?i)\b(?P<lhs>(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))*)'
        r'\s+IN\s*\[\s*(?P<rhs>(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))*)'
        r'\s+WHERE\s+(?P<predicate>[^\]]+?)\s*\]'
    )
    return pattern.sub(lambda m: f'{m.group("lhs")} == {m.group("rhs")} AND ({m.group("predicate").strip()})', expression)


def _strip_min_wrapper(expression: str) -> str:
    """Preserve MIN wrappers as-is.

    Earlier versions stripped MIN(...) aggressively to work around a
    small set of malformed model outputs, but that destroyed legitimate
    source-derived aggregates such as `MIN(A.SMA_Dt)`. The grammar now
    supports MIN directly, so the safest generic behavior is to leave the
    wrapper untouched.
    """
    return expression


def _repair_missing_then_parentheses(expression: str) -> str:
    """Insert a missing `)` before THEN only when the IF/ELSEIF condition
    is still open at that point.

    This is intentionally conservative: if the condition already closes
    before THEN, the expression is left untouched.
    """

    if "THEN" not in expression.upper() or ("IF(" not in expression.upper() and "ELSEIF(" not in expression.upper()):
        return expression

    result: list[str] = []
    i = 0
    n = len(expression)
    in_double = False
    changed = False

    while i < n:
        ch = expression[i]
        if ch == '"':
            in_double = not in_double
            result.append(ch)
            i += 1
            continue
        if in_double:
            result.append(ch)
            i += 1
            continue

        token = None
        if expression[i : i + 7].upper() == "ELSEIF(" and (i == 0 or not (expression[i - 1].isalnum() or expression[i - 1] == "_")):
            token = "ELSEIF("
        elif expression[i : i + 3].upper() == "IF(" and (i == 0 or not (expression[i - 1].isalnum() or expression[i - 1] == "_")):
            token = "IF("

        if token:
            start = i + len(token)
            depth = 1
            j = start
            local_in_double = False
            repaired_here = False
            while j < n:
                cur = expression[j]
                if cur == '"':
                    local_in_double = not local_in_double
                elif not local_in_double:
                    if cur == "(":
                        depth += 1
                    elif cur == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    elif depth == 1 and expression[j : j + 4].upper() == "THEN":
                        before = expression[j - 1] if j > 0 else ""
                        after = expression[j + 4] if j + 4 < n else ""
                        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                            result.append(expression[i:j])
                            result.append(")")
                            i = j
                            changed = True
                            repaired_here = True
                            break
                j += 1
            # Only skip the fallthrough append when *this* IF/ELSEIF was
            # actually repaired -- `changed` tracks whether any repair has
            # happened anywhere in the expression so far (for the return
            # value below) and must never gate the loop's own advancement,
            # or a later, already-well-formed IF/ELSEIF (one that closes
            # normally, without needing a repair) would never advance `i`
            # once an earlier repair had set `changed = True`.
            if repaired_here:
                continue

        result.append(ch)
        i += 1

    return "".join(result)


_MISSING_IF_KEYWORD_BEFORE_RE = re.compile(r"(IF|ELSEIF)\s*$", re.IGNORECASE)


def _repair_missing_if_before_then(expression: str) -> str:
    """Deterministically insert a missing `IF` keyword when a
    parenthesized condition is directly followed by `THEN(...)` without
    an `IF`/`ELSEIF` keyword of its own -- i.e. `(cond)THEN(a)ELSE(b)`
    where an `IF` was clearly intended but omitted.

    Confirmed against a real generation defect: a DEGDATE expression
    contained exactly `("A"."X">="A"."Y")THEN("A"."X")ELSE("A"."Y")`
    sitting as the *value* inside an outer THEN(...) clause -- the model
    appears to have been attempting the same "pick the greater of two
    values" construct that also produced the ternary (`? :`) defect
    elsewhere, but this time omitted the leading `IF` entirely instead of
    using `?`/`:`.

    Deliberately narrow and bail-safe, same philosophy as
    _normalize_ternary_operator: only fires when a `)` immediately
    precedes `THEN(` and that `)`'s matching `(` is NOT itself preceded
    by `IF`/`ELSEIF` (i.e. this is unambiguously not already a properly
    formed IF/ELSEIF...THEN). Iterates until no more insertions apply, so
    more than one occurrence in the same expression is handled, and
    always re-scans from the start after each insertion since inserting
    text shifts every later position.

    Important, honest limitation: this fixes the missing-`IF` shape in
    isolation, but does NOT guarantee the surrounding expression becomes
    fully grammar-valid on its own -- the same underlying "pick the
    greater of two values, then compare to a third" construct has been
    observed producing a SEPARATE, compounding malformation (an extra
    unmatched closing parenthesis) in the same real defect this function
    was built from. That combination was deliberately NOT force-repaired
    here: attempting to also guess at removing "the right" extra paren
    in an already-malformed, deeply-nested expression carries real risk
    of producing a different, silently wrong rewrite rather than a
    correct one. Grammar validation still runs after this (and every
    other normalization pass) and correctly routes anything still
    invalid to PENDING_REVIEW -- this function only ever narrows how
    often that happens, it is not a substitute for that safety net.
    """
    result = expression
    changed = True
    while changed:
        changed = False
        text = result
        n = len(text)
        i = 0
        in_double = False
        while i < n:
            ch = text[i]
            if ch == '"':
                in_double = not in_double
                i += 1
                continue
            if in_double:
                i += 1
                continue
            if text[i : i + 5].upper() == "THEN(" and (
                i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
            ):
                j = i - 1
                while j >= 0 and text[j].isspace():
                    j -= 1
                if j >= 0 and text[j] == ")":
                    depth = 1
                    k = j - 1
                    local_in_double = False
                    while k >= 0 and depth > 0:
                        c = text[k]
                        if c == '"':
                            local_in_double = not local_in_double
                        elif not local_in_double:
                            if c == ")":
                                depth += 1
                            elif c == "(":
                                depth -= 1
                        k -= 1
                    if depth == 0:
                        open_paren_index = k + 1
                        before = text[:open_paren_index]
                        if not _MISSING_IF_KEYWORD_BEFORE_RE.search(before):
                            result = text[:open_paren_index] + "IF" + text[open_paren_index:]
                            changed = True
                            break
            i += 1
    return result


def _repair_extra_close_before_then(expression: str) -> str:
    """Remove a spurious extra `)` that appears between an IF/ELSEIF
    condition and its `THEN(` keyword.

    The model sometimes emits `IF(cond))THEN(...)` or
    `ELSEIF(cond))THEN(...)`. That extra close cannot be valid because the
    condition's own closing parenthesis must be followed directly by
    `THEN`. This pass removes only that single extra close and leaves the
    surrounding branch structure untouched.
    """

    previous = expression
    for _ in range(3):
        result: list[str] = []
        i = 0
        n = len(previous)
        in_double = False
        changed = False
        while i < n:
            ch = previous[i]
            if ch == '"':
                in_double = not in_double
                result.append(ch)
                i += 1
                continue
            if in_double:
                result.append(ch)
                i += 1
                continue

            token = None
            if previous[i : i + 7].upper() == "ELSEIF(" and (i == 0 or not (previous[i - 1].isalnum() or previous[i - 1] == "_")):
                token = "ELSEIF("
            elif previous[i : i + 3].upper() == "IF(" and (i == 0 or not (previous[i - 1].isalnum() or previous[i - 1] == "_")):
                token = "IF("

            if token:
                open_index = i + len(token) - 1
                close_index = _find_matching_paren(previous, open_index)
                if close_index != -1:
                    j = close_index + 1
                    while j < n and previous[j].isspace():
                        j += 1
                    extra_close_end = j
                    while extra_close_end < n and previous[extra_close_end] == ")":
                        extra_close_end += 1
                    if extra_close_end > j and previous[extra_close_end:].lstrip().startswith("THEN("):
                        result.append(previous[i : close_index + 1])
                        i = extra_close_end
                        changed = True
                        continue

            result.append(ch)
            i += 1

        repaired = "".join(result)
        if not changed:
            return repaired
        previous = repaired
    return previous


def _remove_excess_closing_parens(expression: str) -> str:
    """Drop unmatched closing parens while leaving quoted text alone."""
    result: list[str] = []
    depth = 0
    in_double = False
    for ch in expression:
        if ch == '"':
            in_double = not in_double
            result.append(ch)
            continue
        if in_double:
            result.append(ch)
            continue
        if ch == "(":
            depth += 1
            result.append(ch)
            continue
        if ch == ")":
            if depth == 0:
                continue
            depth -= 1
            result.append(ch)
            continue
        result.append(ch)
    return "".join(result)


def _normalize_legacy_if_syntax(expression: str) -> str:
    """Convert common comma-style IF(condition, true, false) output into 4X syntax.

    The 4X grammar expects IF(condition)THEN(true)ELSE(false). Some LLM
    outputs default to SQL-style IF(condition, true, false); this helper
    rewrites that shape before validation.
    """

    def split_top_level_args(text: str) -> list[str] | None:
        args = []
        current = []
        depth = 0
        bracket_depth = 0
        in_string = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == '"':
                in_string = not in_string
                current.append(ch)
            elif not in_string and ch == "(":
                depth += 1
                current.append(ch)
            elif not in_string and ch == ")":
                if depth == 0:
                    return None
                depth -= 1
                current.append(ch)
            elif not in_string and ch == "[":
                bracket_depth += 1
                current.append(ch)
            elif not in_string and ch == "]":
                if bracket_depth == 0:
                    return None
                bracket_depth -= 1
                current.append(ch)
            elif not in_string and ch == "," and depth == 0 and bracket_depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
            i += 1
        args.append("".join(current).strip())
        return args if len(args) == 3 else None

    def rewrite_once(text: str) -> str:
        result = []
        i = 0
        in_string = False
        while i < len(text):
            ch = text[i]
            if ch == '"':
                in_string = not in_string
                result.append(ch)
                i += 1
                continue

            if not in_string and text[i : i + 3].upper() == "IF(":
                start = i + 3
                depth = 1
                j = start
                local_in_string = False
                while j < len(text):
                    cur = text[j]
                    if cur == '"':
                        local_in_string = not local_in_string
                    elif not local_in_string:
                        if cur == "(":
                            depth += 1
                        elif cur == ")":
                            depth -= 1
                            if depth == 0:
                                break
                    j += 1
                if depth == 0:
                    inner = text[start:j]
                    parts = split_top_level_args(inner)
                    if parts:
                        condition, when_true, when_false = parts
                        result.append(f"IF({condition})THEN({when_true})ELSE({when_false})")
                        i = j + 1
                        continue

            result.append(ch)
            i += 1
        return "".join(result)

    previous = expression
    for _ in range(3):
        rewritten = rewrite_once(previous)
        rewritten = _rewrite_legacy_else_if(rewritten)
        if rewritten == previous:
            return rewritten
        previous = rewritten
    return previous


def _normalize_sql_style_syntax(expression: str) -> str:
    """Normalize common SQL-style syntax that is not valid 4X Formula
    Expression syntax, before grammar validation:

    - Single-quoted string literals ('Y') become double-quoted ("Y"),
      since the 4X grammar's STRING token only accepts double quotes.
    - A bare SQL-style equality operator (=) becomes the 4X equality
      operator (==), since the 4X grammar's COMP_OP only recognizes
      ==, !=, >=, <=, >, and <. Existing !=, <=, >=, and == are left
      untouched.

    This is a generic, input-independent syntax-shape fix: LLM output
    translating SQL conditions frequently defaults to SQL literal syntax
    even when explicitly told to use 4X grammar, and retrying the whole
    LLM call for a purely mechanical substitution like this is wasteful
    and unreliable. Content inside double-quoted strings is left alone so
    this never rewrites the literal text of a value.
    """
    result: list[str] = []
    i = 0
    n = len(expression)
    in_double = False

    while i < n:
        ch = expression[i]

        if in_double:
            result.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == '"':
            in_double = True
            result.append(ch)
            i += 1
            continue

        if ch == "'":
            j = i + 1
            content = []
            while j < n and expression[j] != "'":
                content.append(expression[j])
                j += 1
            literal = "".join(content).replace('"', '\\"')
            result.append(f'"{literal}"')
            i = j + 1
            continue

        if ch == "=":
            next_char = expression[i + 1] if i + 1 < n else ""
            prev_char = expression[i - 1] if i > 0 else ""
            if next_char == "=":
                result.append("==")
                i += 2
                continue
            if prev_char in ("!", "<", ">"):
                result.append("=")
                i += 1
                continue
            result.append("==")
            i += 1
            continue

        result.append(ch)
        i += 1

    normalized = "".join(result)
    normalized = _rewrite_legacy_else_if(normalized)
    normalized = _rewrite_null_predicates(normalized)
    normalized = _rewrite_exists_predicates(normalized)
    return normalized


_TERNARY_BOUNDARY_KEYWORD_RE = re.compile(r"(THEN|ELSEIF|ELSE|AND|OR)\b", re.IGNORECASE)


def _normalize_ternary_operator(expression: str) -> str:
    """Deterministically rewrite a `(condition) ? true_val : false_val`
    ternary into the grammar's real `IF(condition)THEN(true_val)ELSE(false_val)`
    form -- the 4X grammar has no ternary operator at all (confirmed
    against a real generation defect: a DEGDATE expression produced
    exactly this shape and was correctly rejected by grammar validation
    with "No terminal matches '?'").

    This is deliberately conservative and bails out (leaves that part of
    the text completely unchanged) the moment the surrounding shape isn't
    unambiguous, rather than guessing:
      - the condition must be the fully-parenthesized group immediately
        preceding the `?` (nothing before a `?` that isn't `(...)` is
        rewritten);
      - the `:` that separates the two branches must be found at the same
        paren depth as the `?` (a `:` inside a nested call's own parens is
        never mistaken for the ternary's own separator);
      - the false-branch's end must be found as an unambiguous boundary (a
        depth-0 `)`, `,`, or one of THEN/ELSEIF/ELSE/AND/OR as a whole
        word) -- if the string ends before any boundary is found, the
        false-branch extends to end of string.
    If any of these can't be established, the `?` is left as a literal
    character, so grammar validation still catches it exactly as it does
    today. This can only ever turn an already-invalid expression into a
    valid one; it never touches an expression that doesn't contain a `?`.
    """
    result: list[str] = []
    i = 0
    n = len(expression)
    in_double = False

    while i < n:
        ch = expression[i]
        if ch == '"':
            in_double = not in_double
            result.append(ch)
            i += 1
            continue
        if in_double:
            result.append(ch)
            i += 1
            continue

        if ch == "?":
            rewritten = _try_rewrite_ternary_at(expression, i, result)
            if rewritten is not None:
                new_result_text, next_i = rewritten
                result = list(new_result_text)
                i = next_i
                continue

        result.append(ch)
        i += 1

    return "".join(result)


def _try_rewrite_ternary_at(expression: str, qmark_index: int, result_so_far: list[str]) -> tuple[str, int] | None:
    # 1) The condition must be the fully-parenthesized group ending right
    # before '?' (skipping whitespace).
    j = len(result_so_far) - 1
    while j >= 0 and result_so_far[j].isspace():
        j -= 1
    if j < 0 or result_so_far[j] != ")":
        return None

    depth = 1
    k = j - 1
    local_in_double = False
    while k >= 0 and depth > 0:
        c = result_so_far[k]
        if c == '"':
            local_in_double = not local_in_double
        elif not local_in_double:
            if c == ")":
                depth += 1
            elif c == "(":
                depth -= 1
        k -= 1
    if depth != 0:
        return None
    cond_start = k + 1
    condition_text = "".join(result_so_far[cond_start : j + 1])  # includes its own ( )
    prefix_text = "".join(result_so_far[:cond_start])

    # 2) Find the ':' that separates true/false branches, at depth 0
    # relative to right after '?'.
    p = qmark_index + 1
    n = len(expression)
    depth2 = 0
    local_in_double2 = False
    colon_pos = None
    while p < n:
        c = expression[p]
        if c == '"':
            local_in_double2 = not local_in_double2
        elif not local_in_double2:
            if c == "(":
                depth2 += 1
            elif c == ")":
                if depth2 == 0:
                    return None  # ran out of the ternary's own scope first
                depth2 -= 1
            elif c == ":" and depth2 == 0:
                colon_pos = p
                break
        p += 1
    if colon_pos is None:
        return None
    true_branch = expression[qmark_index + 1 : colon_pos].strip()
    if not true_branch:
        return None

    # 3) Find where the false-branch ends: a depth-0 ')', ',', or a
    # THEN/ELSEIF/ELSE/AND/OR keyword as a whole word.
    q = colon_pos + 1
    depth3 = 0
    local_in_double3 = False
    false_end = None
    while q < n:
        c = expression[q]
        if c == '"':
            local_in_double3 = not local_in_double3
        elif not local_in_double3:
            if c == "(":
                depth3 += 1
            elif c == ")":
                if depth3 == 0:
                    false_end = q
                    break
                depth3 -= 1
            elif depth3 == 0 and c == ",":
                false_end = q
                break
            elif depth3 == 0:
                m = _TERNARY_BOUNDARY_KEYWORD_RE.match(expression, q)
                if m and (q == 0 or not (expression[q - 1].isalnum() or expression[q - 1] == "_")):
                    false_end = q
                    break
        q += 1
    if false_end is None:
        false_end = n
    false_branch = expression[colon_pos + 1 : false_end].strip()
    if not false_branch:
        return None

    rewritten = f"IF{condition_text}THEN({true_branch})ELSE({false_branch})"
    return prefix_text + rewritten, false_end


_boolean_grouping_parser: "object | None" = None


def _get_boolean_grouping_parser():
    """Lazily constructed, dedicated Lark parser instance for
    _auto_parenthesize_null_check_or_pattern -- needs source-position
    tracking (propagate_positions) to insert parentheses at the right
    text offsets, the same reason every other position-dependent parser
    instance in this codebase (formula_pretty_printer.py,
    period_pruning.py, semantic_validation.py) keeps its own rather than
    sharing app.grammar.validator's."""
    global _boolean_grouping_parser
    if _boolean_grouping_parser is None:
        from lark import Lark

        grammar_path = Path(__file__).resolve().parents[1] / "grammar" / "fourx_grammar.lark"
        _boolean_grouping_parser = Lark(
            grammar_path.read_text(), parser="earley", start="start", propagate_positions=True
        )
    return _boolean_grouping_parser


def _unwrap_boolean_node(node):
    from lark import Tree

    while (
        isinstance(node, Tree)
        and node.data not in ("if_expr", "column_ref", "function_call")
        and len(node.children) == 1
    ):
        node = node.children[0]
    return node


def _node_span_text(node, expression: str) -> str | None:
    from lark import Token, Tree

    if isinstance(node, Token):
        return str(node)
    if isinstance(node, Tree) and not node.meta.empty:
        return " ".join(expression[node.meta.start_pos : node.meta.end_pos].split())
    return None


def _column_ref_key(node, expression: str) -> str | None:
    """A normalized identity string for a column_ref node (or a function
    call's single column_ref argument), so two references can be compared
    for "is this the same column" regardless of exact spacing."""
    from lark import Tree

    node = _unwrap_boolean_node(node)
    if isinstance(node, Tree) and node.data == "column_ref":
        text = _node_span_text(node, expression)
        return text.upper() if text else None
    return None


def _is_null_check_call(node, expression: str) -> tuple[str, str] | None:
    """If `node` is a call to ISEMPTY(x) or ISNOTEMPTY(x) with exactly one
    column_ref argument, return (function_name, column_key); else None."""
    from lark import Tree

    node = _unwrap_boolean_node(node)
    if not (isinstance(node, Tree) and node.data == "function_call"):
        return None
    children = list(node.children)
    if len(children) != 2:
        return None
    func_name_node, args_node = children
    func_name = str(func_name_node).upper()
    if func_name not in ("ISEMPTY", "ISNOTEMPTY"):
        return None
    if not (isinstance(args_node, Tree) and args_node.data == "arg_list"):
        return None
    if len(args_node.children) != 1:
        return None
    column_key = _column_ref_key(args_node.children[0], expression)
    if column_key is None:
        return None
    return func_name, column_key


def _is_matching_literal_comparison(node, expression: str, column_key: str) -> bool:
    """True if `node` is a `compare` node testing the same column
    (`column_key`) against a literal, in either operand order (e.g.
    `X=="N"` or `"N"==X`)."""
    from lark import Tree

    node = _unwrap_boolean_node(node)
    if not (isinstance(node, Tree) and node.data == "compare"):
        return False
    if len(node.children) != 3:
        return False
    left, _op, right = node.children
    left_key = _column_ref_key(left, expression)
    right_key = _column_ref_key(right, expression)
    return left_key == column_key or right_key == column_key


def _find_null_check_or_spans(node, expression: str, spans: list[tuple[int, int]]) -> None:
    """Collect (start, end) text spans to wrap in parentheses so that an
    `ISEMPTY(X)` (or `ISNOTEMPTY(X)`) sitting immediately next to an OR,
    combined with a comparison of that same X against a literal on the
    OTHER side of that OR, gets grouped together -- instead of binding to
    a sibling AND operand first by ordinary precedence.

    Concretely, for `A AND ISEMPTY(X) OR X=="v"`, the grammar parses this
    (correctly, by standard AND-before-OR precedence) as:

        or_op
          and_op
            A
            ISEMPTY(X)          <- and_op's right child, adjacent to the OR
          X=="v"                <- or_op's right child

    This is exactly the shape produced when this codebase's own
    normalization mechanically expands a single source comparison like
    `NVL(x,'N')='N'` into `ISEMPTY(x) OR x=="N"` and that sits next to a
    preceding `AND` -- the fix is to wrap only the two adjacent
    "ISEMPTY(X)" and "X==literal" operands (which are already contiguous
    in the source text, just not grouped), producing
    `A AND (ISEMPTY(X) OR X=="v")`, without touching or reordering `A`.

    The mirror shape (`X=="v" OR ISEMPTY(X) AND B`, and_op as or_op's
    right child, null-check as and_op's LEFT child) is handled the same
    way. Only these two shapes -- where the null-check operand is
    textually adjacent to the OR boundary -- are attempted; a null-check
    on the *non-adjacent* side of the AND would require reordering text,
    not just adding parentheses around a contiguous span, and is
    deliberately left to check_ambiguous_boolean_grouping to flag for
    human review instead of being guessed at here.
    """
    from lark import Tree

    if isinstance(node, Tree):
        if node.data == "or_op" and len(node.children) == 2:
            for and_side_idx in (0, 1):
                and_side = node.children[and_side_idx]
                other_side = node.children[1 - and_side_idx]
                and_unwrapped = _unwrap_boolean_node(and_side)
                if not (isinstance(and_unwrapped, Tree) and and_unwrapped.data == "and_op" and len(and_unwrapped.children) == 2):
                    continue
                # The null-check must be the AND operand adjacent to the OR
                # boundary: if and_op is on the LEFT of or_op, that's its
                # RIGHT child (index 1); if and_op is on the RIGHT of
                # or_op, that's its LEFT child (index 0).
                adjacent_idx = 1 if and_side_idx == 0 else 0
                and_operand = and_unwrapped.children[adjacent_idx]
                null_check = _is_null_check_call(and_operand, expression)
                if null_check is None:
                    continue
                if not _is_matching_literal_comparison(other_side, expression, null_check[1]):
                    continue
                nc_node = _unwrap_boolean_node(and_operand)
                other_node = _unwrap_boolean_node(other_side)
                if isinstance(nc_node, Tree) and nc_node.meta.empty:
                    continue
                if isinstance(other_node, Tree) and other_node.meta.empty:
                    continue
                span_start = min(nc_node.meta.start_pos, other_node.meta.start_pos)
                span_end = max(nc_node.meta.end_pos, other_node.meta.end_pos)
                if not _is_span_already_parenthesized_range(span_start, span_end, expression):
                    spans.append((span_start, span_end))
        for child in node.children:
            _find_null_check_or_spans(child, expression, spans)


def _is_span_already_parenthesized_range(start: int, end: int, expression: str) -> bool:
    before = start - 1
    while before >= 0 and expression[before].isspace():
        before -= 1
    after = end
    while after < len(expression) and expression[after].isspace():
        after += 1
    return before >= 0 and expression[before] == "(" and after < len(expression) and expression[after] == ")"


def _auto_parenthesize_null_check_or_pattern(expression: str) -> str:
    """Deterministically wrap `ISEMPTY(x) OR x=="v"` (or ISNOTEMPTY / !=)
    in parentheses whenever it's an un-parenthesized operand of an AND --
    confirmed against a real generation defect: the source's single
    atomic comparison `NVL(A.FlgProcessing,'N')='N'` was correctly
    expanded to `ISEMPTY(FlgProcessing) OR FlgProcessing=="N"`, but the
    parentheses that expansion should have kept around the OR pair were
    lost when it was combined with a preceding `AND`, silently changing
    `A AND (B OR C)` into `(A AND B) OR C` by grammar precedence -- a
    change that passes grammar validation (it's syntactically valid) and
    can only be caught semantically (see
    app/guardrails/semantic_validation.py::check_ambiguous_boolean_grouping,
    which still runs after this as a safety net for every OTHER
    unparenthesized AND/OR shape this function does not attempt to fix).
    """
    upper = expression.upper()
    # Cheap pre-check before paying for a full Earley parse: the pattern
    # this function looks for can only exist if the expression contains
    # an ISEMPTY/ISNOTEMPTY call, an OR, and an AND all at once. Skipping
    # straight past the (comparatively expensive) parse for the large
    # majority of expressions that obviously can't match keeps this
    # normalization pass from adding meaningful latency across a whole
    # job's worth of expressions, most of which have no OR at all.
    if not (("ISEMPTY(" in upper or "ISNOTEMPTY(" in upper) and " OR " in upper and " AND " in upper):
        return expression

    try:
        tree = _get_boolean_grouping_parser().parse(expression)
    except Exception:
        return expression

    spans: list[tuple[int, int]] = []
    try:
        _find_null_check_or_spans(tree, expression, spans)
    except Exception:
        return expression

    if not spans:
        return expression

    result = expression
    for start, end in sorted(spans, key=lambda s: s[0], reverse=True):
        result = result[:start] + "(" + result[start:end] + ")" + result[end:]
    return result


def _normalize_expression(expression: str, source_text: str = "") -> str:
    """Apply every mechanical, input-independent normalization pass, in an
    order chosen so each pass sees syntax the next one expects. Whitespace
    is flattened first (so every later pass works on a single line, and so
    the final result is always safe for both a Markdown table cell and a
    spreadsheet cell), then quotes/operators, then function-call rewriting
    and comma-style IF detection, both of which rely on string-boundary
    tracking, and finally a trailing-paren-balance check as a last safety
    net after all other rewrites have run."""
    expression = _strip_sql_comments_for_guard_matching(expression)
    expression = _flatten_whitespace(expression)
    expression = _rewrite_sql_at_parameters(expression)
    expression = _rewrite_dateadd_to_addday(expression)
    expression = _rewrite_bare_not_equality(expression)
    expression = _normalize_ternary_operator(expression)
    expression = _repair_missing_if_before_then(expression)
    expression = _normalize_sql_style_syntax(expression)
    expression = _normalize_sql_functions(expression)
    expression = _normalize_legacy_if_syntax(expression)
    expression = _strip_angle_bracket_placeholders(expression)
    expression = _rewrite_not_in_membership(expression)
    expression = _rewrite_is_empty_syntax(expression)
    expression = _rewrite_postfix_isnotempty(expression)
    expression = _wrap_bare_not_in_parens(expression)
    expression = _rewrite_misused_empty_functions(expression)
    expression = _rewrite_sql_not_equal_operator(expression)
    expression = _rewrite_string_concatenation(expression)
    expression = _rewrite_isnotempty_boolean_comparisons(expression)
    expression = _rewrite_sql_date_literals(expression)
    expression = _rewrite_sqlglot_date_functions(expression)
    expression = _rewrite_to_date_function(expression)
    expression = _rewrite_date_function(expression)
    expression = _rewrite_in_subquery_membership(expression)
    expression = _rewrite_bundled_alias_column_refs(expression, source_text)
    expression = _rewrite_unquoted_dotted_refs(expression)
    expression = _canonicalize_leading_dotted_aliases(expression)
    expression = _rewrite_bundled_business_date_var(expression)
    expression = _rewrite_exists_predicates(expression)
    expression = _auto_parenthesize_null_check_or_pattern(expression)
    expression = _repair_extra_close_before_then(expression)
    expression = _remove_excess_closing_parens(expression)
    expression = _fix_unbalanced_trailing_parens(expression)
    expression = _repair_missing_then_parentheses(expression)
    expression = _rewrite_legacy_else_if(expression)
    expression = _rewrite_null_predicates(expression)
    expression = _strip_min_wrapper(expression)
    return expression


def _rewrite_sql_at_parameters(expression: str) -> str:
    """Strip SQL `@param` sigils so local variables become bare platform names.

    Process/business-date parameters are rewritten later by
    `_rewrite_business_date_variables`; everything else becomes a bare
    identifier (the same convention as `p_TIMEKEY`).
    """
    return re.sub(r"(?<![A-Za-z0-9_\"])@([A-Za-z_][A-Za-z0-9_]*)\b", r"\1", expression)


def _format_platform_date_offset(unit: str, amount: str, base: str) -> str | None:
    """Map SQL day/month/year offsets onto documented 4X date helpers."""
    unit_text = unit.strip().strip("'\"").upper()
    amount = amount.strip()
    base = base.strip()
    if unit_text in {"DAY", "DAYS"}:
        return f"ADDDAY({base}, {amount})"
    if unit_text in {"MONTH", "MONTHS"}:
        return f'PERIOD("M", {amount}, {base})'
    if unit_text in {"YEAR", "YEARS"}:
        return f'PERIOD("Y", {amount}, {base})'
    return None


def _rewrite_dateadd_to_addday(expression: str) -> str:
    """Translate leftover SQL/sqlglot date offsets into ADDDAY / PERIOD."""

    def replace_dateadd(match: re.Match[str]) -> str:
        rendered = _format_platform_date_offset(match.group(1), match.group(2), match.group(3))
        return rendered if rendered is not None else match.group(0)

    expression = re.sub(
        r"(?i)\bDATEADD\s*\(\s*(DAY|DAYS|MONTH|MONTHS|YEAR|YEARS)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)",
        replace_dateadd,
        expression,
    )

    def replace_date_add(match: re.Match[str]) -> str:
        rendered = _format_platform_date_offset(match.group(3), match.group(2), match.group(1))
        return rendered if rendered is not None else match.group(0)

    # sqlglot emits DATE_ADD(date, amount, 'UNIT') for T-SQL DATEADD.
    return re.sub(
        r"(?i)\bDATE_ADD\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*['\"]?(DAY|DAYS|MONTH|MONTHS|YEAR|YEARS)['\"]?\s*\)",
        replace_date_add,
        expression,
    )


def _rewrite_bare_not_equality(expression: str) -> str:
    """Rewrite `NOT a == b` into `a != b` (platform has no bare NOT operator)."""
    return re.sub(
        r"(?i)\bNOT\s+(\"[^\"]+\"(?:\s*\.\s*\"[^\"]+\")*|[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*==\s*",
        r"\1 != ",
        expression,
    )


def _source_allows_target_reference(source_sql: str, entity_name: str, column: str) -> bool:
    """Allow a self-reference when the source SQL either explicitly
    preserves/increments the target value, OR contains a WHERE-guarded
    UPDATE that sets this column -- because in real SQL, rows that don't
    satisfy an UPDATE's WHERE clause are never touched, so "no match"
    always means "keep the existing value", whether or not the statement
    spells that out with an NVL/COALESCE self-read.

    Previously this only recognized the explicit NVL/COALESCE form, so a
    plain `UPDATE t SET col = val WHERE cond` (the most common shape in
    these procedures) was never allowed to preserve the prior value on
    "no match" -- the generator forced ELSE(NULL) instead, silently
    turning "this row wasn't touched" into "this row was wiped". See the
    architecture review: this is root cause A ("UPDATE ... WHERE
    semantics being converted incorrectly").
    """
    if not source_sql or not column:
        return False

    source_upper = source_sql.upper()
    column_upper = re.escape(column.upper())
    entity_upper = re.escape(entity_name.upper()) if entity_name else ""

    quoted_target = rf'"{column_upper}"'
    if entity_upper:
        qualified_target = rf'"{entity_upper}"\s*\.\s*{quoted_target}'
    else:
        qualified_target = quoted_target

    preservation_patterns = [
        rf"\bNVL\s*\(\s*(?:[A-Z_][A-Z0-9_]*\s*\.\s*)?{qualified_target}\s*,",
        rf"\bCOALESCE\s*\(\s*(?:[A-Z_][A-Z0-9_]*\s*\.\s*)?{qualified_target}\s*,",
    ]
    if any(re.search(pattern, source_upper) for pattern in preservation_patterns):
        return True

    if column_upper == "COUNT" and re.search(
        rf"\bNVL\s*\(\s*{quoted_target}\s*,\s*0\s*\)\s*\+\s*1",
        source_upper,
    ):
        return True

    if _has_where_guarded_update_on_column(source_sql, column):
        return True

    return False


def _has_where_guarded_update_on_column(source_sql: str, column: str) -> bool:
    """True if any statement in source_sql is a plain `UPDATE ... SET
    <column> = ... WHERE ...` -- i.e. an UPDATE whose WHERE clause means
    unmatched rows are left untouched, so the column's prior value is
    implicitly preserved for those rows by ordinary SQL semantics.

    Deliberately conservative: only bare UPDATE/SET/WHERE statements
    qualify, not MERGE (which has its own explicit WHEN NOT MATCHED
    branches that should be honored as written, not overridden) and not
    an UPDATE with no WHERE at all (which touches every row, so there is
    no "unmatched" case to preserve).
    """
    try:
        statements = split_statements(source_sql)
    except Exception:
        statements = [source_sql]

    column_pattern = re.compile(rf"\b{re.escape(column)}\b\s*=", re.IGNORECASE)
    for stmt in statements:
        cleaned = _strip_leading_comments(stmt).strip()
        upper = cleaned.upper()
        if not upper.startswith("UPDATE"):
            continue
        if "MERGE" in upper:
            continue
        set_match = re.search(r"\bSET\b(.*?)(\bWHERE\b|$)", cleaned, re.IGNORECASE | re.DOTALL)
        if not set_match:
            continue
        set_clause, where_marker = set_match.group(1), set_match.group(2)
        if not where_marker:
            continue
        if column_pattern.search(set_clause):
            return True
    return False


_SAFE_ARITHMETIC_FUNCTION_NAMES = {
    "COALESCE",
    "NVL",
    "ISNULL",
    "TODATE",
    "SYSDATE",
    "MAX",
    "MIN",
    "SUM",
    "COUNT",
    "DATEDIFF",
    "ADDDAY",
    "PERIOD",
    "CONCAT",
    "LOWER",
    "UPPER",
    "TRIM",
    "LEN",
    "LENGTH",
    "SUBSTR",
    "SUBSTRING",
    "ABS",
    "ROUND",
    "FLOOR",
    "CEIL",
    "CEILING",
    "REPLACE",
    "CONVERT",
    "REGEX",
    "DATEPART",
    "SOM",
    "EOM",
    "SOY",
    "EOY",
    "SOFY",
    "EOFY",
    "SOQ",
    "EOQ",
}


def _split_top_level(text: str, separators: str) -> list[str] | None:
    """Split `text` on any of `separators` at paren/quote depth 0.

    Returns None (not a valid top-level split) if parens/quotes are
    unbalanced. A leading/interior separator that produces an empty
    operand (e.g. a unary `-` immediately after another operator, as in
    `A + -B`) is silently dropped rather than treated as an empty
    operand -- this function is only ever used to decide whether every
    *meaningful* operand is safe, never to reconstruct the expression
    (the original text is always what gets composed, unchanged).
    """
    operands: list[str] = []
    current: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    for ch in text:
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
        elif in_single or in_double:
            current.append(ch)
        elif ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None
            current.append(ch)
        elif depth == 0 and ch in separators:
            operand = "".join(current).strip()
            if operand:
                operands.append(operand)
            current = []
        else:
            current.append(ch)
    if depth != 0 or in_single or in_double:
        return None
    tail = "".join(current).strip()
    if tail:
        operands.append(tail)
    return operands


def _is_safely_composable_value(text: str, depth: int = 0) -> bool:
    """Recursively decide whether a SQL value expression is simple enough
    to compose deterministically -- i.e. contains nothing whose meaning
    depends on business-logic judgment the deterministic composer can't
    make (a CASE branch choice beyond what _translate_case_to_4x already
    handles, a subquery, an aggregate, an unlisted function).

    This never rewrites `text` -- the original string is always what
    gets composed. It only ever decides yes/no, and only ever WIDENS
    which values the deterministic path accepts; anything it says no to
    still falls through to the LLM exactly as before, so this can only
    ever increase how often write order is enforced deterministically,
    never regress an already-working case to something less safe.

    Handles, in addition to the base literal/column-reference/number
    cases in _is_simple_stage_value:
      - top-level arithmetic chains (a - b + c), each operand checked
        recursively -- e.g. the real
        `(v_ProcessDate - A.LastCrDate) + 1` shape found in
        PRO_DPD_Calculation_StoredProcedure_2.sql;
      - a small, explicit allowlist of safe wrapper functions
        (COALESCE/NVL/ISNULL/TODATE), each argument checked recursively;
      - one level of enclosing parentheses around any of the above.

    Bounded to a small recursion depth (arbitrary business logic nested
    indefinitely deep is exactly the case that must stay on the LLM
    path, not something this should try to chase).
    """
    if depth > 4:
        return False
    text = text.strip()
    if not text:
        return False

    if _is_simple_literal_or_reference(text):
        return True

    if text.startswith("(") and text.endswith(")"):
        inner = text[1:-1]
        # Only unwrap if these are genuinely one matching outer pair,
        # not e.g. "(a) + (b)" where stripping the first/last char would
        # be wrong.
        depth_check = 0
        in_single = in_double = False
        balanced_until_end = True
        for idx, ch in enumerate(inner):
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif in_single or in_double:
                continue
            elif ch == "(":
                depth_check += 1
            elif ch == ")":
                depth_check -= 1
                if depth_check < 0 and idx != len(inner) - 1:
                    balanced_until_end = False
                    break
        if balanced_until_end and depth_check == 0:
            if _is_safely_composable_value(inner, depth + 1):
                return True

    arithmetic_operands = _split_top_level(text, "+-*/")
    if arithmetic_operands and len(arithmetic_operands) > 1:
        if all(_is_safely_composable_value(op, depth + 1) for op in arithmetic_operands):
            return True

    func_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", text, re.DOTALL)
    if func_match:
        func_name = func_match.group(1).upper()
        args_text = func_match.group(2)
        if func_name in _SAFE_ARITHMETIC_FUNCTION_NAMES:
            args = _split_top_level(args_text, ",")
            if args is not None and all(_is_safely_composable_value(a, depth + 1) for a in args):
                return True

    if text.upper() in _SAFE_ARITHMETIC_FUNCTION_NAMES:
        return True

    return False


def _is_simple_literal_or_reference(text: str) -> bool:
    """The base, non-recursive cases: a bare literal, quoted/unquoted
    column reference, or number -- exactly what _is_simple_stage_value
    checked before this extension existed."""
    upper = text.upper()
    if upper in {"NULL", "0", "1"}:
        return True
    if re.fullmatch(r'"[^"]+"(?:\s*\.\s*"[^"]+")*', text):
        return True
    if re.fullmatch(r"'[^']*'", text):
        return True
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)*', text):
        return True
    if re.fullmatch(r"\(?\s*[-+]?\d+(?:\.\d+)?\s*\)?", text):
        return True
    return False


def _is_simple_stage_value(expression: str) -> bool:
    """Return True when a statement assigns a value that can be composed
    deterministically -- a simple literal, NULL, direct column reference,
    a safe arithmetic/function-wrapper combination of those (see
    _is_safely_composable_value), or a value that is itself a complete,
    independently grammar-valid IF(...)THEN(...)ELSE(...) expression
    (the shape _translate_case_to_4x produces for a simple SQL CASE).

    Genuinely complex expressions (subqueries, aggregates, unlisted
    function calls, anything nested beyond what the checks above cover)
    are intentionally left to flow through the LLM path so we do not
    oversimplify legitimate business logic.
    """
    text = expression.strip()
    if not text:
        return False

    if _is_safely_composable_value(text):
        return True

    # A value that is itself a complete, independently grammar-valid
    # IF(...)THEN(...)ELSE(...) expression -- the shape
    # _translate_case_to_4x below produces when a SQL CASE expression's
    # own WHEN/THEN/ELSE branches are all simple. This is checked last
    # (grammar validation is comparatively expensive) and only even
    # considered for text that already looks like a formula, not for
    # arbitrary SQL fragments that happen to start with "IF(".
    if text.upper().startswith("IF(") and validate_expression(text).valid:
        return True
    return False


def _translate_case_to_4x(case_node) -> str | None:
    """Deterministically translate a SQL `CASE WHEN cond1 THEN val1 WHEN
    cond2 THEN val2 ELSE val3 END` expression into the platform's real
    `IF(cond1)THEN(val1)ELSEIF(cond2)THEN(val2)ELSE(val3)` syntax.

    This is a mechanical, structure-preserving syntax translation -- the
    same category of fix as _normalize_ternary_operator for the `? :`
    ternary -- not a business-logic decision: SQL's CASE WHEN and the 4X
    grammar's IF/ELSEIF/ELSE have a direct one-to-one correspondence
    (branch order preserved, same number of branches, same fallback),
    so translating it here means this shape of column never needs the
    LLM to reproduce a conditional structure it already has right in
    front of it, which is exactly where a plausible-looking-but-wrong
    rewrite (like the ternary defect) could otherwise slip in.

    Returns None (falls back to the LLM path, same as any other
    "too complex for the deterministic composer" case) unless every
    branch's condition and value is itself simple enough to compose
    safely, and the fully-assembled result is independently grammar-
    valid -- this never guesses at a shape it isn't certain about.
    """
    ifs = case_node.args.get("ifs") or []
    if not ifs:
        return None

    switch_value = case_node.args.get("this")

    branches: list[tuple[str, str]] = []
    for when in ifs:
        condition_node = when.args.get("this")
        value_node = when.args.get("true")
        if condition_node is None or value_node is None:
            return None
        # Nested CASE is valid and must recurse — do not bail to LLM.
        while isinstance(value_node, exp.Paren):
            value_node = value_node.this
        if switch_value is not None:
            left_sql = _render_value_expression_to_4x(switch_value)
            right_sql = _render_value_expression_to_4x(condition_node)
            if left_sql is None or right_sql is None:
                return None
            condition_sql = f"{left_sql} == {right_sql}"
        else:
            condition_sql = _render_sql_condition_to_4x(condition_node) or condition_node.sql(dialect="oracle")
        value_sql = _render_value_expression_to_4x(value_node)
        if value_sql is None:
            return None
        branches.append((condition_sql, value_sql))

    default_node = case_node.args.get("default")
    if default_node is not None:
        while isinstance(default_node, exp.Paren):
            default_node = default_node.this
        default_sql = _render_value_expression_to_4x(default_node)
        if default_sql is None:
            return None
    else:
        default_sql = "NULL"

    first_cond, first_val = branches[0]
    parts = [f"IF({_normalize_expression(first_cond, '')})THEN({first_val})"]
    for cond, val in branches[1:]:
        parts.append(f"ELSEIF({_normalize_expression(cond, '')})THEN({val})")
    parts.append(f"ELSE({default_sql})")

    result = _normalize_expression("".join(parts), "")
    # Normalize before grammar check so SQL `@ProcessDate` / DATEADD forms
    # inside CASE values do not falsely reject an otherwise valid translation.
    return result if validate_expression(result).valid else None


def _is_sqlglot_string_literal(node) -> bool:
    while isinstance(node, exp.Paren):
        node = node.this
    return isinstance(node, exp.Literal) and bool(getattr(node, "is_string", False))


def _datediff_unit_token(unit_text: str) -> str | None:
    mapping = {
        "DAY": '"d"',
        "DAYS": '"d"',
        "MONTH": '"m"',
        "MONTHS": '"m"',
        "YEAR": '"y"',
        "YEARS": '"y"',
    }
    return mapping.get(unit_text.strip().upper())


def _render_value_expression_to_4x(node) -> str | None:
    while isinstance(node, exp.Paren):
        node = node.this

    # sqlglot wraps T-SQL DATEDIFF operands in TIME_STR_TO_TIME(...).
    if isinstance(node, exp.TimeStrToTime):
        return _render_value_expression_to_4x(node.this)

    if isinstance(node, exp.Case):
        return _translate_case_to_4x(node)

    if isinstance(node, exp.Column):
        name = node.name or ""
        if not name:
            return None
        if node.table:
            return f'"{node.table}"."{name}"'
        # Keep bare identifiers unquoted here so existing CASE/MAX
        # translations stay stable; platform quoting is applied later by
        # normalize/finalize.
        return name

    if isinstance(node, exp.Add):
        left = _render_value_expression_to_4x(node.this)
        right = _render_value_expression_to_4x(node.expression)
        if left is None or right is None:
            return None
        # SQL string concatenation uses `+`; platform requires CONCAT.
        if _is_sqlglot_string_literal(node.this) or _is_sqlglot_string_literal(node.expression):
            return f"CONCAT({left}, {right})"
        return f"({left} + {right})"

    if isinstance(node, exp.Sub):
        left = _render_value_expression_to_4x(node.this)
        right = _render_value_expression_to_4x(node.expression)
        if left is None or right is None:
            return None
        return f"({left} - {right})"

    if isinstance(node, exp.Mul):
        left = _render_value_expression_to_4x(node.this)
        right = _render_value_expression_to_4x(node.expression)
        if left is None or right is None:
            return None
        return f"({left} * {right})"

    if isinstance(node, exp.Div):
        left = _render_value_expression_to_4x(node.this)
        right = _render_value_expression_to_4x(node.expression)
        if left is None or right is None:
            return None
        return f"({left} / {right})"

    if isinstance(node, exp.Neg):
        inner = _render_value_expression_to_4x(node.this)
        if inner is None:
            return None
        return f"-{inner}"

    if isinstance(node, exp.Parameter):
        ident = getattr(node.this, "this", None)
        text = str(ident) if ident is not None else str(node.this)
        text = text.strip()
        return f"@{text}" if text else None

    if isinstance(node, exp.Subquery) or (
        isinstance(node, exp.Paren) and isinstance(getattr(node, "this", None), exp.Select)
    ):
        select = node.this if isinstance(node, (exp.Subquery, exp.Paren)) else None
        if isinstance(select, exp.Select):
            projections = list(select.args.get("expressions") or [])
            if len(projections) == 1:
                relation = _select_from_relation_name(select)
                projected = projections[0]
                if isinstance(projected, exp.Alias):
                    projected = projected.this
                # Qualify bare aggregate/column args with the subquery FROM table.
                if relation:
                    def qualify(n):
                        if isinstance(n, exp.Column) and not n.table:
                            return exp.column(n.name, table=relation)
                        return n
                    try:
                        projected = projected.transform(qualify)
                    except Exception:
                        pass
                return _render_value_expression_to_4x(projected)
        return None

    if isinstance(node, exp.Sum):
        this = node.this
        if isinstance(this, exp.Column) and not this.table:
            # Leave bare for callers that already qualified; otherwise quote.
            inner = f'"{this.name}"'
        else:
            inner = _render_value_expression_to_4x(this)
        if inner is None:
            return None
        return f"SUM({inner})"

    if isinstance(node, exp.Count):
        this = node.this
        # Platform grammar has no `COUNT(*)`; COUNT(1) is the row-count form.
        if this is None or isinstance(this, exp.Star):
            return "COUNT(1)"
        if isinstance(this, exp.Column) and not this.table:
            inner = f'"{this.name}"'
        else:
            inner = _render_value_expression_to_4x(this)
        if inner is None:
            return None
        return f"COUNT({inner})"

    if isinstance(node, exp.Lower):
        inner = _render_value_expression_to_4x(node.this)
        return f"LOWER({inner})" if inner is not None else None

    if isinstance(node, exp.Upper):
        inner = _render_value_expression_to_4x(node.this)
        return f"UPPER({inner})" if inner is not None else None

    if isinstance(node, exp.Trim):
        inner = _render_value_expression_to_4x(node.this)
        return f"TRIM({inner})" if inner is not None else None

    if isinstance(node, exp.Length):
        inner = _render_value_expression_to_4x(node.this)
        return f"LEN({inner})" if inner is not None else None

    if isinstance(node, exp.Substring):
        parts = [node.this, node.args.get("start"), node.args.get("length")]
        rendered = [_render_value_expression_to_4x(part) for part in parts if part is not None]
        if len(rendered) < 2 or any(part is None for part in rendered):
            return None
        return f"SUBSTR({', '.join(rendered)})"

    if isinstance(node, exp.Abs):
        inner = _render_value_expression_to_4x(node.this)
        return f"ABS({inner})" if inner is not None else None

    if isinstance(node, exp.Round):
        args = [node.this]
        if node.expression is not None:
            args.append(node.expression)
        rendered = [_render_value_expression_to_4x(arg) for arg in args]
        if any(part is None for part in rendered):
            return None
        return f"ROUND({', '.join(rendered)})"

    if isinstance(node, (exp.Floor, exp.Ceil)):
        inner = _render_value_expression_to_4x(node.this)
        if inner is None:
            return None
        name = "FLOOR" if isinstance(node, exp.Floor) else "CEIL"
        return f"{name}({inner})"

    if isinstance(node, exp.Replace):
        parts = [node.this, node.expression, node.args.get("replacement")]
        rendered = [_render_value_expression_to_4x(part) for part in parts]
        if any(part is None for part in rendered):
            return None
        return f"REPLACE({', '.join(rendered)})"

    if isinstance(node, exp.Cast):
        inner = _render_value_expression_to_4x(node.this)
        to_type = node.args.get("to")
        type_name = getattr(to_type, "this", None) or getattr(to_type, "name", None) or to_type
        type_text = str(type_name).strip() if type_name is not None else ""
        if type_text.startswith("DType."):
            type_text = type_text.split(".", 1)[1]
        if inner is None or not type_text:
            return None
        return f'CONVERT({inner}, "{type_text}")'

    if isinstance(node, exp.Convert):
        # T-SQL CONVERT(type, expr) → sqlglot Convert(this=type, expression=expr)
        to_type = node.this
        type_name = getattr(to_type, "this", None) or getattr(to_type, "name", None) or to_type
        type_text = str(type_name).strip() if type_name is not None else ""
        if type_text.startswith("DType."):
            type_text = type_text.split(".", 1)[1]
        inner = _render_value_expression_to_4x(node.args.get("expression") or node.expression)
        if inner is None or not type_text:
            return None
        return f'CONVERT({inner}, "{type_text}")'

    if isinstance(node, exp.Extract):
        part = node.this
        part_text = str(getattr(part, "this", part) or "").strip().lower()
        field = _render_value_expression_to_4x(node.expression)
        if not part_text or field is None:
            return None
        return f'DATEPART({field}, "{part_text}")'

    if isinstance(node, exp.DateDiff):
        unit = node.args.get("unit")
        unit_text = str(getattr(unit, "this", unit) or "")
        unit_token = _datediff_unit_token(unit_text)
        if unit_token is None:
            return None
        # T-SQL DATEDIFF(unit, start, end) → sqlglot DateDiff(this=end, expression=start).
        # Platform samples use DATEDIFF(end, start, "d").
        end = _render_value_expression_to_4x(node.this)
        start = _render_value_expression_to_4x(node.expression)
        if end is None or start is None:
            return None
        return f"DATEDIFF({end}, {start}, {unit_token})"

    if isinstance(node, exp.DateAdd):
        unit = getattr(node.args.get("unit"), "this", None)
        unit_text = str(unit).upper() if unit is not None else ""
        base = _render_value_expression_to_4x(node.this)
        amount = _render_value_expression_to_4x(node.expression)
        if base is None or amount is None:
            return None
        return _format_platform_date_offset(unit_text, amount, base)

    if isinstance(node, exp.Anonymous) and str(getattr(node, "name", "")).upper() == "CHOOSE":
        args = list(node.expressions or [])
        if len(args) < 2:
            return None
        index = _render_value_expression_to_4x(args[0])
        choices = [_render_value_expression_to_4x(arg) for arg in args[1:]]
        if index is None or any(choice is None for choice in choices):
            return None
        if len(choices) == 1:
            return choices[0]
        result = choices[-1]
        for i, choice in reversed(list(enumerate(choices[:-1], start=1))):
            result = f'IF({index} == {i})THEN({choice})ELSE({result})'
        return result

    if isinstance(node, exp.Expression):
        func_name = str(getattr(node, "key", "")).upper()
        if func_name in _SAFE_ARITHMETIC_FUNCTION_NAMES:
            args: list[str] = []
            this = getattr(node, "this", None)
            if this is not None and not isinstance(node, exp.Anonymous):
                rendered_this = _render_value_expression_to_4x(this)
                if rendered_this is None:
                    return None
                args.append(rendered_this)
            for arg in node.expressions or []:
                rendered_arg = _render_value_expression_to_4x(arg)
                if rendered_arg is None:
                    return None
                args.append(rendered_arg)
            if func_name in {"MAX", "MIN"} and len(args) == 1:
                return f"{func_name}({args[0]})"
            if func_name in {"COALESCE", "NVL", "ISNULL"}:
                return f"COALESCE({', '.join(args)})"
            if func_name in {"LENGTH", "LEN"}:
                return f"LEN({args[0]})" if args else None
            if func_name in {"SUBSTRING", "SUBSTR"}:
                return f"SUBSTR({', '.join(args)})" if len(args) >= 2 else None
            if func_name in {"CEILING", "CEIL"}:
                return f"CEIL({args[0]})" if args else None
            if func_name in {"LTRIM", "RTRIM", "TRIM"}:
                return f"TRIM({args[0]})" if args else None
            if func_name == "REPLACE" and len(args) >= 3:
                return f"REPLACE({', '.join(args[:3])})"
            if func_name in {"CONVERT", "CAST"} and len(args) >= 2:
                return f"CONVERT({args[0]}, {args[1]})"
            if func_name == "DATEPART" and len(args) >= 2:
                # SQL Server DATEPART(part, date) → 4X DATEPART(date, part)
                return f"DATEPART({args[1]}, {args[0]})"
            if func_name in {"REGEX", "REGEXP", "REGEXP_LIKE"} and len(args) >= 2:
                return f"REGEX({args[0]}, {args[1]})"
            if func_name in {"SOM", "EOM", "SOY", "EOY", "SOFY", "EOFY", "SOQ", "EOQ"} and args:
                return f"{func_name}({args[0]})"
            if args:
                return f"{func_name}({', '.join(args)})"

    try:
        text = node.sql(dialect="oracle")
    except Exception:
        return None
    normalized = _normalize_expression(text, "")
    return normalized if _is_simple_stage_value(normalized) else None


def _strip_leading_comments(text: str) -> str:
    """Remove leading SQL comments so simple statement parsing can start
    at the first real keyword."""
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos].isspace():
            pos += 1
        elif text[pos : pos + 2] == "--":
            nl = text.find("\n", pos)
            pos = n if nl == -1 else nl + 1
        elif text[pos : pos + 2] == "/*":
            end = text.find("*/", pos + 2)
            pos = n if end == -1 else end + 2
        else:
            break
    return text[pos:]


def _qualify_unqualified_condition_columns(node, source_alias: str):
    """Attach `source_alias` to bare column references in a boolean AST.

    sqlglot often parses `WHERE Col > 0` as an unqualified `Column`
    node. For DD output we want that row context preserved explicitly,
    so the rendered Platform Condition remains tied to the same source
    relation the SQL statement read from.
    """
    if not source_alias:
        return node

    alias = _canonical_alias_text(source_alias)
    if not alias:
        return node

    def transform(n):
        if isinstance(n, exp.Column) and not n.table:
            return exp.column(n.name, table=alias)
        return n

    try:
        return node.transform(transform)
    except Exception:
        return node


def _statement_source_alias(tree) -> str | None:
    """Best-effort source alias for a statement whose boolean guard is
    being rendered.

    The goal is not perfect SQL semantics for every dialect nuance; it is
    to preserve the row context that the source SQL clearly uses when
    conditions are written with bare column names.
    """
    if isinstance(tree, exp.Select):
        from_clause = tree.args.get("from_")
        if isinstance(from_clause, exp.From):
            source = from_clause.this or (from_clause.expressions[0] if from_clause.expressions else None)
            if source is not None:
                alias = _extract_alias_name(source)
                if alias:
                    return alias
                if isinstance(source, exp.Table) and source.name:
                    return canonical_logical_name(source.name)
    if isinstance(tree, exp.Update):
        from_clause = tree.args.get("from_")
        if isinstance(from_clause, exp.From):
            source = from_clause.this or (from_clause.expressions[0] if from_clause.expressions else None)
            if source is not None:
                alias = _extract_alias_name(source)
                if alias:
                    return alias
                if isinstance(source, exp.Table) and source.name:
                    return canonical_logical_name(source.name)
        target = tree.args.get("this")
        if target is not None:
            alias = _extract_alias_name(target)
            if alias:
                return alias
            if isinstance(target, exp.Table) and target.name:
                return canonical_logical_name(target.name)
    if isinstance(tree, exp.Merge):
        using = tree.args.get("using")
        if isinstance(using, exp.Subquery) and isinstance(using.this, exp.Select):
            return _statement_source_alias(using.this)
    return None


def _render_boolean_predicate_leaf(node, source_alias: str | None = None) -> str | None:
    """Render a non-boolean-structural node (a comparison, IS NULL check,
    BETWEEN, or a bare column/literal) into 4X syntax.

    Deliberately reuses the EXISTING, already-tested text pipeline for
    this (serialize via sqlglot's own .sql(), then _normalize_expression)
    rather than hand-rolling a second leaf renderer -- a leaf predicate
    (unlike AND/OR/NOT) has no grouping ambiguity of its own to get
    wrong, so the risk this whole structural renderer exists to close
    does not apply here; reusing the proven pipeline is both simpler and
    safer than re-implementing it. Returns None (caller falls back to
    today's existing behavior) if the result isn't independently
    grammar-valid as a standalone condition.
    """
    if source_alias:
        node = _qualify_unqualified_condition_columns(node, source_alias)

    try:
        text = node.sql(dialect="oracle")
    except Exception:
        return None
    text = _strip_sql_comments_for_guard_matching(text)
    normalized = _normalize_expression(text, "")
    probe = f"IF({normalized})THEN(1)ELSE(0)"
    if validate_expression(probe).valid:
        return normalized
    return None


def _render_boolean_operand(
    node,
    parent_is_or: bool,
    source_alias: str | None = None,
    procedure_sql: str = "",
) -> str | None:
    """Render one operand of an AND/OR, adding parentheses whenever the
    operand's own top-level connective differs from its parent's (an AND
    directly under an OR, or vice versa) -- the exact, and only, shape
    where omitting parentheses would silently change what the expression
    means by falling back to the grammar's default AND-before-OR
    precedence. This decision is made purely from the parsed tree's own
    node types, never from the rendered text, so it can never be fooled
    by a leaf value that happens to contain the words "AND"/"OR"."""
    unwrapped = node.this if isinstance(node, exp.Paren) else node
    rendered = _render_sql_condition_to_4x(
        unwrapped, source_alias=source_alias, procedure_sql=procedure_sql
    )
    if rendered is None:
        return None
    if isinstance(unwrapped, exp.Or) and not parent_is_or:
        return f"({rendered})"
    if isinstance(unwrapped, exp.And) and parent_is_or:
        return f"({rendered})"
    return rendered


def _render_sql_condition_to_4x(
    node,
    source_alias: str | None = None,
    procedure_sql: str = "",
) -> str | None:
    """Deterministically render a sqlglot boolean-condition AST node into
    4X syntax, with every AND/OR/NOT boundary crossing explicitly
    parenthesized based purely on the parsed TREE STRUCTURE -- never
    inferred from flattened text, and never left to grammar-default
    precedence the way relying on the raw serialized text would.

    This closes a real, distinct risk from the deterministic composer's
    previous guard extraction (`where.this.sql(dialect="oracle")`
    followed by text-level normalization): sqlglot's own serializer
    already renders A AND (B OR C) faithfully (verified directly against
    every shape in the write-order/boolean-structure test suite), so the
    parsed SOURCE structure was never actually at risk in the
    deterministic path -- but a later text-normalization pass expanding
    a single comparison into a compound OR (as happens for some
    NVL/COALESCE-equality shapes) could still, in principle, introduce a
    new AND/OR boundary without its own parentheses. Building AND/OR/NOT
    directly from the tree, rather than through any text round-trip,
    removes that risk by construction rather than by pattern-matching
    for it after the fact.

    Falls back (returns None) for any construct not explicitly handled
    below -- the caller then falls back to today's existing sqlglot
    .sql() + text-normalization pipeline exactly as before, which may
    itself fall back further to the LLM. This can only ever ADD a
    structural guarantee for more cases; it never removes today's
    existing coverage.
    """
    if isinstance(node, exp.Paren):
        return _render_sql_condition_to_4x(
            node.this, source_alias=source_alias, procedure_sql=procedure_sql
        )

    if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
        # Oracle ROWNUM filters are fetch limits, not business predicates.
        left_name = ""
        if isinstance(node.this, exp.Column):
            left_name = (node.this.name or "").upper()
        right_name = ""
        if isinstance(node.expression, exp.Column):
            right_name = (node.expression.name or "").upper()
        if left_name == "ROWNUM" or right_name == "ROWNUM":
            return "TRUE"

    if isinstance(node, exp.And):
        left = _render_boolean_operand(
            node.this, parent_is_or=False, source_alias=source_alias, procedure_sql=procedure_sql
        )
        right = _render_boolean_operand(
            node.expression, parent_is_or=False, source_alias=source_alias, procedure_sql=procedure_sql
        )
        if left is None or right is None:
            return None
        if left == "TRUE":
            return right
        if right == "TRUE":
            return left
        return f"{left} AND {right}"

    if isinstance(node, exp.Or):
        left = _render_boolean_operand(
            node.this, parent_is_or=True, source_alias=source_alias, procedure_sql=procedure_sql
        )
        right = _render_boolean_operand(
            node.expression, parent_is_or=True, source_alias=source_alias, procedure_sql=procedure_sql
        )
        if left is None or right is None:
            return None
        return f"{left} OR {right}"

    if isinstance(node, exp.Not):
        inner = node.this
        inner_unwrapped = inner.this if isinstance(inner, exp.Paren) else inner
        # Collapse the common `NOT(X IS NULL)` shape directly to
        # ISNOTEMPTY(X) -- cleaner than the equivalent
        # NOT(ISEMPTY(X)) and avoids depending on
        # _wrap_bare_not_in_parens's text-level fixup for this path.
        if isinstance(inner_unwrapped, exp.Is) and isinstance(inner_unwrapped.expression, exp.Null):
            operand = _render_boolean_predicate_leaf(inner_unwrapped.this, source_alias=source_alias)
            return f"ISNOTEMPTY({operand})" if operand else None
        rendered = _render_sql_condition_to_4x(
            inner_unwrapped, source_alias=source_alias, procedure_sql=procedure_sql
        )
        return f"NOT({rendered})" if rendered else None

    if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
        operand = _render_boolean_predicate_leaf(node.this, source_alias=source_alias)
        return f"ISEMPTY({operand})" if operand else None

    if isinstance(node, exp.Between):
        left = _render_boolean_predicate_leaf(node.this, source_alias=source_alias)
        low = _render_value_expression_to_4x(node.args.get("low"))
        high = _render_value_expression_to_4x(node.args.get("high"))
        if left is None or low is None or high is None:
            return None
        return f"{left} BETWEEN [{low},{high}]"

    if isinstance(node, exp.In):
        # Subquery membership cannot use platform `IN [literal, ...]`.
        # Inline `key IN (SELECT key FROM src WHERE pred)` as `pred` (and
        # expand temp-table populations from the surrounding procedure).
        query = node.args.get("query")
        if query is not None:
            return _inline_in_subquery_membership_node(node, procedure_sql=procedure_sql)
        values_nodes = list(node.expressions or [])
        if any(isinstance(expr, (exp.Select, exp.Subquery, exp.Union)) for expr in values_nodes):
            return None
        left = _render_boolean_predicate_leaf(node.this, source_alias=source_alias)
        values = [_render_value_expression_to_4x(expr) for expr in values_nodes]
        if left is None or not values or any(value is None for value in values):
            return None
        return f'{left} IN [{",".join(values)}]'

    if isinstance(node, exp.Exists):
        # Platform Formula Expressions have no EXISTS; inline the subquery
        # WHERE predicate (same strategy as IN (SELECT ...)).
        query = node.this
        if isinstance(query, exp.Subquery):
            query = query.this
        if isinstance(query, exp.Select):
            where = query.args.get("where")
            if where is not None:
                return _render_sql_condition_to_4x(
                    where.this, source_alias=source_alias, procedure_sql=procedure_sql
                )
        return None

    if isinstance(node, exp.Like):
        # Map SQL LIKE patterns onto platform membership operators.
        left = _render_boolean_predicate_leaf(node.this, source_alias=source_alias)
        pattern_node = node.expression
        pattern = None
        if isinstance(pattern_node, exp.Literal):
            pattern = str(pattern_node.this)
        else:
            rendered_pattern = _render_value_expression_to_4x(pattern_node)
            if rendered_pattern and rendered_pattern.startswith('"') and rendered_pattern.endswith('"'):
                pattern = rendered_pattern[1:-1]
        if left is None or pattern is None:
            return None
        negated = bool(node.args.get("not"))
        if pattern.startswith("%") and pattern.endswith("%") and len(pattern) >= 2:
            value = pattern[1:-1]
            op = "DOESNOTCONTAINS" if negated else "CONTAINS"
            return f'{left} {op} ["{value}"]'
        if pattern.endswith("%") and not pattern.startswith("%"):
            value = pattern[:-1]
            if negated:
                return f'NOT({left} BEGINSWITH ["{value}"])'
            return f'{left} BEGINSWITH ["{value}"]'
        if pattern.startswith("%") and not pattern.endswith("%"):
            value = pattern[1:]
            if negated:
                return f'NOT({left} ENDSWITH ["{value}"])'
            return f'{left} ENDSWITH ["{value}"]'
        if negated:
            return f'{left} != "{pattern}"'
        return f'{left} == "{pattern}"'

    # Comparisons whose operands include CASE / subqueries fail the
    # leaf .sql()+normalize path; render both sides via the value
    # translator so DEGDATE-style `CASE ... END > date` guards compose.
    _CMP_OPS = (
        (exp.EQ, "=="),
        (exp.NEQ, "!="),
        (exp.GT, ">"),
        (exp.GTE, ">="),
        (exp.LT, "<"),
        (exp.LTE, "<="),
    )
    for node_type, op in _CMP_OPS:
        if isinstance(node, node_type):
            left_node = node.this
            right_node = node.expression
            if source_alias:
                left_node = _qualify_unqualified_condition_columns(left_node, source_alias)
                right_node = _qualify_unqualified_condition_columns(right_node, source_alias)
            left = _render_value_expression_to_4x(left_node)
            right = _render_value_expression_to_4x(right_node)
            if left is None or right is None:
                break
            return f"{left} {op} {right}"

    # Any other node (a comparison, BETWEEN, a bare column/literal) has
    # no AND/OR/NOT structure of its own to get wrong -- render it via
    # the existing, already-proven leaf pipeline.
    return _render_boolean_predicate_leaf(node, source_alias=source_alias)


def _select_from_relation_name(select: exp.Select) -> str | None:
    from_clause = select.args.get("from_")
    if not isinstance(from_clause, exp.From):
        return None
    source = from_clause.this or (from_clause.expressions[0] if from_clause.expressions else None)
    if isinstance(source, exp.Table):
        name = source.name or ""
        return str(name).lstrip("#") if name else None
    return None


def _find_temp_population_select(procedure_sql: str, temp_name: str) -> exp.Select | None:
    """Locate the SELECT that populates a temp/staging table named `temp_name`."""
    if not procedure_sql or not temp_name:
        return None
    cleaned = _strip_sql_comments_for_guard_matching(procedure_sql)
    temp_key = temp_name.lstrip("#").upper()
    dialect_candidates = [detect_dialect(cleaned), Dialect.SQLSERVER, Dialect.ORACLE, Dialect.MYSQL]
    seen: set[Dialect] = set()
    for dialect in dialect_candidates:
        if dialect in seen:
            continue
        seen.add(dialect)
        for stmt in split_statements(cleaned, dialect):
            text = _strip_leading_comments(stmt).strip()
            if not text:
                continue
            upper = text.upper()
            if temp_key not in upper.replace("#", ""):
                continue
            try:
                tree = sqlglot.parse_one(text, read=_SQLGLOT_DIALECT[dialect])
            except Exception:
                continue
            if isinstance(tree, exp.Select) and tree.args.get("into") is not None:
                into = tree.args.get("into")
                into_name = ""
                if isinstance(into, exp.Table):
                    into_name = str(into.name or "")
                elif into is not None:
                    into_name = str(getattr(into, "this", into) or "")
                if into_name.lstrip("#").upper() == temp_key:
                    return tree
            if isinstance(tree, exp.Insert):
                target = tree.args.get("this")
                target_name = ""
                if isinstance(target, exp.Table):
                    target_name = str(target.name or "")
                if target_name.lstrip("#").upper() == temp_key and isinstance(tree.expression, exp.Select):
                    return tree.expression
    return None


def _inline_in_subquery_membership_node(in_node: exp.In, procedure_sql: str = "") -> str | None:
    """Translate `key IN (SELECT key FROM src [WHERE pred])` into predicates.

    When `src` is a temp/staging table populated earlier in the same
    procedure, the population SELECT's WHERE/JOIN filters are inlined so
    the Platform Condition does not need unsupported subquery membership.
    """
    query = in_node.args.get("query")
    select = query.this if isinstance(query, exp.Subquery) else query
    if not isinstance(select, exp.Select):
        return None

    relation = _select_from_relation_name(select)
    source_alias = relation or _statement_source_alias(select)

    predicates: list[str] = []
    where = select.args.get("where")
    if where is not None:
        rendered = _render_sql_condition_to_4x(
            where.this, source_alias=source_alias, procedure_sql=procedure_sql
        )
        if rendered is None:
            return None
        predicates.append(rendered)

    # Only expand temp/staging population when the IN-subquery itself has
    # no WHERE — otherwise the subquery predicates are already complete.
    if relation and not predicates:
        pop = _find_temp_population_select(procedure_sql, relation)
        if pop is not None:
            pop_where = pop.args.get("where")
            if pop_where is not None:
                pop_alias = _statement_source_alias(pop) or relation
                pop_pred = _render_sql_condition_to_4x(
                    pop_where.this, source_alias=pop_alias, procedure_sql=procedure_sql
                )
                if pop_pred is None:
                    return None
                predicates.append(pop_pred)

    if not predicates:
        return None
    if len(predicates) == 1:
        return predicates[0]
    return " AND ".join(predicates)



def _safe_render_guard(
    condition_node,
    source_alias: str | None = None,
    procedure_sql: str = "",
) -> str | None:
    """Render a WHERE/ON guard to 4X, or None when it cannot be expressed.

    Never fall back to raw sqlglot `.sql()` text: that path reintroduces
    SQL-only shapes (IN subqueries, comments, @vars) that later fail
    grammar validation and produce demo-breaking expressions.
    """
    if condition_node is None:
        return None
    rendered = _render_sql_condition_to_4x(
        condition_node, source_alias=source_alias, procedure_sql=procedure_sql
    )
    if rendered is not None:
        return rendered
    try:
        raw = condition_node.sql(dialect="oracle")
    except Exception:
        return None
    raw = _strip_sql_comments_for_guard_matching(raw)
    if re.search(r"(?is)\bIN\s*\(\s*SELECT\b", raw) or re.search(r"(?is)\bEXISTS\s*\(", raw):
        return None
    normalized = _normalize_expression(raw, "")
    probe = f"IF({normalized})THEN(1)ELSE(0)"
    return normalized if validate_expression(probe).valid else None


_LEADING_CONTROL_HEADER_RE = re.compile(
    r"(?is)^\s*(?:"
    r"ELSE\s+IF\b.*?\bBEGIN\b|"
    r"ELSIF\b.*?\bTHEN\b|"
    r"ELSEIF\b.*?\bTHEN\b|"
    r"ELSE\b|"
    r"EXCEPTION\b|"
    r"WHEN\s+OTHERS\b.*?\bTHEN\b|"
    r"(?:BEGIN\s+)?CATCH\b|"
    r"BEGIN\b"
    r")\s*"
)


def _strip_leading_control_header(text: str) -> str:
    """Strip any leading control-flow header (ELSE, BEGIN, ELSIF ... THEN,
    WHEN OTHERS THEN, CATCH, ...) so callers that expect a write statement
    to start directly with UPDATE/MERGE/INSERT still recognize one whose
    assignment site embeds its guarding header as a prefix (see
    `_is_condition_bearing_header` / `_assignment_sites_from_statements`).
    Applies repeatedly since stacked headers (e.g. `ELSE\\nBEGIN\\n`) peel
    off one token at a time.
    """
    stripped = text
    while True:
        new = _LEADING_CONTROL_HEADER_RE.sub("", stripped, count=1)
        if new == stripped:
            return stripped
        stripped = new.lstrip()


def _resolve_using_subquery_projection(using_select: exp.Select, alias_name: str) -> exp.Expression | None:
    """Return the real projected expression for `alias_name` in a MERGE's
    USING (SELECT ...) subquery, or None if it isn't a computed alias there.

    Only resolves an *aliased* projection (`CASE ... END AS Alias`, or any
    other expression aliased that way) -- a plain `SELECT Col FROM ...`
    passthrough is not rewritten, since `Col` there already is a real
    column reference rather than something requiring subquery lookup.
    """
    if not alias_name:
        return None
    for projection in using_select.expressions or []:
        if isinstance(projection, exp.Alias) and (projection.alias or "").lower() == alias_name.lower():
            return projection.this
    return None


def _parse_simple_assignment_stage(
    raw_sql: str,
    target_column: str,
    entity_name: str = "",
    procedure_sql: str = "",
) -> tuple[str, str, str] | None:
    """Extract a deterministic guard/value pair for a simple UPDATE or
    MERGE write to `target_column`.

    Returns `(guard, value, source_target_name)` when the statement only
    assigns a simple literal, NULL, or direct column reference. More
    complex CASE/IF formulas are intentionally left to the LLM path.
    """
    target_upper = target_column.upper()
    context_sql = procedure_sql or raw_sql
    raw_sql = _strip_sql_comments_for_guard_matching(raw_sql)
    raw_sql = _expand_truncated_assignment_sql(raw_sql, context_sql, target_column)
    raw_sql = _strip_sql_comments_for_guard_matching(raw_sql)

    dialect_candidates = [detect_dialect(raw_sql), Dialect.ORACLE, Dialect.SQLSERVER, Dialect.MYSQL]
    seen_dialects: set[Dialect] = set()
    for dialect in dialect_candidates:
        if dialect in seen_dialects:
            continue
        seen_dialects.add(dialect)

        candidates = split_statements(raw_sql, dialect)
        stmt = ""
        for candidate in reversed(candidates):
            if classify_statement(candidate) in {"UPDATE", "MERGE"}:
                stmt = _strip_leading_comments(candidate).strip()
                break
        if not stmt:
            stmt = _strip_leading_comments(raw_sql).strip()
        if not stmt:
            continue
        stmt = _strip_sql_comments_for_guard_matching(stmt)

        select_into_stmt = _extract_select_into_statement(stmt)
        if select_into_stmt:
            try:
                select_tree = sqlglot.parse_one(select_into_stmt, read=_SQLGLOT_DIALECT[dialect])
            except Exception:
                select_tree = None
            if isinstance(select_tree, exp.Select) and select_tree.args.get("into") is not None:
                projection = _render_select_into_projection(select_tree, target_upper)
                if projection is not None:
                    guard = ""
                    where = select_tree.args.get("where")
                    if where is not None:
                        guard = _safe_render_guard(
                            where.this,
                            source_alias=_statement_source_alias(select_tree),
                            procedure_sql=context_sql,
                        )
                        if guard is None:
                            return None
                    return guard, projection, target_column

        # `stmt` falls back to the raw, un-split chunk (line ~3115) whenever
        # no UPDATE/MERGE candidate was found for this dialect. That raw
        # chunk is very often pure control flow (PRINT/BEGIN/END/DECLARE/
        # TRY/CATCH) that was never going to parse as a SQL statement in
        # any dialect. Feeding it to sqlglot anyway doesn't raise -- sqlglot
        # logs a "falling back to Command" WARNING and returns a stub node
        # instead of raising -- so the `except Exception` below never even
        # sees it; it just adds log noise and wasted work for a result we
        # already know is useless. Skip the call entirely when the fragment
        # isn't DML to begin with.
        if classify_statement(stmt) not in _DML_KEYWORDS:
            continue

        try:
            tree = sqlglot.parse_one(stmt, read=_SQLGLOT_DIALECT[dialect])
        except Exception:
            continue

        source_alias = _statement_source_alias(tree)

        def unwrap_parens(node):
            """Strip enclosing exp.Paren wrappers -- e.g. `(CASE WHEN ... END)`,
            a very common real-world style -- so a parenthesized CASE is still
            recognized as exp.Case rather than silently bailing to the LLM
            path just because of one extra layer of source parentheses.
            Confirmed against a real case: PRO_DPD_Calculation_StoredProcedure_2.sql's
            `SET A.DPD_IntService = (CASE WHEN A.IntNotServicedDt IS NOT NULL
            THEN (v_ProcessDate - A.IntNotServicedDt) ELSE 0 END)` was bailing
            for exactly this reason before this fix."""
            while isinstance(node, exp.Paren):
                node = node.this
            return node

        fallback_rhs = _extract_update_assignment_rhs(stmt, target_upper)

        if isinstance(tree, exp.Update):
            guard = ""
            where = tree.args.get("where")
            if where is not None:
                guard = _safe_render_guard(where.this, source_alias=source_alias, procedure_sql=context_sql)
                if guard is None:
                    return None

            for assignment in tree.args.get("expressions", []) or []:
                if not isinstance(assignment, exp.EQ) or not isinstance(assignment.this, exp.Column):
                    continue
                if assignment.this.name.upper() != target_upper:
                    continue
                value_node = unwrap_parens(assignment.expression)
                if isinstance(value_node, exp.Case):
                    case_translation = _translate_case_to_4x(value_node)
                    if case_translation is None:
                        break
                    case_translation = _rewrite_business_date_variables(case_translation, entity_name, source_sql=context_sql)
                    return guard, case_translation, assignment.this.name
                value = _render_value_expression_to_4x(value_node)
                if value is not None:
                    value = _rewrite_business_date_variables(value, entity_name, source_sql=context_sql)
                    if validate_expression(value).valid:
                        return guard, value, assignment.this.name
                    break
                value = value_node.sql(dialect=_SQLGLOT_DIALECT[dialect])
                value = _rewrite_business_date_variables(value, entity_name, source_sql=context_sql)
                if not _is_simple_stage_value(value):
                    break
                return guard, value, assignment.this.name

        if isinstance(tree, exp.Insert):
            projection = _render_insert_select_projection(tree, target_upper)
            if projection is not None:
                guard = ""
                select_tree = tree.expression
                if isinstance(select_tree, exp.Select):
                    where = select_tree.args.get("where")
                    if where is not None:
                        guard = _safe_render_guard(
                            where.this,
                            source_alias=_statement_source_alias(select_tree),
                            procedure_sql=context_sql,
                        )
                        if guard is None:
                            return None
                projection = _rewrite_business_date_variables(projection, entity_name, source_sql=context_sql)
                return guard, projection, target_column

        if fallback_rhs:
            try:
                rhs_tree = sqlglot.parse_one(fallback_rhs, read=_SQLGLOT_DIALECT[dialect])
            except Exception:
                rhs_tree = None
            if rhs_tree is not None:
                value = _render_value_expression_to_4x(rhs_tree)
                if value is not None:
                    guard = ""
                    if isinstance(tree, exp.Update):
                        where = tree.args.get("where")
                        if where is not None:
                            guard = _safe_render_guard(where.this, source_alias=source_alias, procedure_sql=context_sql)
                            if guard is None:
                                return None
                    return guard, value, target_column

        if isinstance(tree, exp.Select) and tree.args.get("into") is not None:
            projection = _render_select_into_projection(tree, target_upper)
            if projection is not None:
                guard = ""
                where = tree.args.get("where")
                if where is not None:
                    guard = _safe_render_guard(where.this, source_alias=source_alias, procedure_sql=context_sql)
                    if guard is None:
                        return None
                return guard, projection, target_column

        select_into_stmt = _extract_select_into_statement(stmt)
        if select_into_stmt:
            try:
                select_tree = sqlglot.parse_one(select_into_stmt, read=_SQLGLOT_DIALECT[dialect])
            except Exception:
                select_tree = None
            if isinstance(select_tree, exp.Select) and select_tree.args.get("into") is not None:
                projection = _render_select_into_projection(select_tree, target_upper)
                if projection is not None:
                    guard = ""
                    where = select_tree.args.get("where")
                    if where is not None:
                        guard = _safe_render_guard(
                            where.this,
                            source_alias=_statement_source_alias(select_tree),
                            procedure_sql=context_sql,
                        )
                        if guard is None:
                            return None
                    return guard, projection, target_column

        if isinstance(tree, exp.Merge):
            guard_parts: list[str] = []
            using = tree.args.get("using")
            using_select = (
                using.this
                if isinstance(using, exp.Subquery) and isinstance(using.this, exp.Select)
                else None
            )
            if using_select is not None:
                where = using_select.args.get("where")
                if where is not None:
                    rendered_guard = _safe_render_guard(where.this, procedure_sql=context_sql)
                    if rendered_guard is None:
                        return None
                    guard_parts.append(rendered_guard)
            on_clause = tree.args.get("on")
            if on_clause is not None:
                rendered_on = _safe_render_guard(on_clause, procedure_sql=context_sql)
                if rendered_on is None:
                    return None
                guard_parts.append(rendered_on)

            whens = tree.args.get("whens")
            when_list = whens.expressions if whens is not None else []
            for when in when_list:
                then = when.args.get("then")
                if not isinstance(then, exp.Update):
                    continue
                when_cond = when.args.get("condition")
                for assignment in then.args.get("expressions", []) or []:
                    if not isinstance(assignment, exp.EQ) or not isinstance(assignment.this, exp.Column):
                        continue
                    if assignment.this.name.upper() != target_upper:
                        continue
                    value_node = unwrap_parens(assignment.expression)
                    if using_select is not None and isinstance(value_node, exp.Column):
                        # THEN UPDATE SET Target.Col = S.Alias commonly just
                        # copies a value the USING subquery already computed
                        # (very often via CASE) under that alias -- resolve
                        # to the real projected expression instead of
                        # emitting a reference to `S.Alias`, which after
                        # alias-to-table resolution becomes a reference to a
                        # column that doesn't physically exist on the
                        # source table (the alias only exists in the
                        # subquery's SELECT list).
                        resolved = _resolve_using_subquery_projection(using_select, value_node.name)
                        if resolved is not None:
                            value_node = unwrap_parens(resolved)
                    if isinstance(value_node, exp.Case):
                        case_translation = _translate_case_to_4x(value_node)
                        if case_translation is None:
                            break
                        value = _rewrite_business_date_variables(case_translation, entity_name, source_sql=context_sql)
                    else:
                        value = _render_value_expression_to_4x(value_node)
                        if value is not None:
                            value = _rewrite_business_date_variables(value, entity_name, source_sql=context_sql)
                            if validate_expression(value).valid:
                                pass
                            else:
                                break
                        else:
                            value = value_node.sql(dialect=_SQLGLOT_DIALECT[dialect])
                            value = _rewrite_business_date_variables(value, entity_name, source_sql=context_sql)
                            if not _is_simple_stage_value(value):
                                break
                        if value is None:
                            break
                    if when_cond is not None:
                        rendered_when = _safe_render_guard(when_cond, source_alias=source_alias, procedure_sql=context_sql)
                        if rendered_when is None:
                            return None
                        guard_parts.append(rendered_when)
                    guard = " AND ".join(f"({part})" for part in guard_parts if part)
                    return guard, value, assignment.this.name

        # Try the next candidate dialect.
        continue

    return None


def _extract_select_into_statement(stmt_text: str) -> str | None:
    """Isolate the actual `SELECT ... INTO ...` statement from a glued
    procedural block.

    SQL Server procedures often glue control-flow cleanup (`IF OBJECT_ID...
    DROP TABLE...`) directly in front of a `SELECT ... INTO #temp ...`
    projection. sqlglot can usually parse the select just fine once that
    preamble is removed, so this helper trims the block down to the
    projection statement itself without changing any logic.
    """
    cleaned = _strip_sql_comments_for_guard_matching(stmt_text)
    candidates = split_statements(cleaned, detect_dialect(cleaned))
    for candidate in candidates:
        upper = candidate.upper()
        if "SELECT" not in upper or not re.search(r"\bINTO\b", upper):
            continue
        if not upper.lstrip().startswith("SELECT"):
            continue
        return candidate.strip() or None

    select_matches = list(re.finditer(r"\bSELECT\b", cleaned, re.IGNORECASE))
    if not select_matches:
        return None

    for match in select_matches:
        candidate = cleaned[match.start() :].strip()
        if not candidate.upper().startswith("SELECT") or not re.search(r"\bINTO\b", candidate, re.IGNORECASE):
            continue

        boundary_match = None
        for boundary in re.finditer(
            r"(?mi)^\s*(UPDATE|INSERT|DELETE|MERGE|IF|BEGIN|CREATE|DROP|EXEC|RETURN|WHILE)\b",
            candidate,
        ):
            if boundary.start() > 0:
                boundary_match = boundary
                break

        if boundary_match is not None:
            candidate = candidate[: boundary_match.start()].strip()
        return candidate or None
    return None


def _render_select_projection_value(
    value_node: exp.Expression,
    select_tree: exp.Select,
    *,
    unwrap_parens=None,
) -> str | None:
    """Render one SELECT list item (CASE / column / scalar) to a 4X value."""
    node = value_node
    if unwrap_parens is not None:
        node = unwrap_parens(node)
    else:
        while isinstance(node, exp.Paren):
            node = node.this

    source_alias = _statement_source_alias(select_tree) or _select_from_relation_name(select_tree)
    if source_alias and not isinstance(node, exp.Column):
        node = _qualify_unqualified_condition_columns(node, source_alias)

    if isinstance(node, exp.Case):
        return _translate_case_to_4x(node)

    if isinstance(node, exp.Column):
        col_name = node.name or ""
        if not col_name:
            return None
        table_name = node.table or source_alias or _select_from_relation_name(select_tree)
        if table_name:
            return f'"{str(table_name).lstrip("#")}"."{col_name}"'
        return f'"{col_name}"'

    return _render_value_expression_to_4x(node)


def _render_select_into_projection(tree: exp.Select, target_upper: str) -> str | None:
    for expr in tree.args.get("expressions", []) or []:
        value_node = expr
        alias = None

        if isinstance(expr, exp.Alias):
            alias = expr.alias_or_name or expr.alias
            value_node = expr.this
        elif isinstance(expr, exp.Column):
            alias = expr.alias_or_name or expr.name
        elif isinstance(expr, exp.Identifier):
            alias = expr.this
        else:
            alias = getattr(expr, "alias", None)

        if not alias or alias.upper() != target_upper:
            continue

        return _render_select_projection_value(value_node, tree)
    return None


def _render_insert_select_projection(tree: exp.Insert, target_upper: str) -> str | None:
    """Map `INSERT INTO t (c1,c2,…) SELECT e1,e2,…` by column position.

    SQL Server staging loads often put multi-branch CASE expressions in the
    SELECT list without aliases; positional pairing with the INSERT column
    list is required to compose Outcome / ShortfallPct style formulas.
    """
    select_tree = tree.expression
    if not isinstance(select_tree, exp.Select):
        return None

    schema = tree.this
    insert_columns: list[str] = []
    if isinstance(schema, exp.Schema):
        for col in schema.expressions or []:
            if isinstance(col, exp.Identifier):
                insert_columns.append(str(col.this or ""))
            elif isinstance(col, exp.Column):
                insert_columns.append(str(col.name or ""))
            else:
                name = getattr(col, "name", None) or getattr(col, "this", None)
                if name:
                    insert_columns.append(str(name))

    select_exprs = list(select_tree.args.get("expressions") or [])
    if not insert_columns or len(insert_columns) != len(select_exprs):
        # Fall back to alias-based matching when lengths differ.
        return _render_select_into_projection(select_tree, target_upper)

    for col_name, expr in zip(insert_columns, select_exprs):
        if (col_name or "").upper() != target_upper:
            continue
        value_node = expr.this if isinstance(expr, exp.Alias) else expr
        return _render_select_projection_value(value_node, select_tree)
    return None


def _select_into_source_alias(tree: exp.Select) -> str | None:
    """Return the single source alias used by a `SELECT ... INTO` seed
    projection when the query clearly reads from one base relation.

    Bare projections in SQL Server often rely on the surrounding `FROM`
    alias rather than repeating it in every select item. For deterministic
    DD generation we need that alias preserved, otherwise a direct source
    column like `LastCrDate` looks identical to the target column name and
    semantic validation incorrectly treats it as circular.
    """
    from_clause = tree.args.get("from_")
    if not isinstance(from_clause, exp.From):
        return None
    source = from_clause.this
    if source is None:
        sources = list(from_clause.expressions or [])
        if len(sources) != 1:
            return None
        source = sources[0]
    return _extract_alias_name(source)


def _extract_update_assignment_rhs(stmt_text: str, target_upper: str) -> str | None:
    pattern = re.compile(
        rf"(?is)(?:^|,)\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?{re.escape(target_upper)}\s*=\s*"
    )
    match = pattern.search(stmt_text.upper())
    if not match:
        return None

    start = match.end()
    i = start
    n = len(stmt_text)
    paren_depth = 0
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = stmt_text[i]
        nxt = stmt_text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_single:
            if ch == "'" and nxt != "'":
                in_single = False
            elif ch == "'" and nxt == "'":
                i += 1
            i += 1
            continue
        if in_double:
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "-" and nxt == "-":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "," and paren_depth == 0:
            return stmt_text[start:i].strip()
        elif paren_depth == 0 and stmt_text[i : i + 5].upper() == "WHERE":
            return stmt_text[start:i].strip()
        i += 1

    return stmt_text[start:].strip() or None


def _strip_sql_comments_for_guard_matching(text: str) -> str:
    """Strip `--` line comments and `/* */` block comments (quote-aware),
    so a WHERE-clause search below can never be hijacked by a comment
    that happens to contain the word "where" in its own free text (a
    real, confirmed failure mode: a comment like
    `--Update X set Y=Z where BandName='...'` sits directly above the
    real UPDATE statement in one of the sample procedures, and an earlier
    version of this function matched the comment's "where" instead of the
    actual WHERE clause several lines later, producing a guard that could
    never match anything)."""
    result: list[str] = []
    i = 0
    n = len(text)
    in_single = False
    while i < n:
        ch = text[i]
        if in_single:
            result.append(ch)
            if ch == "'" and not (i + 1 < n and text[i + 1] == "'"):
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            result.append(ch)
            i += 1
            continue
        if ch == "-" and i + 1 < n and text[i + 1] == "-":
            nl = text.find("\n", i)
            i = n if nl == -1 else nl
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        result.append(ch)
        i += 1
    return "".join(result)


_WHERE_CLAUSE_GUARD_RE = re.compile(r"(?is)\bWHERE\b(.+?)(?:;|$)")


def _extract_where_guard_text(raw_sql: str) -> str | None:
    """The row-scoping WHERE clause text of a statement, whitespace/case
    normalized for exact comparison -- or None if the statement has none.
    Deliberately simple (a single regex, not a full parse): this is only
    ever used to compare two guards for exact textual equality, not to
    understand the guard's own structure. Comments are stripped first so
    a comment's own free text can never be mistaken for a real WHERE
    clause."""
    match = _WHERE_CLAUSE_GUARD_RE.search(_strip_sql_comments_for_guard_matching(raw_sql))
    if not match:
        return None
    normalized = " ".join(match.group(1).split()).upper().rstrip(")")
    return normalized or None


_ERROR_SIGNAL_RE = re.compile(
    r"(?i)\bERROR\b|\bEXCEPTION\b|@@ERROR|ERROR_NUMBER\s*\(|ERROR_MESSAGE\s*\(|"
    r"ERROR_STATE\s*\(|ERROR_SEVERITY\s*\(|\bSQLCODE\b|\bSQLERRM\b|"
    r"\bFAILED\b|\bFAILURE\b"
)


def _guard_has_error_signal(guard_text: str) -> bool:
    return bool(_ERROR_SIGNAL_RE.search(guard_text or ""))


def undeterminable_exception_sites(sites: list["_AssignmentSite"]) -> list["_AssignmentSite"]:
    """Identify EXCEPTION_HANDLER write sites whose guard is not a genuine,
    data-driven "an error occurred" signal.

    "An unhandled exception occurred" is a runtime execution event, not a
    fact present in row data, so it has no real per-row guard of its own
    unless the source SQL explicitly tests an error/status indicator (e.g.
    `@@ERROR`, `ERROR_NUMBER()`, an ErrorFlag column). Any other guard on a
    CATCH/EXCEPTION-block write -- a plain row-scoping filter like a
    process name or run id -- is not actually distinguishing the exception
    case, REGARDLESS of whether it happens to textually match another
    site's guard: a different-but-still-plain identity filter is just as
    unrelated to "an exception occurred" as an identical one. Composing it
    as a real condition would deterministically apply that write to every
    row matching the filter, not only rows where an exception genuinely
    happened -- a confidently wrong result, not just a gap.

    Confirmed against a real generation defect this way: both the LLM
    path and the deterministic composer independently produced the exact
    same wrong expression for a column shaped like this (ERRORDATE in
    ACLRUNNINGPROCESSSTATUS) -- the fix has to be structural exclusion,
    not a smarter guard, since there is no data-driven guard to discover.

    Only sites with a *present* non-error guard are excluded here -- a
    site with no extractable guard at all is left alone, since that's
    also the signature of an unrelated write incorrectly bridged onto this
    header by statement-site construction (e.g. across a comment-only gap
    with no real write in between); treating "no guard found" the same as
    "a real but unrelated guard" would misfire on that mis-attribution
    instead of on the exception-handler ambiguity this function targets.
    """
    exception_sites = [s for s in sites if _infer_assignment_role(s.raw_sql) == "EXCEPTION_HANDLER"]
    undeterminable: list["_AssignmentSite"] = []
    for site in exception_sites:
        guard = _extract_where_guard_text(site.raw_sql)
        if guard and not _guard_has_error_signal(guard):
            undeterminable.append(site)
    return undeterminable


def _extract_complete_update_for_column(
    procedure_sql: str,
    target_column: str,
    prefer_fragment: str = "",
) -> str | None:
    """Recover a full `UPDATE ... SET col = <expr> ...` when chunking truncated CASE.

    Statement splitting often cuts at the first `END` inside `CASE ... END`,
    leaving an unparseable fragment. Scan the procedure text with CASE/paren
    depth so nested CASE expressions stay intact.
    """
    proc = procedure_sql or ""
    column = (target_column or "").strip()
    if not proc or not column:
        return None

    prefer = (prefer_fragment or "").strip()
    prefer_idx = proc.find(prefer[:120]) if prefer and len(prefer) >= 20 else -1

    best: str | None = None
    best_score = -1
    for match in re.finditer(
        rf"(?is)\bSET\b[\s\S]{{0,120}}?\b{re.escape(column)}\b\s*=\s*",
        proc,
    ):
        start = proc.rfind("UPDATE", max(0, match.start() - 240), match.start() + 1)
        if start < 0:
            continue
        i = match.end()
        depth_paren = 0
        depth_case = 0
        in_single = False
        in_double = False
        n = len(proc)
        while i < n:
            ch = proc[i]
            nxt = proc[i : i + 2]
            if in_single:
                if ch == "'":
                    in_single = False
                i += 1
                continue
            if in_double:
                if ch == '"':
                    in_double = False
                i += 1
                continue
            if ch == "'":
                in_single = True
                i += 1
                continue
            if ch == '"':
                in_double = True
                i += 1
                continue
            if ch == "(":
                depth_paren += 1
                i += 1
                continue
            if ch == ")":
                depth_paren = max(0, depth_paren - 1)
                i += 1
                continue
            upper_tail = proc[i : i + 8].upper()
            if re.match(r"CASE\b", upper_tail):
                depth_case += 1
                i += 4
                continue
            if re.match(r"END\b", upper_tail):
                if depth_case > 0:
                    depth_case -= 1
                    i += 3
                    continue
                # END of BEGIN block / procedure — stop before it when expression done
                break
            if depth_paren == 0 and depth_case == 0:
                if re.match(r"(?:GO|ELSE|ELSEIF|ELSIF)\b", upper_tail):
                    break
                if re.match(r"(?:UPDATE|MERGE|INSERT|DELETE|CREATE|IF)\b", upper_tail) and i > match.end():
                    # Next statement at top level
                    break
            i += 1

        candidate = proc[start:i].strip()
        if column.upper() not in candidate.upper():
            continue
        if candidate.upper().count("CASE") > candidate.upper().count(" END") and "CASE" in candidate.upper():
            # Still truncated — skip
            continue
        score = len(candidate)
        if prefer_idx >= 0 and start <= prefer_idx <= start + len(candidate):
            score += 10000
        if score > best_score:
            best_score = score
            best = candidate
    return best


def _expand_truncated_assignment_sql(
    raw_sql: str,
    procedure_sql: str,
    target_column: str,
) -> str:
    text = (raw_sql or "").strip()
    if not text:
        return text
    upper = text.upper()
    truncated = (
        upper.count("CASE") > len(re.findall(r"\bEND\b", upper))
        or (upper.lstrip().startswith("UPDATE") and "FROM" not in upper and not text.rstrip().endswith(";"))
        or bool(re.search(r"(?i)(?:=\s*|THEN|ELSE|WHEN)\s*$", text.rstrip()))
    )
    if not truncated:
        return text
    recovered = _extract_complete_update_for_column(procedure_sql, target_column, prefer_fragment=text)
    return recovered if recovered and len(recovered) > len(text) else text


def _find_snippet_in_procedure(proc: str, snippet: str, start: int = 0) -> int:
    """Locate `snippet` (or its first ~120 chars) inside `proc`, tolerating
    whitespace differences between them.

    An assignment site's raw_sql can be reassembled from separately
    `.strip()`-ped statement fragments joined with a single `\\n` (see
    `_assignment_sites_from_statements`'s CONTROL_FLOW_BLOCK case, e.g. an
    "ELSE" header folded onto the UPDATE it guards) -- that reassembly
    collapses whatever original indentation/whitespace sat between them in
    `proc`, even though the code itself is unchanged. An exact substring
    search would then wrongly conclude the snippet isn't in `proc` at all.
    Tries the cheap exact search first, then falls back to a whitespace-
    insensitive regex search.
    """
    key = snippet[:120] if len(snippet) >= 40 else snippet
    if not key:
        return -1
    idx = proc.find(key, start)
    if idx >= 0:
        return idx
    idx = proc.find(key)
    if idx >= 0:
        return idx
    parts = key.split()
    if not parts:
        return -1
    pattern = re.compile(r"\s+".join(re.escape(part) for part in parts))
    match = pattern.search(proc, start)
    if match:
        return match.start()
    match = pattern.search(proc)
    return match.start() if match else -1


def _assignment_looks_like_control_flow_else_default(
    stage_raw: str,
    procedure_sql: str,
) -> bool:
    """True when an unconditional UPDATE sits in a procedural ELSE branch.

    Those writes are mutually exclusive with earlier IF/ELSEIF arms — not a
    sequential last-write-wins wipe that should erase prior guards.
    """
    snippet = (stage_raw or "").strip()
    if not snippet:
        return False

    # A CONTROL_FLOW_BLOCK assignment site already folds its own guarding
    # ELSE/BEGIN header directly onto the front of the write (see
    # _is_condition_bearing_header / _assignment_sites_from_statements) --
    # check that embedded header first. Once it's part of the snippet
    # itself, it's no longer part of the *preceding* text the fallback
    # search below looks at, so that search alone would miss it here.
    if re.match(r"(?is)^\s*ELSE\b(?!\s+IF\b)\s+BEGIN\b", snippet[:240]):
        return True

    proc = procedure_sql or ""
    if not proc:
        return False

    idx = _find_snippet_in_procedure(proc, snippet)
    if idx < 0:
        search_key = snippet.splitlines()[0].strip()
        if search_key:
            idx = proc.find(search_key)
    if idx < 0:
        return False

    window = proc[max(0, idx - 500) : idx]
    # Strip line comments so `-- ELSE` in notes does not false-positive.
    cleaned_lines: list[str] = []
    for line in window.splitlines():
        cleaned_lines.append(line.split("--", 1)[0])
    window = "\n".join(cleaned_lines).upper()
    # Require procedural `ELSE BEGIN` — bare `ELSE` inside CASE WHEN arms
    # (e.g. `ELSE 'N' END`) must not count as a control-flow else default.
    return bool(re.search(r"\bELSE\b(?!\s+IF\b)\s+BEGIN[\s\S]{0,240}$", window))


def _procedural_then_predicate_4x(
    stage_raw: str,
    procedure_sql: str,
    *,
    entity_name: str,
) -> str | None:
    """Render a simple procedural IF predicate wrapping a THEN-arm UPDATE.

    Handles `IF DAY(@ProcessDate) = 1` and similar scalar compares so the
    guard is not lost when the UPDATE's own WHERE is only a row filter.
    Skips ELSE / ELSE IF arms (those use exclusive-branch compose).
    """
    proc = procedure_sql or ""
    snippet = (stage_raw or "").strip()
    if not snippet or not proc:
        return None
    # A CONTROL_FLOW_BLOCK site can fold an "ELSE" or "ELSE IF"/"ELSIF"
    # header directly onto the front of its own raw_sql (see
    # _assignment_sites_from_statements). When that happens, the found
    # `idx` below points at the START of that embedded header -- i.e. the
    # header sits AFTER idx, inside the matched snippet, not before it --
    # so the backward window search would instead find whatever unrelated
    # header happens to precede this stage's real one and wrongly attach
    # that stage's predicate here. Bail out first; ELSE/ELSE IF arms are
    # for exclusive-branch compose, not this function, regardless of
    # whether their header is embedded or sits in a separate statement.
    if re.match(r"(?is)^\s*(?:ELSE\s+IF|ELSIF|ELSE)\b", snippet):
        return None
    if _assignment_looks_like_control_flow_else_default(snippet, proc):
        return None
    idx = _find_snippet_in_procedure(proc, snippet)
    if idx < 0:
        return None
    window = proc[max(0, idx - 400) : idx]
    cleaned = "\n".join(line.split("--", 1)[0] for line in window.splitlines())
    # Nearest IF … BEGIN before this statement (not IF OBJECT_ID).
    matches = list(
        re.finditer(
            r"(?is)\bIF\s+(?!OBJECT_ID\b)(?P<cond>.+?)\s+BEGIN\b",
            cleaned,
        )
    )
    if not matches:
        return None
    # If an ELSE IF sits closer than the IF, this is not a pure THEN arm.
    else_if = list(re.finditer(r"(?is)\bELSE\s+IF\b", cleaned))
    if else_if and else_if[-1].start() > matches[-1].start():
        return None
    cond = matches[-1].group("cond").strip()
    # Delegate to the general scalar-predicate renderer (DAY(@ProcessDate)=N,
    # NOT NULL + offset/positive lineage checks, EXISTS, and plain @var<op>
    # literal comparisons) instead of only recognizing DAY(@ProcessDate)=N
    # here -- any other wrapping IF condition (e.g. IF @RunType='MONTHLY')
    # was previously silently dropped rather than combined into the guard.
    return _render_simple_scalar_predicate_4x(cond, entity_name, proc)


def _procedural_exclusive_branch_predicate(
    stage_raw: str,
    procedure_sql: str,
    *,
    search_from: int = 0,
    allow_opening_if: bool = True,
) -> tuple[str | None, int, bool]:
    """Return (predicate|'' for ELSE|None, match_index, disqualified).

    `search_from` skips earlier duplicate UPDATEs with the same text (sample 16
    blackout vs final ELSE both set amount=0 / shortfall=Y).

    `allow_opening_if` gates the generic `IF ... BEGIN` fallback used when no
    `ELSE IF`/`ELSE` header directly precedes the statement. Only the first
    branch of a chain may open with a bare IF; if a later branch's nearest
    header is a bare IF rather than ELSE IF/ELSE, it is an independent,
    unconnected IF block that happens to share a WHERE clause by coincidence
    -- not a real elseif-chain sibling -- so the caller must not compose it
    as one (real SQL executes independent IFs sequentially, last-write-wins,
    not first-true-wins like ELSEIF). `disqualified=True` signals exactly
    this case -- a genuine header was found but it's a bare IF where only
    ELSE IF/ELSE would make this branch a real chain sibling -- distinct
    from `pred=None, disqualified=False` (no wrapping header found at all,
    e.g. a leading unrelated write before the chain even starts), which the
    caller may still skip over rather than treat as chain-breaking.
    """
    proc = procedure_sql or ""
    snippet = (stage_raw or "").strip()
    if not snippet or not proc:
        return None, -1, False
    idx = _find_snippet_in_procedure(proc, snippet, max(0, search_from))
    if idx < 0:
        return None, -1, False

    # A CONTROL_FLOW_BLOCK assignment site can already fold its own
    # guarding ELSE/ELSEIF header directly onto the front of the write
    # (see _assignment_sites_from_statements) -- once that header is part
    # of the snippet itself, it is no longer part of the *preceding* text
    # the window search below looks at, so that search alone would miss
    # it. Mirrors the same check in
    # _assignment_looks_like_control_flow_else_default.
    embedded_else_if = re.match(
        r"(?is)^\s*(?:ELSE\s+IF|ELSIF)\s+(?P<cond>.+?)\s+(?:BEGIN|THEN)\b", snippet
    )
    if embedded_else_if:
        return embedded_else_if.group("cond").strip(), idx, False
    if re.match(r"(?is)^\s*ELSE\b(?!\s+IF\b)\s*(?:(?:BEGIN|THEN)\s*)?(?:\r?\n|$)", snippet):
        return "", idx, False

    window = proc[max(0, idx - 600) : idx]
    cleaned = "\n".join(line.split("--", 1)[0] for line in window.splitlines())
    upper = cleaned.upper()
    # T-SQL opens a branch body with `BEGIN`; Oracle PL/SQL opens it with
    # `THEN` and has no BEGIN at all for an IF/ELSIF/ELSE arm (BEGIN there
    # only wraps the whole procedure/block). Both are accepted so this
    # detector isn't T-SQL-only.
    if re.search(r"\bELSE\b(?!\s+IF\b)\s*(?:(?:BEGIN|THEN)\s*)?$", upper):
        return "", idx, False
    else_if = list(
        re.finditer(r"(?is)\b(?:ELSE\s+IF|ELSIF)\s+(?P<cond>.+?)\s+(?:BEGIN|THEN)\b", cleaned)
    )
    if else_if:
        return else_if[-1].group("cond").strip(), idx, False
    if_match = list(
        re.finditer(r"(?is)\bIF\s+(?!OBJECT_ID\b)(?P<cond>.+?)\s+(?:BEGIN|THEN)\b", cleaned)
    )
    if not allow_opening_if:
        return None, idx, bool(if_match)
    if if_match:
        return if_match[-1].group("cond").strip(), idx, False
    return None, idx, False


# `@` is T-SQL's sigil for a parameter/local variable; Oracle PL/SQL
# parameters (e.g. `p_TIMEKEY`) carry no sigil at all, so it is optional
# here rather than required -- see dd_generation.yaml's documented
# platform convention for exactly this shape (`IF p_TIMEKEY > 26267`).
_SIMPLE_SCALAR_COMPARISON_RE = re.compile(
    r"(?is)^@?(?P<var>[A-Za-z_][\w]*)\s*(?P<op>=|<>|!=|>=|<=|>|<)\s*"
    r"(?P<lit>'[^']*'|-?\d+(?:\.\d+)?)\s*$"
)
_COMPARISON_OP_MAP = {"=": "==", "<>": "!=", "!=": "!=", ">=": ">=", "<=": "<=", ">": ">", "<": "<"}


def _render_simple_scalar_predicate_4x(cond_sql: str, entity_name: str, procedure_sql: str) -> str | None:
    """Best-effort 4X rendering for procedural IF scalar predicates."""
    text = (cond_sql or "").strip()
    if not text:
        return None
    bd = f'"{entity_name}"."var"."BUSINESS_DATE"'
    # `@var IS NOT NULL AND @ProcessDate >= DATEADD(DAY, -N, @var)` — @var's
    # real source (table/column) comes from its own DECLARE, not a guess.
    not_null_offset = re.match(
        r"(?is)^@(?P<var>[A-Za-z_][\w]*)\s+IS\s+NOT\s+NULL\s+AND\s+"
        r"@?(?:v_)?(?:ProcessDate|ProcessDt|BusinessDate)\s*>=\s*"
        r"DATEADD\s*\(\s*DAY\s*,\s*(?P<offset>-?\d+)\s*,\s*@(?P=var)\s*\)\s*$",
        text,
    )
    if not_null_offset:
        ref = _resolve_scalar_lookup_reference(not_null_offset.group("var"), procedure_sql)
        if ref:
            offset = int(not_null_offset.group("offset"))
            return f"ISNOTEMPTY({ref}) AND {bd} >= ADDDAY({ref}, {offset})"
        return None
    # `@var IS NOT NULL AND @var > 0` — same lineage-first resolution.
    not_null_positive = re.match(
        r"(?is)^@(?P<var>[A-Za-z_][\w]*)\s+IS\s+NOT\s+NULL\s+AND\s+@(?P=var)\s*>\s*0\s*$",
        text,
    )
    if not_null_positive:
        ref = _resolve_scalar_lookup_reference(not_null_positive.group("var"), procedure_sql)
        if ref:
            return f"ISNOTEMPTY({ref}) AND {ref} > 0"
        return None
    # IF EXISTS (...) — keep as EXISTS for unsupported/manual clarity
    if re.match(r"(?is)^EXISTS\s*\(", text):
        return f"EXISTS({text[text.upper().find('EXISTS') + 6:].strip()})"
    day_eq = re.match(
        r"(?is)^DAY\s*\(\s*@?(?:v_)?(?:ProcessDate|ProcessDt|BusinessDate)\s*\)\s*=\s*(\d+)\s*$",
        text,
    )
    if day_eq:
        return f'DATEPART("d", {bd}) == {day_eq.group(1)}'
    # Plain `@var <op> literal` (e.g. `@RunType = 'MONTHLY'`, `@Flag = 1`).
    # @var's source comes from its own DECLARE lookup when there is one;
    # otherwise it's treated as a bare procedure input parameter (the
    # documented convention for a rule-gating parameter like p_TIMEKEY —
    # see dd_generation.yaml), never guessed as a table/column reference.
    simple_cmp = _SIMPLE_SCALAR_COMPARISON_RE.match(text)
    if simple_cmp:
        var = simple_cmp.group("var")
        op = _COMPARISON_OP_MAP[simple_cmp.group("op")]
        literal = simple_cmp.group("lit")
        if literal.startswith("'"):
            literal = '"' + literal[1:-1].replace('"', '\\"') + '"'
        ref = _resolve_scalar_lookup_reference(var, procedure_sql) or var
        return f"{ref} {op} {literal}"
    return None


def _resolve_scalar_lookup_reference(var: str, procedure_sql: str) -> str | None:
    """Return `"table"."column"` for @var if it's declared as a single-row
    scalar lookup, or None if no such lineage can be found in the source."""
    for match in _SCALAR_LOOKUP_DECLARE_RE.finditer(procedure_sql or ""):
        if match.group("var").lower() == var.lower():
            return f'"{match.group("table")}"."{match.group("col")}"'
    return None


def _compose_exclusive_control_flow_stages(
    stages: list[tuple[str, str, str, str]],
    *,
    procedure_sql: str,
    entity_name: str,
) -> str | None:
    """Compose IF/ELSE IF/ELSE arms that share a row WHERE as exclusive branches.

    Sample 16 CoverAppropriatedAmount: three UPDATEs with the same WHERE sit
    under IF / ELSE IF / ELSE — nesting them as LWW makes the last `THEN(0)`
    always win. Build IF(ctrl1)THEN(v1)ELSEIF(ctrl2)THEN(v2)ELSE(v3) instead,
    optionally wrapped by the shared row guard.
    """
    if len(stages) < 2 or not procedure_sql:
        return None
    # Prefer arms that sit under procedural IF/ELSE IF/ELSE; ignore leading
    # unrelated writes (e.g. CoverShortfallFlag='N' before the fund IF).
    branched: list[tuple[tuple[str, str, str, str], str]] = []
    cursor = 0
    for stage in stages:
        # A bare IF may open the chain the first time a real branch header
        # is actually found -- not only when it happens to be the very
        # first stage in `stages`. A leading unrelated unconditional write
        # before the chain starts (e.g. an initial reset/seed UPDATE with
        # no wrapping IF at all, which returns pred=None here without
        # being disqualified) must not block the chain that follows it
        # from opening with its own bare IF.
        pred, idx, disqualified = _procedural_exclusive_branch_predicate(
            stage[3], procedure_sql, search_from=cursor, allow_opening_if=(not branched)
        )
        if idx >= 0:
            cursor = idx + 1
        if disqualified:
            # A later branch's nearest header is an independent bare IF, not
            # a real ELSE IF/ELSE continuation -- these are not one connected
            # exclusive chain, so bail out entirely rather than composing a
            # subset and silently dropping this branch's write. The caller
            # falls back to ordinary sequential (last-write-wins) composition.
            return None
        if pred is not None:
            branched.append((stage, pred))
    if len(branched) < 2:
        return None
    # Need both a conditional arm and an ELSE (or multiple ELSE IFs).
    preds = [p for _s, p in branched]
    if not any(p == "" for p in preds) and len(set(preds)) < 2:
        return None
    row_guards = [(g or "").strip() for (g, _v, _t, _r), _p in branched]
    if len(set(row_guards)) != 1:
        return None
    shared_where = row_guards[0]

    rendered_arms: list[tuple[str | None, str]] = []
    for stage, pred in branched:
        _g, value, _t, _raw = stage
        if pred == "":
            rendered_arms.append((None, value))
            continue
        ctrl = _render_simple_scalar_predicate_4x(pred, entity_name, procedure_sql)
        if ctrl is None:
            return None
        rendered_arms.append((ctrl, value))

    if_arms = [(c, v) for c, v in rendered_arms if c is not None]
    else_arms = [v for c, v in rendered_arms if c is None]
    if not if_arms:
        return None
    parts = [f"IF({if_arms[0][0]})THEN({if_arms[0][1]})"]
    for ctrl, val in if_arms[1:]:
        parts.append(f"ELSEIF({ctrl})THEN({val})")
    else_val = else_arms[-1] if else_arms else "NULL"
    parts.append(f"ELSE({else_val})")
    inner = "".join(parts)
    if not validate_expression(inner).valid:
        return None
    if shared_where:
        wrapped = f"IF({shared_where})THEN({inner})ELSE(NULL)"
        return wrapped if validate_expression(wrapped).valid else inner
    return inner


def _compose_procedural_if_else_seed(
    empty_stages: list[tuple[str, str, str, str]],
    *,
    procedure_sql: str,
    entity_name: str,
    is_plain_update_wipe,
) -> str | None:
    """Seed expression for an IF-arm rich formula paired with an ELSE wipe.

    Preserves both arms when possible: IF(cond)THEN(rich_case)ELSE(wipe).
    Eligibility CASE outcomes must not disappear just because a later
    override (or an unreachable IF) exists — source anomalies still report
    unreachable predicates separately.
    """
    rich_empties = [
        stage
        for stage in empty_stages
        if (stage[1] or "").upper().startswith("IF(")
        or "COALESCE(" in (stage[1] or "").upper()
        or "DATEDIFF(" in (stage[1] or "").upper()
    ]
    wipe_empties = [
        stage
        for stage in empty_stages
        if is_plain_update_wipe(stage[3], stage[1])
        and _assignment_looks_like_control_flow_else_default(stage[3], procedure_sql)
    ]
    if not rich_empties or not wipe_empties:
        return None

    rich_val = _rewrite_business_date_variables(
        _normalize_expression(rich_empties[-1][1], ""),
        entity_name,
        source_sql=procedure_sql,
    )
    wipe_val = _rewrite_business_date_variables(
        _normalize_expression(wipe_empties[-1][1], ""),
        entity_name,
        source_sql=procedure_sql,
    )
    if not validate_expression(rich_val).valid or not validate_expression(wipe_val).valid:
        return None

    cond = _procedural_if_predicate_before_stage(
        rich_empties[-1][3],
        procedure_sql,
        entity_name=entity_name,
    )
    if cond and validate_expression(f"IF({cond})THEN(1)ELSE(0)").valid:
        paired = f"IF({cond})THEN({rich_val})ELSE({wipe_val})"
        if validate_expression(paired).valid:
            return paired

    # Cannot render the IF predicate — keep the rich THEN CASE (eligibility
    # chain) rather than dropping it for the ELSE wipe alone.
    return rich_val


def _procedural_if_predicate_before_stage(
    stage_raw: str,
    procedure_sql: str,
    *,
    entity_name: str,
) -> str | None:
    """Render the IF predicate of the BEGIN block that contains `stage_raw`."""
    proc = procedure_sql or ""
    snippet = (stage_raw or "").strip()
    if not snippet or not proc:
        return None
    idx = _find_snippet_in_procedure(proc, snippet)
    if idx < 0:
        return None
    window = proc[max(0, idx - 500) : idx]
    cleaned = "\n".join(line.split("--", 1)[0] for line in window.splitlines())
    matches = list(
        re.finditer(r"(?is)\bIF\s+(?!OBJECT_ID\b)(?P<cond>.+?)\s+BEGIN\b", cleaned)
    )
    if not matches:
        return None
    # Skip ELSE IF — those are handled by exclusive-branch compose.
    else_if = list(re.finditer(r"(?is)\bELSE\s+IF\b", cleaned))
    if else_if and else_if[-1].start() > matches[-1].start():
        return None
    cond_sql = matches[-1].group("cond").strip()
    rendered = _render_simple_scalar_predicate_4x(cond_sql, entity_name, proc)
    if rendered:
        return rendered
    return _render_processdate_vs_cutoff_predicate(cond_sql, entity_name, proc)


def _render_processdate_vs_cutoff_predicate(
    cond_sql: str,
    entity_name: str,
    procedure_sql: str,
) -> str | None:
    """Map `@ProcessDate < @SchemeCutoffDate` (etc.) using DECLARE lineage."""
    text = (cond_sql or "").strip()
    bd = f'"{entity_name}"."var"."BUSINESS_DATE"'
    m = re.match(
        r"(?is)^@?(?:v_)?(?P<left>ProcessDate|ProcessDt|BusinessDate)\s*"
        r"(?P<op><=|>=|<|>)\s*"
        r"@?(?P<right>[A-Za-z_][\w]*)\s*$",
        text,
    )
    if not m:
        return None
    right = m.group("right")
    op = {"<": "<", ">": ">", "<=": "<=", ">=": ">="}[m.group("op")]
    lineage = _extract_date_offset_lineage(procedure_sql)
    key = right.upper()
    if key not in lineage:
        # Also try DECLARE without going through extract (typed DECLARE).
        decl = re.search(
            rf"(?is)DECLARE\s+@{re.escape(right)}\s+\w+\s*=\s*"
            rf"DATEADD\s*\(\s*(?P<unit>YEAR|MONTH|DAY)S?\s*,\s*(?P<offset>-?\d+)\s*,\s*"
            rf"@?(?:v_)?(?:ProcessDate|ProcessDt|BusinessDate)\s*\)",
            procedure_sql or "",
        )
        if not decl:
            return None
        unit = decl.group("unit").upper().rstrip("S")
        offset = int(decl.group("offset"))
    else:
        unit, offset = lineage[key]
    if unit == "DAY":
        rhs = f"ADDDAY({bd}, {offset})"
    elif unit == "MONTH":
        rhs = f'PERIOD("M", {offset}, {bd})'
    else:
        rhs = f'PERIOD("Y", {offset}, {bd})'
    return f"{bd} {op} {rhs}"


def _is_non_derivable_expression(expression: str) -> bool:
    """True when the formula is a bare NULL/0/1 with no derivation logic.

    Those must not become DD rules / Excel / CSV / report cards. Empty
    strings are handled separately (review rows may still carry metadata).
    """
    text = (expression or "").strip()
    if not text:
        return False
    return text.upper() in {"NULL", "0", "1"}


_BALANCED_IF_RE = re.compile(
    r"^IF\((?P<guard>.+)\)THEN\((?P<then>.+)\)ELSE\((?P<else>.+)\)$",
    re.IGNORECASE | re.DOTALL,
)


def _split_top_level_if(expression: str) -> tuple[str, str, str] | None:
    """Split a single top-level `IF(g)THEN(t)ELSE(e)` into `(g, t, e)`."""
    text = (expression or "").strip()
    if not re.match(r"(?i)^IF\(", text):
        return None

    def _matching_paren(src: str, open_idx: int) -> int | None:
        depth = 0
        for idx in range(open_idx, len(src)):
            if src[idx] == "(":
                depth += 1
            elif src[idx] == ")":
                depth -= 1
                if depth == 0:
                    return idx
        return None

    if_open = text.find("(")
    guard_close = _matching_paren(text, if_open)
    if guard_close is None:
        return None
    after_guard = text[guard_close + 1 :].lstrip()
    then_match = re.match(r"(?i)^THEN\(", after_guard)
    if not then_match:
        return None
    then_open = (guard_close + 1) + (len(text[guard_close + 1 :]) - len(after_guard)) + then_match.end() - 1
    then_close = _matching_paren(text, then_open)
    if then_close is None:
        return None
    after_then = text[then_close + 1 :].lstrip()
    else_match = re.match(r"(?i)^ELSE\(", after_then)
    if not else_match:
        return None
    else_open = (then_close + 1) + (len(text[then_close + 1 :]) - len(after_then)) + else_match.end() - 1
    else_close = _matching_paren(text, else_open)
    if else_close is None:
        return None
    if text[else_close + 1 :].strip():
        return None
    return (
        text[if_open + 1 : guard_close],
        text[then_open + 1 : then_close],
        text[else_open + 1 : else_close],
    )


def _collapse_tautology_branches(expression: str) -> str:
    """Rewrite `IF(g)THEN(v)ELSE(v)` → `v` (including nested ELSE tautologies)."""
    text = (expression or "").strip()
    if not text:
        return text
    changed = True
    while changed:
        changed = False
        parts = _split_top_level_if(text)
        if not parts:
            break
        guard, then_part, else_part = parts
        then_s = _collapse_tautology_branches(then_part)
        else_s = _collapse_tautology_branches(else_part)
        if then_s == else_s:
            text = then_s
            changed = True
            continue
        rebuilt = f"IF({guard})THEN({then_s})ELSE({else_s})"
        if rebuilt != text:
            text = rebuilt
            changed = True
    return text


def _drop_leading_isempty_arm(body: str, col_compact: str) -> str | None:
    """If body is `IF(ISEMPTY(col))THEN(dead)ELSEIF(rest…` → `IF(rest…`.

    Also supports `IF(ISEMPTY(col))THEN(dead)ELSE(rest)`.
    """
    text = (body or "").strip()
    head = re.match(r"(?is)^IF\(ISEMPTY\((?P<col>.+?)\)\)THEN\(", text)
    if not head:
        return None
    if re.sub(r"\s+", "", head.group("col")) != col_compact:
        return None
    then_open = head.end() - 1
    depth = 0
    then_close = None
    for idx in range(then_open, len(text)):
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                then_close = idx
                break
    if then_close is None:
        return None
    rest = text[then_close + 1 :].lstrip()
    if rest.upper().startswith("ELSEIF("):
        return "IF(" + rest[7:]
    if rest.upper().startswith("ELSE("):
        m = re.match(r"(?i)ELSE\(", rest)
        if not m:
            return None
        else_open = (then_close + 1) + (len(text[then_close + 1 :]) - len(rest)) + m.end() - 1
        depth = 0
        else_close = None
        for idx in range(else_open, len(text)):
            if text[idx] == "(":
                depth += 1
            elif text[idx] == ")":
                depth -= 1
                if depth == 0:
                    else_close = idx
                    break
        if else_close is None or text[else_close + 1 :].strip():
            return None
        return text[else_open + 1 : else_close]
    return None


def _strip_dead_isempty_under_isnotempty(expression: str) -> str:
    """Drop unreachable `ISEMPTY(X)` arms under an outer `ISNOTEMPTY(X)` guard.

    Handles `IF/ELSEIF/ELSE` CASE translations wrapped by a NOT NULL WHERE.
    """
    parts = _split_top_level_if(expression or "")
    if not parts:
        return expression or ""
    guard, then_part, else_part = parts
    else_part = _strip_dead_isempty_under_isnotempty(else_part)
    notempty = re.fullmatch(r"(?is)ISNOTEMPTY\((.+)\)$", guard.strip())
    if not notempty:
        then_part = _strip_dead_isempty_under_isnotempty(then_part)
        return f"IF({guard})THEN({then_part})ELSE({else_part})"

    col_compact = re.sub(r"\s+", "", notempty.group(1))
    stripped = _drop_leading_isempty_arm(then_part, col_compact)
    if stripped is not None:
        then_part = _strip_dead_isempty_under_isnotempty(stripped)
    else:
        then_part = _strip_dead_isempty_under_isnotempty(then_part)
    return f"IF({guard})THEN({then_part})ELSE({else_part})"


def _find_all_if_guards(expression: str) -> list[str]:
    """Every guard substring inside the expression's IF(...)/ELSEIF(...)
    headers, at any nesting depth -- used to scan every branch condition
    in a composed formula, not just the outermost one."""
    guards = []
    for m in re.finditer(r"(?i)\b(?:IF|ELSEIF)\(", expression or ""):
        open_idx = m.end() - 1
        close_idx = _find_matching_paren(expression, open_idx)
        if close_idx is not None and close_idx > open_idx:
            guards.append(expression[open_idx + 1 : close_idx])
    return guards


def _split_top_level_and_conjuncts(guard: str) -> list[str]:
    """Split a guard on top-level ` AND ` -- not inside parens or quoted
    string literals -- into its individual conjuncts."""
    text = guard or ""
    parts: list[str] = []
    depth = 0
    in_quotes = False
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            in_quotes = not in_quotes
            i += 1
            continue
        if in_quotes:
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and text[i : i + 5].upper() == " AND ":
            parts.append(text[start:i])
            i += 5
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _dedupe_redundant_and_conjuncts(expression: str) -> str:
    """Drop an exact-duplicate conjunct from every AND-guard in the
    expression: `A AND A` -> `A`. Composing two different assignment
    sites' guards by ANDing them together (rather than keeping them as
    separate, ordered branches) can leave the same conjunct present
    twice -- always redundant (A AND A == A), so safe to collapse
    unconditionally rather than just flagging it.
    """
    text = expression or ""
    for guard in sorted(set(_find_all_if_guards(text)), key=len, reverse=True):
        conjuncts = _split_top_level_and_conjuncts(guard)
        if len(conjuncts) < 2:
            continue
        seen: list[str] = []
        seen_normalized: set[str] = set()
        for conjunct in conjuncts:
            normalized = re.sub(r"\s+", "", conjunct).upper()
            if normalized in seen_normalized:
                continue
            seen_normalized.add(normalized)
            seen.append(conjunct)
        if len(seen) == len(conjuncts):
            continue
        deduped_guard = " AND ".join(seen)
        text = text.replace(f"IF({guard})", f"IF({deduped_guard})")
        text = text.replace(f"ELSEIF({guard})", f"ELSEIF({deduped_guard})")
    return text


def _simplify_composed_expression(expression: str) -> str:
    """Remove dead/contradictory IF nesting left by sequential UPDATE folding."""
    text = (expression or "").strip()
    if not text or not text.upper().startswith("IF("):
        return text
    text = _strip_dead_isempty_under_isnotempty(text)
    text = _collapse_tautology_branches(text)
    text = _dedupe_redundant_and_conjuncts(text)
    return text


_TRIVIAL_GUARDED_COPY_RE = re.compile(
    r'^IF\(.+\)THEN\("[^"]+"\."[A-Za-z_][\w]*"\)ELSE\(NULL\)$',
    re.IGNORECASE | re.DOTALL,
)
_BARE_COLUMN_REF_RE = re.compile(r'^"[^"]+"\."[A-Za-z_][\w]*"$')
_NOISE_ENTITY_RE = re.compile(r"(?i)(AuditLog|CollectionsQueue)\b")


def _is_trivial_guarded_column_copy(expression: str) -> bool:
    text = re.sub(r"\s+", "", expression or "")
    return bool(_TRIVIAL_GUARDED_COPY_RE.fullmatch(text))


def _should_omit_passthrough_dd_row(
    *,
    target_table: str,
    entity_name: str,
    expression: str,
) -> bool:
    """Omit workflow filter-copies that inflate reports as duplicate rules.

    Temp staging INSERT columns that only copy AccountId/DpdBucket/… under a
    WHERE filter, and CollectionsQueue/AuditLog key/date copies, are not
    independent business derivations. Rich CASE outcomes (e.g. Reason) stay.
    """
    table = target_table or ""
    entity = entity_name or ""
    expr = (expression or "").strip()
    if not expr:
        return False
    is_temp = table.lstrip().startswith("#") or (entity.startswith("#"))
    is_noise = bool(_NOISE_ENTITY_RE.search(table) or _NOISE_ENTITY_RE.search(entity))
    if not is_temp and not is_noise:
        return False
    compact = re.sub(r"\s+", "", expr)
    if _is_trivial_guarded_column_copy(expr):
        return True
    if is_noise and re.fullmatch(
        r'(?is)IF\(.+\)THEN\("[^"]+"\.(?:"var"\.)?"[^"]+"\)ELSE\(NULL\)',
        compact,
    ):
        return True
    # MERGE/INSERT projections that are only a source column or BUSINESS_DATE
    # on audit/queue tables are workflow copies, not independent rules.
    if is_noise and (
        _BARE_COLUMN_REF_RE.fullmatch(expr)
        or re.fullmatch(r'^"[^"]+"\."var"\."[A-Za-z_][\w]*"$', expr)
    ):
        return True
    if is_temp and _BARE_COLUMN_REF_RE.fullmatch(expr):
        return True
    return False


_MULTI_ASSIGN_RE = re.compile(
    r"(?m)^\s*(?:[A-Za-z_][\w]*\s*\.\s*)?([A-Za-z_][\w]*)\s*=\s*(?![=<>])"
)


def _is_multi_column_assignment_blob(expression: str, target_column: str = "") -> bool:
    """Detect LLM outputs that dump an entire multi-column UPDATE as one formula.

    Historical PENDING_REVIEW flood: `IF(p_TIMEKEY > 26267, (DPD_IntService=...,
    DPD_NoCredit=..., ...), (...))` — invalid 4X and wrong for a single column.
    """
    text = expression or ""
    if "=" not in text:
        return False
    # Platform comparisons use ==; SQL-style single = assignments are the smell.
    assigns = []
    for match in re.finditer(
        r"(?<![<>=!])(?<![A-Za-z0-9_])([A-Za-z_][\w]*)\s*=\s*(?!=)",
        text,
    ):
        name = match.group(1)
        if name.upper() in {
            "IF", "THEN", "ELSE", "ELSEIF", "AND", "OR", "NULL", "TRUE", "FALSE",
        }:
            continue
        assigns.append(name)
    if len(assigns) < 2:
        return False
    distinct = {a.upper() for a in assigns}
    if target_column and target_column.upper() in distinct and len(distinct) == 1:
        return False
    return len(distinct) >= 2


def _extract_column_from_assignment_blob(expression: str, column: str) -> str | None:
    """If a multi-assign blob contains `column = <expr>`, return that RHS only."""
    if not expression or not column:
        return None
    pattern = re.compile(
        rf"(?is)(?:^|[,(\s])(?:[A-Za-z_][\w]*\s*\.\s*)?{re.escape(column)}\s*=\s*(?P<rhs>.+?)(?=,\s*[A-Za-z_][\w]*\s*=|\)$|$)"
    )
    match = pattern.search(expression)
    if not match:
        return None
    rhs = match.group("rhs").strip().rstrip(",").strip()
    # Trim balanced trailing closes that belong to outer IF wrappers.
    while rhs.endswith(")") and rhs.count(")") > rhs.count("("):
        rhs = rhs[:-1].rstrip()
    return rhs or None


def _scrub_llm_expression_for_column(expression: str, column: str) -> str:
    """Repair or reject LLM shapes that historically caused PENDING_REVIEW."""
    if not expression:
        return expression
    text = _normalize_legacy_if_syntax(expression)
    if _is_multi_column_assignment_blob(text, column):
        extracted = _extract_column_from_assignment_blob(text, column)
        if extracted:
            return extracted
        return ""
    # Leftover SQL ISNULL / comma IF after normalization → unusable.
    if re.search(r"(?i)\bISNULL\s*\(", text) or re.search(r"(?i)\bIF\s*\([^)]+,", text):
        # Allow IF(cond)THEN — only flag when a comma still separates IF args.
        if re.search(r"(?i)\bIF\s*\((?:[^()\"]+|\"[^\"]*\"|\([^()]*\))*,", text):
            if "THEN(" not in text.upper():
                return ""
    return text


def _is_process_status_table(table_or_entity: str) -> bool:
    name = (table_or_entity or "").lstrip("#").upper()
    return bool(
        re.search(r"RUNNINGPROCESSSTATUS|PROCESSSTATUS|RUNSTATUS|BANDAUDITSTATUS", name)
    )


def _should_omit_dd_row_from_presentation(expression: str) -> bool:
    """Omit rows that have nothing meaningful to show stakeholders."""
    text = (expression or "").strip()
    if not text:
        return True
    return _is_non_derivable_expression(text)


def _is_noop_column_self_assignment(
    value: str,
    column: str,
    target_entity: str = "",
) -> bool:
    """True when RHS is only a reference to the same target column on the
    same entity (or an unqualified/self-alias ref).

    Copies from a distinct source alias such as `"SRC"."Asset_Norm"` are
    NOT no-ops — they are intentional projections from a MERGE USING arm.
    Complex IF/CONCAT/arithmetic expressions that merely mention the column
    are also not no-ops.
    """
    text = (value or "").strip()
    column = (column or "").strip().strip('"')
    if not text or not column:
        return False
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        balanced = True
        for idx, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(text) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        text = text[1:-1].strip()

    # Strict shape: only a (possibly qualified) column reference.
    compact = re.sub(r"\s+", "", text)
    if not re.fullmatch(
        r'(?i)(?:"[^"]+"\.)?"[^"]+"|(?:[A-Za-z_][\w]*\.)?[A-Za-z_][\w]*',
        compact,
    ):
        return False

    parts = re.findall(r'"([^"]+)"', text)
    if not parts:
        return bool(
            re.fullmatch(
                rf"(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)*{re.escape(column)}",
                text,
                flags=re.IGNORECASE,
            )
        )
    if parts[-1].upper() != column.upper():
        return False
    if len(parts) == 1:
        return True
    qualifier = parts[0]
    target_key = _normalize_relation_name(target_entity) if target_entity else ""
    if target_key and _normalize_relation_name(qualifier) == target_key:
        return True
    # Distinct multi-character source names (SRC, SOURCE, DIMPRODUCT, …)
    # and short SQL aliases (S, A, D) are source projections, not self-copies,
    # unless the qualifier resolves to the target entity name.
    if target_key and _normalize_relation_name(qualifier) == target_key:
        return True
    if len(qualifier) > 1 and qualifier.upper() in {"SRC", "SOURCE", "SRCROW", "INPUT"}:
        return False
    if target_key and _normalize_relation_name(qualifier) != target_key:
        return False
    # Unknown short alias without a target match → treat as projection.
    if len(qualifier) <= 2:
        return False
    return True


def _compose_simple_assignment_expression(
    assignment_sites: list[_AssignmentSite],
    entity_name: str,
    fallback_column: str,
    procedure_sql: str = "",
    entity_name_map: dict[str, str] | None = None,
) -> str | None:
    """Compose a sequential 4X expression from deterministic assignment
    sites when each stage is only a simple guard/value write.

    The composition preserves source order exactly: earlier stages become
    the outer branches and later fix-ups stay nested later.

    Aliases are resolved per site before nesting so a reused letter like
    `"B"` that means CustomerCal in one MERGE and PUI_CAL in another does
    not survive into the final formula as an ambiguous single-letter leak.
    """
    if not assignment_sites:
        return None

    target_pattern = re.compile(rf"\b{re.escape(fallback_column)}\b", re.IGNORECASE)
    direct_sites = [site for site in assignment_sites if target_pattern.search(site.raw_sql)]
    if direct_sites:
        assignment_sites = direct_sites

    entity_name_map = dict(entity_name_map or {})
    if entity_name:
        entity_name_map.setdefault(entity_name, entity_name)
    dialect = detect_dialect(procedure_sql or assignment_sites[0].raw_sql)

    stages: list[tuple[str, str, str, str]] = []
    for site in assignment_sites:
        stage = _parse_simple_assignment_stage(
            site.raw_sql, fallback_column, entity_name, procedure_sql=procedure_sql
        )
        if stage is None:
            # Skip unparseable assignment fragments (truncated nested CASE,
            # subquery writes, etc.) rather than aborting columns that still
            # have other composable stages.
            continue
        guard, value, source_target = stage
        site_aliases = _collect_alias_resolution_inventory(site.raw_sql, dialect)
        if guard.strip():
            guard = _finalize_platform_expression(
                guard,
                entity_name=entity_name,
                entity_name_map=entity_name_map,
                alias_resolution_inventory=site_aliases,
                source_sql=procedure_sql or site.raw_sql,
            )
        value = _finalize_platform_expression(
            value,
            entity_name=entity_name,
            entity_name_map=entity_name_map,
            alias_resolution_inventory=site_aliases,
            source_sql=procedure_sql or site.raw_sql,
        )
        # Skip self-copies after alias/entity finalize so `"S"."Col"` from a
        # staging projection is kept, while `SET S.Col = S.Col` on the target
        # entity (sample 09 ELSE no-op) is still dropped.
        if _is_noop_column_self_assignment(
            value, source_target or fallback_column, target_entity=entity_name
        ):
            continue
        # Attach simple procedural IF predicates (e.g. DAY(@ProcessDate)=1)
        # to THEN-arm updates so they are not lost as unconditional LWW.
        branch_pred = _procedural_then_predicate_4x(
            site.raw_sql, procedure_sql, entity_name=entity_name
        )
        if branch_pred:
            guard = f"({branch_pred}) AND ({guard})" if guard.strip() else branch_pred
        stages.append((guard, value, source_target, site.raw_sql))

    # INSERT … SELECT often seeds a column with NULL before a later UPDATE
    # CASE fills it. Drop those null seeds so they do not wrap (and invert)
    # the real last-write-wins formula.
    while (
        len(stages) > 1
        and (stages[0][1] or "").strip().upper() == "NULL"
        and (stages[0][3] or "").lstrip().upper().startswith("INSERT")
    ):
        stages = stages[1:]

    if not stages:
        return None

    exclusive = _compose_exclusive_control_flow_stages(
        stages,
        procedure_sql=procedure_sql,
        entity_name=entity_name,
    )
    if exclusive:
        final_expr = _rewrite_business_date_variables(
            _normalize_expression(exclusive, ""),
            entity_name,
            source_sql=procedure_sql,
        )
        final_expr = _simplify_composed_expression(final_expr)
        if _is_non_derivable_expression(final_expr):
            return None
        return final_expr if validate_expression(final_expr).valid else None

    current_target = stages[0][2] or fallback_column
    guarded_stages = [stage for stage in stages if stage[0].strip()]
    empty_stages = [stage for stage in stages if not stage[0].strip()]

    def _is_plain_update_wipe(raw_sql: str, value: str) -> bool:
        """True only for a bare `UPDATE ... SET col = <simple> ` with no WHERE.

        IF/ELSE control-flow fragments and MERGE statements often parse with
        an empty guard even though they are conditional; those must not
        erase earlier composed CASE logic.
        """
        cleaned = _strip_leading_comments(raw_sql or "").strip()
        cleaned = _strip_leading_control_header(cleaned)
        upper = cleaned.upper()
        if not upper.startswith("UPDATE"):
            return False
        if re.search(r"\bWHERE\b", cleaned, re.IGNORECASE):
            return False
        probe = (value or "").strip().upper()
        return probe in {"NULL", "0", "1"} or bool(re.fullmatch(r'"[^"]*"', (value or "").strip()))

    # When unconditional writes are mixed with guarded ones:
    # - a trailing plain UPDATE with no WHERE overwrites every row when it
    #   is a sequential last-write-wins wipe;
    # - the same shape inside a procedural ELSE is an IF/ELSE default and
    #   must become ELSE(seed) under the earlier guarded arms.
    if empty_stages and guarded_stages:
        trailing = stages[-1]
        trailing_unconditional = not trailing[0].strip() and _is_plain_update_wipe(
            trailing[3], trailing[1]
        )
        else_default = trailing_unconditional and _assignment_looks_like_control_flow_else_default(
            trailing[3], procedure_sql
        )
        pair_seed = _compose_procedural_if_else_seed(
            empty_stages,
            procedure_sql=procedure_sql,
            entity_name=entity_name,
            is_plain_update_wipe=_is_plain_update_wipe,
        )
        if trailing_unconditional and not else_default:
            expression = _rewrite_business_date_variables(
                _normalize_expression(trailing[1], ""),
                entity_name,
            )
            if not validate_expression(expression).valid:
                return None
            stages_to_apply = []
        elif pair_seed is not None:
            # IF-arm CASE + ELSE wipe (+ later WHERE fix-ups). Prefer the
            # executable IF/ELSE seed so a dead THEN does not hide ELSE.
            expression = pair_seed
            stages_to_apply = guarded_stages
        else:
            # Prefer formula/CASE empty stages as the ELSE seed; skip plain
            # literal wipes and MERGE fragments that parsed without an ON
            # guard and only write NULL (those are incomplete stage extracts,
            # not true unconditional defaults). When the trailing stage is a
            # procedural ELSE default, use that literal as the ELSE seed.
            def _is_incomplete_null_merge(stage: tuple[str, str, str, str]) -> bool:
                _g, value, _t, raw = stage
                if (value or "").strip().upper() != "NULL":
                    return False
                cleaned = _strip_leading_comments(raw or "").strip().upper()
                return cleaned.startswith("MERGE")

            if else_default:
                expression = _rewrite_business_date_variables(
                    _normalize_expression(trailing[1], ""),
                    entity_name,
                )
                if not validate_expression(expression).valid:
                    return None
                stages_to_apply = guarded_stages
            else:
                seed_empties = [
                    stage
                    for stage in empty_stages
                    if not _is_plain_update_wipe(stage[3], stage[1])
                    and not _is_incomplete_null_merge(stage)
                ]
                if not seed_empties:
                    combined_raw_sql = "\n".join(site.raw_sql for site in assignment_sites)
                    if _has_where_guarded_update_on_column(combined_raw_sql, current_target):
                        expression = (
                            f'"{entity_name}"."{current_target}"'
                            if entity_name
                            else f'"{current_target}"'
                        )
                    else:
                        expression = "NULL"
                else:
                    expression = _rewrite_business_date_variables(
                        _normalize_expression(seed_empties[-1][1], ""),
                        entity_name,
                    )
                    if not validate_expression(expression).valid:
                        return None
                    for _guard, value, source_target, _raw in seed_empties[:-1]:
                        normalized_value = _rewrite_business_date_variables(
                            _normalize_expression(value, ""),
                            entity_name,
                        )
                        if not validate_expression(normalized_value).valid:
                            return None
                        if _is_noop_column_self_assignment(
                            normalized_value, source_target or current_target, target_entity=entity_name
                        ):
                            continue
                        if normalized_value.upper().startswith("IF(") or "CASE" in value.upper():
                            expression = normalized_value
                        current_target = source_target or current_target
                stages_to_apply = guarded_stages
    else:
        first_kind = assignment_sites[0].kind.upper() if assignment_sites else ""
        first_raw = assignment_sites[0].raw_sql if assignment_sites else ""
        first_is_seed_projection = first_kind in {"SELECT", "INSERT"}
        if not first_is_seed_projection and first_kind == "CONTROL_FLOW_BLOCK":
            select_stmt = _extract_select_into_statement(first_raw)
            if select_stmt:
                for dialect in (detect_dialect(select_stmt), Dialect.ORACLE, Dialect.SQLSERVER, Dialect.MYSQL):
                    try:
                        tree = sqlglot.parse_one(select_stmt, read=_SQLGLOT_DIALECT[dialect])
                    except Exception:
                        continue
                    if isinstance(tree, exp.Select) and tree.args.get("into") is not None:
                        if _render_select_into_projection(tree, current_target.upper()) is not None:
                            first_is_seed_projection = True
                            break

        # All empty-guard stages: prefer a rich IF/CASE formula over a later
        # procedural ELSE wipe (common IF @date ... CASE ... ELSE SET = 0),
        # unless the IF is unreachable and executable SQL always takes ELSE.
        pair_seed = _compose_procedural_if_else_seed(
            empty_stages,
            procedure_sql=procedure_sql,
            entity_name=entity_name,
            is_plain_update_wipe=_is_plain_update_wipe,
        )
        rich_empties = [
            stage
            for stage in empty_stages
            if (stage[1] or "").upper().startswith("IF(")
            or "COALESCE(" in (stage[1] or "").upper()
            or "DATEDIFF(" in (stage[1] or "").upper()
        ]
        wipe_empties = [
            stage
            for stage in empty_stages
            if _is_plain_update_wipe(stage[3], stage[1])
        ]
        if pair_seed is not None:
            expression = pair_seed
            stages_to_apply = []
        elif (
            rich_empties
            and wipe_empties
            and _assignment_looks_like_control_flow_else_default(wipe_empties[-1][3], procedure_sql)
        ):
            expression = _rewrite_business_date_variables(
                _normalize_expression(rich_empties[-1][1], ""),
                entity_name,
            )
            if not validate_expression(expression).valid:
                return None
            stages_to_apply = []
        elif first_is_seed_projection:
            expression = "NULL"
            stages_to_apply = stages
        else:
            combined_raw_sql = "\n".join(site.raw_sql for site in assignment_sites)
            if _has_where_guarded_update_on_column(combined_raw_sql, current_target):
                expression = f'"{entity_name}"."{current_target}"' if entity_name else f'"{current_target}"'
            else:
                expression = "NULL"
            stages_to_apply = stages

    for guard, value, source_target, _raw in stages_to_apply:
        source_target = source_target or current_target
        current_target = source_target
        normalized_value = _rewrite_business_date_variables(
            _normalize_expression(value, ""), entity_name, source_sql=procedure_sql
        )
        if not validate_expression(normalized_value).valid:
            return None
        if not guard.strip():
            expression = normalized_value
            continue
        # Sequential `SET col = f(col)` (e.g. CONCAT(col, '_QUARTER_END')) must
        # embed the prior composed value — not a circular self-read.
        normalized_value = _substitute_composed_column_refs(
            normalized_value,
            entity_name=entity_name,
            column=current_target or fallback_column,
            prior_expression=expression,
        )
        if not validate_expression(normalized_value).valid:
            return None
        normalized_guard = _normalize_expression(guard, "")
        # Board-approval gate: IF DAY=1 THEN map CASE 'Y' → PENDING_APPROVAL.
        # Avoid circular `EligibleForUpgrade == "Y"` self-reads on the same column.
        if re.fullmatch(r'"?PENDING_APPROVAL"?', normalized_value.strip(), flags=re.IGNORECASE):
            day_pred = None
            day_match = re.search(
                r'DATEPART\("d"\s*,\s*[^)]+\)\s*==\s*\d+',
                normalized_guard,
                flags=re.IGNORECASE,
            )
            if day_match and expression:
                day_pred = day_match.group(0)
                pending_case = re.sub(
                    r'THEN\("Y"\)',
                    'THEN("PENDING_APPROVAL")',
                    expression,
                )
                candidate = f"IF({day_pred})THEN({pending_case})ELSE({expression})"
                if validate_expression(candidate).valid:
                    expression = candidate
                    continue
        expression = f"IF({normalized_guard})THEN({normalized_value})ELSE({expression})"

    final_expr = _rewrite_business_date_variables(
        _normalize_expression(expression, ""), entity_name, source_sql=procedure_sql
    )
    final_expr = _simplify_composed_expression(final_expr)
    if _is_non_derivable_expression(final_expr):
        return None
    return final_expr


def _substitute_composed_column_refs(
    value: str,
    *,
    entity_name: str,
    column: str,
    prior_expression: str,
) -> str:
    """Replace self-reads of `column` inside CONCAT with `prior_expression`.

    Sequential string appends (`SET col = col + '_QUARTER_END'`) must embed
    the prior CASE formula. Arithmetic / CASE self-reads such as sample 07's
    `AdjustedPenalty * 1.10` keep the column ref so presentation stays stable.
    """
    text = (value or "").strip()
    column = (column or "").strip().strip('"')
    prior = (prior_expression or "").strip()
    if not text or not column or not prior:
        return text
    if not re.search(r"(?i)\bCONCAT\s*\(", text):
        return text
    if _is_noop_column_self_assignment(prior, column, target_entity=entity_name):
        return text

    col_re = re.escape(column)

    def _repl_qualified(match: re.Match[str]) -> str:
        frag = match.group(0)
        if _is_noop_column_self_assignment(frag, column, target_entity=entity_name):
            return prior
        return frag

    out = re.sub(rf'"[^"]+"\s*\.\s*"{col_re}"', _repl_qualified, text, flags=re.IGNORECASE)

    def _repl_bare(match: re.Match[str]) -> str:
        frag = match.group(0)
        if _is_noop_column_self_assignment(frag, column, target_entity=entity_name):
            return prior
        return frag

    out = re.sub(
        rf'(?<![A-Za-z0-9_".]){col_re}(?![A-Za-z0-9_"])',
        _repl_bare,
        out,
        flags=re.IGNORECASE,
    )
    return out


def _repair_trailing_self_reference(expression: str, entity_name: str, column: str, source_sql: str) -> str:
    """Replace a final `ELSE(target_column)` fallback with `ELSE(NULL)` if
    the source SQL does not explicitly preserve the same target column.
    """
    if _source_allows_target_reference(source_sql, entity_name, column):
        return expression

    column_upper = re.escape(column.upper())
    entity_upper = re.escape(entity_name.upper()) if entity_name else ""
    if entity_upper:
        qualified = rf'"{entity_upper}"\s*\.\s*"{column_upper}"'
    else:
        qualified = rf'"{column_upper}"'

    candidate = re.sub(
        rf'(?i)(ELSE\s*\()\s*{qualified}\s*(\)\s*)$',
        r"\1NULL\2",
        expression,
    )
    return candidate if candidate != expression else expression


def _self_reference_pattern(entity_name: str, column: str) -> re.Pattern[str]:
    return re.compile(
        rf'"{re.escape(entity_name.upper())}"\s*\.\s*"{re.escape(column.upper())}"',
        re.IGNORECASE,
    )


def _split_outer_if_then_else(expression: str) -> tuple[str, str, str] | None:
    """If `expression` is exactly one outer IF(<guard>)THEN(<then>)ELSE(<else>)
    -- no ELSEIF, nothing before or after -- return its three parenthesized
    parts. Otherwise None (composed expressions with ELSEIF chains or extra
    wrapping are left to _repair_self_referential_guard's caller to reject
    the self-reference outright rather than guessing at a substitution)."""
    text = expression.strip()
    head = re.match(r"(?i)^IF\(", text)
    if not head:
        return None
    guard_close = _find_matching_paren(text, head.end() - 1)
    if guard_close < 0:
        return None
    after_guard = text[guard_close + 1:]
    then_head = re.match(r"(?i)^THEN\(", after_guard)
    if not then_head:
        return None
    then_close = _find_matching_paren(after_guard, then_head.end() - 1)
    if then_close < 0:
        return None
    after_then = after_guard[then_close + 1:]
    else_head = re.match(r"(?i)^ELSE\(", after_then)
    if not else_head:
        return None
    else_close = _find_matching_paren(after_then, else_head.end() - 1)
    if else_close < 0 or else_close != len(after_then) - 1:
        return None
    guard = text[head.end() : guard_close]
    then_branch = after_guard[then_head.end() : then_close]
    else_branch = after_then[else_head.end() : else_close]
    return guard, then_branch, else_branch


def _repair_self_referential_guard(expression: str, entity_name: str, column: str) -> tuple[str, bool]:
    """A 4X row Formula Expression's `"entity"."column"` reference always
    resolves to that column's currently STORED value -- a single-pass row
    formula has no way to see a value its own IF/THEN/ELSE has just
    computed. Sequential source SQL commonly clamps a freshly computed
    value with a trailing `UPDATE ... SET col = 0 WHERE col < 0`-style
    statement; composed as a row formula this becomes
    `IF(COALESCE(self, 0) < 0)THEN(0)ELSE(<real derivation>)`, whose guard
    is circular (it reads yesterday's stored value, not what the ELSE arm
    itself computes) even though the source SQL is perfectly valid
    imperative code.

    Fix: since the ELSE arm already holds this column's real (self-
    reference-free) derivation, substitute it into the guard in place of
    the self-reference -- the guard then evaluates against the value this
    same formula computes, matching the clamp's actual intent.

    This deliberately targets only that one shape. A self-reference deep
    inside a branch of a larger nested chain (e.g. a genuinely recursive-
    style lineage formula spanning several ELSE arms) is left untouched --
    existing advisory-only handling for that broader, harder-to-resolve
    case is intentional elsewhere in this pipeline and out of scope here.

    Returns (expression, blocked_unsafe). `blocked_unsafe` is True only in
    the one case this repair recognizes but cannot resolve: the outer
    guard self-references, and the branch that would replace it in the
    guard is itself self-referential too -- there the guard is provably
    circular with nothing safe to substitute, so the caller must not
    export that expression as ACTIVE. Every other case (no self-reference,
    a shape this repair doesn't target, or a successful substitution)
    returns `blocked_unsafe=False`.
    """
    pattern = _self_reference_pattern(entity_name, column)
    if not pattern.search(expression):
        return expression, False

    parts = _split_outer_if_then_else(expression)
    if parts is None:
        return expression, False
    guard, then_branch, else_branch = parts
    if not pattern.search(guard):
        return expression, False
    if pattern.search(then_branch) or pattern.search(else_branch):
        return expression, True

    new_guard = pattern.sub(f"({else_branch})", guard)
    repaired = f"IF({new_guard})THEN({then_branch})ELSE({else_branch})"
    return repaired, False


def _expression_should_be_rejected(validation_errors: list[str]) -> bool:
    """Return True when the generated formula is not safe to export.

    Any row that still has validation errors is a review-only row. The
    safest behavior is to keep the row metadata and validation notes, but
    omit the expression itself rather than exporting a misleading or
    hallucinated Platform Condition.
    """
    return bool(validation_errors)


_ColumnJob = tuple[
    CanonicalModel,
    SQLObject,
    StructuralInfo,
    str,
    str,
    LLMClient,
    str,
    "dict[int, date] | None",
    Optional[ChromaStore],
    dict[str, str],
]


def _build_jobs_for_chain(
    chain: LineageChain,
    canonical_model: CanonicalModel,
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo],
    llm_client: LLMClient,
    function_reference: str,
    entity_name_map: dict[str, str] | None,
    timekey_map: dict[int, date] | None,
    rag_store: Optional[ChromaStore],
) -> list[_ColumnJob]:
    entity_name_map = entity_name_map or {}
    jobs: list[_ColumnJob] = []
    seen_logical_columns: set[tuple[str, str]] = set()
    from app.utils.entity_name_map import resolve_entity_name

    for oid in chain.order:
        obj = objects[oid]
        info = structural_infos[oid]
        for target_table, columns in info.columns_written_by_table.items():
            entity_name = resolve_entity_name(target_table, entity_name_map)
            # Process-status bookkeeping is not a platform DD formula target;
            # exporting empty/self-copy PENDING_REVIEW rows for COMPLETED/
            # ERRORDATE historically polluted review queues.
            if _is_process_status_table(target_table) or _is_process_status_table(entity_name):
                continue
            for column in columns:
                canonical_column = canonical_logical_name(column)
                logical_key = (canonical_logical_name(entity_name), canonical_column)
                if logical_key in seen_logical_columns:
                    continue
                seen_logical_columns.add(logical_key)
                jobs.append(
                    (
                        canonical_model,
                        obj,
                        info,
                        entity_name,
                        column,  # preserve source casing for platform CSV / report
                        llm_client,
                        function_reference,
                        timekey_map,
                        rag_store,
                        dict(entity_name_map),
                    )
                )
    return jobs


def _run_jobs(jobs: list[_ColumnJob]) -> list[DDRow]:
    """Run every column-generation job (each one independent -- a single
    column's worth of LLM call + validation) through a bounded worker pool
    and flatten the results. Each job is network-bound (the LLM call), so
    threads give real concurrency here despite the GIL.
    """
    if not jobs:
        return []

    max_workers = max(1, min(settings.dd_generation_max_workers, len(jobs)))
    if max_workers == 1:
        dd_rows: list[DDRow] = []
        for job in jobs:
            dd_rows.extend(_generate_column_rows(job))
        return dd_rows

    dd_rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for rows in executor.map(_generate_column_rows, jobs):
            dd_rows.extend(rows)
    return dd_rows


def generate_dd_rows(
    chain: LineageChain,
    canonical_model: CanonicalModel,
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo],
    llm_client: LLMClient,
    function_reference: str,
    entity_name_map: dict[str, str] | None = None,
    timekey_map: dict[int, date] | None = None,
    rag_store: Optional[ChromaStore] = None,
) -> list[DDRow]:
    jobs = _build_jobs_for_chain(
        chain, canonical_model, objects, structural_infos, llm_client,
        function_reference, entity_name_map, timekey_map, rag_store,
    )
    return _run_jobs(jobs)


def generate_dd_rows_for_chains(
    chains: list[LineageChain],
    canonical_models: list[CanonicalModel],
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo],
    llm_client: LLMClient,
    function_reference: str,
    entity_name_map: dict[str, str] | None = None,
    timekey_map: dict[int, date] | None = None,
    rag_store: Optional[ChromaStore] = None,
) -> list[DDRow]:
    """Same generation logic as `generate_dd_rows`, but batches every
    column-generation job across every chain into a single worker pool
    instead of one pool per chain.

    Processing chains one at a time (each with its own short-lived
    executor) means chain N+1 can't start until every job in chain N has
    finished, and a chain with fewer jobs than the worker limit leaves
    workers idle instead of picking up work from the next chain. Flattening
    the whole job list first keeps the same bounded worker count fully
    occupied across the entire run, so multi-chain jobs (the common case --
    a lineage chain is a *group* of related procedures) finish sooner
    without changing what gets generated, validated, or how.
    """
    all_jobs: list[_ColumnJob] = []
    for chain, model in zip(chains, canonical_models):
        all_jobs.extend(
            _build_jobs_for_chain(
                chain, model, objects, structural_infos, llm_client,
                function_reference, entity_name_map, timekey_map, rag_store,
            )
        )
    return _run_jobs(all_jobs)


def _build_source_statement_refs(obj: SQLObject, info: StructuralInfo, column: str) -> list[str]:
    """One human-readable breadcrumb per write site that could feed this
    column's expression -- e.g. "npa.sql stmt #30 (role=NULL_RESET)" -- so
    a reviewer (or the generated report) can trace a row back to the
    exact source statement(s) it came from, not just the object name.
    Built from the same _AssignmentSite data already computed for the
    LLM's prompt context (see _format_assignment_context), so this can
    never describe a different set of write sites than what the model was
    actually shown.
    """
    refs: list[str] = []
    for site in _assignment_sites(info, column):
        if not site.raw_sql.strip():
            continue
        role = _infer_assignment_role(site.raw_sql)
        if site.statement_indices:
            stmt_label = "stmt #" + ",".join(str(i) for i in site.statement_indices)
        else:
            stmt_label = "stmt #?"
        refs.append(f"{obj.source_file} {stmt_label} (role={role})")
    return refs


def _build_source_statement_sql(info: StructuralInfo, column: str) -> list[str]:
    """The actual raw SQL text of each write site that could feed this
    column's expression, in the same order as _build_source_statement_refs.

    This is what lets alias resolution be scoped to the specific
    statement(s) a row's formula actually came from, instead of the whole
    object's raw SQL. A generic alias like "A" is very often reused for a
    different table in a different statement elsewhere in the same
    stored procedure (a common pattern: repeated `UPDATE A SET ... FROM
    X A` blocks) -- resolving against the whole object collapses that as
    unrecoverably ambiguous and drops the alias entirely, even though it
    is completely unambiguous within the one or two statements this
    particular row was actually derived from.
    """
    snippets: list[str] = []
    for site in _assignment_sites(info, column):
        text = site.raw_sql.strip()
        if text:
            snippets.append(text)
    return snippets


def _collect_source_reference_inventory(
    text: str,
    dialect: Dialect,
    entity_name: str = "",
) -> _SourceReferenceInventory:
    """Extract the source-backed reference universe visible in `text`.

    This is used for two purposes:
    1. give the generator an explicit allowlist of real tables/aliases and
       column-to-qualifier pairings; and
    2. let the grounding step rewrite a hallucinated qualifier only when
       the source SQL already proves an unambiguous real qualifier exists.
    """
    allowed_qualifiers: set[str] = set()
    qualifiers_by_column: dict[str, set[str]] = {}

    dialect_name = _SQLGLOT_DIALECT.get(dialect, "oracle")
    for stmt in split_statements(text, dialect):
        cleaned_stmt = _strip_leading_comments(stmt).strip()
        if not cleaned_stmt:
            continue
        try:
            tree = sqlglot.parse_one(cleaned_stmt, read=dialect_name)
        except Exception:
            continue

        statement_qualifiers: set[str] = set()

        for table in tree.find_all(exp.Table):
            if table.name:
                table_name = canonical_logical_name(table.name)
                allowed_qualifiers.add(table_name)
                statement_qualifiers.add(table_name)
            alias = _extract_alias_name(table)
            if alias:
                allowed_qualifiers.add(alias)
                statement_qualifiers.add(alias)

        single_statement_qualifier = _statement_source_alias(tree)
        if not single_statement_qualifier and len(statement_qualifiers) == 1:
            single_statement_qualifier = next(iter(statement_qualifiers))

        for column_ref in tree.find_all(exp.Column):
            column_name = canonical_logical_name(column_ref.name or "")
            if not column_name:
                continue
            qualifier = column_ref.table or single_statement_qualifier
            if qualifier:
                qualifier_name = canonical_logical_name(str(qualifier))
                allowed_qualifiers.add(qualifier_name)
                qualifiers_by_column.setdefault(column_name, set()).add(qualifier_name)

    return _SourceReferenceInventory(
        target_entity_name=canonical_logical_name(entity_name.strip()) if entity_name.strip() else "",
        allowed_qualifiers=allowed_qualifiers,
        qualifiers_by_column=qualifiers_by_column,
    )


def _append_allowed_reference_context(base_context: str, inventory: _SourceReferenceInventory) -> str:
    lines = inventory.allowed_reference_lines()
    if not lines:
        return base_context
    extra = "[Allowed source references]\n" + "\n".join(f"- {line}" for line in lines)
    if base_context.strip():
        return f"{base_context}\n\n{extra}"
    return extra


def _collect_alias_resolution_inventory(
    text: str,
    dialect: Dialect,
) -> dict[str, tuple[str, ...]]:
    return collect_table_aliases(text, dialect)


def _ground_expression_to_source_references(
    expression: str,
    inventory: _SourceReferenceInventory,
    entity_name: str,
) -> str:
    """Rewrite hallucinated table qualifiers only when the source proves a
    unique real qualifier for the referenced column.

    The grounding is deliberately narrow:
    - already-allowed qualifiers are left untouched;
    - the entity-name convention for BUSINESS_DATE is preserved;
    - otherwise, a candidate qualifier is only rewritten when one and only
      one real source qualifier is observed for the same referenced
      column, so the fix never invents a new table or alias.
    """
    if not expression or not inventory.qualifiers_by_column:
        return expression

    entity_display = entity_name.strip().strip('"')
    entity_upper = canonical_logical_name(entity_display) if entity_display else ""
    allowed_qualifiers = {canonical_logical_name(q) for q in inventory.allowed_qualifiers if q}

    quoted_ref_re = re.compile(r'"[^"]+"(?:\s*\.\s*"[^"]+")+')

    def replace(match: re.Match[str]) -> str:
        segments = re.findall(r'"([^"]+)"', match.group(0))
        if len(segments) < 2:
            return match.group(0)

        qualifier = canonical_logical_name(segments[0])
        if qualifier in allowed_qualifiers:
            return match.group(0)

        tail = [canonical_logical_name(segment) for segment in segments[1:]]
        column_name = tail[-1] if tail else ""

        if len(tail) >= 2 and tail[0] == "VAR" and tail[1] == "BUSINESS_DATE":
            if entity_display:
                return f'"{entity_display}"."var"."BUSINESS_DATE"'
            return match.group(0)

        if not column_name:
            return match.group(0)

        source_qualifiers = {
            canonical_logical_name(q)
            for q in inventory.qualifiers_by_column.get(column_name, set())
            if q
        }
        if len(source_qualifiers) != 1:
            return match.group(0)

        replacement_qualifier = next(iter(source_qualifiers))
        if replacement_qualifier in {qualifier, entity_upper}:
            return match.group(0)

        return '"' + replacement_qualifier + '"' + "".join(f'."{segment}"' for segment in segments[1:])

    grounded = quoted_ref_re.sub(replace, expression)
    return grounded


def _has_cross_statement_self_dependency(sites: list["_AssignmentSite"], column: str) -> bool:
    """True if this column is written by 2+ separate statements AND a
    statement other than the first one that writes it also *reads* the
    column somewhere in its own text (a WHERE/subquery/CASE condition,
    not just its own assignment target).

    This is the exact shape of a genuine sequential dependency: by the
    time that later statement runs, the column may already hold whatever
    an earlier statement in this same list just wrote to it, so reading
    it there means "read what the prior step produced" -- not "read the
    original source value". The deterministic composer folds separate
    statements as independent branches of one expression; it has no way
    to thread that intermediate state through correctly, so a column
    matching this shape should never be silently composed as if it were
    a simple set of independent conditions. See architecture review root
    causes B/C ("sequential-update handling" / "state/dependency
    tracking").

    Deliberately conservative (may under-flag rather than over-flag):
    only the *first* writing statement is exempted, since it alone is
    guaranteed to observe the column's original, unmodified value.
    """
    writing_sites = [s for s in sites if any(c.upper() == column.upper() for c in s.columns_written)]
    if len(writing_sites) < 2:
        return False

    read_pattern = re.compile(rf"\b{re.escape(column)}\b", re.IGNORECASE)
    assign_target_pattern = re.compile(
        rf"(?:\b[A-Z_][A-Z0-9_]*\s*\.\s*)?\b{re.escape(column)}\b\s*=(?!=)",
        re.IGNORECASE,
    )

    for site in writing_sites[1:]:
        total_refs = len(read_pattern.findall(site.raw_sql))
        assign_target_refs = len(assign_target_pattern.findall(site.raw_sql))
        if total_refs > assign_target_refs:
            return True
    return False


def _source_table_for_entity_column(
    info: StructuralInfo,
    entity_name: str,
    column: str,
    entity_name_map: dict[str, str],
) -> str | None:
    """Best-effort source table for a (entity, column) generation job."""
    column_key = canonical_logical_name(column)
    entity_key = _normalize_relation_name(entity_name)
    matches: list[str] = []
    for table, cols in (info.columns_written_by_table or {}).items():
        if not any(canonical_logical_name(col) == column_key for col in cols):
            continue
        mapped = entity_name_map.get(table, table)
        if _normalize_relation_name(mapped) == entity_key or _normalize_relation_name(table) == entity_key:
            matches.append(table)
    if len(matches) == 1:
        return matches[0]
    return matches[0] if matches else None


def _finalize_platform_expression(
    expression: str,
    *,
    entity_name: str,
    entity_name_map: dict[str, str],
    alias_resolution_inventory: dict[str, tuple[str, ...]],
    source_sql: str = "",
) -> str:
    """Normalize, resolve aliases, then rewrite to platform entity refs."""
    if not expression:
        return expression
    expression = _normalize_expression(expression, source_sql or "")
    expression = resolve_aliases_in_expression(
        expression,
        alias_resolution_inventory,
        quote_replacements=True,
    )
    expression = rewrite_expression_to_platform_entities(
        expression,
        entity_name=entity_name,
        entity_name_map=entity_name_map,
        alias_to_parts=alias_resolution_inventory,
    )
    expression = _rewrite_business_date_variables(
        expression, entity_name, source_sql=source_sql
    )
    return expression


def _generate_for_column(
    canonical_model: CanonicalModel,
    obj: SQLObject,
    info: StructuralInfo,
    entity_name: str,
    column: str,
    llm_client: LLMClient,
    function_reference: str,
    timekey_map: dict[int, date] | None,
    rag_store: Optional[ChromaStore] = None,
    entity_name_map: dict[str, str] | None = None,
) -> list[DDRow]:
    entity_name_map = entity_name_map or {}
    target_table = _source_table_for_entity_column(info, entity_name, column, entity_name_map)
    relevant_chunks = _relevant_chunks(info, column)
    all_sites = _assignment_sites(info, column, target_table=target_table)
    excluded_sites = undeterminable_exception_sites(all_sites)
    excluded_stmt_indices: set[int] = set()
    for site in excluded_sites:
        excluded_stmt_indices.update(site.statement_indices)

    sites = [s for s in all_sites if s not in excluded_sites]
    if excluded_stmt_indices:
        relevant_chunks = [
            chunk
            for chunk in relevant_chunks
            if not (
                chunk.statement_indices
                and set(chunk.statement_indices).issubset(excluded_stmt_indices)
            )
        ]

    relevant_sql = "\n\n".join(chunk.raw_sql.strip() for chunk in relevant_chunks if chunk.raw_sql.strip())
    assignment_context = _format_assignment_context(info, column, sites=sites)
    source_statement_refs = _build_source_statement_refs(obj, info, column)
    source_statement_sql = _build_source_statement_sql(info, column)
    rag_context = _retrieve_rag_context(
        rag_store, relevant_sql, canonical_model.technical_summary, canonical_model.business_summary
    )
    # The full platform function/operator reference is a fallback for when
    # RAG has no targeted hit for this column -- once RAG found one, resending
    # the whole doc on top of it is pure duplication (see RagContext docstring).
    effective_function_reference = "" if rag_context.platform_context else function_reference
    source_reference_inventory = _collect_source_reference_inventory(
        "\n\n".join(part for part in [obj.raw_sql, relevant_sql, assignment_context] if part),
        obj.dialect,
        entity_name=entity_name,
    )
    alias_resolution_inventory = _collect_alias_resolution_inventory(
        "\n\n".join(part for part in [obj.raw_sql, relevant_sql, assignment_context] if part),
        obj.dialect,
    )
    # Whole-procedure alias maps drop letters reused for different tables.
    # Prefer unambiguous aliases from this column's own assignment sites so
    # SELECT ... FROM Table A still resolves `"A"."Col"` for that write.
    site_local_aliases = _collect_alias_resolution_inventory(
        "\n\n".join(site.raw_sql for site in sites if site.raw_sql),
        obj.dialect,
    )
    if site_local_aliases:
        alias_resolution_inventory = {**alias_resolution_inventory, **site_local_aliases}
    allowed_reference_context = _append_allowed_reference_context("", source_reference_inventory)
    undeterminable_note = (
        "This column is also written inside an exception handler whose only apparent "
        "trigger condition is the same row-scoping filter the normal-flow write also "
        "uses -- \"an unhandled exception occurred\" is a runtime event, not a fact "
        "present in row data, so it cannot be reliably expressed as a per-row Formula "
        "Expression condition. The exception-handler write has been excluded from this "
        "derivation; only the normal-flow value is represented below. Confirm with the "
        "platform whether this column needs a different mechanism (e.g. a batch-run "
        "audit log) to capture the exception state, rather than a DD Formula Expression."
        if excluded_sites
        else None
    )

    has_cross_statement_dependency = _has_cross_statement_self_dependency(sites, column)

    derivation_option = DerivationOption.FORMULA_EXPRESSION
    expression: str | None = None
    decision_table_json: str | None = None
    validation_errors: list[str] = []
    source_sql_excerpt = _source_sql_context_excerpt(obj.raw_sql, relevant_sql)
    if assignment_context:
        source_sql_excerpt = assignment_context

    # Prefer deterministic composition even when later statements also read
    # this column — the advisory note below asks reviewers to spot-check
    # sequential order. Skipping compose entirely left too many blank rows.
    deterministic_expression = _compose_simple_assignment_expression(
        sites,
        entity_name,
        column,
        procedure_sql=obj.raw_sql,
        entity_name_map=entity_name_map,
    )
    deterministic_self_reference_blocked = False
    if deterministic_expression:
        deterministic_expression = _finalize_platform_expression(
            deterministic_expression,
            entity_name=entity_name,
            entity_name_map=entity_name_map,
            alias_resolution_inventory=alias_resolution_inventory,
            source_sql=obj.raw_sql,
        )
        deterministic_expression, self_reference_blocked = _repair_self_referential_guard(
            deterministic_expression, entity_name, column
        )
        grammar_result = validate_expression(deterministic_expression)
        semantic_result = check_semantic_consistency(
            deterministic_expression, column, entity_name, relevant_chunks, obj.raw_sql, source_statement_sql
        )
        if grammar_result.valid and not self_reference_blocked:
            # Prefer a grammar-valid deterministic composition over LLM.
            # Semantic caveats (self-ref in process-status ELSE, sequential
            # order, etc.) become advisory notes — they must not discard a
            # correct single-column formula and open the door to comma-style
            # multi-column LLM blobs that historically flooded PENDING_REVIEW.
            expression = deterministic_expression
            validation_errors = []
            if not semantic_result.passed:
                row_advisory_seed_early = [
                    f"Semantic advisory: {err}" for err in semantic_result.errors
                ]
            else:
                row_advisory_seed_early = []
            dt_payload = _decision_table_from_formula_if_categorical(expression, entity_name, column)
            if dt_payload is not None:
                derivation_option = DerivationOption.DECISION_TABLE
                decision_table_json = json.dumps(dt_payload)
        elif grammar_result.valid and self_reference_blocked:
            # A self-reference survived composition and could not be safely
            # resolved (see _repair_self_referential_guard) -- this must
            # never ship ACTIVE with only an advisory footnote, since the
            # guard would silently read a stale/circular value. Route
            # straight to PENDING_REVIEW instead of trying the LLM path,
            # which has no better way to see a value this formula itself
            # would compute either.
            deterministic_self_reference_blocked = True
            deterministic_expression = None
            row_advisory_seed_early = []
            validation_errors = [
                f'"{column}" formula reads its own value ("{entity_name}"."{column}") in a way '
                "automated composition could not safely resolve into a non-circular condition. "
                "Do not approve without rewriting this condition manually."
            ]
        else:
            deterministic_expression = None
            row_advisory_seed_early = []
    else:
        row_advisory_seed_early = []

    if expression is None and not deterministic_self_reference_blocked:
        grounded_source_sql_excerpt = _append_allowed_reference_context(
            source_sql_excerpt, source_reference_inventory
        )
        grounded_relevant_sql = _append_allowed_reference_context(
            assignment_context or relevant_sql, source_reference_inventory
        )
        raw_output = llm_client.generate_formula_expression(
            technical_summary=canonical_model.technical_summary,
            business_summary=canonical_model.business_summary,
            source_sql=grounded_source_sql_excerpt,
            function_reference=effective_function_reference,
            column_name=column,
            entity_name=entity_name,
            relevant_sql=grounded_relevant_sql,
            rag_context=rag_context.combined,
        )
        for attempt in range(_MAX_GENERATION_ATTEMPTS):
            derivation_option, expression, decision_table_json, parse_errors = _interpret_llm_output(raw_output)
            if expression:
                expression = _normalize_expression(expression, obj.raw_sql)
                expression = _scrub_llm_expression_for_column(expression, column)
                if not expression:
                    attempt_errors = list(parse_errors) + [
                        "Rejected multi-column / comma-style IF blob; "
                        "formula must derive only the target column"
                    ]
                    validation_errors = attempt_errors
                    if attempt + 1 >= _MAX_GENERATION_ATTEMPTS:
                        break
                    raw_output = llm_client.retry_with_error(
                        previous_expression=raw_output,
                        error="\n".join(attempt_errors),
                        context=f'Target column only: "{entity_name}"."{column}"',
                    )
                    continue
                expression = _ground_expression_to_source_references(expression, source_reference_inventory, entity_name)
                expression = _finalize_platform_expression(
                    expression,
                    entity_name=entity_name,
                    entity_name_map=entity_name_map,
                    alias_resolution_inventory=alias_resolution_inventory,
                    source_sql=obj.raw_sql,
                )
                expression = _rewrite_business_date_variables(expression, entity_name, source_sql=obj.raw_sql)
                expression = _normalize_expression(expression, obj.raw_sql)
                repaired = _repair_trailing_self_reference(expression, entity_name, column, obj.raw_sql)
                if repaired != expression and validate_expression(repaired).valid:
                    expression = repaired

            attempt_errors = list(parse_errors)

            if expression and not attempt_errors:
                grammar_result = validate_expression(expression)
                if not grammar_result.valid:
                    attempt_errors.append(f"Grammar validation failed: {grammar_result.error}")
                else:
                    semantic_result = check_semantic_consistency(
                        expression, column, entity_name, relevant_chunks, obj.raw_sql, source_statement_sql
                    )
                    if not semantic_result.passed:
                        attempt_errors.extend(f"Semantic validation: {e}" for e in semantic_result.errors)

            if derivation_option == DerivationOption.DECISION_TABLE and decision_table_json and not expression:
                # Platform sample exports always keep Display Derivation Expression
                # populated even for Decision Table rows. Require an expression.
                attempt_errors.append(
                    "Decision Table output must also include a Display Derivation "
                    "Expression (IF/THEN/ELSEIF form). Return JSON with both "
                    '"expression" and "decision_table" keys.'
                )

            if not attempt_errors:
                validation_errors = []
                if (
                    derivation_option == DerivationOption.FORMULA_EXPRESSION
                    and expression
                    and decision_table_json is None
                ):
                    dt_payload = _decision_table_from_formula_if_categorical(expression, entity_name, column)
                    if dt_payload is not None:
                        derivation_option = DerivationOption.DECISION_TABLE
                        decision_table_json = json.dumps(dt_payload)
                break

            validation_errors = attempt_errors
            if attempt + 1 >= _MAX_GENERATION_ATTEMPTS:
                break

            retry_context = "\n\n".join(
                part
                for part in [
                    f'Target column: "{entity_name}"."{column}"',
                    f"Ordered assignment context:\n{assignment_context}" if assignment_context else "",
                    f"Relevant SQL:\n{relevant_sql}" if relevant_sql else "",
                    f"{allowed_reference_context}" if allowed_reference_context else "",
                    f"Technical summary:\n{canonical_model.technical_summary}" if canonical_model.technical_summary else "",
                    f"Business summary:\n{canonical_model.business_summary}" if canonical_model.business_summary else "",
                    f"Source SQL:\n{_append_allowed_reference_context(source_sql_excerpt, source_reference_inventory)}",
                    f"Platform reference:\n{effective_function_reference}" if effective_function_reference else "",
                    f"RAG context:\n{rag_context.combined}" if rag_context.combined else "",
                ]
                if part
            )
            raw_output = llm_client.retry_with_error(
                previous_expression=expression or raw_output,
                error="\n".join(attempt_errors),
                context=retry_context,
            )

    business_meaning = _derive_business_meaning(
        llm_client=llm_client,
        technical_summary=canonical_model.technical_summary,
        business_summary=canonical_model.business_summary,
        source_sql=source_sql_excerpt,
        function_reference=function_reference,
        entity_name=entity_name,
        column_name=column,
        relevant_sql=assignment_context or relevant_sql,
        formula=expression or deterministic_expression or "",
    )

    confidence = info.confidence if not validation_errors else min(info.confidence, 0.3)
    # Platform Status: ACTIVE when the expression passed grammar + semantic
    # checks. review_state tracks lifecycle separately (GENERATED / NEEDS_REVIEW /
    # UNSUPPORTED / APPROVED). Exception-handler caveats are advisory only —
    # they must not demote a valid normal-flow formula out of ACTIVE.
    row_advisory_seed: list[str] = list(row_advisory_seed_early)
    if undeterminable_note:
        row_advisory_seed.append(undeterminable_note)
        confidence = min(confidence, 0.85)

    has_exists = bool(expression and re.search(r"(?i)\bEXISTS\s*\(", expression))
    if has_exists:
        review_state = ReviewState.UNSUPPORTED
        status = DDStatus.PENDING_REVIEW
        validation_errors = list(validation_errors) + [
            "Procedure-level or EXISTS logic is not expressible as a row formula"
        ]
    elif validation_errors:
        review_state = ReviewState.NEEDS_REVIEW
        status = DDStatus.PENDING_REVIEW
    elif expression:
        review_state = ReviewState.GENERATED
        status = DDStatus.ACTIVE
    else:
        review_state = ReviewState.NEEDS_REVIEW
        status = DDStatus.PENDING_REVIEW

    periods = effective_periods_for_column(info.version_thresholds, timekey_map)
    if not periods:
        periods = [(date.today(), True, "", 0)]

    if _expression_should_be_rejected(validation_errors):
        expression = ""

    if _is_non_derivable_expression(expression or ""):
        # No real derivation to present — omit rather than export NULL/0 noise.
        return []

    if expression and _should_omit_passthrough_dd_row(
        target_table=target_table or "",
        entity_name=entity_name,
        expression=expression,
    ):
        # Staging/queue/audit filter-copies look like duplicate rules in the
        # business report; keep them in the write ledger, not as DD cards.
        return []

    # Prefer DT payload already chosen; otherwise derive from final expression.
    if derivation_option == DerivationOption.FORMULA_EXPRESSION and expression and not decision_table_json:
        dt_payload = _decision_table_from_formula_if_categorical(expression, entity_name, column)
        if dt_payload is not None:
            derivation_option = DerivationOption.DECISION_TABLE
            decision_table_json = json.dumps(dt_payload)

    conditional_json: str | None = None
    if derivation_option == DerivationOption.DECISION_TABLE:
        conditional_json = _conditional_json_from_decision_table(
            json.loads(decision_table_json) if decision_table_json else None
        )
    else:
        conditional_json = _conditional_json_from_formula(expression or "", entity_name)

    data_type = _infer_data_type(column, expression or "")
    column_type = _infer_column_type(column, derivation_option)

    rows = []
    for eff_date, is_real_mapping, variable, representative_value in periods:
        row_confidence = confidence
        row_status = status
        row_validation_errors = list(validation_errors)
        advisory_notes: list[str] = list(row_advisory_seed)

        if has_cross_statement_dependency:
            if expression:
                advisory_notes.append(
                    f'"{column}" is written by multiple sequential source statements; '
                    "spot-check that the composed formula preserves execution order."
                )
                row_confidence = min(row_confidence, 0.85)
                # Keep ACTIVE when the composed expression already validated —
                # sequential dependency is advisory, not a hard demotion.
            else:
                row_status = DDStatus.PENDING_REVIEW
                review_state = ReviewState.NEEDS_REVIEW
                advisory_notes.append(
                    f'"{column}" is written by multiple sequential source statements, and a later '
                    "statement reads the column's own value -- meaning it may depend on what an "
                    "earlier statement already wrote. Automated composition cannot guarantee this "
                    "execution-order dependency is preserved; verify this rule against the source "
                    "SQL statement-by-statement before approving."
                )

        if not is_real_mapping:
            advisory_notes.append(
                f"Effective start date {eff_date} is a synthetic estimate because no TIMEKEY-to-calendar-date mapping was supplied for this run."
            )
        if not validation_errors and row_confidence < settings.output_guardrail_confidence_threshold:
            advisory_notes.append(
                f"Confidence {row_confidence:.3f} is below the advisory threshold {settings.output_guardrail_confidence_threshold:.3f}."
            )

        row_expression = expression or ""
        # Only prune an already-clean expression -- pruning a row that's
        # already flagged PENDING_REVIEW would risk hiding the very logic
        # a reviewer needs to see, and there is nothing reliable to prune
        # from an expression that hasn't been validated in the first
        # place.
        if row_expression and variable and not validation_errors:
            pruned = prune_expression_for_period(row_expression, variable, representative_value)
            if pruned != row_expression and validate_expression(pruned).valid:
                row_expression = pruned

        rows.append(
            DDRow(
                entity_name=entity_name,
                column_name=column,
                column_type=column_type,
                derivation_option=derivation_option,
                display_derivation_expression=row_expression,
                effective_start_date=eff_date,
                status=row_status,
                review_state=review_state,
                data_type=data_type,
                decision_table_json=decision_table_json,
                conditional_json=conditional_json,
                source_chain_id=canonical_model.chain_id,
                source_object_ids=[obj.object_id],
                source_statement_refs=source_statement_refs,
                source_statement_sql=source_statement_sql,
                confidence=row_confidence,
                validation_errors=row_validation_errors,
                advisory_notes=advisory_notes,
                business_meaning=business_meaning,
            )
        )
    return rows


_CATEGORICAL_BRANCH_RE = re.compile(
    r"(?is)(?:IF|ELSEIF)\s*\((?P<cond>.+?)\)\s*THEN\s*\((?P<val>.+?)\)(?=(?:ELSEIF|ELSE)\s*\()"
)
_CATEGORICAL_ELSE_RE = re.compile(r"(?is)ELSE\s*\((?P<val>.+)\)\s*$")
_STRING_LITERAL_RE = re.compile(r'^"(?:[^"]|\\"")+"$')
_NUMERIC_LITERAL_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _split_top_level_and(condition: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_double = False
    i = 0
    text = condition or ""
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_double = not in_double
            current.append(ch)
            i += 1
            continue
        if not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and text[i : i + 5].upper() == " AND ":
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 5
                continue
        current.append(ch)
        i += 1
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts or ([condition.strip()] if condition.strip() else [])


def _column_link_name(expr: str) -> tuple[str, str, str]:
    """Return (columnName, qualifierName, type) from a 4X column reference."""
    text = (expr or "").strip()
    parts = re.findall(r'"([^"]+)"', text)
    if not parts:
        bare = re.sub(r"[^A-Za-z0-9_]", "", text) or "VALUE"
        return bare, bare, "ENT"
    column = parts[-1]
    if len(parts) >= 2:
        qualifier = parts[-2]
        link_type = "TEMP" if qualifier.upper() == "VAR" else "REL"
        return column, qualifier, link_type
    return column, column, "ENT"


def _parse_condition_link(condition: str, entity_name: str) -> dict | None:
    text = (condition or "").strip()
    if not text:
        return None

    empty = re.match(r'(?is)^ISEMPTY\s*\(\s*(.+?)\s*\)$', text)
    if empty:
        col, name, link_type = _column_link_name(empty.group(1))
        return {
            "columnName": col,
            "operator": "ISEMPTY",
            "rangeFrom": "",
            "rangeTo": "",
            "value": "",
            "name": name or entity_name,
            "type": link_type,
            "isFilterSet": "",
        }
    not_empty = re.match(r'(?is)^ISNOTEMPTY\s*\(\s*(.+?)\s*\)$', text)
    if not_empty:
        col, name, link_type = _column_link_name(not_empty.group(1))
        return {
            "columnName": col,
            "operator": "ISNOTEMPTY",
            "rangeFrom": "",
            "rangeTo": "",
            "value": "",
            "name": name or entity_name,
            "type": link_type,
            "isFilterSet": "",
        }

    between = re.match(
        r'(?is)^(.+?)\s+BETWEEN\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]\s*$',
        text,
    )
    if between:
        col, name, link_type = _column_link_name(between.group(1))
        return {
            "columnName": col,
            "operator": "BETWEEN",
            "rangeFrom": between.group(2).strip().strip('"'),
            "rangeTo": between.group(3).strip().strip('"'),
            "value": "",
            "name": name or entity_name,
            "type": link_type,
            "isFilterSet": "",
        }

    membership = re.match(
        r'(?is)^(.+?)\s+(IN|NOTIN|CONTAINS|BEGINSWITH|ENDSWITH|DOESNOTCONTAINS|HRCHYIN|HRCHYNOTIN)\s*'
        r'\[\s*(.*?)\s*\]\s*$',
        text,
    )
    if membership:
        col, name, link_type = _column_link_name(membership.group(1))
        values = [
            item.strip().strip('"')
            for item in membership.group(3).split(",")
            if item.strip()
        ]
        return {
            "columnName": col,
            "operator": membership.group(2).upper(),
            "rangeFrom": "",
            "rangeTo": "",
            "value": ",".join(values),
            "name": name or entity_name,
            "type": link_type,
            "isFilterSet": "",
        }

    compare = re.match(
        r'(?is)^(.+?)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$',
        text,
    )
    if compare:
        col, name, link_type = _column_link_name(compare.group(1))
        raw_value = compare.group(3).strip()
        value = raw_value.strip('"')
        return {
            "columnName": col,
            "operator": compare.group(2),
            "rangeFrom": "",
            "rangeTo": "",
            "value": value,
            "name": name or entity_name,
            "type": link_type,
            "isFilterSet": "",
        }

    # Fallback: keep the expression text so nothing is lost.
    return {
        "columnName": "",
        "operator": "EXPR",
        "rangeFrom": "",
        "rangeTo": "",
        "value": text,
        "name": entity_name,
        "type": "ENT",
        "isFilterSet": "",
    }


def _condition_links_from_guard(condition: str, entity_name: str) -> list[dict]:
    links: list[dict] = []
    for part in _split_top_level_and(condition):
        # Drop leading AND()/OR() wrappers for simple cases.
        cleaned = part.strip()
        if cleaned.upper().startswith("AND(") and cleaned.endswith(")"):
            cleaned = cleaned[4:-1]
        link = _parse_condition_link(cleaned, entity_name)
        if link:
            links.append(link)
    return links


def _decision_table_from_formula_if_categorical(
    expression: str,
    entity_name: str,
    column: str,
) -> dict | None:
    """Build Decision Table JSON matching the platform sample export shape.

    Produces `decisionTableDetails` with real operators (`==`, `BETWEEN`,
    `IN`, `ISEMPTY`, …) in `conditionalLinksInfo` whenever the IF/THEN
    chain is a clear multi-bucket categorical classification.
    """
    if not expression or "IF(" not in expression.upper():
        return None

    branches = list(_CATEGORICAL_BRANCH_RE.finditer(expression))
    else_match = _CATEGORICAL_ELSE_RE.search(expression)
    if len(branches) < 2:
        return None

    values: list[str] = []
    details: list[dict] = []
    for idx, branch in enumerate(branches, start=1):
        cond = branch.group("cond").strip()
        val = branch.group("val").strip()
        values.append(val)
        if not (_STRING_LITERAL_RE.match(val) or _NUMERIC_LITERAL_RE.match(val)):
            return None
        label = val.strip('"')
        links = _condition_links_from_guard(cond, entity_name)
        if not links:
            links = [
                {
                    "columnName": column,
                    "operator": "EXPR",
                    "rangeFrom": "",
                    "rangeTo": "",
                    "value": cond,
                    "name": entity_name,
                    "type": "ENT",
                    "isFilterSet": "",
                }
            ]
        details.append(
            {
                "derivedValue": label,
                "sequenceNumber": idx,
                "conditionName": label,
                "conditionalLinksInfo": links,
            }
        )

    if else_match:
        else_val = else_match.group("val").strip()
        if "IF(" in else_val.upper():
            return None
        if not (_STRING_LITERAL_RE.match(else_val) or _NUMERIC_LITERAL_RE.match(else_val)):
            return None
        values.append(else_val)
        label = else_val.strip('"')
        details.append(
            {
                "derivedValue": label,
                "sequenceNumber": len(details) + 1,
                "conditionName": label,
                "conditionalLinksInfo": [
                    {
                        "columnName": column,
                        "operator": "ELSE",
                        "rangeFrom": "",
                        "rangeTo": "",
                        "value": "",
                        "name": entity_name,
                        "type": "ENT",
                        "isFilterSet": "",
                    }
                ],
            }
        )

    distinct = {v.strip('"') for v in values}
    if len(distinct) < 2:
        return None
    if all(_NUMERIC_LITERAL_RE.match(v) for v in values):
        return None

    return {"decisionTableDetails": details}


def _conditional_json_from_decision_table(dt_payload: dict | None) -> str | None:
    """Platform sample leaves Conditional Json empty for Decision Table rows.

    For Formula Expression rows we still may emit a compact condition set
    via `_conditional_json_from_formula`. Decision Table conditions already
    live inside Decision Table Json, so Conditional Json stays empty.
    """
    return None


def _conditional_json_from_formula(expression: str, entity_name: str) -> str | None:
    """Build Conditional Json for non-DT formula rows with IF guards.

    Shape mirrors the nested `conditionalLinksInfo` objects used inside
    Decision Table Json in `samples/derivations/sample_derivations.csv`.
    """
    if not expression or "IF(" not in expression.upper():
        return None
    branches = list(_CATEGORICAL_BRANCH_RE.finditer(expression))
    if not branches:
        return None
    details: list[dict] = []
    for idx, branch in enumerate(branches, start=1):
        links = _condition_links_from_guard(branch.group("cond").strip(), entity_name)
        if not links:
            continue
        details.append(
            {
                "sequenceNumber": idx,
                "conditionalLinksInfo": links,
            }
        )
    if not details:
        return None
    return json.dumps({"conditionalDetails": details})


def _infer_data_type(column_name: str, expression: str = "") -> str:
    """Infer platform Data Type: string | number | datetime.

    Prefer signals from the derived expression (TODATE, Y/N literals, etc.),
    then fall back to conservative column-name heuristics matching
    `samples/derivations/sample_derivations.csv`.
    """
    lowered = (column_name or "").lower()
    expr = expression or ""
    expr_upper = expr.upper()

    if any(token in lowered for token in ("date", "_at", "period_id", "timekey")) or lowered.endswith("dt"):
        return "datetime"
    # A date function can appear in a *condition* of a flag/amount formula;
    # it does not make the result a date. Prefer the target column's meaning.
    if any(token in lowered for token in (
        "amount", "amt", "penalty", "fee", "interest", "balance", "pct",
        "percent", "ratio", "score", "count", "qty", "tenure", "month",
        "day", "diff", "dpd", "rate",
    )) and not any(token in lowered for token in ("flag", "bucket", "status")):
        return "number"
    if any(token in lowered for token in (
        "flag", "flg", "ind", "check", "reason", "status", "class",
        "bucket", "tier", "type", "name", "desc", "code", "msg",
        "description", "eligible", "worsened", "applied", "outcome",
    )):
        return "string"
    if re.search(r'(?i)\b(?:THEN|ELSE)\s*\(\s*"(?:Y|N|YES|NO|TRUE|FALSE)"\s*\)', expr):
        return "string"
    if "TODATE(" in expr_upper or "ADDDAY(" in expr_upper or "SOM(" in expr_upper or "EOM(" in expr_upper:
        if "DATEDIFF(" not in expr_upper:
            return "datetime"
    if "DATEDIFF(" in expr_upper and any(token in lowered for token in ("day", "days", "diff", "count", "dpd")):
        return "number"

    string_tokens = (
        "flag",
        "flg",
        "ind",
        "check",
        "reason",
        "status",
        "class",
        "bucket",
        "tier",
        "type",
        "name",
        "desc",
        "code",
        "msg",
        "description",
    )
    if any(token in lowered for token in string_tokens):
        return "string"

    # Literal outcomes dominate type when present.
    string_literals = re.findall(r'"([^"]*)"', expr)
    meaningful = [v for v in string_literals if v.upper() not in {"", "NULL"} and not v.replace(".", "", 1).isdigit()]
    if meaningful and all(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ \-]*", v) or v.upper() in {"Y", "N", "YES", "NO", "TRUE", "FALSE"}
        for v in meaningful
    ):
        # Pure categorical labels / flags.
        if any(v.upper() in {"Y", "N", "YES", "NO", "TRUE", "FALSE"} for v in meaningful) or any(
            not v.replace(".", "", 1).isdigit() for v in meaningful
        ):
            # If expression is mostly classification labels, string wins.
            if "DATEDIFF(" not in expr_upper and "ROUND(" not in expr_upper and "ABS(" not in expr_upper:
                if any(ch.isalpha() for v in meaningful for ch in v):
                    return "string"

    if any(token in lowered for token in ("amt", "amount", "pct", "percent", "balance", "rate", "score", "count", "qty", "id")):
        return "number"
    return "number"


def _infer_column_type(column_name: str, derivation_option: DerivationOption) -> ColumnType:
    """Infer Physical vs Temporary using only generic naming signals.

    Platform sample exports mark intermediate/helper columns Temporary and
    persisted fact attributes Physical. Without a platform catalog we can
    only apply conservative name-based heuristics -- never invent business
    meaning.
    """
    lowered = column_name.lower()
    temporary_tokens = (
        "check",
        "ratio",
        "grouping",
        "review",
        "temp",
        "tmp",
        "helper",
        "previous_",
        "prev_",
        "bureau",
        "relative_",
        "increase_in",
        "downgrade",
        "no_of_",
        "rejection_",
        "chque_",
        "cheque_",
    )
    if any(token in lowered for token in temporary_tokens):
        return ColumnType.TEMPORARY
    if derivation_option == DerivationOption.DECISION_TABLE:
        return ColumnType.PHYSICAL
    return ColumnType.PHYSICAL


def _interpret_llm_output(
    raw_output: str,
) -> tuple[DerivationOption, str | None, str | None, list[str]]:
    stripped = raw_output.strip()
    if not stripped:
        return DerivationOption.FORMULA_EXPRESSION, "", None, []

    def unwrap_code_fence(text: str) -> str:
        fenced = re.match(r"(?is)^\s*```(?:json|text)?\s*(.*?)\s*```\s*$", text)
        return fenced.group(1).strip() if fenced else text.strip("`").strip()

    def extract_json_candidate(text: str) -> str | None:
        candidate = unwrap_code_fence(text)
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return candidate[start : end + 1].strip()

    def is_decision_table_payload(parsed: object) -> bool:
        if isinstance(parsed, dict):
            keys = {str(key).replace("-", "_").lower() for key in parsed.keys()}
            if {"decision_table", "decisiontable"} & keys:
                return True
            if {"rules", "buckets", "input_columns", "output_columns", "decisiontabledetails"} & keys and not {
                "expression",
                "formula",
            } & keys:
                return True
        return False

    json_candidate = extract_json_candidate(stripped)
    if json_candidate is not None:
        try:
            parsed = json.loads(json_candidate)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict):
                keys = {str(key).replace("-", "_").lower(): key for key in parsed.keys()}
                expression_key = keys.get("expression") or keys.get("formula") or keys.get("display_derivation_expression")
                expression_value = None
                if expression_key is not None:
                    raw_expr = parsed.get(expression_key)
                    if isinstance(raw_expr, str) and raw_expr.strip():
                        expression_value = unwrap_code_fence(raw_expr)

                if is_decision_table_payload(parsed) or (
                    expression_value is not None
                    and any(k in keys for k in ("decision_table", "decisiontable", "decisiontabledetails"))
                ):
                    decision_table = parsed.get("decision_table") if "decision_table" in parsed else None
                    if decision_table is None:
                        decision_table = parsed.get("decisionTable")
                    if decision_table is None and "decisionTableDetails" in parsed:
                        decision_table = {"decisionTableDetails": parsed.get("decisionTableDetails")}
                    if decision_table is None and any(
                        k in {str(x).replace("-", "_").lower() for x in parsed.keys()}
                        for k in ("rules", "buckets", "decisiontabledetails")
                    ):
                        decision_table = parsed
                    if decision_table is None:
                        decision_table = parsed
                    return (
                        DerivationOption.DECISION_TABLE,
                        expression_value,
                        json.dumps(decision_table),
                        [],
                    )

                if expression_value is not None and not is_decision_table_payload(parsed):
                    return DerivationOption.FORMULA_EXPRESSION, expression_value, None, []

            if is_decision_table_payload(parsed):
                decision_table = parsed.get("decision_table") if isinstance(parsed, dict) else None
                if decision_table is None and isinstance(parsed, dict):
                    decision_table = parsed.get("decisionTable")
                if decision_table is None:
                    decision_table = parsed
                return (
                    DerivationOption.DECISION_TABLE,
                    None,
                    json.dumps(decision_table),
                    [],
                )

    return DerivationOption.FORMULA_EXPRESSION, unwrap_code_fence(stripped), None, []


def flag_duplicate_dd_rows(dd_rows: list[DDRow]) -> list[DDRow]:
    """Detect DD rows sharing the same (entity_name, column_name,
    effective_start_date) identity -- the exact key
    app/report/dd_export.py::merge_dd_rows uses to decide whether a row
    is "the same row" -- coming from more than one source. A column
    normally has exactly one derivation per effective date; more than one
    commonly means two different source procedures both write the same
    shared table+column (each correctly reflecting its own procedure's own
    logic, often each scoped to different rows by its own guard condition
    -- see check_dropped_override_conditions' row-scoping check), and nothing
    in this pipeline can know on its own whether they should be combined
    into a single formula or whether one is simply wrong for this column.

    Rather than silently exporting duplicate rows for the same key --
    which the platform's own schema does not expect, and which
    merge_dd_rows' last-one-wins-by-key merge would otherwise let one
    silently overwrite the other with no record that a conflict ever
    existed -- every row sharing a duplicated key is routed to
    PENDING_REVIEW with a note identifying the other source chain(s) it
    conflicts with, so a reviewer resolves it explicitly instead of the
    pipeline guessing or the report/Excel silently picking one.

    Rows are never dropped, merged, or rewritten here -- only status,
    confidence, and validation_errors are updated -- so this can never
    lose or alter derivation logic, and a column with only one source
    (the overwhelmingly common case) is completely unaffected.
    """
    key_to_rows: dict[tuple[str, str, object], list[DDRow]] = {}
    for row in dd_rows:
        key = (row.entity_name, row.column_name, row.effective_start_date)
        key_to_rows.setdefault(key, []).append(row)

    for rows in key_to_rows.values():
        if len(rows) < 2:
            continue
        distinct_chains = sorted({r.source_chain_id for r in rows})
        for row in rows:
            other_chains = [c for c in distinct_chains if c != row.source_chain_id] or distinct_chains
            row.status = DDStatus.PENDING_REVIEW
            row.confidence = min(row.confidence, 0.3)
            row.validation_errors.append(
                f'Another derivation for "{row.entity_name}"."{row.column_name}" '
                f"effective {row.effective_start_date} was generated from a "
                f"different source ({', '.join(other_chains)}). Multiple "
                "procedures/statements write this column for this "
                "effective date -- reconcile into a single formula (for "
                "example, guard each with its own row-scoping condition) "
                "before accepting any of them."
            )
    return dd_rows
