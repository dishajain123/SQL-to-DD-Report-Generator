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


def _assignment_sites(info: StructuralInfo, column: str) -> list[_AssignmentSite]:
    """Return ordered write sites for the target column.

    Prefer statement-level assignments when available so later fix-up
    UPDATEs remain separate from earlier MERGE calculations. Fall back to
    the older chunk view only when the structural info does not expose
    statements.
    """
    statements = getattr(info, "statements", None)
    if statements:
        return _assignment_sites_from_statements(statements, column)
    return _assignment_sites_from_chunks(_relevant_chunks(info, column), column)


def _assignment_sites_from_statements(statements: list[StatementInfo], column: str) -> list[_AssignmentSite]:
    column_upper = column.upper()
    sites: list[_AssignmentSite] = []
    pending_headers: list[StatementInfo] = []
    bridge_context: list[StatementInfo] = []

    for stmt in statements:
        writes_target = _statement_writes_column(stmt, column_upper)

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

        if stmt.statement_type == "CONTROL_FLOW" and not stmt.columns and not stmt.set_columns_by_table:
            pending_headers.append(stmt)
            continue

        if pending_headers and not stmt.set_columns_by_table and not stmt.tables_written:
            bridge_context.append(stmt)
            continue

        pending_headers = []
        bridge_context = []

    return sites


def _assignment_sites_from_chunks(chunks: list[SmartChunk], column: str) -> list[_AssignmentSite]:
    sites: list[_AssignmentSite] = []
    for chunk in chunks:
        raw = chunk.raw_sql.strip()
        if not raw:
            continue
        sites.append(
            _AssignmentSite(
                kind=chunk.chunk_kind,
                statement_indices=list(chunk.statement_indices),
                raw_sql=raw,
                columns_written=[column],
            )
        )
    return sites


def _statement_writes_column(stmt: StatementInfo, column_upper: str) -> bool:
    for cols in stmt.set_columns_by_table.values():
        if any(col.upper() == column_upper for col in cols):
            return True
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
    ]
) -> list[DDRow]:
    canonical_model, obj, info, entity_name, column, llm_client, function_reference, timekey_map, rag_store = job
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
    )


def _retrieve_rag_context(
    rag_store: Optional[ChromaStore],
    relevant_sql: str,
    technical_summary: str,
    business_summary: str,
) -> str:
    """Query the platform (4X function/operator) and domain RAG
    collections for the chunks most relevant to this specific column,
    instead of handing the model the entire reference document every time.

    Falls back to an empty string -- letting the caller rely on the full
    function_reference instead -- if no RAG store was supplied, the store
    can't be reached, or nothing has been ingested yet. This keeps the
    pipeline fully functional whether or not `ingest_platform_doc` /
    `ingest_domain_doc` has ever been run; RAG is a targeting aid on top of
    the existing full-reference behavior, not a replacement that could
    break generation if it's unavailable.
    """
    if rag_store is None:
        return ""

    platform_query = (relevant_sql or technical_summary).strip()
    domain_query = (business_summary or technical_summary).strip()

    sections: list[str] = []
    try:
        if platform_query:
            platform_hits = rag_store.query(PLATFORM_COLLECTION, platform_query, n_results=4)
            if platform_hits:
                sections.append(
                    "Relevant platform function/operator reference:\n" + "\n---\n".join(platform_hits)
                )
        if domain_query:
            domain_hits = rag_store.query(DOMAIN_COLLECTION, domain_query, n_results=2)
            if domain_hits:
                sections.append("Relevant domain glossary:\n" + "\n---\n".join(domain_hits))
    except Exception as exc:  # pragma: no cover - defensive: RAG must never break generation
        logger.warning("RAG retrieval failed, continuing without it: %s", exc)
        return ""

    return "\n\n".join(sections)


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
        if re.fullmatch(r'"[^"]*"', stripped):
            return True
        if re.fullmatch(r'\(\s*"[^"]*"\s*\)', stripped):
            return True
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


def _rewrite_business_date_variables(expression: str, entity_name: str) -> str:
    if not expression or not entity_name:
        return expression
    replacement = f'"{entity_name}"."var"."BUSINESS_DATE"'
    rewritten = re.sub(
        r'(?<![A-Za-z0-9_".])@?(?:v_)?(?:PROCESSDATE|PROCESSDT|BUSINESSDATE)\b',
        replacement,
        expression,
        flags=re.IGNORECASE,
    )
    return rewritten.replace('."VAR"."BUSINESS_DATE"', '."var"."BUSINESS_DATE"')


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


def _rewrite_exists_predicates(expression: str) -> str:
    """Drop unsupported EXISTS wrappers but keep the predicate body."""
    result: list[str] = []
    i = 0
    in_double = False
    while i < len(expression):
        ch = expression[i]
        if ch == '"':
            in_double = not in_double
            result.append(ch)
            i += 1
            continue

        if not in_double and expression[i : i + 7].upper() == "EXISTS(":
            open_index = i + 6
            close_index = _find_matching_paren(expression, open_index)
            if close_index != -1:
                inner = expression[i + 7 : close_index].strip()
                tail = expression[close_index + 1 :].lstrip()
                needs_if_close = tail.upper().startswith("THEN")
                suffix = ")" if needs_if_close else ""
                where_match = re.search(r"(?is)\bWHERE\b", inner)
                if where_match:
                    predicate = inner[where_match.end() :].strip()
                    result.append(predicate + suffix)
                else:
                    comma_match = re.search(r"(?s),", inner)
                    if comma_match:
                        predicate = inner[comma_match.end() :].strip()
                        result.append(predicate + suffix)
                    else:
                        result.append(inner + suffix)
                i = close_index + 1
                continue

        result.append(ch)
        i += 1
    return "".join(result)


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
    expression = _flatten_whitespace(expression)
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


def _source_allows_target_reference(source_sql: str, entity_name: str, column: str) -> bool:
    """Allow a self-reference only when the source SQL explicitly
    preserves the target value or increments it.

    This mirrors the semantic validation rule so the generator can
    mechanically repair the most common false-positive shape: an
    otherwise-correct expression whose final ELSE clause falls back to the
    target column even though the source SQL never does that.
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

    return False


_SAFE_ARITHMETIC_FUNCTION_NAMES = {"COALESCE", "NVL", "ISNULL", "TODATE", "SYSDATE", "MAX", "MIN"}


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
        if isinstance(value_node, exp.Case):
            return None
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
        if isinstance(default_node, exp.Case):
            return None
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

    result = "".join(parts)
    return result if validate_expression(result).valid else None


def _render_value_expression_to_4x(node) -> str | None:
    while isinstance(node, exp.Paren):
        node = node.this

    if isinstance(node, exp.Case):
        return _translate_case_to_4x(node)

    if isinstance(node, exp.Add):
        left = _render_value_expression_to_4x(node.this)
        right = _render_value_expression_to_4x(node.expression)
        if left is None or right is None:
            return None
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

    if isinstance(node, exp.DateAdd):
        unit = getattr(node.args.get("unit"), "this", None)
        unit_text = str(unit).upper() if unit is not None else ""
        if unit_text not in {"DAY", "DAYS"}:
            return None
        base = _render_value_expression_to_4x(node.this)
        amount = _render_value_expression_to_4x(node.expression)
        if base is None or amount is None:
            return None
        return f"ADDDAY({base}, {amount})"

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
    if isinstance(tree, exp.Update):
        from_clause = tree.args.get("from_")
        if isinstance(from_clause, exp.From):
            source = from_clause.this or (from_clause.expressions[0] if from_clause.expressions else None)
            if source is not None:
                alias = _extract_alias_name(source)
                if alias:
                    return alias
        target = tree.args.get("this")
        if target is not None:
            alias = _extract_alias_name(target)
            if alias:
                return alias
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


def _render_boolean_operand(node, parent_is_or: bool, source_alias: str | None = None) -> str | None:
    """Render one operand of an AND/OR, adding parentheses whenever the
    operand's own top-level connective differs from its parent's (an AND
    directly under an OR, or vice versa) -- the exact, and only, shape
    where omitting parentheses would silently change what the expression
    means by falling back to the grammar's default AND-before-OR
    precedence. This decision is made purely from the parsed tree's own
    node types, never from the rendered text, so it can never be fooled
    by a leaf value that happens to contain the words "AND"/"OR"."""
    unwrapped = node.this if isinstance(node, exp.Paren) else node
    rendered = _render_sql_condition_to_4x(unwrapped, source_alias=source_alias)
    if rendered is None:
        return None
    if isinstance(unwrapped, exp.Or) and not parent_is_or:
        return f"({rendered})"
    if isinstance(unwrapped, exp.And) and parent_is_or:
        return f"({rendered})"
    return rendered


def _render_sql_condition_to_4x(node, source_alias: str | None = None) -> str | None:
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
        return _render_sql_condition_to_4x(node.this, source_alias=source_alias)

    if isinstance(node, exp.And):
        left = _render_boolean_operand(node.this, parent_is_or=False, source_alias=source_alias)
        right = _render_boolean_operand(node.expression, parent_is_or=False, source_alias=source_alias)
        if left is None or right is None:
            return None
        return f"{left} AND {right}"

    if isinstance(node, exp.Or):
        left = _render_boolean_operand(node.this, parent_is_or=True, source_alias=source_alias)
        right = _render_boolean_operand(node.expression, parent_is_or=True, source_alias=source_alias)
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
        rendered = _render_sql_condition_to_4x(inner_unwrapped, source_alias=source_alias)
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
        left = _render_boolean_predicate_leaf(node.this, source_alias=source_alias)
        values = [_render_value_expression_to_4x(expr) for expr in (node.expressions or [])]
        if left is None or not values or any(value is None for value in values):
            return None
        return f'{left} IN [{",".join(values)}]'

    # Any other node (a comparison, BETWEEN, a bare column/literal) has
    # no AND/OR/NOT structure of its own to get wrong -- render it via
    # the existing, already-proven leaf pipeline.
    return _render_boolean_predicate_leaf(node, source_alias=source_alias)


def _parse_simple_assignment_stage(
    raw_sql: str,
    target_column: str,
    entity_name: str = "",
) -> tuple[str, str, str] | None:
    """Extract a deterministic guard/value pair for a simple UPDATE or
    MERGE write to `target_column`.

    Returns `(guard, value, source_target_name)` when the statement only
    assigns a simple literal, NULL, or direct column reference. More
    complex CASE/IF formulas are intentionally left to the LLM path.
    """
    target_upper = target_column.upper()

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
                        guard = _render_sql_condition_to_4x(
                            where.this, source_alias=_statement_source_alias(select_tree)
                        ) or where.this.sql(dialect=_SQLGLOT_DIALECT[dialect])
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
                guard = _render_sql_condition_to_4x(where.this, source_alias=source_alias) or where.this.sql(
                    dialect=_SQLGLOT_DIALECT[dialect]
                )

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
                    case_translation = _rewrite_business_date_variables(case_translation, entity_name)
                    return guard, case_translation, assignment.this.name
                value = _render_value_expression_to_4x(value_node)
                if value is not None:
                    value = _rewrite_business_date_variables(value, entity_name)
                    if validate_expression(value).valid:
                        return guard, value, assignment.this.name
                    break
                value = value_node.sql(dialect=_SQLGLOT_DIALECT[dialect])
                value = _rewrite_business_date_variables(value, entity_name)
                if not _is_simple_stage_value(value):
                    break
                return guard, value, assignment.this.name

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
                            guard = _render_sql_condition_to_4x(where.this, source_alias=source_alias) or where.this.sql(
                                dialect=_SQLGLOT_DIALECT[dialect]
                            )
                    return guard, value, target_column

        if isinstance(tree, exp.Select) and tree.args.get("into") is not None:
            projection = _render_select_into_projection(tree, target_upper)
            if projection is not None:
                guard = ""
                where = tree.args.get("where")
                if where is not None:
                    guard = _render_sql_condition_to_4x(where.this, source_alias=source_alias) or where.this.sql(
                        dialect=_SQLGLOT_DIALECT[dialect]
                    )
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
                        guard = _render_sql_condition_to_4x(where.this, source_alias=_statement_source_alias(select_tree)) or where.this.sql(
                            dialect=_SQLGLOT_DIALECT[dialect]
                        )
                    return guard, projection, target_column

        if isinstance(tree, exp.Merge):
            guard_parts: list[str] = []
            using = tree.args.get("using")
            if isinstance(using, exp.Subquery) and isinstance(using.this, exp.Select):
                where = using.this.args.get("where")
                if where is not None:
                    guard_parts.append(_render_sql_condition_to_4x(where.this) or where.this.sql(dialect=_SQLGLOT_DIALECT[dialect]))
            on_clause = tree.args.get("on")
            if on_clause is not None:
                guard_parts.append(_render_sql_condition_to_4x(on_clause) or on_clause.sql(dialect=_SQLGLOT_DIALECT[dialect]))

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
                    if isinstance(value_node, exp.Case):
                        case_translation = _translate_case_to_4x(value_node)
                        if case_translation is None:
                            break
                        value = _rewrite_business_date_variables(case_translation, entity_name)
                    else:
                        value = _render_value_expression_to_4x(value_node)
                        if value is not None:
                            value = _rewrite_business_date_variables(value, entity_name)
                            if validate_expression(value).valid:
                                pass
                            else:
                                break
                        else:
                            value = value_node.sql(dialect=_SQLGLOT_DIALECT[dialect])
                            value = _rewrite_business_date_variables(value, entity_name)
                            if not _is_simple_stage_value(value):
                                break
                        if value is None:
                            break
                    if when_cond is not None:
                        guard_parts.append(
                            _render_sql_condition_to_4x(when_cond, source_alias=source_alias)
                            or when_cond.sql(dialect=_SQLGLOT_DIALECT[dialect])
                        )
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


def _render_select_into_projection(tree: exp.Select, target_upper: str) -> str | None:
    source_alias = _select_into_source_alias(tree)
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

        if isinstance(value_node, exp.Case):
            case_translation = _translate_case_to_4x(value_node)
            if case_translation is not None:
                return case_translation

        if isinstance(value_node, exp.Column) and source_alias and not value_node.table:
            return f'"{source_alias}"."{value_node.name}"'

        rendered = _render_value_expression_to_4x(value_node)
        if rendered is not None:
            return rendered
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


def undeterminable_exception_sites(sites: list["_AssignmentSite"]) -> list["_AssignmentSite"]:
    """Identify EXCEPTION_HANDLER write sites whose only apparent guard is
    textually identical (once normalized) to a guard some non-exception
    site for the same column also uses.

    This is the structural signature of a column that cannot honestly be
    derived as a per-row Formula Expression at all: "an unhandled
    exception occurred" is a runtime execution event, not a fact present
    in row data, so it has no real data-driven guard of its own. When the
    exception-handler statement's only extracted condition is literally
    the same row-scoping filter the normal-flow statement also uses (for
    example both restricted to `WHERE RUNNINGPROCESSNAME = 'X'`, since
    they write the same process's status row), that WHERE clause isn't
    actually distinguishing the exception case -- it just happens to be
    present on both statements for an unrelated reason (scoping to one
    process's row in a table several different procedures share).

    Confirmed against a real generation defect this way: both the LLM
    path and the deterministic composer independently produced the exact
    same wrong expression for a column shaped like this (ERRORDATE in
    ACLRUNNINGPROCESSSTATUS) -- because both derive their "exception
    occurred" guard from the same coincidentally-identical WHERE text,
    there is no guard for either path to discover here; the fix has to
    be structural exclusion, not a smarter guard.
    """
    exception_sites = [s for s in sites if _infer_assignment_role(s.raw_sql) == "EXCEPTION_HANDLER"]
    if not exception_sites:
        return []
    other_guards = {
        g
        for g in (_extract_where_guard_text(s.raw_sql) for s in sites if _infer_assignment_role(s.raw_sql) != "EXCEPTION_HANDLER")
        if g
    }
    if not other_guards:
        return []
    return [s for s in exception_sites if _extract_where_guard_text(s.raw_sql) in other_guards]


def _compose_simple_assignment_expression(
    assignment_sites: list[_AssignmentSite],
    entity_name: str,
    fallback_column: str,
) -> str | None:
    """Compose a sequential 4X expression from deterministic assignment
    sites when each stage is only a simple guard/value write.

    The composition preserves source order exactly: earlier stages become
    the outer branches and later fix-ups stay nested later.
    """
    if not assignment_sites:
        return None

    target_pattern = re.compile(rf"\b{re.escape(fallback_column)}\b", re.IGNORECASE)
    direct_sites = [site for site in assignment_sites if target_pattern.search(site.raw_sql)]
    if direct_sites:
        assignment_sites = direct_sites

    stages: list[tuple[str, str, str]] = []
    for site in assignment_sites:
        stage = _parse_simple_assignment_stage(site.raw_sql, fallback_column, entity_name)
        if stage is None:
            if re.search(rf"\b{re.escape(fallback_column)}\b\s*=", site.raw_sql, re.IGNORECASE):
                return None
            continue
        stages.append(stage)

    if not stages:
        return None

    current_target = stages[0][2] or fallback_column
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
    expression = "NULL" if first_is_seed_projection else "NULL"

    for guard, value, source_target in stages:
        source_target = source_target or current_target
        current_target = source_target
        normalized_value = _rewrite_business_date_variables(_normalize_expression(value, ""), entity_name)
        if not validate_expression(normalized_value).valid:
            return None
        if not guard.strip():
            expression = normalized_value
            continue
        normalized_guard = _normalize_expression(guard, "")
        expression = f"IF({normalized_guard})THEN({normalized_value})ELSE({expression})"

    return _rewrite_business_date_variables(_normalize_expression(expression, ""), entity_name)


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


def _expression_should_be_rejected(validation_errors: list[str]) -> bool:
    """Return True when the generated formula is not safe to export.

    Any row that still has validation errors is a review-only row. The
    safest behavior is to keep the row metadata and validation notes, but
    omit the expression itself rather than exporting a misleading or
    hallucinated Platform Condition.
    """
    return bool(validation_errors)


_ColumnJob = tuple[
    CanonicalModel, SQLObject, StructuralInfo, str, str, LLMClient, str, "dict[int, date] | None", Optional[ChromaStore]
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
    for oid in chain.order:
        obj = objects[oid]
        info = structural_infos[oid]
        for target_table, columns in info.columns_written_by_table.items():
            entity_name = entity_name_map.get(target_table, target_table)
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
                        canonical_column,
                        llm_client,
                        function_reference,
                        timekey_map,
                        rag_store,
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
) -> list[DDRow]:
    relevant_chunks = _relevant_chunks(info, column)
    all_sites = _assignment_sites(info, column)
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
    source_reference_inventory = _collect_source_reference_inventory(
        "\n\n".join(part for part in [obj.raw_sql, relevant_sql, assignment_context] if part),
        obj.dialect,
        entity_name=entity_name,
    )
    alias_resolution_inventory = _collect_alias_resolution_inventory(
        "\n\n".join(part for part in [obj.raw_sql, relevant_sql, assignment_context] if part),
        obj.dialect,
    )
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

    derivation_option = DerivationOption.FORMULA_EXPRESSION
    expression: str | None = None
    decision_table_json: str | None = None
    validation_errors: list[str] = []
    source_sql_excerpt = _source_sql_context_excerpt(obj.raw_sql, relevant_sql)
    if assignment_context:
        source_sql_excerpt = assignment_context

    deterministic_expression = _compose_simple_assignment_expression(sites, entity_name, column)
    if deterministic_expression:
        deterministic_expression = resolve_aliases_in_expression(
            deterministic_expression,
            alias_resolution_inventory,
            quote_replacements=True,
        )
        grammar_result = validate_expression(deterministic_expression)
        semantic_result = check_semantic_consistency(
            deterministic_expression, column, entity_name, relevant_chunks, obj.raw_sql, source_statement_sql
        )
        if grammar_result.valid and semantic_result.passed:
            expression = deterministic_expression
            validation_errors = []
        else:
            deterministic_expression = None

    if expression is None:
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
            function_reference=function_reference,
            column_name=column,
            entity_name=entity_name,
            relevant_sql=grounded_relevant_sql,
            rag_context=rag_context,
        )
        for attempt in range(_MAX_GENERATION_ATTEMPTS):
            derivation_option, expression, decision_table_json, parse_errors = _interpret_llm_output(raw_output)
            if expression:
                expression = _normalize_expression(expression, obj.raw_sql)
                expression = _ground_expression_to_source_references(expression, source_reference_inventory, entity_name)
                expression = resolve_aliases_in_expression(
                    expression,
                    alias_resolution_inventory,
                    quote_replacements=True,
                )
                expression = _rewrite_business_date_variables(expression, entity_name)
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

            if not attempt_errors:
                validation_errors = []
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
                    f"Platform reference:\n{function_reference}",
                    f"RAG context:\n{rag_context}" if rag_context else "",
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
    status = DDStatus.PENDING_REVIEW if validation_errors else DDStatus.ACTIVE
    if undeterminable_note:
        status = DDStatus.PENDING_REVIEW
        confidence = min(confidence, 0.5)
        validation_errors = [*validation_errors, undeterminable_note]

    periods = effective_periods_for_column(info.version_thresholds, timekey_map)
    if not periods:
        periods = [(date.today(), True, "", 0)]

    if _expression_should_be_rejected(validation_errors):
        expression = ""

    data_type = _infer_data_type(column)

    rows = []
    for eff_date, is_real_mapping, variable, representative_value in periods:
        row_confidence = confidence
        row_status = status
        row_validation_errors = list(validation_errors)
        advisory_notes: list[str] = []

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
                column_type=ColumnType.PHYSICAL,
                derivation_option=derivation_option,
                display_derivation_expression=row_expression,
                effective_start_date=eff_date,
                status=row_status,
                data_type=data_type,
                decision_table_json=decision_table_json,
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
            if {"rules", "buckets", "input_columns", "output_columns"} & keys and not {"expression", "formula"} & keys:
                return True
        return False

    json_candidate = extract_json_candidate(stripped)
    if json_candidate is not None:
        try:
            parsed = json.loads(json_candidate)
        except json.JSONDecodeError:
            pass
        else:
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


def _infer_data_type(column_name: str) -> str:
    lowered = column_name.lower()
    if any(token in lowered for token in ("date", "dt", "_at")):
        return "datetime"
    if any(token in lowered for token in ("flag", "flg", "ind", "check", "reason")):
        return "string"
    return "number"


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
