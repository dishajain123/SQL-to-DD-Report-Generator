"""Write / coverage ledger for a SQL procedure.

Every INSERT/UPDATE/MERGE/DELETE must appear here either as a covered
DD-formula candidate or as an explicit unsupported / parse-failure item.
Generation must not silently omit an unparsed write.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from app.models.core import StatementInfo, StructuralInfo


class WriteKind(str, Enum):
    ROW_FORMULA = "row_formula"
    DECISION_TABLE = "decision_table"
    SET_BASED_INSERT = "set_based_insert"
    SET_BASED_MERGE = "set_based_merge"
    SET_BASED_DELETE = "set_based_delete"
    CROSS_ROW = "cross_row"
    PROCEDURE_BRANCH = "procedure_branch"
    EXCEPTION_HANDLER = "exception_handler"
    TEMP_STAGING = "temp_staging"
    PROCESS_STATUS = "process_status"
    UNSUPPORTED = "unsupported"
    PARSE_FAILURE = "parse_failure"


class ConditionScope(str, Enum):
    ROW = "row"
    PROCEDURE = "procedure"
    NONE = "none"


@dataclass
class WriteLedgerEntry:
    statement_index: int
    statement_type: str
    target_table: str
    columns: list[str] = field(default_factory=list)
    kind: WriteKind = WriteKind.UNSUPPORTED
    condition_scope: ConditionScope = ConditionScope.NONE
    parsed_ok: bool = True
    parse_error: str | None = None
    source_excerpt: str = ""
    source_sql: str = ""
    notes: list[str] = field(default_factory=list)
    covered_by_dd: bool = False


@dataclass
class CoverageLedger:
    object_id: str
    entries: list[WriteLedgerEntry] = field(default_factory=list)
    source_anomalies: list[str] = field(default_factory=list)
    inventory_errors: list[str] = field(default_factory=list)
    dd_coverage_checked: bool = False

    @property
    def uncovered_writes(self) -> list[WriteLedgerEntry]:
        return [e for e in self.entries if not e.covered_by_dd]

    @property
    def blockers(self) -> list[str]:
        out: list[str] = list(self.inventory_errors)
        for entry in self.entries:
            if (
                self.dd_coverage_checked
                and entry.kind in {WriteKind.ROW_FORMULA, WriteKind.DECISION_TABLE}
                and not entry.covered_by_dd
            ):
                out.append(
                    f"stmt #{entry.statement_index} {entry.statement_type} "
                    f"→ {entry.target_table}: no validated DD row covers all assigned columns"
                )
            if entry.kind in {WriteKind.PARSE_FAILURE, WriteKind.UNSUPPORTED, WriteKind.PROCEDURE_BRANCH, WriteKind.EXCEPTION_HANDLER}:
                out.append(
                    f"stmt #{entry.statement_index} {entry.statement_type} "
                    f"→ {entry.target_table or '(unknown)'}: {entry.kind.value}"
                    + (f" ({entry.parse_error})" if entry.parse_error else "")
                )
            elif not entry.covered_by_dd and entry.kind not in {WriteKind.ROW_FORMULA, WriteKind.DECISION_TABLE}:
                out.append(
                    f"stmt #{entry.statement_index} {entry.statement_type} "
                    f"→ {entry.target_table}: requires manual / workflow coverage"
                )
        out.extend(self.source_anomalies)
        return out

    @property
    def ready_to_present(self) -> bool:
        return self.dd_coverage_checked and not self.blockers and all(
            e.covered_by_dd
            for e in self.entries
            if e.statement_type in {"INSERT", "UPDATE", "MERGE", "DELETE"}
        )

    def to_markdown(self) -> str:
        lines = [
            f"# Coverage ledger — {self.object_id}",
            "",
            f"Ready to present: **{'yes' if self.ready_to_present else 'no'}**",
            "",
            "| Stmt | Type | Target | Columns | Kind | Scope | Covered | Notes |",
            "|------|------|--------|---------|------|-------|---------|-------|",
        ]
        for e in self.entries:
            cols = ", ".join(e.columns) if e.columns else "—"
            notes = "; ".join(e.notes + ([e.parse_error] if e.parse_error else []))
            lines.append(
                f"| {e.statement_index} | {e.statement_type} | `{e.target_table or '—'}` | "
                f"{cols} | {e.kind.value} | {e.condition_scope.value} | "
                f"{'yes' if e.covered_by_dd else 'no'} | {notes or '—'} |"
            )
        if self.source_anomalies:
            lines.extend(["", "## Source anomalies", ""])
            for anomaly in self.source_anomalies:
                lines.append(f"- {anomaly}")
        if self.blockers:
            lines.extend(["", "## Blockers", ""])
            for blocker in self.blockers:
                lines.append(f"- {blocker}")
        return "\n".join(lines) + "\n"


# `\bIF\b` alone never matches inside ELSIF/ELSEIF (the preceding letter
# blocks the left word boundary) -- match those explicitly so every branch
# of an if/elsif/else chain is recognized as procedure-guarded, not just
# whichever branch's guard text happens to start with a bare "IF"/"ELSE".
_PROC_BRANCH_RE = re.compile(r"(?is)\b(?:ELSE|ELSIF|ELSEIF)\b|\bIF\s+(?!OBJECT_ID\b)")
_CATCH_RE = re.compile(r"(?is)\b(?:BEGIN\s+CATCH|EXCEPTION\b|WHEN\s+OTHERS)\b")
_TEMP_RE = re.compile(r"^#")
_STATUS_RE = re.compile(r"(?i)RUNNINGPROCESSSTATUS|PROCESSSTATUS|RUNSTATUS\b")
_CROSS_ROW_RE = re.compile(
    r"(?is)\bOVER\s*\(|\bGROUP\s+BY\b|\bHAVING\b|\bPARTITION\s+BY\b|"
    r"\bSUM\s*\(|\bCOUNT\s*\(|\bAVG\s*\("
)
_WRITE_TYPES = {"INSERT", "UPDATE", "MERGE", "DELETE", "TRUNCATE", "SELECT"}

# A join condition rendered from the parsed tree can carry an inline
# comment sqlglot attached to the AST node (e.g. `S.Id = A.Id /* Rule 4: ... */`)
# -- stripped before checking the shape is a plain equality, not a real
# part of the join predicate.
_JOIN_COND_COMMENT_RE = re.compile(r"(?s)/\*.*?\*/|--[^\n]*")
_EQUALITY_JOIN_COND_RE = re.compile(
    r'(?is)^[\w."\[\]]+\s*=\s*[\w."\[\]]+(?:\s+AND\s+[\w."\[\]]+\s*=\s*[\w."\[\]]+)*$'
)


def _join_is_lookup(stmt: StatementInfo) -> bool:
    """True when every JOIN in this UPDATE is a lookup -- reached by plain
    equality on its full key, contributing no aggregate/window/GROUP BY --
    as opposed to a genuine fan-out join.

    T-SQL routinely expresses a single-row update-with-dimension-lookup as
    `UPDATE a SET ... FROM t a INNER JOIN dim d ON a.key = d.key`; that is
    semantically a foreign-key reference (4X's "Entity"."FK_DIM"."Column"),
    not a relational/set-based write, and must not be classified the same
    as a genuine multi-row fan-out join.
    """
    raw = stmt.raw_text or ""
    if re.search(r"(?is)\bGROUP\s+BY\b|\bOVER\s*\(|\bHAVING\b", raw):
        return False
    if not stmt.join_conditions:
        return False
    for cond in stmt.join_conditions:
        cleaned = _JOIN_COND_COMMENT_RE.sub(" ", cond).strip()
        if not _EQUALITY_JOIN_COND_RE.match(cleaned):
            return False
    return True


def _global_temp_merge_sources(statements: list[StatementInfo]) -> set[str]:
    """`##` global temp tables in this object that a later INSERT/UPDATE/
    MERGE reads from while writing to a persistent (non-#) table -- i.e.
    a working copy of an entity that this same procedure merges back into
    its real table, not session-local scratch space.

    Session-local `#temp` scratch tables are excluded on purpose: a `##`
    global temp is shared across a whole batch of procedures the way a
    real entity is, so when one of them is provably the source a
    persistent write reads from, it is materially different from a
    disposable `#temp` intermediate and should not be automatically
    excluded from DD coverage the same way.

    This only sees the merge-back when it happens within this same
    object's own statements -- a `##` table whose merge into its real
    table happens in a *different* procedure later in the batch (a real,
    common pattern in this corpus) isn't detectable from one object's
    StructuralInfo alone, so it is conservatively left as TEMP_STAGING.
    """
    sources: set[str] = set()
    for stmt in statements:
        if stmt.statement_type not in {"INSERT", "UPDATE", "MERGE"}:
            continue
        targets = stmt.tables_written or []
        if not any(not _TEMP_RE.match(t) for t in targets):
            continue
        for table in stmt.tables_read or []:
            if table.startswith("##"):
                sources.add(table.upper())
    return sources


def _classify_write(
    stmt: StatementInfo,
    preceding: Iterable[StatementInfo],
    global_temp_merge_sources: frozenset[str] = frozenset(),
) -> tuple[WriteKind, ConditionScope, list[str]]:
    notes: list[str] = []
    raw = stmt.raw_text or ""
    target = (stmt.tables_written[0] if stmt.tables_written else "") or ""
    is_select_into = stmt.statement_type == "SELECT" and bool(target)

    if not target and stmt.statement_type in _WRITE_TYPES - {"SELECT"}:
        return WriteKind.PARSE_FAILURE, ConditionScope.NONE, ["unparsed or missing write target"]

    if not stmt.parsed_ok and target:
        notes.append("parse incomplete; write target recovered for coverage")

    if any(_CATCH_RE.search(p.raw_text or "") for p in preceding) or _CATCH_RE.search(raw):
        for prev in reversed(list(preceding)):
            if _CATCH_RE.search(prev.raw_text or ""):
                return WriteKind.EXCEPTION_HANDLER, ConditionScope.PROCEDURE, notes + ["exception / CATCH path"]
            if (prev.statement_type or "").upper() in _WRITE_TYPES:
                break

    if _CROSS_ROW_RE.search(raw):
        return WriteKind.CROSS_ROW, ConditionScope.ROW, notes + [
            "join/aggregate/window — not expressible as a plain per-row formula"
        ]

    if any(
        _PROC_BRANCH_RE.search(p.raw_text or "") for p in _recent_control(preceding)
    ):
        notes.append("guarded by procedure-level IF/ELSE")
        return WriteKind.PROCEDURE_BRANCH, ConditionScope.PROCEDURE, notes

    if _TEMP_RE.match(target):
        if target.startswith("##") and target.upper() in global_temp_merge_sources:
            notes = notes + [
                "global temp table, but this procedure later merges it into a persistent "
                "table -- treated as the entity's working copy, not disposable scratch space"
            ]
        else:
            return WriteKind.TEMP_STAGING, ConditionScope.ROW, notes + ["temporary table write"]

    if _STATUS_RE.search(target):
        return WriteKind.PROCESS_STATUS, ConditionScope.PROCEDURE, notes + ["process-status bookkeeping"]

    if stmt.statement_type == "MERGE":
        return WriteKind.SET_BASED_MERGE, ConditionScope.ROW, notes + ["set-based MERGE upsert"]
    if stmt.statement_type == "INSERT" or is_select_into:
        return WriteKind.SET_BASED_INSERT, ConditionScope.ROW, notes + (
            ["SELECT INTO write"] if is_select_into else ["set-based INSERT"]
        )
    if stmt.statement_type in {"DELETE", "TRUNCATE"}:
        return WriteKind.SET_BASED_DELETE, ConditionScope.ROW, notes + [f"set-based {stmt.statement_type}"]
    if stmt.statement_type == "UPDATE":
        if stmt.join_tables or re.search(r"(?is)\bJOIN\b", raw):
            if _join_is_lookup(stmt):
                return WriteKind.ROW_FORMULA, ConditionScope.ROW, notes + [
                    "UPDATE…FROM/JOIN on a plain equality key — dimension/reference lookup, "
                    "expressible as a per-row formula"
                ]
            return WriteKind.CROSS_ROW, ConditionScope.ROW, notes + [
                "UPDATE…FROM/JOIN — relational write; requires workflow or manual mapping"
            ]
        return WriteKind.ROW_FORMULA, ConditionScope.ROW, notes
    return WriteKind.UNSUPPORTED, ConditionScope.NONE, notes + ["unsupported write shape"]


def _recent_control(preceding: Iterable[StatementInfo], limit: int = 4) -> list[StatementInfo]:
    out: list[StatementInfo] = []
    for stmt in reversed(list(preceding)):
        if stmt.statement_type == "CONTROL_FLOW":
            out.append(stmt)
            if len(out) >= limit:
                break
        elif stmt.statement_type in _WRITE_TYPES:
            break
    return out


def reconcile_write_inventory(source_sql: str, entries: Iterable[WriteLedgerEntry]) -> list[str]:
    """Compare each operation and target, including repeated writes and temp identity.

    A table-name set cannot distinguish five UPDATEs of a table from four.
    This scanner is intentionally independent of the production SQL parser.
    """
    from app.parsing.write_inventory_scan import scan_expected_writes

    def key(operation: str, table: str) -> tuple[str, str]:
        target = (table or "").strip().strip('"').strip("[]")
        if "." in target:
            target = target.split(".")[-1].strip().strip('"').strip("[]")
        return operation.upper(), target.upper()

    expected = Counter(key(w.operation, w.target_table) for w in scan_expected_writes(source_sql))
    parsed = Counter(key(e.statement_type, e.target_table) for e in entries if e.target_table)
    errors: list[str] = []
    for (operation, table), count in sorted((expected - parsed).items()):
        errors.append(f"Missing {count} {operation} write(s) to {table}")
    for (operation, table), count in sorted((parsed - expected).items()):
        errors.append(f"Unexpected {count} parsed {operation} write(s) to {table}")
    return errors


def build_coverage_ledger(info: StructuralInfo, source_sql: str = "") -> CoverageLedger:
    ledger = CoverageLedger(object_id=info.object_id)
    statements = list(info.statements or [])
    global_temp_merge_sources = frozenset(_global_temp_merge_sources(statements))
    for i, stmt in enumerate(statements):
        is_select_into = (
            stmt.statement_type == "SELECT"
            and bool(stmt.tables_written)
            and re.search(r"(?is)\bINTO\b", stmt.raw_text or "")
        )
        if stmt.statement_type not in {"INSERT", "UPDATE", "MERGE", "DELETE", "TRUNCATE"} and not is_select_into:
            continue
        targets = list(stmt.tables_written) or [""]
        kind, scope, notes = _classify_write(stmt, statements[:i], global_temp_merge_sources)
        for target in targets:
            columns = list((stmt.set_columns_by_table or {}).get(target, []))
            if not columns and stmt.set_columns_by_table:
                for table, cols in stmt.set_columns_by_table.items():
                    if table.lstrip("#").upper() == (target or "").lstrip("#").upper():
                        columns = list(cols)
                        break
            entry = WriteLedgerEntry(
                statement_index=stmt.statement_index,
                statement_type="SELECT_INTO" if is_select_into else stmt.statement_type,
                target_table=target,
                columns=columns,
                kind=kind,
                condition_scope=scope,
                parsed_ok=stmt.parsed_ok,
                parse_error=stmt.parse_error,
                source_excerpt=(stmt.raw_text or "")[:240].replace("\n", " "),
                source_sql=stmt.raw_text or "",
                notes=list(notes),
                covered_by_dd=False,
            )
            ledger.entries.append(entry)

    # Prefer whole-procedure anomaly scan when SQL is available.
    from app.guardrails.source_anomalies import detect_source_anomalies

    full_sql = source_sql or "\n".join(s.raw_text or "" for s in statements)
    ledger.source_anomalies.extend(detect_source_anomalies(full_sql))
    if source_sql:
        ledger.inventory_errors.extend(reconcile_write_inventory(source_sql, ledger.entries))
    return ledger
