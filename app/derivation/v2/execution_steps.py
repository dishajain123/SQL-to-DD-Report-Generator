"""Per-column execution steps and overwrite (shadowing) detection.

The Phase-3 fold collapses every write to a column into one formula, which is
the right platform artefact but hides the sequence: an assignment made in an
early statement and replaced by a later one simply disappears into an
unreachable ELSEIF arm. This module keeps the sequence explicit:

* one :class:`ExecutionStep` per source write, in SQL execution order, with
  its scope (main path vs ``CATCH`` handler) and procedure-wide gate;
* a conservative shadowing check: when the statement behind step *i* also
  assigns literals to other columns, those literals are substituted into a
  later step *j*'s row condition. If the condition then provably holds, every
  row step *i* touched is re-matched and overwritten by step *j* — the
  assignment from step *i* never survives the procedure.

The formula itself is never changed here; the DD must show what the SQL
actually executes. The notes explain why a branch in it is unreachable.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.derivation.derivation_option import format_expression_syntax
from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.phase2_mutation_folder import MutationPass, join_context
from app.derivation.v2.phase3_ast_generator import parse_sql_expression_to_ast
from app.derivation.v2.sql_text import (
    bare_ident,
    extract_update_statements,
    iter_set_assignments,
    stripped_offset_to_line,
)
from app.models.core import ExecutionStep

SCOPE_MAIN = "Main"
SCOPE_CATCH = "Exception handler (CATCH)"

_UNKNOWN = object()


@lru_cache(maxsize=4096)
def _parse(sql: str, entity: str, column: str, as_condition: bool = False) -> dict[str, Any]:
    """Memoised parse for the pairwise checks below. Callers only read the
    returned tree; never mutate it."""
    return parse_sql_expression_to_ast(
        sql, default_entity=entity, target_column=column, as_condition=as_condition
    )


def is_identity_write(mutation: MutationPass, entity: str, column: str) -> bool:
    """``SET col = col`` (or a temp-table copy traced back to the same root
    column) leaves the value unchanged; Phase 3 skips it, and so do steps."""
    try:
        node = _parse(mutation.assigned_expression or "", entity, column)
    except Exception:
        return False
    return _column_key(node, entity) == column.upper()


def build_execution_steps(
    mutations: list[MutationPass],
    *,
    sql_text: str,
    target_entity: str,
    target_column: str,
    statement_label: str = "",
) -> tuple[list[ExecutionStep], list[str]]:
    """Return ``(steps, advisories)`` for one DD column.

    ``advisories`` are row-level notes for overwritten assignments.
    """
    ordered = sorted(
        (m for m in mutations if not is_identity_write(m, target_entity, target_column)),
        key=lambda m: m.source_position,
    )
    steps: list[ExecutionStep] = []
    for number, mutation in enumerate(ordered, 1):
        steps.append(
            ExecutionStep(
                step=number,
                statement_ref=f"{statement_label}stmt #{mutation.statement_index}".strip(),
                source_line=stripped_offset_to_line(sql_text, mutation.source_position),
                operation=mutation.operation,
                scope=SCOPE_CATCH if mutation.is_exception_handler else SCOPE_MAIN,
                workflow_gate=_gate_label(mutation),
                row_condition=_compile_sql(
                    mutation.where_clause, target_entity, target_column, as_condition=True
                ),
                join_conditions=join_context(mutation.joins, mutation.alias_map),
                assigned_value=_compile_sql(
                    mutation.assigned_expression, target_entity, target_column
                ),
                notes=[
                    f"State reset (line "
                    f"{stripped_offset_to_line(sql_text, mutation.state_reset_position or 0)}): "
                    f"{mutation.state_reset}."
                ]
                if mutation.state_reset
                else [],
            )
        )

    advisories: list[str] = []
    for i, earlier in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            later = ordered[j]
            try:
                outcome = _overwrite_outcome(
                    earlier, later, sql_text, target_entity, target_column
                )
            except Exception:  # advisory only — never fail the DD row over it
                outcome = None
            if outcome is None:
                continue
            reason, new_value = outcome
            old_value = steps[i].assigned_value or earlier.assigned_expression
            steps[i].notes.append(
                f"Overwritten by step {j + 1}: {reason}, so every row this step "
                f"sets to {old_value} is re-matched and set to {new_value}. "
                "This assignment does not survive the procedure."
            )
            steps[j].notes.append(f"Overwrites the value assigned in step {i + 1}.")
            advisories.append(
                f"{target_entity}.{target_column}: the value {old_value} assigned at step "
                f"{i + 1} (line {steps[i].source_line}) is always overwritten with "
                f"{new_value} at step {j + 1} (line {steps[j].source_line}) because "
                f"{reason}. The formula shows the executed result; keeping {old_value} "
                "requires a source SQL change."
            )
            break
    return steps, advisories


def _gate_label(mutation: MutationPass) -> str | None:
    if mutation.workflow_gate:
        return mutation.workflow_gate
    kind = (mutation.control_branch_kind or "").upper()
    if kind == "ELSE":
        return "ELSE branch (no preceding IF condition held)"
    if kind in {"IF", "ELSEIF"} and mutation.outer_condition:
        return f"IF {mutation.outer_condition}"
    return None


def _compile_sql(
    sql: str | None, entity: str, column: str, *, as_condition: bool = False
) -> str:
    if not sql or not sql.strip():
        return ""
    try:
        node = parse_sql_expression_to_ast(
            sql, default_entity=entity, target_column=column, as_condition=as_condition
        )
        return format_expression_syntax(
            compile_ast_to_4x_string(
                node, target_entity=entity, target_column=column
            )
        )
    except Exception:  # an unparseable step is still worth listing verbatim
        return " ".join(sql.split())


def _overwrite_outcome(
    earlier: MutationPass,
    later: MutationPass,
    sql_text: str,
    entity: str,
    column: str,
) -> tuple[str, str] | None:
    """``(reason, new_value)`` when ``later`` provably overwrites ``earlier``."""
    if earlier.is_exception_handler != later.is_exception_handler:
        return None
    if (
        earlier.control_branch_group
        and earlier.control_branch_group == later.control_branch_group
        and earlier.control_branch_index != later.control_branch_index
    ):
        return None  # mutually exclusive IF/ELSE arms
    if later.control_branch_group and later.control_branch_group != earlier.control_branch_group:
        return None  # later write only runs when its procedure-level branch is taken
    if earlier.operation != "UPDATE" or later.operation != "UPDATE":
        return None

    value_node = _parse(later.assigned_expression or "", entity, column)
    if _references(value_node, column.upper(), entity):
        return None  # later step transforms the earlier value rather than replacing it
    if not later.effective_condition:
        return (
            "the later statement updates every row unconditionally",
            _compile_sql(later.assigned_expression, entity, column),
        )

    assigned, display = _co_assigned_literals(earlier, entity)
    facts = {
        col: value
        for col, value in assigned.items()
        if not _rewritten_between(sql_text, earlier, later, col)
    }
    if not facts or not later.where_clause:
        return None
    condition = _parse(later.where_clause, entity, column, True)
    if _evaluate(condition, facts, entity) is not True:
        return None

    resolved = _evaluate(value_node, facts, entity)
    if resolved is _UNKNOWN or resolved is None or isinstance(resolved, bool):
        new_value = _compile_sql(later.assigned_expression, entity, column)
    else:
        new_value = _literal_text(resolved)

    used = [
        col
        for col in facts
        if col != column.upper()
        and (_references(condition, col, entity) or _references(value_node, col, entity))
    ]
    if not used:
        return "the later statement's row condition holds for every row this step changed", new_value
    shown = ", ".join(f"{display[col]} = {_literal_text(facts[col])}" for col in used)
    return (
        f"this step's statement also sets {shown}, which satisfies the later "
        "statement's row condition",
        new_value,
    )


def _co_assigned_literals(
    mutation: MutationPass, entity: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Literal values the mutation's statement assigns.

    Returns ``({COLUMN: value}, {COLUMN: column name as written})``.
    """
    facts, display = _co_assigned_literals_cached(mutation.raw_sql or "", entity)
    return dict(facts), dict(display)


@lru_cache(maxsize=1024)
def _co_assigned_literals_cached(
    raw_sql: str, entity: str
) -> tuple[dict[str, Any], dict[str, str]]:
    statements = extract_update_statements(raw_sql)
    if not statements:
        return {}, {}
    facts: dict[str, Any] = {}
    display: dict[str, str] = {}
    for assign in iter_set_assignments(statements[0].get("set_clause") or ""):
        node = _parse(assign["expr"], entity, "")
        if node.get("type") == "LITERAL" and str(node.get("value_type")).upper() != "NULL":
            name = bare_ident(assign["column"])
            facts[name.upper()] = _literal_value(node)
            display[name.upper()] = name
    return facts, display


@lru_cache(maxsize=8)
def _assigned_columns_by_statement(sql_text: str) -> tuple[tuple[int, frozenset[str]], ...]:
    """``(start, {COLUMN, …})`` per UPDATE — computed once per procedure, not
    once per step pair (a column with 50 steps has over 1,200 pairs)."""
    return tuple(
        (
            int(stmt.get("start") or 0),
            frozenset(
                bare_ident(assign["column"]).upper()
                for assign in iter_set_assignments(stmt.get("set_clause") or "")
            ),
        )
        for stmt in extract_update_statements(sql_text)
    )


def _rewritten_between(
    sql_text: str, earlier: MutationPass, later: MutationPass, column: str
) -> bool:
    return any(
        earlier.source_position < start < later.source_position and column in columns
        for start, columns in _assigned_columns_by_statement(sql_text)
    )


def _literal_value(node: dict[str, Any]) -> Any:
    value = node.get("value")
    if str(node.get("value_type")).upper() == "STRING":
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def _literal_text(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _column_key(node: dict[str, Any], entity: str) -> str | None:
    if node.get("type") != "COLUMN_REF" or node.get("relationship"):
        return None
    if str(node.get("entity") or "").upper() != entity.upper():
        return None
    return str(node.get("column") or "").upper()


def _references(node: Any, column: str, entity: str) -> bool:
    if isinstance(node, dict):
        if _column_key(node, entity) == column:
            return True
        return any(_references(v, column, entity) for v in node.values())
    if isinstance(node, list):
        return any(_references(v, column, entity) for v in node)
    return False


def _evaluate(node: Any, facts: dict[str, Any], entity: str) -> Any:
    """Three-valued evaluation: a value/bool, or ``_UNKNOWN``."""
    if not isinstance(node, dict):
        return _UNKNOWN
    kind = node.get("type")
    if kind == "LITERAL":
        if str(node.get("value_type")).upper() == "NULL":
            return None
        return _literal_value(node)
    if kind == "COLUMN_REF":
        key = _column_key(node, entity)
        return facts[key] if key in facts else _UNKNOWN
    if kind == "IF_THEN_ELSE":
        cond = _evaluate(node.get("condition"), facts, entity)
        if cond is True:
            return _evaluate(node.get("then_branch"), facts, entity)
        if cond is False:
            return _evaluate(node.get("else_branch"), facts, entity)
        return _UNKNOWN
    if kind == "FUNCTION_CALL":
        func = str(node.get("function_name") or "").upper()
        args = node.get("arguments") or []
        if func in {"ISEMPTY", "ISNOTEMPTY"} and len(args) == 1:
            value = _evaluate(args[0], facts, entity)
            if value is _UNKNOWN:
                return _UNKNOWN
            empty = value is None or value == ""
            return empty if func == "ISEMPTY" else not empty
        if func == "NOT" and len(args) == 1:
            value = _evaluate(args[0], facts, entity)
            return (not value) if isinstance(value, bool) else _UNKNOWN
        return _UNKNOWN
    if kind == "MEMBERSHIP_OP":
        value = _evaluate(node.get("column"), facts, entity)
        op = str(node.get("operator") or "IN").upper().replace(" ", "")
        if value is _UNKNOWN or op not in {"IN", "NOTIN"}:
            return _UNKNOWN
        hit = any(_same(value, member) for member in node.get("values") or [])
        return hit if op == "IN" else not hit
    if kind == "BINARY_OP":
        op = str(node.get("operator") or "").strip().upper()
        if op in {"AND", "OR"}:
            left = _evaluate(node.get("left"), facts, entity)
            right = _evaluate(node.get("right"), facts, entity)
            if op == "AND":
                if left is False or right is False:
                    return False
                return True if left is True and right is True else _UNKNOWN
            if left is True or right is True:
                return True
            return False if left is False and right is False else _UNKNOWN
        left = _evaluate(node.get("left"), facts, entity)
        right = _evaluate(node.get("right"), facts, entity)
        if left is _UNKNOWN or right is _UNKNOWN or left is None or right is None:
            return _UNKNOWN
        try:
            if op in {"=", "=="}:
                return _same(left, right)
            if op in {"!=", "<>"}:
                return not _same(left, right)
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
        except TypeError:
            return _UNKNOWN
        return _UNKNOWN
    return _UNKNOWN


def _same(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).strip().upper() == str(right).strip().upper()
