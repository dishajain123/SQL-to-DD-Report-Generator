"""Phase 3 — structured 4X JSON AST generator.

Prefers an LLM JSON response when available; always falls back to a
deterministic mutation→AST folder so offline / mocked runs stay usable.

Allowed node types only:
  IF_THEN_ELSE, BINARY_OP, FUNCTION_CALL, MEMBERSHIP_OP, COLUMN_REF, LITERAL
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.derivation.v2.phase2_mutation_folder import MutationPass
from app.derivation.v2.sql_text import (
    bare_ident,
    extract_subquery_dependency_refs,
    normalize_table_name,
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_ALLOWED_TYPES = {
    "IF_THEN_ELSE",
    "BINARY_OP",
    "FUNCTION_CALL",
    "MEMBERSHIP_OP",
    "COLUMN_REF",
    "VARIABLE_REF",
    "LITERAL",
}

_AST_SYSTEM_PROMPT = """\
You convert a chronological sequence of T-SQL UPDATE mutations for ONE
target column into a single JSON Abstract Syntax Tree for the 4X platform.

Return ONLY valid JSON (no markdown fences) using EXACTLY these node types:

- IF_THEN_ELSE: {"type":"IF_THEN_ELSE","condition":<Node>,"then_branch":<Node>,"else_branch":<Node>}
- BINARY_OP: {"type":"BINARY_OP","operator":"=="|"!="|">"|">="|"<"|"<="|"AND"|"OR"|"+"|"-"|"*"|"/","left":<Node>,"right":<Node>}
- FUNCTION_CALL: {"type":"FUNCTION_CALL","function_name":"ISEMPTY"|"ISNOTEMPTY"|"COALESCE"|"SOM"|"EOM"|"ADDDAY","arguments":[<Node>]}
- MEMBERSHIP_OP: {"type":"MEMBERSHIP_OP","operator":"IN"|"NOTIN","column":<Node>,"values":["v1","v2"]}
- COLUMN_REF: {"type":"COLUMN_REF","entity":"ENTITY","relationship":null|"REL","column":"COL"}
- LITERAL: {"type":"LITERAL","value_type":"STRING"|"NUMBER"|"NULL","value":...}

Mapping rules:
- T-SQL IS NULL → FUNCTION_CALL ISEMPTY
- T-SQL IS NOT NULL → FUNCTION_CALL ISNOTEMPTY
- T-SQL IN (...) → MEMBERSHIP_OP
- T-SQL = → operator "=="; <> → "!="
- Join paths → COLUMN_REF.relationship (e.g. "##CUSTOMERCAL")
- Later mutations override earlier ones unless WHERE-guarded; nest as IF/ELSEIF chain
  (outermost IF = last chronological pass).

CRITICAL — arithmetic vs date offset:
- Integer/numeric columns (COUNT, DpdDays, amounts, rates, flags-as-numbers):
  map ``ISNULL(col,0)+1`` / ``col + X`` to BINARY_OP with operator "+"
  (NEVER ADDDAY).
- ADDDAY is ONLY for DATE/DATETIME columns OR explicit SQL DATEADD(DAY, n, expr).
  Example: DATEADD(DAY, 1, ProcessDate) → FUNCTION_CALL ADDDAY(ProcessDate, 1).
"""


# Column-name hints that imply a date/datetime value (ADDDAY allowed).
_DATE_COLUMN_HINTS = (
    "DATE",
    "DATETIME",
    "TIMESTAMP",
    "TIMEKEY",  # not a date value, but often compared — excluded below
    "DT",
    "DOB",
    "SOM",
    "EOM",
)
# Stronger suffixes/tokens that indicate calendar dates (not counters).
_DATE_COLUMN_TOKENS = (
    "DATE",
    "DATETIME",
    "TIMESTAMP",
    "_DT",
    "DOB",
    "PROCESS_DATE",
    "PROCESSDATE",
    "BUSINESS_DATE",
    "BUSINESSDATE",
    "NPADATE",
    "NPA_DATE",
    "OVERDUESINCEDT",
    "EFFECTIVEFROM",
    "EFFECTIVETO",
    "STARTDATE",
    "ENDDATE",
    "UPGRADEDATE",
    "CLASSIFICATIONDATE",
)

# Column-name hints that imply integer/numeric counters (ADDDAY forbidden).
_NUMERIC_COLUMN_HINTS = (
    "COUNT",
    "CNT",
    "DPD",
    "DAYS",
    "AMT",
    "AMOUNT",
    "BAL",
    "BALANCE",
    "PCT",
    "PERCENT",
    "RATE",
    "KEY",
    "ID",
    "NUM",
    "NUMBER",
    "QTY",
    "QUANTITY",
    "SCORE",
    "FLAG",  # often 0/1
    "RUN",
)


def generate_ast(
    mutations: list[MutationPass],
    *,
    target_entity: str,
    target_column: str,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    """Build a 4X JSON AST for the mutation sequence."""
    if not mutations:
        return _column_ref(target_entity, target_column)

    if llm_client is not None and _llm_can_generate_ast(llm_client):
        try:
            raw = _call_llm_for_ast(llm_client, mutations, target_entity, target_column)
            node = _parse_json_ast(raw)
            if node and _validate_ast_shape(node):
                node = _sanitize_addday_misuse(node, target_column)
                return _apply_value_predicate_guard(node, target_entity, target_column)
            logger.warning("phase3 LLM AST failed shape validation; using deterministic fold")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("phase3 LLM AST generation failed: %s", exc)

    try:
        return build_ast_from_mutations(mutations, target_entity, target_column)
    except _ValuePredicateMixingError as exc:
        # Surface through the existing compile-error channel (ast_compiler
        # raises on this sentinel) rather than letting the exception bubble
        # past the caller and silently drop the whole DD row — the pipeline
        # only records validation_errors around compile_ast_to_4x_string.
        logger.warning(
            "phase3 value/predicate validation failed for %s.%s: %s",
            target_entity,
            target_column,
            exc,
        )
        return {
            "type": "FUNCTION_CALL",
            "function_name": "__VALUE_PREDICATE_MIXING__",
            "arguments": [],
            "_validation_error": str(exc),
        }


def _llm_can_generate_ast(llm_client: Any) -> bool:
    return hasattr(llm_client, "generate_ast_json") or hasattr(llm_client, "_complete")


def build_ast_from_mutations(
    mutations: list[MutationPass],
    target_entity: str,
    target_column: str,
) -> dict[str, Any]:
    """Deterministic chronological fold (later guarded passes layer on top).

    Unguarded assignments reset the base value; guarded assignments wrap
    as ``IF(cond)THEN(expr)ELSE(prior)``. Identity write-backs
    (``SET col = same_col``) are skipped.

    Procedural ``IF / ELSE IF / ELSE`` siblings (same ``control_branch_group``)
    are folded as one mutually-exclusive IF/ELSEIF/ELSE tree so a trailing
    ELSE ``UPDATE`` cannot wipe earlier arms.
    """
    if not mutations:
        return _column_ref(target_entity, target_column)

    # Identical assignment expressions across multiple UPDATEs (WHERE only
    # selects rows, e.g. TRY/CATCH both do COUNT = ISNULL(COUNT,0)+1) → emit
    # the assignment alone. Do NOT apply this to a single guarded mutation.
    if (
        len(mutations) > 1
        and not any(m.control_branch_group for m in mutations)
        and len({(m.assigned_expression or "").strip().upper() for m in mutations}) == 1
    ):
        then_node = parse_sql_expression_to_ast(
            mutations[0].assigned_expression,
            default_entity=target_entity,
            target_column=target_column,
        )
        then_node = _sanitize_addday_misuse(then_node, target_column)
        then_node = _unwrap_false_assignment_comparison(then_node, target_entity, target_column)
        _assert_value_not_predicate(then_node, target_column)
        return then_node

    # Default prior value: NULL when every write is conditional (no unguarded
    # base assignment), otherwise self-ref for pass-through.
    has_unguarded = any(
        not m.effective_condition and m.control_branch_kind != "ELSE"
        for m in mutations
    )
    # ELSE arms count as covering defaults inside control groups.
    has_else_arm = any(m.control_branch_kind == "ELSE" for m in mutations)
    if has_unguarded or has_else_arm:
        ast: dict[str, Any] = _column_ref(target_entity, target_column)
    else:
        ast = {"type": "LITERAL", "value_type": "NULL", "value": None}

    for segment in _segment_mutations_by_control_flow(mutations):
        group_id = segment[0].control_branch_group if segment else None
        if group_id and all(m.control_branch_group == group_id for m in segment):
            ast = _fold_control_branch_group(segment, target_entity, target_column)
            continue

        for mutation in segment:
            then_node = parse_sql_expression_to_ast(
                mutation.assigned_expression,
                default_entity=target_entity,
                target_column=target_column,
            )
            then_node = _sanitize_addday_misuse(then_node, target_column)
            then_node = _unwrap_false_assignment_comparison(
                then_node, target_entity, target_column
            )
            _assert_value_not_predicate(then_node, target_column)
            if _is_self_column_ref(then_node, target_entity, target_column):
                continue

            cond_sql = mutation.effective_condition
            if cond_sql:
                cond = parse_sql_expression_to_ast(
                    cond_sql,
                    default_entity=target_entity,
                    target_column=target_column,
                    as_condition=True,
                )
                ast = {
                    "type": "IF_THEN_ELSE",
                    "condition": cond,
                    "then_branch": then_node,
                    "else_branch": ast,
                }
            else:
                # Unguarded pass resets the base value for subsequent guards.
                ast = then_node
    return ast


def _segment_mutations_by_control_flow(
    mutations: list[MutationPass],
) -> list[list[MutationPass]]:
    """Group consecutive mutations that share a control_branch_group."""
    segments: list[list[MutationPass]] = []
    current: list[MutationPass] = []
    current_group: str | None = None

    for mutation in mutations:
        group = mutation.control_branch_group
        if current and group and group == current_group:
            current.append(mutation)
            continue
        if current:
            segments.append(current)
        current = [mutation]
        current_group = group
    if current:
        segments.append(current)
    return segments


def _fold_control_branch_group(
    mutations: list[MutationPass],
    target_entity: str,
    target_column: str,
) -> dict[str, Any]:
    """Fold IF / ELSEIF / ELSE siblings into one IF_THEN_ELSE tree.

    Evaluation order matches T-SQL: first IF condition, then ELSEIF arms,
    then ELSE default. A trailing unguarded ELSE assignment becomes the
    default branch — it must NOT chronologically overwrite earlier arms.
    """
    ordered = sorted(
        mutations,
        key=lambda m: (
            m.control_branch_index if m.control_branch_index is not None else m.ordinal
        ),
    )

    else_muts = [m for m in ordered if m.control_branch_kind == "ELSE"]
    if_muts = [m for m in ordered if m.control_branch_kind != "ELSE"]

    if else_muts:
        # Last ELSE assignment wins within the ELSE arm.
        default = parse_sql_expression_to_ast(
            else_muts[-1].assigned_expression,
            default_entity=target_entity,
            target_column=target_column,
        )
        default = _sanitize_addday_misuse(default, target_column)
        default = _unwrap_false_assignment_comparison(default, target_entity, target_column)
        _assert_value_not_predicate(default, target_column)
    else:
        # Column not written in ELSE → unset / NULL (not self-equality pass-through).
        default = {"type": "LITERAL", "value_type": "NULL", "value": None}

    ast = default
    for mutation in reversed(if_muts):
        then_node = parse_sql_expression_to_ast(
            mutation.assigned_expression,
            default_entity=target_entity,
            target_column=target_column,
        )
        then_node = _sanitize_addday_misuse(then_node, target_column)
        then_node = _unwrap_false_assignment_comparison(
            then_node, target_entity, target_column
        )
        _assert_value_not_predicate(then_node, target_column)
        if _is_self_column_ref(then_node, target_entity, target_column):
            continue
        cond_sql = mutation.effective_condition
        if not cond_sql:
            # IF arm with no usable predicate — treat as unconditional then.
            ast = then_node
            continue
        cond = parse_sql_expression_to_ast(
            cond_sql,
            default_entity=target_entity,
            target_column=target_column,
            as_condition=True,
        )
        ast = {
            "type": "IF_THEN_ELSE",
            "condition": cond,
            "then_branch": then_node,
            "else_branch": ast,
        }
    return ast


_ASSIGNMENT_VALUE_NODE_TYPES = {"LITERAL", "COLUMN_REF", "FUNCTION_CALL", "VARIABLE_REF"}


def _looks_like_assignment_value(node: dict[str, Any]) -> bool:
    """True for node shapes that can legitimately BE an assigned value.

    Broader than "LITERAL only" — a value can just as validly be another
    column's contents (``SET AccountId = Source.AccountId``), a function
    call, or a T-SQL variable. Arithmetic (``+``/``-``/``*``/``/``) is also
    value-shaped. A comparison/logical BINARY_OP is deliberately excluded —
    that shape can never be a legitimate assigned value.
    """
    if not isinstance(node, dict):
        return False
    if node.get("type") in _ASSIGNMENT_VALUE_NODE_TYPES:
        return True
    if node.get("type") == "BINARY_OP" and node.get("operator") in {"+", "-", "*", "/"}:
        return True
    return False


def _unwrap_false_assignment_comparison(
    node: dict[str, Any],
    target_entity: str,
    target_column: str,
) -> dict[str, Any]:
    """Rewrite mistaken ``col == value`` assignment values to bare ``value``.

    ``SET Target = value`` must yield ``value`` directly in THEN/ELSE — never
    a self-equality predicate used as a value. ``value`` may be a literal
    (``'Y'``), but just as often another column reference (e.g.
    ``SET AccountId = Source.AccountId`` folding to a marker that later
    parses as ``AccountId == ##LoanAccountCal.AccountId``) — any
    non-predicate node shape on the non-target-column side qualifies.
    """
    if not isinstance(node, dict):
        return node
    if node.get("type") != "BINARY_OP":
        return node
    op = str(node.get("operator") or "").strip()
    if op not in {"==", "="}:
        return node
    left = node.get("left") or {}
    right = node.get("right") or {}
    if not isinstance(left, dict) or not isinstance(right, dict):
        return node

    def _is_target_col(n: dict[str, Any]) -> bool:
        if n.get("type") != "COLUMN_REF":
            return False
        return str(n.get("column") or "").upper() == str(target_column or "").upper()

    if _is_target_col(left) and _looks_like_assignment_value(right):
        return right
    if _is_target_col(right) and _looks_like_assignment_value(left):
        return left
    return node


def _apply_value_predicate_guard(
    node: Any,
    target_entity: str,
    target_column: str,
) -> Any:
    """Recursively unwrap ``col == value`` mixing anywhere in an AST tree.

    The deterministic fold path (``build_ast_from_mutations``) already calls
    ``_unwrap_false_assignment_comparison`` at each then/else construction
    site. The LLM-generated path does not — an LLM response can just as
    easily produce ``THEN(TargetCol == Value)`` — so this walker re-applies
    the same guard everywhere a THEN/ELSE value slot appears, regardless of
    which producer built the tree.
    """
    if not isinstance(node, dict):
        return node
    node_type = node.get("type")
    if node_type == "IF_THEN_ELSE":
        then_b = _unwrap_false_assignment_comparison(
            node.get("then_branch") or {}, target_entity, target_column
        )
        else_b = _unwrap_false_assignment_comparison(
            node.get("else_branch") or {}, target_entity, target_column
        )
        return {
            **node,
            "condition": _apply_value_predicate_guard(
                node.get("condition"), target_entity, target_column
            ),
            "then_branch": _apply_value_predicate_guard(
                then_b, target_entity, target_column
            ),
            "else_branch": _apply_value_predicate_guard(
                else_b, target_entity, target_column
            ),
        }
    if node_type == "BINARY_OP":
        return {
            **node,
            "left": _apply_value_predicate_guard(
                node.get("left"), target_entity, target_column
            ),
            "right": _apply_value_predicate_guard(
                node.get("right"), target_entity, target_column
            ),
        }
    if node_type == "FUNCTION_CALL":
        return {
            **node,
            "arguments": [
                _apply_value_predicate_guard(a, target_entity, target_column)
                for a in (node.get("arguments") or [])
            ],
        }
    if node_type == "MEMBERSHIP_OP":
        return {
            **node,
            "column": _apply_value_predicate_guard(
                node.get("column"), target_entity, target_column
            ),
        }
    return node


_ORDERING_COMPARISON_OPS = {">", ">=", "<", "<="}


def _assert_value_not_predicate(node: dict[str, Any], target_column: str) -> None:
    """Enforce ``UPDATE SET Col = Value WHERE Condition`` value/predicate separation.

    The assigned ``Value`` must always fold into the AST's THEN node; the
    ``WHERE`` clause (row-level inequalities, join predicates, …) must always
    fold into the IF/ELSEIF condition node — never the other way round. A
    bare ordering comparison (``>``/``>=``/``<``/``<=``) as the *entire*
    assigned value is not a legitimate literal/expression assignment (a
    boolean-from-comparison flag would be wrapped in a CASE, which folds to
    IF_THEN_ELSE, not a raw BINARY_OP) — it is a strong, generalizable signal
    that WHERE-clause boolean logic leaked into the value payload. Fail loud
    here so it surfaces as a validation error instead of silently compiling
    to a boolean where a value belongs.
    """
    if not isinstance(node, dict) or node.get("type") != "BINARY_OP":
        return
    operator = str(node.get("operator") or "").strip()
    if operator not in _ORDERING_COMPARISON_OPS:
        return
    raise _ValuePredicateMixingError(
        f"Value/predicate mixing detected for column '{target_column}': "
        f"assigned expression folded to a bare '{operator}' comparison instead "
        "of a literal/value — WHERE-clause logic must fold into the IF "
        "condition, not the THEN value."
    )


class _ValuePredicateMixingError(ValueError):
    """Raised when an assigned value folds to a bare row-level comparison."""


def _is_self_column_ref(node: dict[str, Any], entity: str, column: str) -> bool:
    if not isinstance(node, dict) or node.get("type") != "COLUMN_REF":
        return False
    if node.get("relationship"):
        return False
    return (
        str(node.get("entity") or "").upper() == str(entity or "").upper()
        and str(node.get("column") or "").upper() == str(column or "").upper()
    )


def parse_sql_expression_to_ast(
    expression: str,
    *,
    default_entity: str,
    target_column: str = "",
    as_condition: bool = False,
) -> dict[str, Any]:
    """Best-effort SQL fragment → AST (CASE, IS NULL, IN, comparisons, refs)."""
    text = (expression or "").strip().rstrip(";")
    if not text:
        return {"type": "LITERAL", "value_type": "NULL", "value": None}

    # Strip wrapping parentheses.
    while text.startswith("(") and text.endswith(")") and _balanced(text[1:-1]):
        text = text[1:-1].strip()

    # CAST(expr AS type) — 4X has no explicit cast; drop the type and parse
    # the inner expression (tried early so a cast wrapping a CASE/DATEADD/
    # arithmetic expression still resolves that inner shape correctly).
    cast_node = _try_parse_cast(text, default_entity, target_column)
    if cast_node is not None:
        return cast_node

    # Explicit DATEADD → ADDDAY (only date-offset form we promote to ADDDAY).
    dateadd = _try_parse_dateadd(text, default_entity, target_column)
    if dateadd is not None:
        return dateadd

    # Oracle/SQL date literals & constructors → TODATE(...)
    date_lit = _try_parse_date_literal(text, default_entity, target_column)
    if date_lit is not None:
        return date_lit

    # Scalar subqueries used as values: (SELECT COUNT(*) …) / (SELECT SUM(…) …)
    subquery = _try_parse_scalar_subquery(text, default_entity, target_column)
    if subquery is not None:
        return subquery

    # EXISTS(...) as a condition — project WHERE or fall back to tautology.
    # Dependency refs are preserved on the node for lineage / HITL.
    if as_condition and re.match(r"(?is)^EXISTS\s*\(", text):
        from app.derivation.v2.sql_text import (
            exists_condition_to_row_predicate,
            extract_subquery_dependency_refs,
        )

        deps = extract_subquery_dependency_refs(text)
        pred = exists_condition_to_row_predicate(text)
        if pred and not re.match(r"(?is)^EXISTS\b", pred.strip()):
            node = parse_sql_expression_to_ast(
                pred,
                default_entity=default_entity,
                target_column=target_column,
                as_condition=True,
            )
            if deps:
                node = {**node, "_dependency_refs": deps}
            return node
        return _fallback_tautology(dependency_refs=deps)

    # GETDATE() / SYSDATE / CURRENT_TIMESTAMP → process-date variable token
    if re.match(r"(?is)^(GETDATE|SYSDATE|CURRENT_TIMESTAMP|CURRENT_DATE)\s*\(\s*\)\s*$", text) or re.match(
        r"(?is)^(SYSDATE|CURRENT_DATE)\s*$", text
    ):
        return {"type": "VARIABLE_REF", "name": "@ProcessDate"}

    # CASE WHEN ... END
    case_ast = _try_parse_case(text, default_entity, target_column)
    if case_ast is not None:
        return case_ast

    # BETWEEN must be tried before AND — otherwise
    # ``col BETWEEN 61 AND 90`` is wrongly split on AND.
    between = re.match(r"(?is)^(.+?)\s+BETWEEN\s+(.+?)\s+AND\s+(.+)$", text)
    if between:
        col = parse_sql_expression_to_ast(
            between.group(1), default_entity=default_entity, target_column=target_column
        )
        low = parse_sql_expression_to_ast(
            between.group(2), default_entity=default_entity, target_column=target_column
        )
        high = parse_sql_expression_to_ast(
            between.group(3), default_entity=default_entity, target_column=target_column
        )
        return {
            "type": "BINARY_OP",
            "operator": "AND",
            "left": {"type": "BINARY_OP", "operator": ">=", "left": col, "right": low},
            "right": {"type": "BINARY_OP", "operator": "<=", "left": col, "right": high},
        }

    # OR / AND (top-level)
    for op in (" OR ", " AND "):
        parts = _split_top_level(text, op.strip())
        if len(parts) > 1:
            node = parse_sql_expression_to_ast(
                parts[0],
                default_entity=default_entity,
                target_column=target_column,
                as_condition=True,
            )
            for part in parts[1:]:
                node = {
                    "type": "BINARY_OP",
                    "operator": op.strip().upper(),
                    "left": node,
                    "right": parse_sql_expression_to_ast(
                        part,
                        default_entity=default_entity,
                        target_column=target_column,
                        as_condition=True,
                    ),
                }
            return node

    # IS NOT NULL / IS NULL
    isnull = re.match(r"(?is)^(.+?)\s+IS\s+NOT\s+NULL\s*$", text)
    if isnull:
        return {
            "type": "FUNCTION_CALL",
            "function_name": "ISNOTEMPTY",
            "arguments": [
                parse_sql_expression_to_ast(
                    isnull.group(1),
                    default_entity=default_entity,
                    target_column=target_column,
                )
            ],
        }
    isnull = re.match(r"(?is)^(.+?)\s+IS\s+NULL\s*$", text)
    if isnull:
        return {
            "type": "FUNCTION_CALL",
            "function_name": "ISEMPTY",
            "arguments": [
                parse_sql_expression_to_ast(
                    isnull.group(1),
                    default_entity=default_entity,
                    target_column=target_column,
                )
            ],
        }

    # NOT IN / IN
    membership = re.match(
        r"(?is)^(.+?)\s+(NOT\s+)?IN\s*\((.+)\)\s*$",
        text,
    )
    if membership:
        inner_list = membership.group(3).strip()
        if re.match(r"(?is)^SELECT\b", inner_list):
            from app.derivation.v2.sql_text import in_subquery_to_row_predicate

            lhs = membership.group(1).strip()
            pred, deps = in_subquery_to_row_predicate(lhs, inner_list)
            if pred:
                node = parse_sql_expression_to_ast(
                    pred,
                    default_entity=default_entity,
                    target_column=target_column,
                    as_condition=True,
                )
                if deps:
                    node = {**node, "_dependency_refs": deps}
                return node
            return _fallback_tautology(dependency_refs=deps)
        values = [
            _literal_from_sql_token(v.strip())
            for v in _split_top_level(inner_list, ",")
        ]
        lit_values = []
        for v in values:
            if v.get("value_type") == "STRING":
                lit_values.append(v.get("value"))
            elif v.get("value_type") == "NUMBER":
                lit_values.append(v.get("value"))
            else:
                lit_values.append(None)
        return {
            "type": "MEMBERSHIP_OP",
            "operator": "NOTIN" if membership.group(2) else "IN",
            "column": parse_sql_expression_to_ast(
                membership.group(1),
                default_entity=default_entity,
                target_column=target_column,
            ),
            "values": lit_values,
        }

    # LIKE / NOT LIKE pattern matching — tried before the arithmetic loop
    # below, since the pattern side is very often a concatenation
    # (``col LIKE '%' + Other + '%'``); splitting on LIKE first leaves a
    # clean sub-expression for the recursive call to hand to arithmetic.
    # The 4X grammar has no LIKE token at all — pattern matching is native
    # only via MEMBERSHIP_OP CONTAINS/BEGINSWITH/ENDSWITH/DOESNOTCONTAINS,
    # which take a literal value list, not an arbitrary expression. So a
    # LIKE with a genuinely dynamic pattern (e.g. concatenated with a
    # column) has no valid 4X representation at all; a fixed-literal
    # pattern (the common case) maps onto those operators cleanly.
    not_like_split = _split_top_level_keyword(text, r"NOT\s+LIKE")
    if len(not_like_split) == 2 and not_like_split[0].strip():
        lhs_sql, rhs_sql = not_like_split
        mapped = _try_map_like_to_membership(
            lhs_sql.strip(), rhs_sql.strip(), negate=True,
            default_entity=default_entity, target_column=target_column,
        )
        if mapped is not None:
            return mapped
        return {
            "type": "BINARY_OP",
            "operator": "NOT LIKE",
            "left": parse_sql_expression_to_ast(
                lhs_sql.strip(), default_entity=default_entity, target_column=target_column
            ),
            "right": parse_sql_expression_to_ast(
                rhs_sql.strip(), default_entity=default_entity, target_column=target_column
            ),
        }
    like_split = _split_top_level_keyword(text, "LIKE")
    if len(like_split) == 2 and like_split[0].strip():
        lhs_sql, rhs_sql = like_split
        mapped = _try_map_like_to_membership(
            lhs_sql.strip(), rhs_sql.strip(), negate=False,
            default_entity=default_entity, target_column=target_column,
        )
        if mapped is not None:
            return mapped
        return {
            "type": "BINARY_OP",
            "operator": "LIKE",
            "left": parse_sql_expression_to_ast(
                lhs_sql.strip(), default_entity=default_entity, target_column=target_column
            ),
            "right": parse_sql_expression_to_ast(
                rhs_sql.strip(), default_entity=default_entity, target_column=target_column
            ),
        }

    # Comparisons
    for sql_op, fourx_op in (
        ("<>", "!="),
        (">=", ">="),
        ("<=", "<="),
        ("!=", "!="),
        ("==", "=="),
        (">", ">"),
        ("<", "<"),
        ("=", "=="),
    ):
        parts = _split_top_level(text, sql_op)
        if len(parts) == 2:
            return {
                "type": "BINARY_OP",
                "operator": fourx_op,
                "left": parse_sql_expression_to_ast(
                    parts[0], default_entity=default_entity, target_column=target_column
                ),
                "right": parse_sql_expression_to_ast(
                    parts[1], default_entity=default_entity, target_column=target_column
                ),
            }

    # ISNULL(a,b) / COALESCE(a,b) / NVL(a,b) / IFNULL(a,b) (MySQL) — never
    # ISEMPTY; keep as COALESCE.
    isnull_fn = re.match(r"(?is)^(?:ISNULL|COALESCE|NVL|IFNULL)\s*\((.+)\)$", text)
    if isnull_fn:
        args = [
            parse_sql_expression_to_ast(
                a, default_entity=default_entity, target_column=target_column
            )
            for a in _split_top_level(isnull_fn.group(1), ",")
        ]
        return {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": args}

    # LEAST(...)/GREATEST(...) (Oracle/MySQL) — semantically identical to
    # MIN/MAX applied to the same argument list; map onto those so the
    # platform-recognized function names are used instead of falling
    # through to a raw string literal.
    least_greatest_fn = re.match(r"(?is)^(?P<fn>LEAST|GREATEST)\s*\((?P<args>.*)\)$", text)
    if least_greatest_fn:
        mapped_fn = "MIN" if least_greatest_fn.group("fn").upper() == "LEAST" else "MAX"
        args = [
            parse_sql_expression_to_ast(
                a, default_entity=default_entity, target_column=target_column
            )
            for a in _split_top_level(least_greatest_fn.group("args"), ",")
        ]
        return {"type": "FUNCTION_CALL", "function_name": mapped_fn, "arguments": args}

    # Generic known functions: MIN/MAX/SUM/COUNT/ABS/ROUND/CONCAT/...
    gen_fn = re.match(
        r"(?is)^(?P<fn>MIN|MAX|SUM|COUNT|ABS|ROUND|CONCAT|DATEDIFF|LEN|UPPER|LOWER)\s*\((?P<args>.*)\)$",
        text,
    )
    if gen_fn:
        raw_args = gen_fn.group("args").strip()
        args = []
        if raw_args:
            for a in _split_top_level(raw_args, ","):
                # COUNT(*) / COUNT(1)
                if a.strip() in {"*", "1"}:
                    args.append({"type": "LITERAL", "value_type": "NUMBER", "value": 1})
                else:
                    args.append(
                        parse_sql_expression_to_ast(
                            a,
                            default_entity=default_entity,
                            target_column=target_column,
                        )
                    )
        return {
            "type": "FUNCTION_CALL",
            "function_name": gen_fn.group("fn").upper(),
            "arguments": args,
        }

    # ERROR_MESSAGE() / @@ERROR — map to @ErrorMessage variable token.
    if re.match(r"(?is)^(?:ERROR_MESSAGE|ERROR_NUMBER|ERROR_LINE)\s*\(\s*\)\s*$", text):
        return {"type": "VARIABLE_REF", "name": "@ErrorMessage"}

    # Arithmetic + - * /  — numeric increments stay BINARY_OP; date + N → ADDDAY
    # only when the operand/target is date-like (never for COUNT/DPD/amounts).
    for op in ("+", "-", "*", "/"):
        parts = _split_top_level(text, op)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            left = parse_sql_expression_to_ast(
                parts[0], default_entity=default_entity, target_column=target_column
            )
            right = parse_sql_expression_to_ast(
                parts[1], default_entity=default_entity, target_column=target_column
            )
            # Guard: never let ISNULL/COALESCE(+N) become ADDDAY/ISEMPTY.
            if op in {"+", "-"} and _should_use_addday_for_arithmetic(
                left, right, target_column
            ):
                # date ± N  →  ADDDAY(date, ±N)
                offset = right
                if op == "-" and offset.get("type") == "LITERAL":
                    try:
                        offset = {
                            "type": "LITERAL",
                            "value_type": "NUMBER",
                            "value": -1 * float(offset.get("value")),
                        }
                        if float(offset["value"]) == int(offset["value"]):
                            offset["value"] = int(offset["value"])
                    except (TypeError, ValueError):
                        offset = {
                            "type": "BINARY_OP",
                            "operator": "*",
                            "left": {"type": "LITERAL", "value_type": "NUMBER", "value": -1},
                            "right": right,
                        }
                elif op == "-":
                    offset = {
                        "type": "BINARY_OP",
                        "operator": "*",
                        "left": {"type": "LITERAL", "value_type": "NUMBER", "value": -1},
                        "right": right,
                    }
                return {
                    "type": "FUNCTION_CALL",
                    "function_name": "ADDDAY",
                    "arguments": [left, offset],
                }
            return {
                "type": "BINARY_OP",
                "operator": op,
                "left": left,
                "right": right,
            }

    # Unary minus / plus (``-A.OverdueDays``, ``-DAY(@ProcessDate)``) — tried
    # only after the binary arithmetic loop above has already had first
    # crack at any top-level operator (so ``-A.OverdueDays + 1`` still
    # splits as BINARY_OP "+" first, recursing into "-A.OverdueDays" for
    # this branch). A pure negative number literal (``-15``) is excluded
    # here and handled by the numeric-literal check below instead. Without
    # this, a leading unary minus falls through every remaining check and
    # silently becomes a STRING literal of the raw SQL text.
    unary = re.match(r"(?is)^([+-])\s*(\S.*)$", text)
    if unary and not re.match(r"^-?\d+(\.\d+)?$", text):
        sign, operand_sql = unary.group(1), unary.group(2).strip()
        operand = parse_sql_expression_to_ast(
            operand_sql, default_entity=default_entity, target_column=target_column
        )
        if sign == "+":
            return operand
        if operand.get("type") == "LITERAL" and operand.get("value_type") == "NUMBER":
            try:
                value = -1 * float(operand["value"])
                operand["value"] = int(value) if value == int(value) else value
            except (TypeError, ValueError):
                pass
            return operand
        return {
            "type": "BINARY_OP",
            "operator": "*",
            "left": {"type": "LITERAL", "value_type": "NUMBER", "value": -1},
            "right": operand,
        }

    # T-SQL scalar variables (@GraceWindowStart) — before string/column fallback.
    if re.match(r"^@[A-Za-z_][A-Za-z0-9_]*$", text):
        return {"type": "VARIABLE_REF", "name": text}

    # Numeric / string literals BEFORE column paths — otherwise ``1.10`` is
    # mistaken for COLUMN_REF(entity="1", column="10").
    if re.match(r"^-?\d+(\.\d+)?$", text):
        return _literal_from_sql_token(text)
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith("N'") and text.endswith("'")
    ) or (text.startswith('"') and text.endswith('"')):
        return _literal_from_sql_token(text)
    if text.upper() in {"NULL", "NONE"}:
        return _literal_from_sql_token(text)

    # Entity::Rel::Col or Entity::Col markers from phase2
    marker = re.match(
        r'^(?P<entity>[#A-Za-z_][#A-Za-z0-9_]*)::(?:(?P<rel>[#A-Za-z_][#A-Za-z0-9_]*)::)?(?P<col>[A-Za-z_][A-Za-z0-9_]*)$',
        text,
    )
    if marker:
        return _column_ref(
            marker.group("entity"),
            marker.group("col"),
            relationship=marker.group("rel"),
        )

    # Qualified SQL col: Entity.Col / alias.Col (identifiers must start with a letter/_/#)
    qual = re.match(
        r'^(?:\[?(?P<e1>[#A-Za-z_][#A-Za-z0-9_]*)\]?\.)?(?:\[?(?P<e2>[#A-Za-z_][#A-Za-z0-9_]*)\]?\.)?\[?(?P<col>[A-Za-z_][A-Za-z0-9_]*)\]?$',
        text,
    )
    if qual and (qual.group("e1") or re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text)):
        if qual.group("e1") and qual.group("e2"):
            return _column_ref(qual.group("e1"), qual.group("col"), relationship=qual.group("e2"))
        if qual.group("e1"):
            # Could be entity.col or just alias.col — treat first as entity.
            return _column_ref(qual.group("e1"), qual.group("col"))
        return _column_ref(default_entity, qual.group("col"))

    # Literals
    return _literal_from_sql_token(text)


def _fallback_tautology(
    *,
    dependency_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Grammar-valid ``1 == 1`` when a subquery cannot be projected further."""
    node: dict[str, Any] = {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {"type": "LITERAL", "value_type": "NUMBER", "value": 1},
        "right": {"type": "LITERAL", "value_type": "NUMBER", "value": 1},
    }
    if dependency_refs:
        node["_dependency_refs"] = list(dependency_refs)
    return node


def _find_top_level_as(text: str) -> int | None:
    """Index of the first ``AS`` keyword at paren-depth 0 (word-boundary aware)."""
    depth = 0
    in_single = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_single:
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and text[i : i + 2].upper() == "AS":
            before_ok = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
            after_idx = i + 2
            after_ok = after_idx >= n or not (text[after_idx].isalnum() or text[after_idx] == "_")
            if before_ok and after_ok:
                return i
        i += 1
    return None


def _try_parse_cast(
    text: str,
    default_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """``CAST(expr AS type)`` -> the inner expr's AST.

    4X has no explicit cast operator; the target SQL type carries no
    derivation-relevant information, so it is dropped and the wrapped
    expression is parsed on its own. Without this, the entire cast
    (including the type/precision) fell through to the final fallback and
    became a STRING literal of the raw SQL text.
    """
    m = re.match(r"(?is)^CAST\s*\((?P<inner>.*)\)\s*$", text)
    if not m or not _balanced(m.group("inner")):
        return None
    inner = m.group("inner")
    split_at = _find_top_level_as(inner)
    if split_at is None:
        return None
    expr_sql = inner[:split_at].strip()
    if not expr_sql:
        return None
    return parse_sql_expression_to_ast(
        expr_sql, default_entity=default_entity, target_column=target_column
    )


def _try_parse_dateadd(
    text: str,
    default_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """Map ``DATEADD(DAY|DD, n, expr)`` → ``ADDDAY(expr, n)``."""
    match = re.match(
        r"(?is)^DATEADD\s*\(\s*(?P<unit>DAY|DAYS|DD)\s*,\s*(?P<offset>.+?)\s*,\s*(?P<base>.+)\s*\)$",
        text,
    )
    if not match:
        # Also accept sqlglot-ish DATE_ADD(base, n, 'day')
        match2 = re.match(
            r"(?is)^DATE_ADD\s*\(\s*(?P<base>.+?)\s*,\s*(?P<offset>.+?)\s*,\s*'?(?:DAY|DAYS|DD)'?\s*\)$",
            text,
        )
        if not match2:
            return None
        base = parse_sql_expression_to_ast(
            match2.group("base"),
            default_entity=default_entity,
            target_column=target_column,
        )
        offset = parse_sql_expression_to_ast(
            match2.group("offset"),
            default_entity=default_entity,
            target_column=target_column,
        )
        return {
            "type": "FUNCTION_CALL",
            "function_name": "ADDDAY",
            "arguments": [base, offset],
        }

    base = parse_sql_expression_to_ast(
        match.group("base"),
        default_entity=default_entity,
        target_column=target_column,
    )
    offset = parse_sql_expression_to_ast(
        match.group("offset"),
        default_entity=default_entity,
        target_column=target_column,
    )
    return {
        "type": "FUNCTION_CALL",
        "function_name": "ADDDAY",
        "arguments": [base, offset],
    }


def _try_parse_date_literal(
    text: str,
    default_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """Map Oracle/SQL date literals to ``TODATE("YYYY-MM-DD")``.

    Handles:
      DATE '1900-01-01'
      TO_DATE('01/01/1900','DD/MM/YYYY')
      TODATE('1900-01-01')
      DATE('1900-01-01')
    """
    del default_entity, target_column  # reserved for nested args later

    # DATE 'YYYY-MM-DD' / DATE "YYYY-MM-DD"
    m = re.match(r"(?is)^DATE\s+'([^']+)'\s*$", text) or re.match(
        r'(?is)^DATE\s+"([^"]+)"\s*$', text
    )
    if m:
        return {
            "type": "FUNCTION_CALL",
            "function_name": "TODATE",
            "arguments": [
                {"type": "LITERAL", "value_type": "STRING", "value": m.group(1).strip()}
            ],
        }

    # TO_DATE('d','fmt') / TODATE('d') / DATE('d')
    m = re.match(
        r"(?is)^(?:TO_DATE|TODATE|DATE)\s*\(\s*'([^']+)'\s*(?:,\s*'([^']*)')?\s*\)\s*$",
        text,
    ) or re.match(
        r'(?is)^(?:TO_DATE|TODATE|DATE)\s*\(\s*"([^"]+)"\s*(?:,\s*"([^"]*)")?\s*\)\s*$',
        text,
    )
    if m:
        args: list[dict[str, Any]] = [
            {"type": "LITERAL", "value_type": "STRING", "value": m.group(1).strip()}
        ]
        if m.group(2) and m.group(2).strip():
            # Keep format as a second string arg when present (platform accepts it).
            args.append(
                {"type": "LITERAL", "value_type": "STRING", "value": m.group(2).strip()}
            )
        return {"type": "FUNCTION_CALL", "function_name": "TODATE", "arguments": args}

    return None


def _try_parse_scalar_subquery(
    text: str,
    default_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """Collapse common scalar aggregations into COUNT/SUM/COALESCE(SUM,…).

    Full multi-table subquery semantics are not expressible in 4X; we keep the
    aggregate intent so arithmetic like ``col + (SELECT COUNT(*) …)`` stays numeric.
    """
    # Unwrap a single pair of outer parens if this is a parenthesized SELECT.
    body = text.strip()
    if body.startswith("(") and body.endswith(")") and _balanced(body[1:-1]):
        body = body[1:-1].strip()
    if not re.match(r"(?is)^SELECT\b", body):
        return None

    # SELECT COUNT(*) … / SELECT COUNT(1) …
    count_m = re.match(
        r"(?is)^SELECT\s+COUNT\s*\(\s*(?:\*|1|[A-Za-z_][A-Za-z0-9_]*)\s*\)(?:\s|$)",
        body,
    )
    if count_m:
        return {
            "type": "FUNCTION_CALL",
            "function_name": "COUNT",
            "arguments": [{"type": "LITERAL", "value_type": "NUMBER", "value": 1}],
            "_dependency_refs": extract_subquery_dependency_refs(body),
        }

    # SELECT ISNULL(SUM(col), 0) … / SELECT COALESCE(SUM(col), 0) … / SELECT NVL(SUM(col), 0)
    sum_null_m = re.match(
        r"(?is)^SELECT\s+(?:ISNULL|COALESCE|NVL)\s*\(\s*SUM\s*\(\s*"
        r"(?:(?:\[?#?#?[A-Za-z_][A-Za-z0-9_]*\]?\.)?\[?[A-Za-z_][A-Za-z0-9_]*\]?)"
        r"\s*\)\s*,\s*(.+?)\s*\)(?:\s|$)",
        body,
    )
    if sum_null_m:
        fallback = parse_sql_expression_to_ast(
            sum_null_m.group(1),
            default_entity=default_entity,
            target_column=target_column,
        )
        # Extract column name for SUM arg
        col_m = re.search(
            r"(?is)SUM\s*\(\s*(?:(?:\[?#?#?[A-Za-z_][A-Za-z0-9_]*\]?\.)?(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?))\s*\)",
            body,
        )
        col_name = bare_ident(col_m.group("col")) if col_m else target_column or "Amount"
        sum_node = {
            "type": "FUNCTION_CALL",
            "function_name": "SUM",
            "arguments": [
                {
                    "type": "COLUMN_REF",
                    "entity": default_entity,
                    "relationship": None,
                    "column": col_name,
                }
            ],
        }
        return {
            "type": "FUNCTION_CALL",
            "function_name": "COALESCE",
            "arguments": [sum_node, fallback],
            "_dependency_refs": extract_subquery_dependency_refs(body),
        }

    # SELECT SUM(col) …
    sum_m = re.match(
        r"(?is)^SELECT\s+SUM\s*\(\s*"
        r"(?:(?:\[?#?#?[A-Za-z_][A-Za-z0-9_]*\]?\.)?(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?))"
        r"\s*\)(?:\s|$)",
        body,
    )
    if sum_m:
        return {
            "type": "FUNCTION_CALL",
            "function_name": "SUM",
            "arguments": [
                {
                    "type": "COLUMN_REF",
                    "entity": default_entity,
                    "relationship": None,
                    "column": bare_ident(sum_m.group("col")),
                }
            ],
        }

    # SELECT MIN/MAX(col) …
    agg_m = re.match(
        r"(?is)^SELECT\s+(?P<fn>MIN|MAX)\s*\(\s*"
        r"(?:(?:\[?#?#?[A-Za-z_][A-Za-z0-9_]*\]?\.)?(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?))"
        r"\s*\)(?:\s|$)",
        body,
    )
    if agg_m:
        return {
            "type": "FUNCTION_CALL",
            "function_name": agg_m.group("fn").upper(),
            "arguments": [
                {
                    "type": "COLUMN_REF",
                    "entity": default_entity,
                    "relationship": None,
                    "column": bare_ident(agg_m.group("col")),
                }
            ],
        }

    # Plain (non-aggregate) single-column lookup: SELECT col FROM table
    # [alias] [WHERE ...] — e.g. a DimAssetClass/lookup-table key fetch.
    # Unlike the aggregate cases above, this has no natural tie to
    # ``default_entity`` (the subquery's real source is a *different*
    # table), so it resolves against that source table directly — still
    # not entity-map-aware here (phase3 has no entity_map), but far better
    # than the previous behaviour of collapsing the whole subquery,
    # including a typed lookup key, into an opaque STRING literal.
    plain_m = re.match(
        r"(?is)^SELECT\s+(?:(?:\[?#?#?[A-Za-z_][A-Za-z0-9_]*\]?\.)?(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?))"
        r"\s+FROM\s+(?P<table>\[?#?#?[A-Za-z_][A-Za-z0-9_]*\]?)"
        r"(?:\s+(?:AS\s+)?(?!WHERE\b|GROUP\b|ORDER\b|JOIN\b|HAVING\b)[A-Za-z_][A-Za-z0-9_]*)?"
        r"(?:\s+WHERE\s.*)?$",
        body,
    )
    if plain_m and not re.search(r"(?is)\bGROUP\s+BY\b|\bJOIN\b", body):
        return {
            "type": "COLUMN_REF",
            "entity": normalize_table_name(plain_m.group("table")),
            "relationship": None,
            "column": bare_ident(plain_m.group("col")),
            "_dependency_refs": extract_subquery_dependency_refs(body),
        }

    return None


def _should_use_addday_for_arithmetic(
    left: dict[str, Any],
    right: dict[str, Any],
    target_column: str,
) -> bool:
    """True only for date ± numeric-day arithmetic — never for counters."""
    if _is_numeric_column_name(target_column):
        return False
    # Right side must look like a day offset (number / numeric literal expr).
    if not _node_looks_numeric(right):
        return False
    if _is_date_like_column_name(target_column):
        return True
    return _node_looks_date_valued(left)


def _is_date_like_column_name(name: str) -> bool:
    upper = (name or "").upper()
    if not upper:
        return False
    if _is_numeric_column_name(upper):
        return False
    return any(tok in upper for tok in _DATE_COLUMN_TOKENS)


def _is_numeric_column_name(name: str) -> bool:
    upper = (name or "").upper()
    if not upper:
        return False
    # Exact / suffix hits for counters and amounts.
    for hint in _NUMERIC_COLUMN_HINTS:
        if upper == hint or upper.endswith(hint) or hint in upper.split("_"):
            # Avoid treating *DATE* columns that contain "DAY" carefully —
            # "DAYS" / "DPD" are numeric; "DATE" is not handled here.
            if hint == "DAYS" and "DATE" in upper and "DPD" not in upper:
                continue
            return True
    return False


def _node_looks_numeric(node: dict[str, Any] | None) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("type") == "LITERAL":
        return str(node.get("value_type") or "").upper() == "NUMBER"
    if node.get("type") == "BINARY_OP" and node.get("operator") in {"+", "-", "*", "/"}:
        return _node_looks_numeric(node.get("left")) and _node_looks_numeric(node.get("right"))
    if node.get("type") == "FUNCTION_CALL":
        func = str(node.get("function_name") or "").upper()
        if func in {"COALESCE", "ABS", "ROUND", "FLOOR", "CEIL", "ISNULL", "COUNT", "SUM", "MIN", "MAX"}:
            args = node.get("arguments") or []
            if func in {"COUNT", "SUM"}:
                return True
            return any(_node_looks_numeric(a) for a in args)
        return False
    if node.get("type") == "COLUMN_REF":
        return _is_numeric_column_name(str(node.get("column") or ""))
    return False


def _node_looks_date_valued(node: dict[str, Any] | None) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("type") == "COLUMN_REF":
        return _is_date_like_column_name(str(node.get("column") or ""))
    if node.get("type") == "FUNCTION_CALL":
        func = str(node.get("function_name") or "").upper()
        if func in {"SOM", "EOM", "ADDDAY", "TODATE", "DATE"}:
            return True
        if func == "COALESCE":
            args = node.get("arguments") or []
            return bool(args) and _node_looks_date_valued(args[0])
    return False


def _sanitize_addday_misuse(node: dict[str, Any], target_column: str) -> dict[str, Any]:
    """Rewrite ADDDAY(x, n) → x + n when the target/operand is numeric, not a date."""

    def walk(n: Any) -> Any:
        if not isinstance(n, dict):
            return n
        ntype = n.get("type")
        if ntype == "FUNCTION_CALL" and str(n.get("function_name") or "").upper() == "ADDDAY":
            args = [walk(a) for a in (n.get("arguments") or [])]
            if len(args) >= 2 and _addday_should_be_numeric_plus(args[0], target_column):
                return {
                    "type": "BINARY_OP",
                    "operator": "+",
                    "left": args[0],
                    "right": args[1],
                }
            return {**n, "arguments": args}
        if ntype == "IF_THEN_ELSE":
            return {
                **n,
                "condition": walk(n.get("condition")),
                "then_branch": walk(n.get("then_branch")),
                "else_branch": walk(n.get("else_branch")),
            }
        if ntype == "BINARY_OP":
            return {**n, "left": walk(n.get("left")), "right": walk(n.get("right"))}
        if ntype == "FUNCTION_CALL":
            return {**n, "arguments": [walk(a) for a in (n.get("arguments") or [])]}
        if ntype == "MEMBERSHIP_OP":
            return {**n, "column": walk(n.get("column"))}
        return n

    return walk(node)


def _addday_should_be_numeric_plus(base: dict[str, Any], target_column: str) -> bool:
    if _is_numeric_column_name(target_column):
        return True
    if _is_date_like_column_name(target_column):
        return False
    if _node_looks_date_valued(base):
        return False
    # COALESCE(COUNT, 0) / COUNT column refs → numeric
    col = _column_name_from_node(base)
    if _node_looks_numeric(base) or (col and _is_numeric_column_name(col)):
        return True
    return False


def _column_name_from_node(node: dict[str, Any] | None) -> str | None:
    if not isinstance(node, dict):
        return None
    if node.get("type") == "COLUMN_REF":
        return str(node.get("column") or "") or None
    if node.get("type") == "FUNCTION_CALL":
        args = node.get("arguments") or []
        for arg in args:
            name = _column_name_from_node(arg)
            if name:
                return name
    if node.get("type") == "BINARY_OP":
        return _column_name_from_node(node.get("left")) or _column_name_from_node(
            node.get("right")
        )
    return None



def _extract_matching_case_body(text: str) -> tuple[str, bool]:
    """Return the text between a leading ``CASE`` and ITS matching ``END``.

    Depth-aware over nested ``CASE ... END`` pairs (and string literals) —
    unlike a ``$``-anchored regex, this correctly stops at the END that
    closes THIS CASE, not wherever the next END substring happens to be.
    """
    m = re.match(r"(?is)^CASE\b", text)
    if not m:
        return "", False
    i = m.end()
    n = len(text)
    depth = 1
    in_single = False
    while i < n:
        ch = text[i]
        if in_single:
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if re.match(r"(?is)^CASE\b", text[i:]):
            depth += 1
            i += 4
            continue
        if re.match(r"(?is)^END\b", text[i:]):
            depth -= 1
            if depth == 0:
                return text[m.end() : i].strip(), True
            i += 3
            continue
        i += 1
    return "", False


def _scan_case_top_level_markers(body: str) -> list[tuple[int, str]]:
    """Positions of ``WHEN``/``THEN``/``ELSE`` at CASE-depth 0, paren-depth 0.

    Keywords belonging to a nested ``CASE ... END`` (inside a THEN/ELSE
    value) are excluded — the nested CASE's own depth tracking absorbs them,
    so they never register here and stay embedded verbatim in the outer
    branch's captured text for a later recursive parse.
    """
    markers: list[tuple[int, str]] = []
    n = len(body)
    i = 0
    depth = 0
    paren_depth = 0
    in_single = False
    while i < n:
        ch = body[i]
        if in_single:
            if ch == "'":
                if i + 1 < n and body[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == "(":
            paren_depth += 1
            i += 1
            continue
        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue
        if re.match(r"(?is)^CASE\b", body[i:]):
            depth += 1
            i += 4
            continue
        if re.match(r"(?is)^END\b", body[i:]):
            depth = max(0, depth - 1)
            i += 3
            continue
        if depth == 0 and paren_depth == 0:
            km = re.match(r"(?is)^(WHEN|THEN|ELSE)\b", body[i:])
            if km:
                markers.append((i, km.group(1).upper()))
                i += len(km.group(1))
                continue
        i += 1
    return markers


def _try_parse_case(
    text: str,
    default_entity: str,
    target_column: str = "",
) -> dict[str, Any] | None:
    """Parse a ``CASE`` expression: searched, simple, and nested forms.

    Searched: ``CASE WHEN cond1 THEN r1 ... [ELSE e] END``
    Simple:   ``CASE operand WHEN v1 THEN r1 ... [ELSE e] END`` — each WHEN
              value folds to an equality against ``operand``.
    Nested:   a THEN/ELSE value that is itself a full CASE expression —
              handled by depth-aware marker scanning, so an inner CASE's own
              WHEN/THEN/ELSE never get mistaken for the outer CASE's.
    """
    body, matched = _extract_matching_case_body(text)
    if not matched:
        return None

    markers = _scan_case_top_level_markers(body)
    first_when_idx = next((i for i, (_, k) in enumerate(markers) if k == "WHEN"), None)
    if first_when_idx is None:
        return None

    # Simple-CASE operand: any text before the first top-level WHEN.
    operand_sql = body[: markers[first_when_idx][0]].strip() or None

    else_node: dict[str, Any] = {"type": "LITERAL", "value_type": "NULL", "value": None}
    whens: list[tuple[str, str]] = []
    idx = first_when_idx
    n_markers = len(markers)
    while idx < n_markers:
        pos, kind = markers[idx]
        if kind == "WHEN" and idx + 1 < n_markers and markers[idx + 1][1] == "THEN":
            cond_end = markers[idx + 1][0]
            then_start = markers[idx + 1][0] + 4
            then_end = markers[idx + 2][0] if idx + 2 < n_markers else len(body)
            cond_sql = body[pos + 4 : cond_end].strip()
            then_sql = body[then_start:then_end].strip()
            whens.append((cond_sql, then_sql))
            idx += 2
            continue
        if kind == "ELSE":
            else_start = pos + 4
            else_end = markers[idx + 1][0] if idx + 1 < n_markers else len(body)
            else_node = parse_sql_expression_to_ast(
                body[else_start:else_end].strip(),
                default_entity=default_entity,
                target_column=target_column,
            )
            idx += 1
            continue
        idx += 1  # stray THEN with no preceding WHEN

    if not whens:
        return None

    ast = else_node
    for cond_sql, then_sql in reversed(whens):
        if operand_sql:
            cond_sql = f"({operand_sql}) = ({cond_sql})"
        ast = {
            "type": "IF_THEN_ELSE",
            "condition": parse_sql_expression_to_ast(
                cond_sql,
                default_entity=default_entity,
                target_column=target_column,
                as_condition=True,
            ),
            "then_branch": parse_sql_expression_to_ast(
                then_sql,
                default_entity=default_entity,
                target_column=target_column,
            ),
            "else_branch": ast,
        }
    return ast


def _call_llm_for_ast(
    llm_client: Any,
    mutations: list[MutationPass],
    target_entity: str,
    target_column: str,
) -> str:
    payload = {
        "target_entity": target_entity,
        "target_column": target_column,
        "mutations": [m.as_dict() for m in mutations],
    }
    user = (
        f"Target: {target_entity}.{target_column}\n\n"
        f"Mutations (chronological JSON):\n{json.dumps(payload, indent=2)}\n\n"
        "Return the JSON AST only."
    )
    if hasattr(llm_client, "generate_ast_json"):
        return str(llm_client.generate_ast_json(target_entity, target_column, payload))
    if hasattr(llm_client, "_complete"):
        return str(
            llm_client._complete(
                _AST_SYSTEM_PROMPT,
                user,
                max_tokens=getattr(llm_client, "max_new_tokens", 2048),
                stage="ast_generation",
            )
        )
    raise RuntimeError("llm_client cannot generate AST JSON")


def _parse_json_ast(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract outermost object.
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _validate_ast_shape(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    node_type = node.get("type")
    if node_type not in _ALLOWED_TYPES:
        return False
    if node_type == "IF_THEN_ELSE":
        return all(
            _validate_ast_shape(node.get(k))
            for k in ("condition", "then_branch", "else_branch")
        )
    if node_type == "BINARY_OP":
        return _validate_ast_shape(node.get("left")) and _validate_ast_shape(node.get("right"))
    if node_type == "FUNCTION_CALL":
        args = node.get("arguments")
        return isinstance(args, list) and all(_validate_ast_shape(a) for a in args)
    if node_type == "MEMBERSHIP_OP":
        return _validate_ast_shape(node.get("column")) and isinstance(node.get("values"), list)
    if node_type == "COLUMN_REF":
        return bool(node.get("entity")) and bool(node.get("column"))
    if node_type == "VARIABLE_REF":
        return bool(node.get("name") or node.get("variable"))
    if node_type == "LITERAL":
        return node.get("value_type") in {"STRING", "NUMBER", "NULL", "FLOAT", "INT", "DECIMAL"}
    return False


def _column_ref(
    entity: str,
    column: str,
    relationship: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "COLUMN_REF",
        "entity": entity,
        "relationship": relationship,
        "column": column,
    }


def _literal_from_sql_token(token: str) -> dict[str, Any]:
    text = token.strip()
    if text.upper() in {"NULL", "NONE"}:
        return {"type": "LITERAL", "value_type": "NULL", "value": None}
    if re.match(r"^@[A-Za-z_][A-Za-z0-9_]*$", text):
        return {"type": "VARIABLE_REF", "name": text}
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith("N'") and text.endswith("'")
    ):
        inner = text[2:-1] if text.upper().startswith("N'") else text[1:-1]
        inner = inner.replace("''", "'")
        return {"type": "LITERAL", "value_type": "STRING", "value": inner}
    if (text.startswith('"') and text.endswith('"')):
        return {"type": "LITERAL", "value_type": "STRING", "value": text[1:-1]}
    if re.match(r"^-?\d+(\.\d+)?$", text):
        number: Any = float(text) if "." in text else int(text)
        return {"type": "LITERAL", "value_type": "NUMBER", "value": number}
    # Bare word → string literal (e.g. STANDARD without quotes in some dialects)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text):
        return {"type": "LITERAL", "value_type": "STRING", "value": text}
    return {"type": "LITERAL", "value_type": "STRING", "value": text}


def _try_map_like_to_membership(
    lhs_sql: str,
    pattern_sql: str,
    *,
    negate: bool,
    default_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """Map a LIKE/NOT LIKE with a fixed-literal pattern onto MEMBERSHIP_OP.

    The 4X grammar's only native pattern-matching is MEMBERSHIP_OP
    CONTAINS/BEGINSWITH/ENDSWITH/DOESNOTCONTAINS, which take a literal
    value — not an arbitrary expression. This only fires when the pattern
    resolves to a plain string literal (optionally wrapped in ``%``
    wildcards); a dynamic pattern (e.g. concatenated with a column) has no
    valid 4X representation and returns None so the caller falls back to
    the honest-failure BINARY_OP shape instead.
    """
    pattern_node = parse_sql_expression_to_ast(
        pattern_sql, default_entity=default_entity, target_column=target_column
    )
    if pattern_node.get("type") != "LITERAL" or pattern_node.get("value_type") != "STRING":
        return None
    raw = str(pattern_node.get("value") or "")
    starts = raw.startswith("%")
    ends = raw.endswith("%")
    needle = raw
    if starts:
        needle = needle[1:]
    if ends and needle:
        needle = needle[:-1]
    lhs_node = parse_sql_expression_to_ast(
        lhs_sql, default_entity=default_entity, target_column=target_column
    )

    if not starts and not ends:
        # No wildcard at all — LIKE degenerates to exact equality.
        op = "!=" if negate else "=="
        return {
            "type": "BINARY_OP",
            "operator": op,
            "left": lhs_node,
            "right": {"type": "LITERAL", "value_type": "STRING", "value": needle},
        }

    if starts and ends:
        operator = "DOESNOTCONTAINS" if negate else "CONTAINS"
    elif ends:  # 'needle%' — starts-with
        if negate:
            return None  # no native "NOT BEGINSWITH" token in the grammar
        operator = "BEGINSWITH"
    else:  # '%needle' — ends-with
        if negate:
            return None  # no native "NOT ENDSWITH" token in the grammar
        operator = "ENDSWITH"

    return {
        "type": "MEMBERSHIP_OP",
        "operator": operator,
        "column": lhs_node,
        "values": [needle],
    }


def _split_top_level_keyword(text: str, keyword_pattern: str) -> list[str]:
    """Split ``text`` on the first top-level (paren/string-depth 0) keyword.

    ``keyword_pattern`` is a regex fragment (word-boundary wrapped, e.g.
    ``r"NOT\\s+LIKE"``), matched case-insensitively. Returns ``[text]``
    unmatched, or ``[before, after]`` on the first depth-0 match.
    """
    pattern = re.compile(rf"(?is)\b(?:{keyword_pattern})\b")
    depth = 0
    in_single = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_single:
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0:
            m = pattern.match(text, i)
            if m:
                return [text[:i], text[m.end():]]
        i += 1
    return [text]


def _split_top_level(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_single = False
    i = 0
    sep = separator
    upper = text
    # Case-insensitive for AND/OR
    sep_upper = sep.upper()
    text_upper = text.upper()
    while i < len(text):
        ch = text[i]
        if ch == "'" and not in_single:
            in_single = True
            buf.append(ch)
            i += 1
            continue
        if ch == "'" and in_single:
            in_single = False
            buf.append(ch)
            i += 1
            continue
        if in_single:
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
            continue
        if depth == 0 and text_upper.startswith(sep_upper, i):
            # Ensure separator is at token boundary for AND/OR
            if sep_upper.strip() in {"AND", "OR"}:
                before_ok = i == 0 or not text[i - 1].isalnum()
                after_idx = i + len(sep_upper)
                after_ok = after_idx >= len(text) or not text[after_idx].isalnum()
                if not (before_ok and after_ok):
                    buf.append(ch)
                    i += 1
                    continue
            parts.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()] if len(parts) > 1 else [text]


def _balanced(text: str) -> bool:
    depth = 0
    in_single = False
    for ch in text:
        if ch == "'" and not in_single:
            in_single = True
            continue
        if ch == "'" and in_single:
            in_single = False
            continue
        if in_single:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
