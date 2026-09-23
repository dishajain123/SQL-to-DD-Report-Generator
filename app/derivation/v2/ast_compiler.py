"""Deterministic JSON AST → 4X Formula Expression compiler.

No LLM involvement. Output is shaped for `app.grammar.fourx_grammar.lark`:

    IF(cond)THEN(expr)ELSEIF(cond2)THEN(expr2)ELSE(default)

Nested ``IF_THEN_ELSE`` else-branches are flattened into ``ELSEIF`` clauses
(not string-sliced ``IF`` prefixes).
"""
from __future__ import annotations

import re
from typing import Any


# NOTE: "LIKE"/"NOT LIKE" are NOT valid tokens in fourx_grammar.lark (it has
# no LIKE operator at all — pattern matching is only native via
# MEMBERSHIP_OP CONTAINS/BEGINSWITH/ENDSWITH/DOESNOTCONTAINS, which phase3
# maps a fixed-literal LIKE pattern onto directly). They're listed here only
# because compile_ast_to_4x_string still renders them as an honest-failure
# fallback for a LIKE whose pattern is genuinely dynamic (e.g. concatenated
# with a column) and therefore has no valid 4X representation at all — the
# resulting string deliberately fails grammar validation so it surfaces as
# a review flag instead of silently compiling to something wrong.
ALLOWED_BINARY_OPS = {"==", "!=", ">", ">=", "<", "<=", "AND", "OR", "+", "-", "*", "/", "LIKE", "NOT LIKE"}
# Every MEMBERSHIP_OP token the grammar (fourx_grammar.lark) actually
# defines — not just IN/NOTIN. CONTAINS/BEGINSWITH/ENDSWITH/
# DOESNOTCONTAINS are how the platform expresses LIKE-style pattern
# matching; HRCHYIN/HRCHYNOTIN are hierarchy membership variants.
ALLOWED_MEMBERSHIP_OPS = {
    "IN",
    "NOTIN",
    "HRCHYIN",
    "HRCHYNOTIN",
    "CONTAINS",
    "BEGINSWITH",
    "ENDSWITH",
    "DOESNOTCONTAINS",
}
_MEMBERSHIP_OP_TOKENS = ALLOWED_MEMBERSHIP_OPS
ALLOWED_FUNCTIONS = {
    "ISEMPTY",
    "ISNOTEMPTY",
    "COALESCE",
    "SOM",
    "EOM",
    "ADDDAY",
    "MAX",
    "MIN",
    "DATEDIFF",
    "TODATE",
    "CONCAT",
    "ABS",
    "ROUND",
    "SUM",
    "COUNT",
}


def compile_ast_to_4x_string(node: dict[str, Any] | None) -> str:
    """Compile a structured AST node into a 4X DSL string."""
    if node is None:
        return "NULL"

    if not isinstance(node, dict):
        raise ValueError(f"AST node must be a dict, got {type(node).__name__}")

    node_type = node.get("type")
    if not node_type:
        raise ValueError(f"AST node missing 'type': {node!r}")

    if node_type == "IF_THEN_ELSE":
        return _compile_if_then_else(node)

    if node_type == "BINARY_OP":
        operator = str(node.get("operator") or "").strip().upper()
        # Comparison ops stay as written; normalize SQL '=' leftovers.
        raw_op = str(node.get("operator") or "").strip()
        if raw_op == "=":
            operator = "=="
        elif raw_op == "<>":
            operator = "!="
        elif raw_op in {"==", "!=", ">", ">=", "<", "<=", "+", "-", "*", "/"}:
            operator = raw_op
        elif operator not in {"AND", "OR"}:
            operator = raw_op

        # Malformed producer output: ISEMPTY/ISNOTEMPTY used as an infix
        # comparison operator (``[Expr] ISEMPTY NULL``) instead of the unary
        # FUNCTION_CALL form. Recover the intended functional syntax around
        # whichever operand is the real expression (not a NULL literal).
        if operator in {"ISEMPTY", "ISNOTEMPTY"}:
            left_node = node.get("left") or {}
            right_node = node.get("right") or {}
            operand = left_node if not _is_null_literal(left_node) else right_node
            return f"{operator}({compile_ast_to_4x_string(operand)})"

        # NULL comparisons (``col >= NULL`` / ``col == NULL``) are not valid
        # 4X grammar — the platform has no NULL literal on the RHS/LHS of a
        # comparison operator. Normalize to ISEMPTY / ISNOTEMPTY on whichever
        # side is the non-NULL operand.
        if operator in {"==", "!=", ">", ">=", "<", "<="}:
            left_node = node.get("left") or {}
            right_node = node.get("right") or {}
            left_is_null = _is_null_literal(left_node)
            right_is_null = _is_null_literal(right_node)
            if left_is_null or right_is_null:
                operand = right_node if left_is_null else left_node
                func = "ISEMPTY" if operator in {"==", "<=", ">="} else "ISNOTEMPTY"
                return f"{func}({compile_ast_to_4x_string(operand)})"

        left = compile_ast_to_4x_string(node["left"])
        right = compile_ast_to_4x_string(node["right"])
        # Preserve the AST's grouping; dropping parentheses changes both
        # arithmetic and mixed AND/OR expressions on the target platform.
        if node["left"].get("type") == "BINARY_OP":
            left = f"({left})"
        if node["right"].get("type") == "BINARY_OP":
            right = f"({right})"
        return f"{left} {operator} {right}"

    if node_type == "FUNCTION_CALL":
        func = str(node.get("function_name") or "").strip().upper()
        args = node.get("arguments") or []
        if func == "__UNSUPPORTED_SQL__":
            raise ValueError(node.get("_validation_error") or "Unsupported SQL expression")
        if func == "__VALUE_PREDICATE_MIXING__":
            # Sentinel from phase3's value/predicate-mixing guard — raise here
            # so the pipeline's existing compile-error handling records a
            # validation error instead of the caller silently dropping the row.
            raise ValueError(node.get("_validation_error") or "value/predicate mixing detected")
        if func == "__UNRESOLVED_SUBQUERY_PREDICATE__":
            # Sentinel from phase3's EXISTS/IN-subquery fallback — raise here
            # instead of ever compiling an always-true "1 == 1" guard, which
            # would silently broaden eligibility to every row.
            raise ValueError(
                node.get("_validation_error")
                or "EXISTS/IN subquery could not be projected into a row-level predicate"
            )
        if func == "ADDDAY" and len(args) != 2:
            # Fail loud instead of emitting invalid grammar (ADDDAY(x) with a
            # dropped offset argument, or extra args the grammar rejects).
            raise ValueError(
                f"ADDDAY requires exactly 2 arguments (date, offset), got {len(args)}: {node!r}"
            )
        rendered = ", ".join(compile_ast_to_4x_string(a) for a in args)
        return f"{func}({rendered})"

    if node_type == "MEMBERSHIP_OP":
        operator = str(node.get("operator") or "IN").strip().upper().replace(" ", "")
        # Preserve every MEMBERSHIP_OP token the grammar actually defines —
        # this previously collapsed ANY operator that wasn't exactly IN/
        # NOTIN-shaped back to bare "IN", silently discarding CONTAINS/
        # BEGINSWITH/ENDSWITH/DOESNOTCONTAINS/HRCHYIN/HRCHYNOTIN even though
        # the platform grammar supports them.
        if operator not in _MEMBERSHIP_OP_TOKENS:
            operator = "NOTIN" if "NOT" in operator and "IN" in operator else "IN"
        col = compile_ast_to_4x_string(node["column"])
        vals = ", ".join(_compile_membership_value(v) for v in (node.get("values") or []))
        return f"{col} {operator} [{vals}]"

    if node_type == "COLUMN_REF":
        entity = _clean_entity_qualifier(str(node.get("entity") or "").strip())
        column = str(node.get("column") or "").strip()
        relationship = node.get("relationship")
        # Local T-SQL scalar variables (@GraceWindowStart, @ProcessDate) must
        # never be emitted as "Entity"."@Var" — grammar accepts them as STRING.
        if _is_at_variable(column) or _is_at_variable(entity):
            return _compile_at_variable(column if _is_at_variable(column) else entity)
        if not entity or not column:
            raise ValueError(f"COLUMN_REF requires entity and column: {node!r}")
        # Guard: decimal literals mis-tagged as COLUMN_REF (e.g. entity="1",
        # column="10" from parsing "1.10") must compile as raw numbers.
        if not relationship and _looks_like_numeric_decimal_parts(entity, column):
            return f"{entity}.{column}"
        if relationship:
            relationship = _clean_entity_qualifier(str(relationship))
            if _is_at_variable(relationship):
                return _compile_at_variable(relationship)
            if relationship.upper() == entity.upper():
                # Redundant self-relationship (e.g. a join path that folded
                # back onto its own entity) collapses to a plain column ref.
                return f'"{entity}"."{column}"'
            return f'"{entity}"."{relationship}"."{column}"'
        return f'"{entity}"."{column}"'

    if node_type == "VARIABLE_REF":
        name = str(node.get("name") or node.get("variable") or "").strip()
        if not name:
            raise ValueError(f"VARIABLE_REF requires name: {node!r}")
        return _compile_at_variable(name)

    if node_type == "LITERAL":
        return _compile_literal(node)

    raise ValueError(f"Unknown AST node type: {node_type}")


_NUMERIC_VALUE_TYPES = frozenset({"NUMBER", "FLOAT", "INT", "INTEGER", "DECIMAL", "DOUBLE"})


def _is_at_variable(text: str) -> bool:
    token = (text or "").strip().strip('"')
    return bool(re.match(r"^@[A-Za-z_][A-Za-z0-9_]*$", token))


def _compile_at_variable(name: str) -> str:
    """Emit a T-SQL scalar as a quoted 4X string token (Lark has no bare ``@``)."""
    text = (name or "").strip().strip('"')
    text = text.split(".")[-1]
    if not text.startswith("@"):
        text = f"@{text.lstrip('@')}"
    return f'"{text}"'


def _compile_literal(node: dict[str, Any]) -> str:
    """Render LITERAL nodes: numbers raw, strings double-quoted, NULL bare."""
    value_type = str(node.get("value_type") or "").strip().upper()
    value = node.get("value")

    if value_type == "NULL" or (value is None and value_type != "STRING"):
        return "NULL"

    # Explicit numeric types, or Python numeric values (incl. numeric strings
    # tagged as NUMBER/FLOAT/INT/DECIMAL).
    if value_type in _NUMERIC_VALUE_TYPES or (
        value_type not in {"STRING", "NULL"} and _is_numeric_value(value)
    ):
        if value is None:
            return "NULL"
        return _format_numeric_literal(value)

    # STRING that is actually an @variable → compile as variable token.
    if isinstance(value, str) and _is_at_variable(value.strip()):
        return _compile_at_variable(value.strip())

    # STRING — always double-quoted (even if the text looks numeric).
    if value_type == "STRING":
        if value is None:
            return '""'
        text = str(value).replace('"', '\\"')
        return f'"{text}"'

    # Untyped / unknown value_type: prefer raw number when payload is numeric.
    if not value_type or value_type not in _NUMERIC_VALUE_TYPES:
        if isinstance(value, str) and _is_at_variable(value.strip()):
            return _compile_at_variable(value.strip())
        if _is_numeric_value(value):
            return _format_numeric_literal(value)
        text = str(value).replace('"', '\\"')
        return f'"{text}"'

    text = str(value).replace('"', '\\"')
    return f'"{text}"'


def _is_numeric_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if text.startswith("-"):
        text = text[1:]
    if not text:
        return False
    # Allow one decimal point: 10, 1.10, 05 (leading zeros ok for column-misparse recovery)
    if text.count(".") > 1:
        return False
    return text.replace(".", "", 1).isdigit()


def _format_numeric_literal(value: Any) -> str:
    """Preserve decimal spelling for string numerics (``\"1.10\"`` → ``1.10``)."""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        # Avoid scientific notation for typical provision rates.
        text = format(value, "f").rstrip("0").rstrip(".")
        return text if text else "0"
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


_KNOWN_SCHEMA_PREFIXES = {"PRO", "DBO", "SYS", "TEMPDB"}


def _clean_entity_qualifier(text: str) -> str:
    """Strip redundant/duplicated schema and entity qualifiers.

    Handles both shapes seen from upstream resolution bugs:
      * a leading SQL schema prefix (``PRO.DpdBucketHistory`` -> ``DpdBucketHistory``)
      * an immediately duplicated segment (``DpdBucketHistory.DpdBucketHistory``
        or ``##Foo.##Foo``) collapsing to a single occurrence.
    Never touches a single bare identifier with no ``.``.
    """
    raw = (text or "").strip()
    if not raw or "." not in raw:
        return raw

    prefix = ""
    if raw.startswith("##"):
        prefix = "##"
        raw = raw[2:]
    elif raw.startswith("#"):
        prefix = "#"
        raw = raw[1:]

    parts = [p for p in raw.split(".") if p]
    if not parts:
        return f"{prefix}{raw}"

    # Drop a leading known SQL schema token (PRO / dbo / …).
    if len(parts) > 1 and parts[0].upper() in _KNOWN_SCHEMA_PREFIXES:
        parts = parts[1:]

    # Collapse immediately-repeated identical segments (case-insensitive).
    deduped: list[str] = []
    for part in parts:
        if deduped and deduped[-1].upper() == part.upper():
            continue
        deduped.append(part)

    return f"{prefix}{'.'.join(deduped) if deduped else raw}"


def _is_null_literal(node: dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("type") != "LITERAL":
        return False
    value_type = str(node.get("value_type") or "").strip().upper()
    return value_type == "NULL" or (node.get("value") is None and value_type != "STRING")


def _looks_like_numeric_decimal_parts(entity: str, column: str) -> bool:
    """True when entity.column is really a dotted decimal (1.10), not a path."""
    return bool(
        entity
        and column
        and entity.isdigit()
        and column.isdigit()
    )

_SELF_EQUALITY_OPS = {"==", "=", "!=", "<>"}


def _repair_value_branch(node: Any) -> Any:
    """Structural safety net for THEN/ELSE scalar-value slots.

    A bare ``==``/``!=`` comparison is never a legitimate 4X scalar value —
    the grammar has no boolean value type at that position. This is a
    context-free backstop (no ``target_column`` is threaded through the
    compiler) that catches the pattern regardless of which AST producer
    built it (deterministic fold, or an LLM response that bypassed phase3's
    own value/predicate guard): in ``col == value`` form the intended value
    is conventionally the right-hand operand, so that side is kept.
    """
    if (
        isinstance(node, dict)
        and node.get("type") == "BINARY_OP"
        and str(node.get("operator") or "").strip() in _SELF_EQUALITY_OPS
    ):
        right = node.get("right")
        if isinstance(right, dict):
            return right
    return node


def _compile_if_then_else(node: dict[str, Any]) -> str:
    """Flatten nested IF_THEN_ELSE else-branches into ELSEIF clauses.

    Produces: ``IF(c0)THEN(t0)ELSEIF(c1)THEN(t1)ELSE(default)``
    """
    clauses: list[tuple[str, str]] = []
    current: dict[str, Any] | None = node
    default_node: dict[str, Any] = {"type": "LITERAL", "value_type": "NULL", "value": None}

    while current is not None and current.get("type") == "IF_THEN_ELSE":
        cond = compile_ast_to_4x_string(current["condition"])
        then_b = compile_ast_to_4x_string(_repair_value_branch(current["then_branch"]))
        clauses.append((cond, then_b))
        else_branch = current.get("else_branch")
        if isinstance(else_branch, dict) and else_branch.get("type") == "IF_THEN_ELSE":
            current = else_branch
            continue
        if isinstance(else_branch, dict):
            default_node = else_branch
        current = None

    if not clauses:
        raise ValueError("IF_THEN_ELSE node produced no clauses")

    first_cond, first_then = clauses[0]
    parts = [f"IF({first_cond})THEN({first_then})"]
    for cond, then_b in clauses[1:]:
        parts.append(f"ELSEIF({cond})THEN({then_b})")
    parts.append(f"ELSE({compile_ast_to_4x_string(_repair_value_branch(default_node))})")
    return "".join(parts)


def _compile_membership_value(value: Any) -> str:
    if isinstance(value, dict):
        # A MEMBERSHIP_OP value can itself be an uncompiled AST node (e.g. a
        # COLUMN_REF from a JSON-authored AST, or an ``IN (col, 'lit')``
        # mixed list) -- recurse through the real compiler instead of
        # falling through to str(value), which would emit the node's raw
        # Python dict repr as a quoted string literal.
        return compile_ast_to_4x_string(value)
    if isinstance(value, bool):
        return f'"{str(value).upper()}"'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text
    if text.replace(".", "", 1).isdigit() or (
        text.startswith("-") and text[1:].replace(".", "", 1).isdigit()
    ):
        return text
    return f'"{text}"'
