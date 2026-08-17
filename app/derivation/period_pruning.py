"""Architecture step 13e: Period-Aware Formula Pruning.

A column's generated Formula Expression can legitimately branch on a
rule-versioning parameter (a TIMEKEY-style threshold check, e.g.
`IF(p_TIMEKEY > 26267)THEN(...)ELSE(...)`), because the source SQL itself
branches that way. But the DD row model already represents "this rule
changed on this date" through the Effective Start Date mechanism (see
app/derivation/versioning.py) -- one DD row per distinct threshold, each
describing the derivation valid from that date forward. Embedding the
FULL threshold-branching tree identically into every one of those rows is
redundant: the row effective from a later date should show only the
branch that actually applies from that date onward, not the whole tree
repeated.

This module prunes that redundancy after generation: given a validated
Formula Expression and a concrete representative TIMEKEY value for one
specific effective-dated row, it walks the real parsed structure (not a
regex) and replaces every IF/ELSEIF whose condition is a simple
comparison against the known rule-versioning parameter with whichever
branch that comparison actually selects at that value -- and leaves
every other condition (Aqua Scheme, override, exception paths, anything
not about the versioning parameter) completely untouched, since those
apply the same way regardless of period and pruning them would risk
silently changing real business logic.

If anything about the expression's structure isn't confidently prunable
(a threshold check mixed into a compound the pruner can't safely isolate,
a parse failure, etc.), the original expression is returned unchanged --
pruning is a display/precision improvement, never a correctness risk.
Reconstructing kept branches by slicing the *original* expression text
(via Lark's propagated source positions) rather than re-serializing the
tree from scratch means this can never drift out of sync with the real
grammar's exact formatting rules.
"""
from __future__ import annotations

from pathlib import Path

from lark import Lark, Token, Tree
from lark.exceptions import LarkError

_GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammar" / "fourx_grammar.lark"

# A separate parser instance from app.grammar.validator's -- pruning needs
# source-position tracking (propagate_positions) to slice the original
# expression text for unpruned branches, which the shared validator parser
# does not enable, and there is no reason for validator.py's parser (used
# on every generated expression, including ones this module never touches)
# to carry that extra overhead.
_pruning_parser = Lark(
    _GRAMMAR_PATH.read_text(), parser="earley", start="start", propagate_positions=True
)

_COMPARISON_OPERATORS = {">", ">=", "<", "<=", "==", "!="}
_FLIPPED_OPERATORS = {">": "<", "<": ">", ">=": "<=", "<=": ">="}


def prune_expression_for_period(expression: str, variable: str, representative_value: int) -> str:
    """Return `expression` with every IF/ELSEIF whose condition tests
    `variable` against a numeric threshold replaced by whichever of its
    branches applies when `variable == representative_value`.

    Returns the original expression unchanged if it doesn't parse, or if
    nothing prunable is found -- callers should treat the result as a
    best-effort simplification, not assume it always differs from the
    input, and should re-validate the result against the grammar before
    trusting it (this module never assumes it produced something correct;
    it is exercised directly against the real grammar in tests, but a
    caller storing the result should still confirm it independently).
    """
    try:
        tree = _pruning_parser.parse(expression)
    except LarkError:
        return expression

    try:
        pruned = _prune_node(tree, expression, variable, representative_value)
    except Exception:
        return expression

    return pruned if pruned is not None else expression


def _span_text(node, expression: str) -> str:
    """The exact original source text a Tree/Token spans."""
    if isinstance(node, Token):
        return str(node)
    meta = node.meta
    if meta.empty:
        return expression
    return expression[meta.start_pos : meta.end_pos]


def _prune_node(node, expression: str, variable: str, value: int) -> str | None:
    """Recursively prune `node`. Returns the pruned text for this node's
    span, or None if nothing changed anywhere inside it (the caller then
    just uses the node's original span text unchanged)."""
    if isinstance(node, Token):
        return None

    if node.data == "if_expr":
        selected = _try_prune_if_expr(node, expression, variable, value)
        if selected is not None:
            return selected
        # Not directly prunable as a whole (e.g. a condition mixes the
        # versioning variable with something else, or isn't about it at
        # all) -- fall through to recursing into this node's own children
        # below, since a nested if_expr inside one of its branches may
        # still be independently prunable.

    meta = node.meta
    if meta.empty:
        return None

    changed = False
    replacements: list[tuple[int, int, str]] = []
    base = meta.start_pos
    for child in node.children:
        if isinstance(child, Token):
            continue
        child_pruned = _prune_node(child, expression, variable, value)
        if child_pruned is not None:
            changed = True
            cmeta = child.meta
            if not cmeta.empty:
                replacements.append((cmeta.start_pos - base, cmeta.end_pos - base, child_pruned))

    if not changed:
        return None

    text = expression[meta.start_pos : meta.end_pos]
    for start, end, replacement in sorted(replacements, key=lambda r: r[0], reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _try_prune_if_expr(node, expression: str, variable: str, value: int) -> str | None:
    """If every condition in this if_expr's IF/ELSEIF chain is a simple
    threshold comparison against `variable`, evaluate the chain at `value`
    and return the exact text of whichever branch wins (recursively
    pruned in turn). Returns None if any condition in the chain can't be
    confidently classified this way, or if no branch matches and there is
    no ELSE (both left conservatively untouched)."""
    children = list(node.children)
    condition, then_branch = children[0], children[1]
    rest = children[2:]

    elseif_clauses = [c for c in rest if isinstance(c, Tree) and c.data == "elseif_clause"]
    else_clause = next((c for c in rest if isinstance(c, Tree) and c.data == "else_clause"), None)

    branches: list[tuple[object, object]] = [(condition, then_branch)]
    for clause in elseif_clauses:
        branches.append((clause.children[0], clause.children[1]))

    thresholds = [_condition_threshold(cond, variable) for cond, _ in branches]
    if not thresholds or any(t is None for t in thresholds):
        return None

    for (_, branch), threshold_info in zip(branches, thresholds):
        operator, threshold = threshold_info
        if _evaluate(operator, value, threshold):
            pruned = _prune_node(branch, expression, variable, value)
            return pruned if pruned is not None else _span_text(branch, expression)

    if else_clause is not None:
        branch = else_clause.children[0]
        pruned = _prune_node(branch, expression, variable, value)
        return pruned if pruned is not None else _span_text(branch, expression)

    return None


def _find_compare_node(node):
    """Unwrap single-child grammar-inlined wrapper nodes down to a
    `compare` node, if the condition is (or reduces to) exactly one
    comparison. Returns None for anything else -- an AND/OR combination,
    a bare function call with no operator, etc. -- since those are left
    alone rather than guessed at."""
    while isinstance(node, Tree) and node.data != "compare" and len(node.children) == 1:
        node = node.children[0]
    if isinstance(node, Tree) and node.data == "compare":
        return node
    return None


def _condition_threshold(condition_node, variable: str) -> tuple[str, int] | None:
    """If `condition_node` is exactly `<variable> <op> <number>` (in
    either operand order), return (operator, threshold) with the operator
    already normalized to the `<variable> <op> <number>` orientation.
    Otherwise None."""
    compare_node = _find_compare_node(condition_node)
    if compare_node is None or len(compare_node.children) != 3:
        return None
    left, op_token, right = compare_node.children
    operator = str(op_token)
    if operator not in _COMPARISON_OPERATORS:
        return None

    left_name = _as_variable_name(left)
    right_name = _as_variable_name(right)
    left_num = _as_number(left)
    right_num = _as_number(right)

    if left_name and left_name.upper() == variable.upper() and right_num is not None:
        return operator, right_num
    if right_name and right_name.upper() == variable.upper() and left_num is not None:
        return _FLIPPED_OPERATORS.get(operator, operator), left_num
    return None


def _unwrap_to_leaf_or_column_ref(node):
    """Walk down through single-child grammar-inlined wrapper nodes,
    stopping at either a `column_ref` node (which always remains its own
    node regardless of child count -- it is not a `?`-prefixed rule) or a
    raw token."""
    while isinstance(node, Tree) and node.data != "column_ref" and len(node.children) == 1:
        node = node.children[0]
    return node


def _node_text(node) -> str | None:
    """Return the normalized text for a token/tree leaf we can compare."""
    node = _unwrap_to_leaf_or_column_ref(node)
    if isinstance(node, Token):
        return str(node).strip('"')
    if isinstance(node, Tree) and node.data in {"column_ref", "path_part"}:
        parts: list[str] = []
        for child in node.children:
            text = _node_text(child)
            if text is None:
                return None
            parts.append(text)
        return ".".join(parts)
    return None


def _as_variable_name(node) -> str | None:
    node = _unwrap_to_leaf_or_column_ref(node)
    if isinstance(node, Tree) and node.data == "column_ref":
        text = _node_text(node)
        if text is None:
            return None
        segments = [segment for segment in text.split(".") if segment]
        return segments[-1] if segments else None
    if isinstance(node, Token) and node.type == "NAME":
        return str(node)
    return None


def _as_number(node) -> int | None:
    node = _unwrap_to_leaf_or_column_ref(node)
    if isinstance(node, Token) and node.type == "NUMBER":
        try:
            return int(float(str(node)))
        except ValueError:
            return None
    return None


def _evaluate(operator: str, value: int, threshold: int) -> bool:
    return {
        ">": value > threshold,
        ">=": value >= threshold,
        "<": value < threshold,
        "<=": value <= threshold,
        "==": value == threshold,
        "!=": value != threshold,
    }.get(operator, False)
