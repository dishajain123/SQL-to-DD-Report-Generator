"""Deterministic, whitespace-only reformatting of an accepted 4X Formula
Expression into an indented IF / ELSE IF / ELSE decision-chain layout for
the generated report.

This is NOT a paraphrase and NOT a second LLM call: it parses the
already-validated expression against the same real grammar
(app/grammar/fourx_grammar.lark) and reconstructs the layout purely by
slicing the *original* expression text at the parser's own tracked source
positions (Lark's propagate_positions) -- the same technique
app/derivation/period_pruning.py already uses, and for the same reason.
Since the grammar itself ignores all whitespace (%ignore WS), inserting
newlines and indentation between tokens can never change what the
expression means, and reusing the original text for every condition/value
(rather than re-serializing tokens from the tree) guarantees the
pretty-printed version can never drift from the exact formula stored on
the DD row.

Limitation, by design: only a nested IF sitting in a THEN/ELSE *branch*
gets its own indented sub-chain. A nested IF used as an operand inside a
condition (e.g. `(IF(...)THEN(...)ELSE(...)) > "table"."var"."DATE"`) is
left as flattened text within that condition line -- this pattern is rare
in generated output and still displays correctly, just without its own
sub-indentation. Extending to that case would add real complexity for a
shape that essentially doesn't occur in practice.
"""
from __future__ import annotations

from pathlib import Path

from lark import Lark, Token, Tree
from lark.exceptions import LarkError

_GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammar" / "fourx_grammar.lark"

# A separate parser instance from app.grammar.validator's, for the same
# reason app/derivation/period_pruning.py keeps its own: this needs
# source-position tracking (propagate_positions) to slice the original
# expression text, which the shared validator parser does not enable, and
# there is no reason for validator.py's parser (used on every generated
# expression) to carry that extra overhead.
_pretty_parser = Lark(
    _GRAMMAR_PATH.read_text(), parser="earley", start="start", propagate_positions=True
)

_INDENT = "    "
_ARROW = "\u2192"  # →


def pretty_print_expression(expression: str) -> str | None:
    """Return an indented IF / ELSE IF / ELSE decision-chain rendering of
    `expression`, or None if it doesn't parse -- callers should fall back
    to showing only the raw formula in that case. This is a display aid,
    never a correctness gate: it never changes, validates, or repairs the
    expression, only reformats it for readability."""
    try:
        tree = _pretty_parser.parse(expression)
    except LarkError:
        return None

    try:
        lines = _render(tree, expression, depth=0)
    except Exception:
        return None
    return "\n".join(lines)


def _span_text(node, expression: str) -> str:
    if isinstance(node, Token):
        return str(node)
    meta = node.meta
    if meta.empty:
        return expression
    return " ".join(expression[meta.start_pos : meta.end_pos].split())


def _unwrap(node):
    """Descend through wrapper nodes that always keep their own Tree even
    with exactly one child -- e.g. the grammar's `membership` rule is not
    `?`-prefixed, so `arith` (and anything nested inside it, including a
    nested if_expr) stays wrapped in a `membership` node even when no
    membership operator was actually used. Without unwrapping this, a
    nested IF sitting inside a THEN/ELSE branch would never be recognized
    as `if_expr` and would incorrectly be flattened to plain text instead
    of getting its own indented sub-chain. Stops at `if_expr` and
    `column_ref`, which are never mere pass-through wrappers."""
    while (
        isinstance(node, Tree)
        and node.data not in ("if_expr", "column_ref")
        and len(node.children) == 1
    ):
        node = node.children[0]
    return node


def _render(node, expression: str, depth: int) -> list[str]:
    """Render `node` starting at indentation `depth`. Only descends into
    if_expr chains -- any other node (a plain function call, a bare
    column reference, an arithmetic expression with no IF at all) is
    rendered as a single line using its exact original text, since there
    is no decision chain to lay out."""
    node = _unwrap(node)
    if isinstance(node, Tree) and node.data == "if_expr":
        return _render_if_chain(node, expression, depth)
    return [_indent(depth) + _span_text(node, expression)]


def _render_if_chain(node, expression: str, depth: int) -> list[str]:
    children = list(node.children)
    condition, then_branch = children[0], children[1]
    rest = children[2:]
    elseif_clauses = [c for c in rest if isinstance(c, Tree) and c.data == "elseif_clause"]
    else_clause = next((c for c in rest if isinstance(c, Tree) and c.data == "else_clause"), None)

    pad = _indent(depth)
    lines: list[str] = [f"{pad}IF ({_span_text(condition, expression)})"]
    lines.append(f"{pad}{_INDENT}{_ARROW} " + _render_branch_inline(then_branch, expression, depth + 1))

    for clause in elseif_clauses:
        clause_condition, clause_branch = clause.children[0], clause.children[1]
        lines.append(f"{pad}ELSE IF ({_span_text(clause_condition, expression)})")
        lines.append(f"{pad}{_INDENT}{_ARROW} " + _render_branch_inline(clause_branch, expression, depth + 1))

    if else_clause is not None:
        branch = else_clause.children[0]
        lines.append(f"{pad}ELSE")
        lines.append(f"{pad}{_INDENT}{_ARROW} " + _render_branch_inline(branch, expression, depth + 1))

    return lines


def _render_branch_inline(branch, expression: str, depth: int) -> str:
    """A branch's value is usually a single value (a column reference, a
    literal, a function call, an arithmetic expression) and reads best on
    one line next to its arrow. Only when the branch is itself a nested
    IF -- a genuinely nested decision, not just a single value -- does it
    get its own indented sub-chain instead."""
    branch = _unwrap(branch)
    if isinstance(branch, Tree) and branch.data == "if_expr":
        nested = _render_if_chain(branch, expression, depth)
        first, *rest = nested
        rendered = first.lstrip()
        if rest:
            rendered += "\n" + "\n".join(rest)
        return rendered
    return _span_text(branch, expression)


def _indent(depth: int) -> str:
    return _INDENT * depth