"""Bound flat-formula expansion without discarding source execution steps."""
from __future__ import annotations


class FormulaExpansionError(ValueError):
    pass


def check_formula_expansion(root, *, max_nodes=12000, max_depth=160, max_text=200000):
    """Count expanded DAG size in O(unique nodes), before recursive consumers.

    Prior-value substitution shares subtrees. Walking every occurrence can
    take exponential time even while the in-memory DAG is small. Saturating
    counts retain that sharing and detect oversized flat representations.
    """
    sizes = {}
    stack = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if not isinstance(node, (dict, list)) or id(node) in sizes:
            continue
        children = ([v for k, v in node.items() if not str(k).startswith('_')]
                    if isinstance(node, dict) else node)
        if not visited:
            stack.append((node, True))
            stack.extend((child, False) for child in children if isinstance(child, (dict, list)))
            continue
        nodes, depth, chars = 1, 1, 0
        for child in children:
            if isinstance(child, (dict, list)):
                n, d, c = sizes[id(child)]
                nodes += n
                depth = max(depth, d + 1)
                chars += c
            elif isinstance(child, str):
                chars += len(child)
        if nodes > max_nodes or depth > max_depth or chars > max_text:
            raise FormulaExpansionError(
                "Flat formula expansion exceeds the supported size/depth; "
                "ordered execution steps and source SQL are retained. "
                "An ordered platform workflow is required; no partial formula was accepted."
            )
        sizes[id(node)] = nodes, depth, chars
