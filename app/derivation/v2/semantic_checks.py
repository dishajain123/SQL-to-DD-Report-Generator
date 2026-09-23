"""Source-based restrictions that an LLM/grammar pass cannot waive.

Grammar-valid formulas ship as GENERATED/ACTIVE — there is no NEEDS_REVIEW
path for simplified SQL constructs (joins, subqueries, MERGE, CAST, ...)
once phases 1-3 have folded them into a complete, valid 4x expression.
The one case a grammar-valid formula cannot self-detect is an empty
derivation: no source assignment matched the target at all, so the
formula falls back to an identity self-reference that is syntactically
valid but not a real derivation.
"""
from __future__ import annotations


def mutation_semantic_errors(mutations, source_sql: str = "") -> list[str]:
    del source_sql
    if not mutations:
        return ["No source assignment was extracted for this target; derivation is incomplete"]
    return []
