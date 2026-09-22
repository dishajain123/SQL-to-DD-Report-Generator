"""Derivation Engine v2 — deterministic 4-phase AST pipeline.

Phases:
  1. Temp-table lineage resolution (`phase1_lineage`)
  2. Chronological column mutation folding (`phase2_mutation_folder`)
  3. Structured 4X JSON AST generation (`phase3_ast_generator`)
  4. Platform metadata + DDRow export (`phase4_metadata`)

Compiled AST strings are validated against `app.grammar.fourx_grammar.lark`
via `app.grammar.validator.validate_expression`.
"""
from __future__ import annotations

from app.derivation.v2.pipeline import generate_dd_rows, generate_dd_rows_for_chains

__all__ = [
    "generate_dd_rows",
    "generate_dd_rows_for_chains",
]
