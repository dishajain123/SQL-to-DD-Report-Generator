"""Phase 3 — structured 4X JSON AST generator.

Deterministic mutation→AST folder only. Formula/condition generation never
calls an LLM, so every AST is reproducible straight from the source SQL.

Allowed node types only:
  IF_THEN_ELSE, BINARY_OP, FUNCTION_CALL, MEMBERSHIP_OP, COLUMN_REF, LITERAL
"""
from __future__ import annotations

from app.derivation.v2.ast_limits import FormulaExpansionError

import re
from typing import Any, Optional

from app.derivation.v2.ast_optimize import (
    enforce_formula_budget,
    enforce_later_update_precedence,
    optimize_expression_ast,
)
from app.derivation.v2.phase2_mutation_folder import (
    MutationPass,
    _ast_signature,
    flatten_and_conjuncts as _flatten_and_conjuncts,
    guard_conjunct_signature,
    prune_redundant_ast,
)
from app.derivation.v2.sql_text import (
    bare_ident,
    extract_subquery_dependency_refs,
    is_ephemeral_sql_alias,
    should_collapse_target_join_hop,
    normalize_table_name,
    split_csv_respecting_parens,
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def _is_single_quoted_string_literal(text: str) -> bool:
    """True only when ``text`` is ENTIRELY one quoted string literal, e.g.
    ``'ACTIVE'`` or ``N'ACTIVE'`` -- not a larger expression that merely
    happens to start and end with a quote character, like the concatenation
    ``'%' + A.Col + '%'`` (its trailing ``'%'`` ends in a quote too, but the
    text as a whole is not a single literal). Honors SQL's ``''`` escaped-
    quote convention when scanning for the real closing quote.
    """
    body = text
    if body[:2].upper() == "N'":
        body = body[1:]
    if len(body) < 2 or body[0] not in ("'", '"'):
        return False
    quote = body[0]
    i = 1
    while i < len(body):
        if body[i] == quote:
            if i + 1 < len(body) and body[i + 1] == quote:
                i += 2
                continue
            return i == len(body) - 1
        i += 1
    return False


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

# T-SQL function name -> documented 4X equivalent
# (samples/platform_docs/4x_functions_operators.md).
_TSQL_TO_4X_FUNCTION_NAMES = {
    "SUBSTRING": "SUBSTR",
    "TRIM": "TRIM",
    "LTRIM": "TRIM",
    "RTRIM": "TRIM",
    "REPLACE": "REPLACE",
    "FLOOR": "FLOOR",
    "CEILING": "CEIL",
}

# CONVERT target types (base name, before any precision) by value family —
# lets the ADDDAY-vs-numeric heuristics see through a conversion.
_NUMERIC_SQL_TYPES = {
    "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "DECIMAL", "NUMERIC",
    "NUMBER", "FLOAT", "REAL", "MONEY", "SMALLMONEY",
}
_DATE_SQL_TYPES = {"DATE", "DATETIME", "DATETIME2", "SMALLDATETIME", "DATETIMEOFFSET"}


def generate_ast(
    mutations: list[MutationPass],
    *,
    target_entity: str,
    target_column: str,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    """Build a 4X JSON AST for the mutation sequence.

    AST/condition generation is deterministic-only: an LLM cannot be trusted
    to reproduce the exact source SQL semantics as a formula, so ``llm_client``
    (kept for signature compatibility) is never used here. The LLM is still
    used elsewhere in the pipeline for report narrative/glossary text.
    """
    del llm_client
    if not mutations:
        return _column_ref(target_entity, target_column)

    try:
        return build_ast_from_mutations(mutations, target_entity, target_column)
    except (_ValuePredicateMixingError, FormulaExpansionError) as exc:
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

    mutations = sorted(mutations, key=lambda m: (m.source_position, m.ordinal))

    # Default prior value: a row not matched by ANY guard keeps whatever
    # value the column already had -- an UPDATE with a WHERE clause never
    # touches non-matching rows, it does not null them out. Self-ref
    # (pass-through) is therefore always the correct base, whether every
    # write in this fold is conditional or not.
    ast: dict[str, Any] = _column_ref(target_entity, target_column)

    # Raw (pre-substitution) (condition, assigned-value) pairs for a run of
    # consecutive self-referential passes that fold flat instead of nesting
    # -- see ``_self_ref_guard_shape`` for the two provably-sound shapes
    # ("A": identical guard + identical value -> OR; "B": guard reads the
    # column only via ISEMPTY(self) and every pass writes a definite
    # non-empty literal -> priority-ordered ELSEIF cascade). A chain only
    # ever extends while every pass keeps classifying the same way;
    # anything else (e.g. a guard comparing the column to a specific value
    # another pass in the same chain just wrote) is Class C and always
    # falls back to the general prior-value substitution below.
    # ``self_ref_chain_base`` is the AST state as of just BEFORE the
    # chain's first pass -- whatever the column already held at that
    # point, not necessarily the bare column ref (an earlier, unrelated
    # guarded write could already have layered something on).
    self_ref_chain: list[tuple[dict[str, Any], dict[str, Any]]] = []
    self_ref_chain_base: dict[str, Any] | None = None
    self_ref_chain_class: str | None = None
    # Literal-only assignments (``SET Col = 'reason' WHERE …`` with no read of
    # ``Col``) that share the same value can OR their guards instead of nesting.
    literal_chain: list[tuple[dict[str, Any], dict[str, Any]]] = []
    literal_chain_base: dict[str, Any] | None = None
    # Class C: guards and assigned values never read the target column — fold as a
    # flat priority cascade (last UPDATE outermost) without inlining ``prior`` into
    # each guard (prevents DAG / string explosion on wide reason-code columns).
    independent_arms: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    independent_base: dict[str, Any] | None = None
    skipped_set_based_source = False
    deferred_clamps: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]] = []
    saw_floor_clamp = False

    for segment in _segment_mutations_by_control_flow(mutations):
        group_id = segment[0].control_branch_group if segment else None
        if group_id and all(m.control_branch_group == group_id for m in segment):
            ast = _fold_control_branch_group(segment, target_entity, target_column, ast)
            self_ref_chain = []
            self_ref_chain_base = None
            self_ref_chain_class = None
            literal_chain = []
            literal_chain_base = None
            independent_arms = []
            independent_base = None
            continue

        for mutation in segment:
            if _skip_mutation_in_ast_fold(mutation, target_entity, target_column):
                if _is_set_based_unresolved_source_assignment(mutation, target_entity):
                    skipped_set_based_source = True
                continue
            ast_before_this_pass = ast
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

            cond_sql = mutation.effective_condition or (
                mutation.where_clause.strip()
                if mutation.where_clause and mutation.where_clause.strip()
                else None
            )
            if cond_sql:
                raw_cond = parse_sql_expression_to_ast(
                    cond_sql,
                    default_entity=target_entity,
                    target_column=target_column,
                    as_condition=True,
                )
                raw_cond = _augment_guard_from_effective_sql(
                    cond_sql,
                    raw_cond,
                    target_entity,
                    target_column,
                )
                # ``SET Col = Bound WHERE Col > Bound`` (cap) on a running multi-pass
                # total: apply it once, over the final total. Folding each cap in
                # place tests a stale ``Col`` (the deep prior is not inlined into
                # guards) and repeats identical caps.
                clamp = _classify_total_clamp(raw_cond, then_node, target_entity, target_column)
                if clamp is not None and clamp[0] == "floor":
                    saw_floor_clamp = True
                if (
                    clamp is not None
                    and clamp[0] == "cap"
                    and isinstance(ast, dict)
                    and _nested_if_depth(ast) >= 1
                ):
                    deferred_clamps.append(clamp)
                    continue
                collapsed_ast = None
                if self_ref_chain and self_ref_chain_base is not None:
                    if self_ref_chain_class == "A":
                        collapsed_ast = _try_collapse_class_a(
                            raw_cond, then_node, self_ref_chain, self_ref_chain_base,
                            target_entity, target_column,
                        )
                    elif self_ref_chain_class == "B":
                        collapsed_ast = _try_collapse_class_b(
                            raw_cond, then_node, self_ref_chain, self_ref_chain_base,
                            target_entity, target_column,
                        )
                if collapsed_ast is not None:
                    ast = collapsed_ast
                    self_ref_chain.append((raw_cond, then_node))
                    literal_chain = []
                    literal_chain_base = None
                    independent_arms = []
                    independent_base = None
                    continue

                collapsed_literal = None
                if literal_chain and literal_chain_base is not None:
                    collapsed_literal = _try_collapse_identical_literal_or(
                        raw_cond,
                        then_node,
                        literal_chain,
                        literal_chain_base,
                        target_entity,
                        target_column,
                    )
                if collapsed_literal is not None:
                    ast = collapsed_literal
                    literal_chain.append((raw_cond, then_node))
                    self_ref_chain = []
                    self_ref_chain_base = None
                    self_ref_chain_class = None
                    independent_arms = []
                    independent_base = None
                    continue

                narrowed_ast = None
                if isinstance(ast_before_this_pass, dict):
                    narrowed_ast = _try_fold_narrowing_chronological_guard(
                        ast_before_this_pass,
                        raw_cond,
                        then_node,
                        target_entity,
                        target_column,
                        mutation.source_position,
                        mutation.ordinal,
                    )
                if narrowed_ast is not None:
                    ast = narrowed_ast
                    self_ref_chain = []
                    self_ref_chain_base = None
                    self_ref_chain_class = None
                    literal_chain = []
                    literal_chain_base = None
                    independent_arms = []
                    independent_base = None
                    continue

                if _is_independent_guard_pass(
                    raw_cond,
                    then_node,
                    target_entity,
                    target_column,
                    ast_before_this_pass,
                ):
                    if independent_base is None:
                        independent_base = ast_before_this_pass
                        independent_arms = []
                    independent_arms.append(
                        (mutation.source_position, mutation.ordinal, raw_cond, then_node)
                    )
                    ast = _rebuild_priority_cascade(independent_arms, independent_base)
                    self_ref_chain = []
                    self_ref_chain_base = None
                    self_ref_chain_class = None
                    literal_chain = []
                    literal_chain_base = None
                    continue

                independent_arms = []
                independent_base = None

                # ``SET Col = ISNULL(Col,0) + Y WHERE g`` after a multi-pass chain is
                # additive: the earlier passes still apply to rows matching ``g``.
                # Folding it as ``IF(g) THEN(Col + Y) ELSE(prior)`` makes the arms
                # mutually exclusive and drops the earlier passes for matching rows.
                increment = _self_increment_operand(then_node, target_entity, target_column)
                if (
                    increment is not None
                    and isinstance(ast, dict)
                    and _nested_if_depth(ast) >= 1
                    and not _references_self_column(raw_cond, target_entity, target_column)
                    and not _references_self_column(increment, target_entity, target_column)
                ):
                    ast = {
                        "type": "BINARY_OP",
                        "operator": "+",
                        "left": ast,
                        "right": {
                            "type": "IF_THEN_ELSE",
                            "condition": raw_cond,
                            "then_branch": increment,
                            "else_branch": {"type": "LITERAL", "value_type": "NUMBER", "value": 0},
                        },
                    }
                    self_ref_chain = []
                    self_ref_chain_base = None
                    self_ref_chain_class = None
                    literal_chain = []
                    literal_chain_base = None
                    continue

                then_node_sub = _substitute_prior_value(
                    then_node, ast, target_entity, target_column
                )
                cond = _substitute_prior_in_guard(
                    raw_cond, ast, target_entity, target_column
                )
                ast = {
                    "type": "IF_THEN_ELSE",
                    "condition": cond,
                    "then_branch": then_node_sub,
                    "else_branch": _else_branch_for_literal_default_guard(
                        ast, raw_cond, target_entity, target_column
                    ),
                    "_source_position": mutation.source_position,
                    "_source_ordinal": mutation.ordinal,
                }
                shape = _self_ref_guard_shape(raw_cond, then_node, target_entity, target_column)
                if shape is not None:
                    self_ref_chain = [(raw_cond, then_node)]
                    self_ref_chain_base = ast_before_this_pass
                    self_ref_chain_class = shape
                else:
                    self_ref_chain = []
                    self_ref_chain_base = None
                    self_ref_chain_class = None
                if _is_identical_literal_assignment(then_node, target_entity, target_column):
                    literal_chain = [(raw_cond, then_node)]
                    literal_chain_base = ast_before_this_pass
                else:
                    literal_chain = []
                    literal_chain_base = None
            else:
                # Unguarded pass resets the base value for subsequent guards.
                ast = _substitute_prior_value(then_node, ast, target_entity, target_column)
                self_ref_chain = []
                self_ref_chain_base = None
                self_ref_chain_class = None
                literal_chain = []
                literal_chain_base = None
                independent_arms = []
                independent_base = None

    if deferred_clamps and isinstance(ast, dict):
        ast = _apply_deferred_clamps(ast, deferred_clamps, with_floor=saw_floor_clamp)

    if isinstance(ast, dict):
        ast = enforce_later_update_precedence(ast)

    ast = (
        optimize_expression_ast(ast, target_entity=target_entity, target_column=target_column)
        if isinstance(ast, dict)
        else ast
    )
    if isinstance(ast, dict):
        try:
            ast = enforce_formula_budget(
                ast, target_entity=target_entity, target_column=target_column
            )
        except Exception:
            pass
    try:
        pruned = prune_redundant_ast(ast)
    except FormulaExpansionError:
        pruned = ast
    folded = pruned if isinstance(pruned, dict) else ast
    ast_out = (
        optimize_expression_ast(folded, target_entity=target_entity, target_column=target_column)
        if isinstance(folded, dict)
        else folded
    )
    if isinstance(ast_out, dict):
        try:
            ast_out = enforce_formula_budget(
                ast_out, target_entity=target_entity, target_column=target_column
            )
        except Exception:
            # Budget pass must never abort derivation; compile phase reports errors.
            pass
    if (
        skipped_set_based_source
        and isinstance(ast_out, dict)
        and _is_self_column_ref(ast_out, target_entity, target_column)
        and not _is_string_flag_column(target_column)
        and _is_numeric_column_name(target_column)
        and not _is_date_like_column_name(target_column)
    ):
        ast_out = _wrap_nullable_identity_default(ast_out, target_entity, target_column)
    return ast_out


_STRING_FLAG_NAME_RE = re.compile(r"(?i)^(?:flg|flag)|(?:flg|flag)$")


def _is_string_flag_column(column: str) -> bool:
    """Character flag columns (``FlgDeg``, ``FlgProcessing``) hold 'Y'/'N'.

    A numeric ``ISEMPTY -> 0`` default is wrong for them; map the flag directly.
    """
    return bool(_STRING_FLAG_NAME_RE.search(bare_ident(column or "")))


def _contains_self_column_ref(node: Any, entity: str, column: str) -> bool:
    """True if ``node`` reads the column being derived anywhere in its tree."""
    if isinstance(node, dict):
        if _is_self_column_ref(node, entity, column):
            return True
        return any(
            _contains_self_column_ref(v, entity, column)
            for k, v in node.items()
            if not str(k).startswith("_")
        )
    if isinstance(node, list):
        return any(_contains_self_column_ref(v, entity, column) for v in node)
    return False


def _self_ref_guard_shape(
    raw_cond: dict[str, Any],
    then_node: dict[str, Any],
    entity: str,
    column: str,
) -> str | None:
    """Classify a single self-referential guarded pass as Class A, B, or
    neither (Class C -- always handled by the general substitution fold).

    Both flattenings below only ever apply while EVERY pass extending the
    chain keeps classifying the same way (checked by the caller); this
    function only says what a single pass, in isolation, is eligible for.

    - **Class A** -- the guard reads the column in any shape at all (a
      plain equality is the common case), and every pass in the eventual
      chain assigns the textually IDENTICAL value. Once any pass fires,
      the guard's outcome for that row can never flip back (the row's
      value is now the one every other pass in the chain would also have
      written), so a flat ``OR`` of the raw guards is exact -- see
      ``_try_collapse_class_a``.
    - **Class B** -- the guard reads the column ONLY via ``ISEMPTY(self)``
      (never bare, never under ``ISNOTEMPTY``, never as a comparison
      operand -- see ``_self_refs_only_via_isempty``), AND this pass's
      assigned value is a literal statically guaranteed non-empty. Once
      such a pass fires, ``ISEMPTY(self)`` becomes permanently false for
      that row, so a later pass with the same shape can never re-match it
      -- the chronological "first UPDATE that matches wins" behaviour is
      then identical to evaluating every guard, in order, against the
      chain's ORIGINAL state, i.e. a priority-ordered ``ELSEIF`` cascade
      using each pass's raw condition -- see ``_try_collapse_class_b``.

      The non-empty-literal requirement is load-bearing: a pass that
      writes ``NULL`` (or any non-literal, whose emptiness can't be
      proven statically) leaves ``ISEMPTY(self)`` still true, so a
      following same-shaped guard would ALSO still match that row in the
      real chronological execution -- collapsing that pair into mutually
      exclusive ELSEIF arms would silently drop the second pass's write.
      Such a pass is therefore never eligible for Class B, in either
      direction (it can't seed one or extend one), and always falls
      through to Class C's exact general-substitution handling instead.
    """
    if _contains_self_column_ref(then_node, entity, column):
        return None
    if not _contains_self_column_ref(raw_cond, entity, column):
        return None
    if (
        _self_refs_only_via_isempty(raw_cond, entity, column)
        and _is_definite_nonempty_literal(then_node)
    ):
        return "B"
    return "A"


def _self_refs_only_via_isempty(node: Any, entity: str, column: str, under_isempty: bool = False) -> bool:
    """True if every self-column-ref in ``node`` sits directly inside an
    ``ISEMPTY(...)`` call's argument (never bare, never under
    ``ISNOTEMPTY``, never as a comparison/arithmetic operand elsewhere).
    """
    if isinstance(node, dict):
        if _is_self_column_ref(node, entity, column):
            return under_isempty
        is_isempty = (
            node.get("type") == "FUNCTION_CALL"
            and str(node.get("function_name") or "").upper() == "ISEMPTY"
        )
        for key, value in node.items():
            if str(key).startswith("_"):
                continue
            child_flag = True if (key == "arguments" and is_isempty) else under_isempty
            if not _self_refs_only_via_isempty(value, entity, column, child_flag):
                return False
        return True
    if isinstance(node, list):
        return all(_self_refs_only_via_isempty(v, entity, column, under_isempty) for v in node)
    return True


def _is_definite_nonempty_literal(node: Any) -> bool:
    """True only for a literal statically guaranteed non-empty (a
    non-empty STRING or a NUMBER) -- never NULL, never a COLUMN_REF or
    FUNCTION_CALL whose runtime value can't be proven non-empty here."""
    if not isinstance(node, dict) or node.get("type") != "LITERAL":
        return False
    value_type = str(node.get("value_type") or "").upper()
    if value_type == "NULL":
        return False
    value = node.get("value")
    if value is None:
        return False
    if value_type == "STRING" and str(value) == "":
        return False
    return True


def _try_collapse_class_a(
    raw_cond: dict[str, Any],
    then_node: dict[str, Any],
    chain: list[tuple[dict[str, Any], dict[str, Any]]],
    base_else: dict[str, Any],
    target_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """OR consecutive self-referential guards instead of nesting them.

    A chain of ``UPDATE ... SET Col = <same value> WHERE <predicate reading
    Col>`` passes (each gated by a different JOIN/population, but with
    textually identical guard and assigned value -- e.g. several UPDATEs all
    doing ``SET ASSET_NORM = 'CONDI_STD' WHERE ASSET_NORM = 'ALWYS_STD'``
    against different cohorts) is semantically an OR of the guards: once any
    pass fires, later passes in the chain no longer match (their own guard
    reads the column they just changed), so the *outcome* for any row is
    "matches guard N for some N in the chain -> same value; otherwise
    unchanged". Folding each pass by substituting the full accumulated prior
    AST into its own guard is precise in the general case, but for this
    identical-condition/identical-value shape it only produces deeper and
    deeper nesting -- an IF_THEN_ELSE tree embedded as a comparison operand
    -- for no semantic gain, and can grow large enough to fail formula
    generation. Collapsing to a flat OR is exact, not an approximation, and
    keeps the formula small.

    Only applies when the assigned value is a LITERAL (not a column read or
    function call) that is structurally identical across the whole chain.
    The literal-only requirement matters: the soundness argument is "every
    pass in the chain assigns the exact same value, so it doesn't matter
    which guard fires first" -- which only holds unconditionally for a
    literal. A shared COLUMN_REF (e.g. two passes both doing
    ``SET Col = Other.Col``) reads whatever ``Other.Col`` holds AT THE TIME
    of each statement; if anything between those two passes writes
    ``Other.Col``, the two reads can differ even though their AST shape
    looks identical, and OR-ing the guards would then silently use
    whichever pass happened to be evaluated, rather than reproducing SQL's
    actual last-write-wins order. Returns the collapsed ``IF_THEN_ELSE``,
    or ``None`` when the chain doesn't apply and the caller should fall
    back to the general prior-value substitution.
    """
    if not chain:
        return None
    if not isinstance(then_node, dict) or then_node.get("type") != "LITERAL":
        return None
    if _contains_self_column_ref(then_node, target_entity, target_column):
        return None
    if not _contains_self_column_ref(raw_cond, target_entity, target_column):
        return None
    then_sig = _ast_signature(then_node)
    if any(_ast_signature(v) != then_sig for _, v in chain):
        return None

    or_condition = raw_cond
    for cond, _ in reversed(chain):
        or_condition = {"type": "BINARY_OP", "operator": "OR", "left": or_condition, "right": cond}
    return {
        "type": "IF_THEN_ELSE",
        "condition": or_condition,
        "then_branch": then_node,
        "else_branch": base_else,
    }


def _try_collapse_class_b(
    raw_cond: dict[str, Any],
    then_node: dict[str, Any],
    chain: list[tuple[dict[str, Any], dict[str, Any]]],
    base_else: dict[str, Any],
    target_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """Flatten a chain of ``ISEMPTY(self)``-guarded passes into one
    priority-ordered ``ELSEIF`` cascade using each pass's RAW (never
    prior-value-substituted) condition, instead of nesting each guard
    around the full accumulated prior AST.

    See ``_self_ref_guard_shape`` for the soundness argument and the
    non-empty-literal requirement this depends on; the caller only invokes
    this once every pass so far (including this one) has already
    classified as Class B, so no shape re-checking happens here beyond
    this pass's own eligibility.
    """
    if not chain:
        return None
    if not _self_refs_only_via_isempty(raw_cond, target_entity, target_column):
        return None
    if not _is_definite_nonempty_literal(then_node):
        return None

    arms = list(chain) + [(raw_cond, then_node)]
    ast: dict[str, Any] = base_else
    for cond, value in reversed(arms):
        ast = {"type": "IF_THEN_ELSE", "condition": cond, "then_branch": value, "else_branch": ast}
    return ast


def _is_identical_literal_assignment(
    then_node: dict[str, Any],
    target_entity: str,
    target_column: str,
) -> bool:
    return _is_literal_ast(then_node) and not _contains_self_column_ref(
        then_node, target_entity, target_column
    )


def _is_independent_guard_pass(
    raw_cond: dict[str, Any],
    then_node: dict[str, Any],
    entity: str,
    column: str,
    prior_ast: Any = None,
) -> bool:
    """Pass does not read the derived column in its guard or assigned value."""
    ent = normalize_table_name(entity or "").upper().lstrip("#")
    col = bare_ident(column or "").upper()
    if ent.endswith("ACCOUNTCAL") and col == "FINALNPADT":
        # Account NPA date must stay strictly chronological (DPD / PUI / restructure
        # before customer SysNPA_Dt write-back); flat priority cascades reorder arms.
        return False
    if _contains_self_column_ref(raw_cond, entity, column):
        return False
    if _contains_self_column_ref(then_node, entity, column):
        return False
    if isinstance(prior_ast, dict) and prior_ast.get("type") == "IF_THEN_ELSE":
        # Chronological folding must nest on an existing guarded tree so
        # later passes (LOS-only NULL, @ProcessDate) keep distinct guards.
        return False
    return True


def _rebuild_and_conjuncts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        return {"type": "LITERAL", "value_type": "BOOLEAN", "value": True}
    out = parts[0]
    for part in parts[1:]:
        out = {"type": "BINARY_OP", "operator": "AND", "left": out, "right": part}
    return out


def _try_fold_narrowing_chronological_guard(
    prior_ast: dict[str, Any],
    raw_cond: dict[str, Any],
    then_node: dict[str, Any],
    target_entity: str,
    target_column: str,
    source_position: int,
    source_ordinal: int,
) -> dict[str, Any] | None:
    """Fold ``IF(G AND Q) THEN v ELSE prior`` as ``IF(G) THEN IF(Q) THEN v ELSE prior_then``.

    Later UPDATE passes often repeat the same join/process guard ``G`` and add a
    column filter ``Q`` (e.g. ``SysAssetClassAlt_Key IN (LOS …)``). Nesting only
    the qualifier under the shared guard keeps LOS-only NULL from shadowing the
    ADDDAY CASE assigned under ``G`` alone.
    """
    if prior_ast.get("type") != "IF_THEN_ELSE":
        return None
    outer_cond = prior_ast.get("condition")
    if not isinstance(outer_cond, dict):
        return None
    outer_parts = _flatten_and_conjuncts(outer_cond)
    raw_parts = _flatten_and_conjuncts(raw_cond)
    outer_sigs = {guard_conjunct_signature(op) for op in outer_parts}
    for op in outer_parts:
        osig = guard_conjunct_signature(op)
        if not any(guard_conjunct_signature(rp) == osig for rp in raw_parts):
            return None
    extra = [
        rp
        for rp in raw_parts
        if guard_conjunct_signature(rp) not in outer_sigs
    ]
    if not extra:
        return None
    narrowing = _rebuild_and_conjuncts(extra)
    prior_then = prior_ast.get("then_branch")
    if not isinstance(prior_then, dict):
        return None
    then_sub = _substitute_prior_value(
        then_node, prior_ast, target_entity, target_column
    )
    inner: dict[str, Any] = {
        "type": "IF_THEN_ELSE",
        "condition": narrowing,
        "then_branch": then_sub,
        "else_branch": prior_then,
    }
    return {
        "type": "IF_THEN_ELSE",
        "condition": outer_cond,
        "then_branch": inner,
        "else_branch": prior_ast.get("else_branch"),
        "_source_position": source_position,
        "_source_ordinal": source_ordinal,
    }


def _rebuild_priority_cascade(
    arms: list[tuple[int, int, dict[str, Any], dict[str, Any]]],
    base_else: dict[str, Any],
) -> dict[str, Any]:
    """Last arm in ``arms`` is the outermost IF (latest UPDATE wins on overlap)."""
    result = base_else
    for source_position, source_ordinal, cond, then_b in reversed(arms):
        result = {
            "type": "IF_THEN_ELSE",
            "condition": cond,
            "then_branch": then_b,
            "else_branch": result,
            "_source_position": source_position,
            "_source_ordinal": source_ordinal,
        }
    return result


def _try_collapse_identical_literal_or(
    raw_cond: dict[str, Any],
    then_node: dict[str, Any],
    chain: list[tuple[dict[str, Any], dict[str, Any]]],
    base_else: dict[str, Any],
    target_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """OR guards that assign the same literal without nesting the prior tree.

    Sound when every pass writes the identical literal and the guard does not
    depend on which pass ran first (same outcome for any matching guard).
    """
    if not chain or not _is_literal_ast(then_node):
        return None
    if _contains_self_column_ref(then_node, target_entity, target_column):
        return None
    then_sig = _ast_signature(then_node)
    if any(_ast_signature(value) != then_sig for _, value in chain):
        return None
    or_condition = raw_cond
    for cond, _ in reversed(chain):
        or_condition = {
            "type": "BINARY_OP",
            "operator": "OR",
            "left": or_condition,
            "right": cond,
        }
    return {
        "type": "IF_THEN_ELSE",
        "condition": or_condition,
        "then_branch": then_node,
        "else_branch": base_else,
    }


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


def _substitute_prior_value(node, prior, entity, column):
    """Resolve reads of the target against the state BEFORE this statement."""
    if isinstance(node, dict):
        if _is_self_column_ref(node, entity, column):
            # After an unguarded literal reset (``SET COUNT=0``), keep the
            # column reference in ``ISNULL(COUNT,0)+1``-style increments —
            # substituting the literal ``0`` would fold the whole expression
            # to a constant ``1``.
            if _is_literal_ast(prior):
                return node
            if isinstance(prior, dict) and prior.get("type") == "IF_THEN_ELSE":
                then_b = prior.get("then_branch")
                if (
                    isinstance(then_b, dict)
                    and then_b.get("type") == "LITERAL"
                    and str(then_b.get("value_type") or "").upper() == "NULL"
                ):
                    # CASE … ELSE Col must not inline an earlier ``IF(g) THEN NULL``
                    # reset — that duplicates ``g`` and NULLs on CASE fall-through.
                    return _column_ref(entity, column)
            if isinstance(prior, dict) and _nested_if_depth(prior) >= 2:
                return _column_ref(entity, column)
            return prior
        return {k: _substitute_prior_value(v, prior, entity, column) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute_prior_value(v, prior, entity, column) for v in node]
    return node


def _is_literal_ast(node: Any) -> bool:
    return isinstance(node, dict) and node.get("type") == "LITERAL"


def _is_concat_call(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and node.get("type") == "FUNCTION_CALL"
        and str(node.get("function_name") or "").upper() == "CONCAT"
        and isinstance(node.get("arguments"), list)
    )


def _is_empty_string_coalesce(node: Any) -> bool:
    if not (
        isinstance(node, dict)
        and node.get("type") == "FUNCTION_CALL"
        and str(node.get("function_name") or "").upper() == "COALESCE"
    ):
        return False
    args = node.get("arguments") or []
    return (
        len(args) == 2
        and isinstance(args[1], dict)
        and args[1].get("type") == "LITERAL"
        and args[1].get("value") == ""
    )


def _null_safe_concat_arguments(args: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``CONCAT(col, ',', 'text')`` → ``CONCAT(COALESCE(col,''), ',', 'text')``."""
    empty = {"type": "LITERAL", "value_type": "STRING", "value": ""}
    wrapped: list[dict[str, Any]] = []
    for arg in args:
        if isinstance(arg, dict) and (_is_string_literal_node(arg) or _is_empty_string_coalesce(arg)):
            wrapped.append(arg)
        else:
            wrapped.append(
                {
                    "type": "FUNCTION_CALL",
                    "function_name": "COALESCE",
                    "arguments": [arg, empty],
                }
            )
    return wrapped


def _references_self_column(node: Any, entity: str, column: str) -> bool:
    if isinstance(node, dict):
        if _is_self_column_ref(node, entity, column):
            return True
        return any(
            _references_self_column(v, entity, column)
            for k, v in node.items()
            if not str(k).startswith("_")
        )
    if isinstance(node, list):
        return any(_references_self_column(v, entity, column) for v in node)
    return False


def _coalesce_or_isnull_of_self(node: Any, entity: str, column: str) -> bool:
    if not isinstance(node, dict) or node.get("type") != "FUNCTION_CALL":
        return False
    fn = str(node.get("function_name") or "").upper()
    if fn not in {"COALESCE", "ISNULL", "NVL", "IFNULL"}:
        return False
    args = node.get("arguments") or []
    return bool(args) and _is_self_column_ref(args[0], entity, column)


def _is_self_negativity_zero_guard(node: Any, entity: str, column: str) -> bool:
    """``ISNULL(Col,0)<0`` / ``COALESCE(Col,0)<0`` / ``Col<0`` style clamps."""
    if not isinstance(node, dict) or node.get("type") != "BINARY_OP":
        return False
    op = str(node.get("operator") or "").strip()
    if op not in {"<", "<="}:
        return False
    left = node.get("left")
    right = node.get("right")
    if _coalesce_or_isnull_of_self(left, entity, column):
        return _is_zero_literal(right)
    if _is_self_column_ref(left, entity, column):
        return _is_zero_literal(right)
    return False


def _is_zero_literal(node: Any) -> bool:
    if not isinstance(node, dict) or node.get("type") != "LITERAL":
        return False
    try:
        return float(node.get("value")) == 0.0
    except (TypeError, ValueError):
        return False


def _replace_negativity_guard_subject(
    node: dict[str, Any],
    prior: Any,
    entity: str,
    column: str,
) -> dict[str, Any]:
    """Point a ``COALESCE(Col,0)<0`` clamp at the already-folded derivation."""
    left = node.get("left")
    op = node.get("operator")
    right = node.get("right")
    zero_lit = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    if isinstance(left, dict) and _coalesce_or_isnull_of_self(left, entity, column):
        pad = (left.get("arguments") or [None, zero_lit])[1] or zero_lit
        if _is_self_column_ref(prior, entity, column) or _is_literal_ast(prior):
            new_left = {
                "type": "FUNCTION_CALL",
                "function_name": "COALESCE",
                "arguments": [prior, pad],
            }
        else:
            # Folded derivation paths for DPD-style ints already end in 0, not
            # NULL — test the computed value directly and avoid COALESCE(IF…)
            # duplication in the exported formula.
            new_left = prior
    elif _is_self_column_ref(left, entity, column):
        new_left = prior
    else:
        return node
    return {"type": "BINARY_OP", "operator": op, "left": new_left, "right": right}


def _substitute_prior_in_guard(node: Any, prior: Any, entity: str, column: str) -> Any:
    """Substitute prior column state into a WHERE/guard predicate.

    When the accumulated state is a compile-time literal (e.g. an unguarded
    ``SET Col = 0`` reset), keep the target column reference in the guard
    instead of folding to ``0 == 0``.

    For ``ISNULL(Col,0)<0`` / ``COALESCE(Col,0)<0`` clamps that run **after**
    the main derivation in source order, test the folded derivation (``prior``),
    not the stale input column — otherwise the clamp appears to run before
    ``DATEDIFF`` logic and can never fire.

    Other complex priors still substitute so join+default flattening can
    detect ``prior == 0`` patterns.
    """
    if isinstance(node, dict) and _is_self_negativity_zero_guard(node, entity, column):
        if _is_literal_ast(prior) or _is_self_column_ref(prior, entity, column):
            return node
        return _replace_negativity_guard_subject(node, prior, entity, column)
    if isinstance(node, dict):
        if _is_self_column_ref(node, entity, column):
            if _is_literal_ast(prior):
                return node
            return _prior_reference_for_guard(prior, entity, column)
        return {
            k: _substitute_prior_in_guard(v, prior, entity, column) for k, v in node.items()
        }
    if isinstance(node, list):
        return [_substitute_prior_in_guard(v, prior, entity, column) for v in node]
    return node


def _prior_reference_for_guard(prior: Any, entity: str, column: str) -> Any:
    """Avoid inlining a deep folded tree into every subsequent WHERE guard."""
    if _is_literal_ast(prior) or _is_self_column_ref(prior, entity, column):
        return prior
    if isinstance(prior, dict) and _nested_if_depth(prior) >= 3:
        return _column_ref(entity, column)
    return prior


def _classify_total_clamp(
    raw_cond: Any,
    then_node: Any,
    entity: str,
    column: str,
) -> tuple[str, str, dict[str, Any], list[dict[str, Any]]] | None:
    """Recognise ``SET Col = Bound WHERE Col > Bound [AND …]`` (cap) or
    ``SET Col = 0 WHERE Col < 0`` (floor); extra conjuncts must not read ``Col``.

    Returns ``(kind, operator, bound, extra_conjuncts)``.
    """
    if not isinstance(raw_cond, dict) or not isinstance(then_node, dict):
        return None
    parts = _flatten_and_conjuncts(raw_cond)
    for i, part in enumerate(parts):
        if not isinstance(part, dict) or part.get("type") != "BINARY_OP":
            continue
        op = str(part.get("operator") or "").strip()
        left, right = part.get("left"), part.get("right")
        if not (
            _is_self_column_ref(left, entity, column)
            or _coalesce_or_isnull_of_self(left, entity, column)
        ):
            continue
        extras = [p for j, p in enumerate(parts) if j != i]
        if any(_references_self_column(e, entity, column) for e in extras):
            return None
        if (
            op in {">", ">="}
            and then_node.get("type") == "COLUMN_REF"
            and not _is_self_column_ref(then_node, entity, column)
            and isinstance(right, dict)
            and _ast_signature(right) == _ast_signature(then_node)
        ):
            return ("cap", op, then_node, extras)
        if op in {"<", "<="} and _is_zero_literal(right) and _is_zero_literal(then_node):
            return ("floor", op, then_node, extras)
    return None


def _floor_subject(node: Any) -> dict[str, Any] | None:
    """``x`` for an already-folded floor ``IF(x < 0) THEN 0 ELSE x``; else ``None``."""
    if not isinstance(node, dict) or node.get("type") != "IF_THEN_ELSE":
        return None
    cond = node.get("condition")
    if not (
        isinstance(cond, dict)
        and cond.get("type") == "BINARY_OP"
        and str(cond.get("operator") or "").strip() == "<"
        and _is_zero_literal(cond.get("right"))
        and _is_zero_literal(node.get("then_branch"))
    ):
        return None
    subject = node.get("else_branch")
    if isinstance(subject, dict) and _ast_signature(cond.get("left")) == _ast_signature(subject):
        return subject
    return None


def _apply_deferred_clamps(
    total: dict[str, Any],
    clamps: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]],
    with_floor: bool = False,
) -> dict[str, Any]:
    """Wrap ``total`` in each distinct deferred cap/floor (later passes outermost)."""
    kept: list[tuple[Any, frozenset]] = []
    result = total
    for kind, op, bound, extras in clamps:
        base_key = (kind, op, _ast_signature(bound))
        extra_sigs = frozenset(_ast_signature(e) for e in extras)
        # An earlier clamp with the same bound and no more conditions already
        # covers every row this one would touch (``WHERE t>NB`` vs the later
        # ``WHERE t>NB AND NB>0``): stacking it only repeats the total.
        if any(bk == base_key and prior <= extra_sigs for bk, prior in kept):
            continue
        kept.append((base_key, extra_sigs))
        if with_floor and kind == "cap" and op == ">" and not extras:
            # Standard clamp: IF(t > Bound) THEN Bound ELSEIF(t < 0) THEN 0 ELSE t.
            # ``_keep_inline`` stops the optimizer hoisting the total's IF terms out
            # of these comparisons (which multiplies the formula into one branch
            # per combination) and stops the floor being rewritten into MAX().
            subject = _floor_subject(result) or result
            zero = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
            result = {
                "type": "IF_THEN_ELSE",
                "condition": {
                    "type": "BINARY_OP", "operator": ">", "left": subject,
                    "right": bound, "_keep_inline": True,
                },
                "then_branch": bound,
                "else_branch": {
                    "type": "IF_THEN_ELSE",
                    "condition": {
                        "type": "BINARY_OP", "operator": "<", "left": subject,
                        "right": zero, "_keep_inline": True,
                    },
                    "then_branch": zero,
                    "else_branch": subject,
                },
            }
            continue
        compare = {
            "type": "BINARY_OP",
            "operator": op,
            "left": result,
            "right": bound if kind == "cap" else {"type": "LITERAL", "value_type": "NUMBER", "value": 0},
        }
        result = {
            "type": "IF_THEN_ELSE",
            "condition": _rebuild_and_conjuncts([compare, *extras]),
            "then_branch": bound,
            "else_branch": result,
        }
    return result


def _self_increment_operand(node: Any, entity: str, column: str) -> dict[str, Any] | None:
    """``Y`` for ``Col + Y`` / ``Y + Col`` / ``COALESCE(Col,0) + Y``; else ``None``."""
    if not isinstance(node, dict) or node.get("type") != "BINARY_OP":
        return None
    if str(node.get("operator") or "").strip() != "+":
        return None
    left, right = node.get("left"), node.get("right")
    for own, other in ((left, right), (right, left)):
        if not isinstance(other, dict):
            continue
        if _is_self_column_ref(own, entity, column) or _coalesce_or_isnull_of_self(
            own, entity, column
        ):
            return other
    return None


def _nested_if_depth(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    if node.get("type") == "IF_THEN_ELSE":
        child = node.get("else_branch")
        return 1 + _nested_if_depth(child if isinstance(child, dict) else None)
    best = 0
    for value in node.values():
        if isinstance(value, dict):
            best = max(best, _nested_if_depth(value))
    return best


def _else_branch_for_literal_default_guard(
    prior_ast: Any,
    raw_cond: dict[str, Any],
    entity: str,
    column: str,
) -> Any:
    """Use pass-through column ref when filling a default for ``Col = <literal>``."""
    if _is_literal_ast(prior_ast) and _references_self_column(raw_cond, entity, column):
        return _column_ref(entity, column)
    return prior_ast


def _wrap_nullable_identity_default(
    node: dict[str, Any],
    entity: str,
    column: str,
) -> dict[str, Any]:
    """``IF(ISEMPTY(col)) THEN 0 ELSE col`` for skipped numeric set-based writes.

    Date/string columns (``SysNPA_Dt``, ``DegReason``) must not inherit this
    numeric zero default — that is only valid for DPD/amount roll-ups.
    """
    col = node if _is_self_column_ref(node, entity, column) else _column_ref(entity, column)
    return {
        "type": "IF_THEN_ELSE",
        "condition": {
            "type": "FUNCTION_CALL",
            "function_name": "ISEMPTY",
            "arguments": [col],
        },
        "then_branch": {"type": "LITERAL", "value_type": "NUMBER", "value": 0},
        "else_branch": col,
    }


def _condition_sql_mentions_column(sql: str | None, column: str) -> bool:
    if not sql:
        return False
    name = bare_ident(column)
    return bool(re.search(rf"(?i)\b{re.escape(name)}\b", sql))


def _is_cross_column_null_default_guard(cond_sql: str, target_column: str) -> bool:
    """True for ``SET Col = <literal> WHERE SiblingCol IS NULL`` (not join filters).

    Join/product predicates (``ISNULL(C.Aqua_Scheme,'N')='Y'``) and process
    filters (``RUNNINGPROCESSNAME = 'DPD_Calculation'``) must still fold into
    the assigned column's formula.
    """
    if not cond_sql:
        return False
    # A null-default fill is gated on a sibling being NULL/0. A guard that tests
    # ``IS NOT NULL`` is a business predicate (``WHERE DPD_Breach_Date IS NOT NULL
    # AND SP_ExpiryDate >= @DATE``), so the assignment must stay in the fold.
    if re.search(r"(?i)\bIS\s+NOT\s+NULL\b", cond_sql):
        return False
    # Every top-level AND term of a fill guard is itself a null/zero test of a sibling.
    # Any other predicate (``<> 'ALWYS_STD'``, ``> 0``, an OR group, ``IN``) makes the
    # guard a business condition that must stay in the formula.
    if any(not _is_null_test_term(term) for term in _split_top_level(cond_sql, "AND")):
        return False
    col = bare_ident(target_column)
    col_re = re.escape(col)
    if re.search(
        rf"(?i)\b(?!{col_re}\b)([A-Za-z_][\w]*)\s+IS\s+(?:NOT\s+)?NULL\b",
        cond_sql,
    ):
        return True
    for match in re.finditer(r"(?i)\bISEMPTY\s*\(\s*([^)]+)\)", cond_sql):
        inner = (match.group(1) or "").strip()
        peer_m = re.search(r'"([^"]+)"\s*$', inner) or re.search(
            r"\.([A-Za-z_][\w]*)\"?\s*$", inner
        )
        if peer_m and peer_m.group(1).upper() != col.upper():
            return True
    for match in re.finditer(r"(?i)\bISNULL\s*\(\s*([A-Za-z_][\w]*)", cond_sql):
        end = match.end()
        if end < len(cond_sql) and cond_sql[end] in {".", ":"}:
            # ``ISNULL(C.Col,…)`` or ``ISNULL(Entity::Rel::Col,…)`` — join / product
            # predicate, not ``SET target WHERE sibling IS NULL``.
            continue
        if _isnull_call_has_ordering_comparison(cond_sql, match.start()):
            # ``ISNULL(WriteOffAmount,0) > 0`` is a value predicate on a
            # sibling column that must gate the assignment, not a null-default
            # fill (``ISNULL(Sibling,0) = 0`` / ``Sibling IS NULL``).
            continue
        if match.group(1).upper() != col.upper():
            return True
    return False


_ORDERING_TAIL_RE = re.compile(r"\s*(?:>=|<=|<>|!=|>|<)")

_NULL_TEST_TERM_RES = (
    re.compile(r"(?is)^[\w.:\[\]\"#]+\s+IS\s+NULL$"),
    re.compile(
        r"(?is)^(?:ISNULL|COALESCE)\s*\(\s*[\w.:\[\]\"#]+\s*,\s*[^,()]+\)\s*==?\s*[^\s()]+$"
    ),
    re.compile(r"(?is)^ISEMPTY\s*\([^()]*\)$"),
)


def _is_null_test_term(term: str) -> bool:
    """``X IS NULL`` / ``ISNULL(X,0)=0`` / ``ISEMPTY(X)`` — a sibling null/zero test."""
    text = (term or "").strip()
    while text.startswith("(") and text.endswith(")") and _balanced(text[1:-1]):
        text = text[1:-1].strip()
    return any(rx.match(text) for rx in _NULL_TEST_TERM_RES)


def _isnull_call_has_ordering_comparison(text: str, start: int) -> bool:
    """True when the ``ISNULL(...)`` call opening at ``start`` is the left
    operand of an ordering / inequality comparison (``>``, ``<``, ``>=``,
    ``<=``, ``<>``, ``!=``)."""
    open_idx = text.find("(", start)
    if open_idx < 0:
        return False
    depth = 0
    in_quote = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "'":
            in_quote = not in_quote
        elif not in_quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return bool(_ORDERING_TAIL_RE.match(text, i + 1))
    return False


def _skip_mutation_in_ast_fold(
    mutation: MutationPass,
    target_entity: str,
    target_column: str,
) -> bool:
    """Drop writes the row-level 4X fold cannot express faithfully.

    - Aggregates over CTE/subquery sources (``UPDATE … FROM (SELECT MAX…) a``)
      leave ``A.Col`` qualifiers that do not resolve to a platform entity.
    - Cross-column null defaults (``SET MaxFin=0 WHERE MaxNonFin IS NULL``)
      must not fold into the wrong column's derivation.
    """
    expr = (mutation.assigned_expression or "").strip()
    if not expr:
        return False

    if _is_set_based_unresolved_source_assignment(mutation, target_entity):
        return True

    cond_sql = mutation.effective_condition or mutation.where_clause or ""
    if mutation.guarded and cond_sql and not _condition_sql_mentions_column(
        cond_sql, target_column
    ):
        try:
            value = parse_sql_expression_to_ast(
                expr, default_entity=target_entity, target_column=target_column
            )
        except Exception:
            value = None
        if isinstance(value, dict) and value.get("type") == "LITERAL":
            if _is_cross_column_null_default_guard(cond_sql, target_column):
                return True
    return False


def _is_set_based_unresolved_source_assignment(
    mutation: MutationPass,
    target_entity: str,
) -> bool:
    """True when the assigned value still references an unmapped statement alias."""
    expr = (mutation.assigned_expression or "").strip()
    target_norm = normalize_table_name(target_entity).upper()
    marker = re.match(
        r"(?i)^(?P<entity>[^:]+)::(?P<rel>[^:]+)::",
        expr,
    )
    if marker and marker.group("entity").upper() == target_norm:
        rel = marker.group("rel") or ""
        if is_ephemeral_sql_alias(rel):
            return False
        if len(rel) <= 3:
            return True
    if expr.startswith('"') and re.search(r'"\s*\.\s*"[^"]+"\s*\.\s*"', expr):
        return False
    if re.match(r"(?is)^MIN\s*\(", expr):
        return False
    match = re.match(
        r"(?i)^(?P<qual>[A-Za-z_][\w]*)\.(?P<col>[A-Za-z_][\w]*)$",
        expr,
    )
    if not match:
        return False
    qual = match.group("qual").upper()
    target_norm = normalize_table_name(target_entity).upper()
    if qual in {target_norm, bare_ident(target_entity).upper()}:
        return False
    alias_map = mutation.alias_map or {}
    if qual in alias_map:
        table = str(alias_map[qual] or "")
        if table.startswith("(") or re.search(r"(?is)\bSELECT\b", table):
            return True
        return False
    # Never rewritten by phase2 (e.g. subquery alias ``A`` with no map entry).
    if len(qual) <= 3:
        return True
    return False


def _fold_control_branch_group(mutations, target_entity, target_column, prior=None):
    """Choose a procedural arm first, then apply its row-filtered updates.

    A false WHERE inside a selected IF arm must not execute the ELSE arm.
    Every arm starts from the same pre-branch column state.

    Each arm is pruned as soon as it is wrapped: an ``IF EXISTS(... WHERE A)``
    arm around ``UPDATE ... WHERE A`` otherwise yields ``IF(A) THEN(IF(A) ...)``,
    and later passes would copy that duplicate into every prior-value slot.
    """
    from dataclasses import replace
    base = prior if prior is not None else _column_ref(target_entity, target_column)
    arms = {}
    for mutation in mutations:
        arms.setdefault(mutation.control_branch_index, []).append(mutation)
    result = base
    for _, arm in sorted(arms.items(), key=lambda item: item[0] or 0, reverse=True):
        local = [replace(m, control_branch_group=None, control_branch_kind=None,
                         outer_condition=None) for m in arm]
        value = build_ast_from_mutations(local, target_entity, target_column)
        value = _substitute_prior_value(value, base, target_entity, target_column)
        if arm[0].control_branch_kind == "ELSE":
            result = value
            continue
        predicate = arm[0].outer_condition
        if not predicate:
            # Malformed IF arm with no condition: apply the writes unguarded
            # rather than poisoning the whole column with an unresolved-EXISTS
            # sentinel (which used to empty FLGDEG / DegReason formulas).
            result = value
            continue
        cond = parse_sql_expression_to_ast(predicate, default_entity=target_entity,
                                          target_column=target_column, as_condition=True)
        cond = _substitute_prior_in_guard(cond, base, target_entity, target_column)
        result = prune_redundant_ast({"type": "IF_THEN_ELSE", "condition": cond,
                                      "then_branch": value, "else_branch": result})
    return result


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


def _plain_entity_name(name: Any) -> str:
    """Upper-cased entity name without the global-temp ``##`` prefix."""
    text = str(name or "").strip().strip('"').upper()
    return text[2:] if text.startswith("##") else text


def _is_self_column_ref(node: dict[str, Any], entity: str, column: str) -> bool:
    if not isinstance(node, dict) or node.get("type") != "COLUMN_REF":
        return False
    relationship = node.get("relationship")
    # A hop from an entity to itself (``AccountCal`` → ``##AccountCal``) is the
    # entity's own column, not a join to another table.
    if relationship and _plain_entity_name(relationship) != _plain_entity_name(node.get("entity")):
        return False
    return (
        _plain_entity_name(node.get("entity")) == _plain_entity_name(entity)
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

    if re.search(r"(?is)\bROW_NUMBER\s*\(|\bOVER\s*\(", text):
        return {
            "type": "FUNCTION_CALL",
            "function_name": "__UNSUPPORTED_SQL__",
            "arguments": [],
            "_validation_error": "ROW_NUMBER()/OVER window syntax is unsupported and must be reviewed",
        }

    # Strip wrapping parentheses.
    while text.startswith("(") and text.endswith(")") and _balanced(text[1:-1]):
        text = text[1:-1].strip()

    # ``X * CASE ... END``: make the CASE an atom before any operator split.
    text = _parenthesize_top_level_case(text)

    # Bracketed column list — the documented ``MIN(<Col>, [<GroupbyColumns>])``
    # / ``MAX(...)`` second argument (app/grammar/fourx_grammar.lark's
    # ``list_literal``). Each item is a bare column name, rendered as a
    # quoted STRING token by the compiler (matching the grammar's
    # ``value: STRING | NUMBER`` for bracketed lists).
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        items = []
        for raw in split_csv_respecting_parens(inner):
            cleaned = raw.strip().strip('"').strip("'").rstrip(";").strip()
            name = bare_ident(cleaned)
            name = name.rstrip(";").strip()
            if name:
                items.append(name)
        return {"type": "LIST_LITERAL", "items": items}

    # CAST(expr AS type) / CONVERT(type, expr) → CONVERT(expr, type) (tried
    # early so a cast wrapping a CASE/DATEADD/arithmetic expression still
    # resolves that inner shape correctly).
    cast_node = _try_parse_cast(text, default_entity, target_column)
    if cast_node is not None:
        return cast_node

    # Explicit DATEADD → ADDDAY (day/week) or PERIOD (month/year/quarter).
    dateadd = _try_parse_dateadd(text, default_entity, target_column)
    if dateadd is not None:
        return dateadd

    string_agg = re.match(r"(?is)^STRING_AGG\s*\(\s*(?P<args>.*)\s*\)$", text.strip())
    if string_agg and _balanced(string_agg.group("args")):
        parts = _split_top_level(string_agg.group("args"), ",")
        if parts:
            return parse_sql_expression_to_ast(
                parts[0].strip(),
                default_entity=default_entity,
                target_column=target_column,
            )

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
        from app.derivation.v2.sql_text import exists_subquery_to_row_predicate

        pred, deps = exists_subquery_to_row_predicate(text)
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

    # Unit arguments are literals, not columns. 4X reverses SQL Server's
    # DATEDIFF / DATEPART argument order (see the bundled function reference).
    date_fn = re.match(r"(?is)^(DATEDIFF|DATEPART|DAY|MONTH|YEAR|EOMONTH)\s*\((.*)\)$", text)
    if date_fn and _balanced(date_fn.group(2)):
        name = date_fn.group(1).upper()
        parts = _split_top_level(date_fn.group(2), ",")
        parse = lambda value: parse_sql_expression_to_ast(value, default_entity=default_entity, target_column=target_column)
        literal = lambda value: {"type": "LITERAL", "value_type": "STRING", "value": value.strip().upper()}
        args = None
        if name == "DATEDIFF" and len(parts) == 3:
            args = [parse(parts[1]), parse(parts[2]), literal(parts[0])]
        elif name == "DATEPART" and len(parts) == 2:
            args = [parse(parts[1]), literal(parts[0])]
        elif name in {"DAY", "MONTH", "YEAR"} and len(parts) == 1:
            args = [parse(parts[0]), literal(name)]
            name = "DATEPART"
        elif name == "EOMONTH" and len(parts) == 1:
            args = [parse(parts[0])]
            name = "EOM"
        elif name == "EOMONTH" and len(parts) == 2:
            # EOMONTH(date, months) = EOM(DATEADD(MONTH, months, date)).
            return {
                "type": "FUNCTION_CALL",
                "function_name": "EOM",
                "arguments": [
                    {
                        "type": "FUNCTION_CALL",
                        "function_name": "PERIOD",
                        "arguments": [
                            {"type": "LITERAL", "value_type": "STRING", "value": "MONTH"},
                            parse(parts[1]),
                            parse(parts[0]),
                        ],
                    }
                ],
            }
        if args is not None:
            return {"type": "FUNCTION_CALL", "function_name": name, "arguments": args}

    # OR / AND (top-level)
    for op in (" OR ", " AND "):
        parts = _split_logical(text, op.strip())
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


    negate = re.match(r"(?is)^NOT\s*(\(.*\)|EXISTS\b.*)$", text)
    if negate:
        return {"type": "FUNCTION_CALL", "function_name": "NOT", "arguments": [
            parse_sql_expression_to_ast(negate.group(1), default_entity=default_entity,
                                        target_column=target_column, as_condition=True)
        ]}

    # The logical splitter preserves the AND belonging to BETWEEN.
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
            from app.derivation.v2.sql_text import (
                dim_asset_class_in_use_hop_only,
                in_subquery_to_row_predicate,
            )

            is_negated = bool(membership.group(2))
            lhs = membership.group(1).strip()
            deps = []
            short_only = dim_asset_class_in_use_hop_only(inner_list)
            node: dict[str, Any] | None = None
            if short_only:
                node = _dim_asset_class_hop_membership_ast(
                    lhs,
                    short_only,
                    default_entity,
                    target_column or "",
                )
                deps = extract_subquery_dependency_refs(inner_list)
            else:
                pred, deps = in_subquery_to_row_predicate(lhs, inner_list)
                if pred:
                    node = parse_sql_expression_to_ast(
                        pred,
                        default_entity=default_entity,
                        target_column=target_column,
                        as_condition=True,
                    )
            if node is not None:
                # `node` encodes the positive membership condition (a
                # matching row exists in the subquery). NOT IN is the
                # logical negation of that whole condition, not the same
                # predicate reused as-is -- reusing it silently drops the
                # negation and NOT IN/IN become indistinguishable. The
                # grammar has no bare NOT keyword, only the NOT(...)
                # function form, so wrap rather than string-prefix.
                if is_negated:
                    node = {
                        "type": "FUNCTION_CALL",
                        "function_name": "NOT",
                        "arguments": [node],
                    }
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
        # No fixed-literal pattern to map onto CONTAINS/etc, and the 4X
        # grammar has no LIKE token at all -- emitting BINARY_OP "NOT LIKE"
        # here used to compile to a string the grammar can't parse (a
        # cryptic Lark "No terminal matches ..." error instead of a clear
        # validation message). Raise the same unsupported-construct
        # sentinel used elsewhere (e.g. ROW_NUMBER/OVER) so this surfaces
        # as an honest, actionable error.
        return {
            "type": "FUNCTION_CALL",
            "function_name": "__UNSUPPORTED_SQL__",
            "arguments": [],
            "_validation_error": (
                f"NOT LIKE with a dynamic (non-literal) pattern has no 4X "
                f"equivalent: {text}"
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
            "type": "FUNCTION_CALL",
            "function_name": "__UNSUPPORTED_SQL__",
            "arguments": [],
            "_validation_error": (
                f"LIKE with a dynamic (non-literal) pattern has no 4X "
                f"equivalent: {text}"
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
    # The balance check matters: ``ISNULL(a,0)+ISNULL(b,0)`` also starts with
    # ``ISNULL(`` and ends with ``)``, but its first "(" closes mid-string, so
    # it is an addition of two calls (handled by the arithmetic split below),
    # not one call with the argument text ``a,0)+ISNULL(b,0``.
    isnull_fn = re.match(r"(?is)^(?:ISNULL|COALESCE|NVL|IFNULL)\s*\((.+)\)$", text)
    if isnull_fn and not _balanced(isnull_fn.group(1)):
        isnull_fn = None
    if isnull_fn:
        args = [
            parse_sql_expression_to_ast(
                a, default_entity=default_entity, target_column=target_column
            )
            for a in _split_top_level(isnull_fn.group(1), ",")
        ]
        return {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": args}

    # NULLIF(a, b) → IF(a == b) THEN NULL ELSE a
    nullif_fn = re.match(r"(?is)^NULLIF\s*\((.+)\)$", text)
    if nullif_fn and _balanced(nullif_fn.group(1)):
        nullif_args = _split_top_level(nullif_fn.group(1), ",")
        if len(nullif_args) == 2:
            left = parse_sql_expression_to_ast(
                nullif_args[0], default_entity=default_entity, target_column=target_column
            )
            right = parse_sql_expression_to_ast(
                nullif_args[1], default_entity=default_entity, target_column=target_column
            )
            else_branch = parse_sql_expression_to_ast(
                nullif_args[0], default_entity=default_entity, target_column=target_column
            )
            return {
                "type": "IF_THEN_ELSE",
                "condition": {
                    "type": "BINARY_OP",
                    "operator": "==",
                    "left": left,
                    "right": right,
                },
                "then_branch": {"type": "LITERAL", "value_type": "NULL", "value": None},
                "else_branch": else_branch,
            }

    # PRO.GETMINIMUMDATE(a, b, NULL) / [db].PRO.GETMINIMUMDATE(...) — scalar
    # least-date UDF. Drop NULL pads, then reuse MIN so aggregate-lowering
    # produces a row-level IF chain (0 set-aggregate leaks).
    min_date_fn = re.match(
        r"(?is)^(?:(?:\[[^\]]+\]|[A-Za-z_][\w]*)\.){0,3}GETMINIMUMDATE\s*\((?P<args>.*)\)$",
        text,
    )
    if min_date_fn and _balanced(min_date_fn.group("args")):
        raw_args = [
            part.strip()
            for part in _split_top_level(min_date_fn.group("args"), ",")
            if part.strip()
        ]
        kept = [part for part in raw_args if part.upper() not in {"NULL", "NONE"}]
        if not kept:
            return {"type": "LITERAL", "value_type": "NULL", "value": None}
        nodes = [
            parse_sql_expression_to_ast(
                part, default_entity=default_entity, target_column=target_column
            )
            for part in kept
        ]
        if len(nodes) == 1:
            return nodes[0]
        return {"type": "FUNCTION_CALL", "function_name": "MIN", "arguments": nodes}

    # LEAST(...)/GREATEST(...) (Oracle/MySQL) — semantically identical to
    # MIN/MAX applied to the same argument list; map onto those so the
    # platform-recognized function names are used instead of falling
    # through to a raw string literal.
    least_greatest_fn = re.match(r"(?is)^(?P<fn>LEAST|GREATEST)\s*\((?P<args>.*)\)$", text)
    if least_greatest_fn and _balanced(least_greatest_fn.group("args")):
        mapped_fn = "MIN" if least_greatest_fn.group("fn").upper() == "LEAST" else "MAX"
        args = [
            parse_sql_expression_to_ast(
                a, default_entity=default_entity, target_column=target_column
            )
            for a in _split_top_level(least_greatest_fn.group("args"), ",")
        ]
        return {"type": "FUNCTION_CALL", "function_name": mapped_fn, "arguments": args}

    # RIGHT(s, n) / LEFT(s, n) → SUBSTR (4X has no RIGHT/LEFT builtins).
    side_fn = re.match(r"(?is)^(?P<side>RIGHT|LEFT)\s*\((?P<args>.*)\)$", text)
    if side_fn and _balanced(side_fn.group("args")):
        parts = _split_top_level(side_fn.group("args"), ",")
        if len(parts) == 2:
            source = parse_sql_expression_to_ast(
                parts[0], default_entity=default_entity, target_column=target_column
            )
            length = parse_sql_expression_to_ast(
                parts[1], default_entity=default_entity, target_column=target_column
            )
            if side_fn.group("side").upper() == "LEFT":
                start = {"type": "LITERAL", "value_type": "NUMBER", "value": 1}
            else:
                len_call = {
                    "type": "FUNCTION_CALL",
                    "function_name": "LEN",
                    "arguments": [source],
                }
                start = {
                    "type": "BINARY_OP",
                    "operator": "+",
                    "left": {
                        "type": "BINARY_OP",
                        "operator": "-",
                        "left": len_call,
                        "right": length,
                    },
                    "right": {"type": "LITERAL", "value_type": "NUMBER", "value": 1},
                }
            return {
                "type": "FUNCTION_CALL",
                "function_name": "SUBSTR",
                "arguments": [source, start, length],
            }

    # T-SQL string/math functions → their 4X names (arguments keep their order;
    # 4X has a single TRIM covering LTRIM/RTRIM).
    renamed_fn = re.match(
        r"(?is)^(?P<fn>SUBSTRING|TRIM|LTRIM|RTRIM|REPLACE|FLOOR|CEILING)\s*\((?P<args>.*)\)$",
        text,
    )
    if renamed_fn and _balanced(renamed_fn.group("args")):
        fn = renamed_fn.group("fn").upper()
        parts = _split_top_level(renamed_fn.group("args"), ",")
        expected_arity = {"SUBSTRING": 3, "REPLACE": 3}.get(fn, 1)
        if len(parts) == expected_arity:
            return {
                "type": "FUNCTION_CALL",
                "function_name": _TSQL_TO_4X_FUNCTION_NAMES[fn],
                "arguments": [
                    parse_sql_expression_to_ast(
                        a, default_entity=default_entity, target_column=target_column
                    )
                    for a in parts
                ],
            }

    # CHOOSE(index, v1, v2, …) → IF(index == 1) THEN(v1) ELSEIF(index == 2) THEN(v2) … ELSE(NULL)
    choose = re.match(r"(?is)^CHOOSE\s*\((?P<args>.*)\)$", text)
    if choose and _balanced(choose.group("args")):
        choose_args = _split_top_level(choose.group("args"), ",")
        if len(choose_args) >= 2:
            index_node = parse_sql_expression_to_ast(
                choose_args[0], default_entity=default_entity, target_column=target_column
            )
            chosen: dict[str, Any] = {"type": "LITERAL", "value_type": "NULL", "value": None}
            for position in range(len(choose_args) - 1, 0, -1):
                chosen = {
                    "type": "IF_THEN_ELSE",
                    "condition": {
                        "type": "BINARY_OP",
                        "operator": "==",
                        "left": index_node,
                        "right": {"type": "LITERAL", "value_type": "NUMBER", "value": position},
                    },
                    "then_branch": parse_sql_expression_to_ast(
                        choose_args[position],
                        default_entity=default_entity,
                        target_column=target_column,
                    ),
                    "else_branch": chosen,
                }
            return chosen

    # Generic known functions: MIN/MAX/SUM/COUNT/ABS/ROUND/CONCAT/...
    gen_fn = re.match(
        r"(?is)^(?P<fn>MIN|MAX|SUM|COUNT|ABS|ROUND|CONCAT|DATEDIFF|LEN|UPPER|LOWER)\s*\((?P<args>.*)\)$",
        text,
    )
    if gen_fn and _balanced(gen_fn.group("args")):
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
        fn_name = gen_fn.group("fn").upper()
        if fn_name == "CONCAT":
            args = _null_safe_concat_arguments(args)
        return {
            "type": "FUNCTION_CALL",
            "function_name": fn_name,
            "arguments": args,
        }

    # ERROR_MESSAGE() / @@ERROR — map to @ErrorMessage variable token.
    if re.match(r"(?is)^(?:ERROR_MESSAGE|ERROR_NUMBER|ERROR_LINE)\s*\(\s*\)\s*$", text):
        return {"type": "VARIABLE_REF", "name": "@ErrorMessage"}

    # SQL + / - share precedence, as do * / /. Split at the LAST
    # top-level operator in each group to preserve left associativity.
    arithmetic = _split_arithmetic(text)
    if arithmetic:
        lhs, op, rhs = arithmetic
        left = parse_sql_expression_to_ast(lhs, default_entity=default_entity, target_column=target_column)
        right = parse_sql_expression_to_ast(rhs, default_entity=default_entity, target_column=target_column)
        if op == "+" and (
            _is_string_literal_node(left)
            or _is_string_literal_node(right)
            or _is_concat_call(left)
            or _is_concat_call(right)
        ):
            # T-SQL overloads "+" for string concatenation, but the 4X
            # grammar treats +/- as numeric-only -- CONCAT is the platform's
            # documented equivalent (see app/grammar/validator.py's
            # "numeric-only operators" check). ``'a' + ' ' + col`` parses as
            # ``('a' + ' ') + col``: an operand that is already a CONCAT is
            # flattened in so the chain stays one CONCAT, never ``CONCAT(..) + col``.
            flat: list[dict[str, Any]] = []
            for side in (left, right):
                flat.extend(side["arguments"] if _is_concat_call(side) else [side])
            return {
                "type": "FUNCTION_CALL",
                "function_name": "CONCAT",
                "arguments": _null_safe_concat_arguments(flat),
            }
        addday = _should_use_addday_for_arithmetic(left, right, target_column)
        if not addday and op == "+":
            addday = _should_use_addday_for_arithmetic(right, left, target_column)
            if addday:
                left, right = right, left
        if op in {"+", "-"} and addday:
            if op == "-":
                right = {
                    "type": "BINARY_OP",
                    "operator": "*",
                    "left": {"type": "LITERAL", "value_type": "NUMBER", "value": -1},
                    "right": right,
                }
            return {"type": "FUNCTION_CALL", "function_name": "ADDDAY", "arguments": [left, right]}
        return {"type": "BINARY_OP", "operator": op, "left": left, "right": right}

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
    if _is_single_quoted_string_literal(text):
        return _literal_from_sql_token(text)
    if text.upper() in {"NULL", "NONE"}:
        return _literal_from_sql_token(text)

    # Entity::Rel::Col or Entity::Col markers from phase2
    marker = re.match(
        r'^(?P<entity>[#A-Za-z_][#A-Za-z0-9_]*)::(?:(?P<rel>[#A-Za-z_][#A-Za-z0-9_]*)::)?(?P<col>[A-Za-z_][A-Za-z0-9_]*)$',
        text,
    )
    if marker:
        ent = marker.group("entity")
        rel = marker.group("rel")
        col = marker.group("col")
        if rel and default_entity:
            ent_norm = normalize_table_name(ent).upper()
            def_norm = normalize_table_name(default_entity).upper()
            if ent_norm == def_norm and should_collapse_target_join_hop(ent, rel, default_entity):
                return _column_ref(rel, col)
        return _column_ref(ent, col, relationship=rel)

    quoted_hop = re.match(
        r'^"(?P<e1>[^"]+)"\s*\.\s*"(?P<e2>[^"]+)"\s*\.\s*"(?P<col>[^"]+)"\s*$',
        text.strip(),
    )
    if quoted_hop:
        return _column_ref(
            quoted_hop.group("e1"),
            quoted_hop.group("col"),
            relationship=quoted_hop.group("e2"),
        )

    # T-SQL allows whitespace around the dot of a qualified name
    # (``A. SRCASSETCLASSALT_KEY``); collapse it so the path regex below sees
    # ``A.SRCASSETCLASSALT_KEY`` instead of falling through to a raw literal.
    if re.fullmatch(r"[#\[A-Za-z_][#\[\]\w]*(?:\s*\.\s*[\[\]#\w]+)+", text) and re.search(r"\s", text):
        text = re.sub(r"\s*\.\s*", ".", text)

    bracket_ref = _try_parse_sql_column_ref(text, default_entity)
    if bracket_ref is not None:
        return bracket_ref

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
    """Sentinel for an EXISTS/IN subquery that cannot be safely projected
    into a row-level predicate.

    This used to emit a grammar-valid ``1 == 1`` tautology -- syntactically
    fine, but semantically it makes the guard always true, silently
    broadening eligibility to every row (e.g. a correlated EXISTS against an
    empty table should mean "never applies", not "always applies"). Emit the
    same kind of compile-time-raising sentinel already used for value/
    predicate mixing instead, so the pipeline records a validation error and
    lowers confidence rather than shipping a plausible-looking wrong guard.
    """
    node: dict[str, Any] = {
        "type": "FUNCTION_CALL",
        "function_name": "__UNRESOLVED_SUBQUERY_PREDICATE__",
        "arguments": [],
        "_validation_error": (
            "EXISTS/IN subquery could not be projected into a row-level "
            "predicate; refusing to fall back to an always-true guard"
        ),
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


def _datatype_literal(sql_type: str) -> dict[str, Any]:
    """SQL type text (``decimal (18, 2)``) -> normalized STRING literal (``DECIMAL(18,2)``)."""
    normalized = re.sub(r"\s*([(),])\s*", r"\1", " ".join(sql_type.split())).upper()
    return {"type": "LITERAL", "value_type": "STRING", "value": normalized}


def _try_parse_cast(
    text: str,
    default_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """``CAST(expr AS type)`` / ``CONVERT(type, expr[, style])`` -> ``CONVERT(expr, type)``.

    4X's ``CONVERT(<FieldName>,<toWhichDataType>)`` takes the value first and
    the type second -- the reverse of T-SQL's CONVERT. The T-SQL style code
    (third argument) has no platform equivalent and is dropped.
    """
    m = re.match(r"(?is)^CAST\s*\((?P<inner>.*)\)\s*$", text)
    if m and _balanced(m.group("inner")):
        inner = m.group("inner")
        split_at = _find_top_level_as(inner)
        if split_at is None:
            return None
        expr_sql = inner[:split_at].strip()
        type_sql = inner[split_at + 2 :].strip()
        if not expr_sql or not type_sql:
            return None
    else:
        m = re.match(r"(?is)^CONVERT\s*\((?P<inner>.*)\)\s*$", text)
        if not m or not _balanced(m.group("inner")):
            return None
        parts = _split_top_level(m.group("inner"), ",")
        if len(parts) not in {2, 3}:
            return None
        type_sql, expr_sql = parts[0].strip(), parts[1].strip()
        if not expr_sql or not type_sql:
            return None
    return {
        "type": "FUNCTION_CALL",
        "function_name": "CONVERT",
        "arguments": [
            parse_sql_expression_to_ast(
                expr_sql, default_entity=default_entity, target_column=target_column
            ),
            _datatype_literal(type_sql),
        ],
    }


# SQL Server DATEADD day-equivalent units (dayofyear / weekday add the same
# number of calendar days as DAY). ``Y`` is dayofyear, not year.
_DATEADD_DAY_UNITS = frozenset({
    "DAY", "DAYS", "DD", "D",
    "DAYOFYEAR", "DY", "Y",
    "WEEKDAY", "DW", "W",
})
_DATEADD_WEEK_UNITS = frozenset({"WEEK", "WEEKS", "WK", "WW"})
# DATEADD calendar units → 4X PERIOD(TimeBasis, Offset, Date).
_DATEADD_PERIOD_UNITS = {
    "MONTH": "MONTH",
    "MONTHS": "MONTH",
    "MM": "MONTH",
    "M": "MONTH",
    "YEAR": "YEAR",
    "YEARS": "YEAR",
    "YY": "YEAR",
    "YYYY": "YEAR",
    "QUARTER": "QUARTER",
    "QUARTERS": "QUARTER",
    "QQ": "QUARTER",
    "Q": "QUARTER",
}


def _unsupported_sql_expression(message: str) -> dict[str, Any]:
    return {
        "type": "FUNCTION_CALL",
        "function_name": "__UNSUPPORTED_SQL__",
        "arguments": [],
        "_validation_error": message,
    }


def _try_parse_dateadd(
    text: str,
    default_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """Map DATEADD units onto ADDDAY (days/weeks) or PERIOD (month/year/quarter)."""
    match = None
    head = re.match(r"(?is)^DATEADD\s*\((?P<args>.*)\)$", text)
    if head and _balanced(head.group("args")):
        # Split on top-level commas only: the offset is often itself a call,
        # e.g. ``DATEADD(DD, -(ISNULL(DPD,0)-1), ReportDate)``.
        dateadd_args = _split_top_level(head.group("args"), ",")
        if len(dateadd_args) == 3:
            unit = dateadd_args[0].strip().upper()
            offset_sql = dateadd_args[1].strip()
            base_sql = dateadd_args[2].strip()
            parse = lambda value: parse_sql_expression_to_ast(
                value, default_entity=default_entity, target_column=target_column
            )
            if unit in _DATEADD_PERIOD_UNITS:
                return {
                    "type": "FUNCTION_CALL",
                    "function_name": "PERIOD",
                    "arguments": [
                        {
                            "type": "LITERAL",
                            "value_type": "STRING",
                            "value": _DATEADD_PERIOD_UNITS[unit],
                        },
                        parse(offset_sql),
                        parse(base_sql),
                    ],
                }
            if unit in _DATEADD_WEEK_UNITS:
                offset = parse(offset_sql)
                return {
                    "type": "FUNCTION_CALL",
                    "function_name": "ADDDAY",
                    "arguments": [
                        parse(base_sql),
                        {
                            "type": "BINARY_OP",
                            "operator": "*",
                            "left": offset,
                            "right": {"type": "LITERAL", "value_type": "NUMBER", "value": 7},
                        },
                    ],
                }
            if unit in _DATEADD_DAY_UNITS:
                match = {"offset": offset_sql, "base": base_sql}
            else:
                return _unsupported_sql_expression(
                    f"DATEADD({unit}, …) has no exact 4X equivalent; "
                    "day/week offsets map to ADDDAY and month/year/quarter to PERIOD"
                )
    if match is None:
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
        match["base"],
        default_entity=default_entity,
        target_column=target_column,
    )
    offset = parse_sql_expression_to_ast(
        match["offset"],
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


def _dim_asset_class_short_name_from_subquery(body: str) -> str | None:
    """Extract ``AssetClassShortName`` / ``AssetClassShortNameEnum`` literal filters."""
    match = re.search(
        r"(?is)AssetClassShortName(?:Enum)?\s*=\s*'([^']+)'",
        body or "",
    )
    if not match:
        return None
    return bare_ident(match.group(1)).upper()


_DIM_ASSET_CLASS_IN_RE = re.compile(
    r"(?is)(?P<lhs>[#A-Za-z_][\w\.]*)\s+IN\s*\(\s*SELECT\b.+?\bDimAssetClass\b.+?"
    r"AssetClassShortName\s*=\s*'(?P<short>[^']+)'",
)


def _dim_asset_class_hop_membership_ast(
    lhs: str,
    short_name: str,
    default_entity: str,
    target_column: str,
) -> dict[str, Any]:
    """``lhs IN (SELECT … DimAssetClass … 'SUB')`` → ``lhs == Entity.SUB.AssetClassAlt_Key``."""
    lhs_ast = parse_sql_expression_to_ast(
        lhs.strip(),
        default_entity=default_entity,
        target_column=target_column,
        as_condition=True,
    )
    hop = _column_ref(default_entity, "AssetClassAlt_Key", relationship=short_name)
    return {
        "type": "BINARY_OP",
        "operator": "==",
        "left": lhs_ast,
        "right": hop,
    }


def _ast_has_dim_short_hop(
    node: Any,
    default_entity: str,
    short_name: str,
) -> bool:
    """True when ``node`` already encodes ``… == <entity>.<SHORT>.AssetClassAlt_Key``."""
    target = normalize_table_name(default_entity).upper()
    short = (short_name or "").upper()
    if not isinstance(node, dict):
        return False
    if node.get("type") == "BINARY_OP" and str(node.get("operator") or "") == "==":
        right = node.get("right")
        if isinstance(right, dict) and right.get("type") == "COLUMN_REF":
            ent = normalize_table_name(str(right.get("entity") or "")).upper()
            rel = str(right.get("relationship") or "").upper()
            col = str(right.get("column") or "").upper()
            if ent == target and rel == short and col == "ASSETCLASSALT_KEY":
                return True
    for value in node.values():
        if isinstance(value, dict) and _ast_has_dim_short_hop(value, default_entity, short_name):
            return True
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and _ast_has_dim_short_hop(
                    item, default_entity, short_name
                ):
                    return True
    return False


def _augment_guard_from_effective_sql(
    cond_sql: str,
    node: dict[str, Any] | None,
    default_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """Re-attach DimAssetClass ``IN`` hops when the SQL text still has them."""
    if not cond_sql:
        return node
    match = _DIM_ASSET_CLASS_IN_RE.search(cond_sql)
    if not match:
        return node
    short = bare_ident(match.group("short")).upper()
    if node is not None and _ast_has_dim_short_hop(node, default_entity, short):
        return node
    lhs = match.group("lhs").strip()
    if "." in lhs:
        lhs = lhs.rsplit(".", 1)[-1]
    hop = _dim_asset_class_hop_membership_ast(
        lhs, short, default_entity, target_column
    )
    if node is None:
        return hop
    if _ast_has_dim_short_hop(node, default_entity, short):
        return node
    return {
        "type": "BINARY_OP",
        "operator": "AND",
        "left": node,
        "right": hop,
    }


def _augment_dim_asset_class_in_membership(
    node: dict[str, Any] | None,
    inner_list: str,
    lhs: str,
    default_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """Legacy entry: prefer hop-only membership for DimAssetClass ``IN`` lists."""
    if not re.search(r"(?is)\bDimAssetClass\b", inner_list or ""):
        return node
    short_name = _dim_asset_class_short_name_from_subquery(inner_list)
    if not short_name:
        return node
    if node is not None and _ast_has_dim_short_hop(node, default_entity, short_name):
        return node
    hop = _dim_asset_class_hop_membership_ast(
        lhs, short_name, default_entity, target_column
    )
    if node is None:
        return hop
    if _ast_has_dim_short_hop(node, default_entity, short_name):
        return node
    return {
        "type": "BINARY_OP",
        "operator": "AND",
        "left": node,
        "right": hop,
    }


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
    # ``SELECT TOP 1 col FROM …`` / ``SELECT DISTINCT col FROM …`` are still
    # scalar lookups; strip the modifier so the existing matchers fire.
    body = re.sub(
        r"(?is)^SELECT\s+(?:DISTINCT\s+)?(?:TOP\s*\(?\s*\d+\s*\)?\s+(?:PERCENT\s+)?)?",
        "SELECT ",
        body,
        count=1,
    )

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
        r"(?is)^SELECT\s+(?:(?:\[?(?P<prefix>[^.\]]+)\]?\.)?\[?(?P<col>[^\]]+)\]?)"
        r"\s+FROM\s+(?P<table>\[?#?#?[A-Za-z_][A-Za-z0-9_]*\]?)"
        r"(?:\s+(?:AS\s+)?(?!WHERE\b|GROUP\b|ORDER\b|JOIN\b|HAVING\b)[A-Za-z_][A-Za-z0-9_]*)?"
        r"(?:\s+WHERE\s.*)?$",
        body,
    )
    if plain_m and not re.search(r"(?is)\bGROUP\s+BY\b|\bJOIN\b", body):
        table = normalize_table_name(plain_m.group("table"))
        column = bare_ident(plain_m.group("col"))
        short_name = _dim_asset_class_short_name_from_subquery(body)
        if short_name and "ASSETCLASS" in table.upper():
            # PRO asset-class procedures pick distinct keys via
            # ``WHERE AssetClassShortName='SUB'`` (etc.). Encode the short
            # name as a relationship hop on the derivation target so CASE
            # branches do not collapse to one undifferentiated key column.
            return {
                "type": "COLUMN_REF",
                "entity": normalize_table_name(default_entity),
                "relationship": short_name,
                "column": column,
                "_dependency_refs": extract_subquery_dependency_refs(body),
            }
        return {
            "type": "COLUMN_REF",
            "entity": table,
            "relationship": None,
            "column": column,
            "_dependency_refs": extract_subquery_dependency_refs(body),
        }

    return None


def _is_string_literal_node(node: dict[str, Any]) -> bool:
    return (
        isinstance(node, dict)
        and node.get("type") == "LITERAL"
        and node.get("value_type") == "STRING"
    )


def _should_use_addday_for_arithmetic(
    left: dict[str, Any],
    right: dict[str, Any],
    target_column: str,
) -> bool:
    """True only for date ± numeric-day arithmetic — never for counters."""
    if not _node_looks_numeric(right):
        return False
    # Date column + day offset stays ADDDAY even when the *target* column is an
    # integer key (e.g. SysAssetClassAlt_Key aging CASE on SysNPA_Dt).
    if _node_looks_date_valued(left):
        return True
    if _is_numeric_column_name(target_column):
        return False
    if _is_date_like_column_name(target_column):
        return True
    return False


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
        if func == "CONVERT":
            return _convert_target_base_type(node) in _NUMERIC_SQL_TYPES
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
        if func in {"SOM", "EOM", "ADDDAY", "TODATE", "DATE", "PERIOD"}:
            return True
        if func == "COALESCE":
            args = node.get("arguments") or []
            return bool(args) and _node_looks_date_valued(args[0])
        if func == "CONVERT":
            return _convert_target_base_type(node) in _DATE_SQL_TYPES
    return False


def _convert_target_base_type(node: dict[str, Any]) -> str:
    """``CONVERT(expr, "DECIMAL(18,2)")`` -> ``"DECIMAL"`` (empty when not a literal type)."""
    args = node.get("arguments") or []
    if len(args) < 2 or not _is_string_literal_node(args[1]):
        return ""
    return str(args[1].get("value") or "").split("(", 1)[0].strip().upper()


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
    if _node_looks_date_valued(base):
        return False
    if _is_date_like_column_name(target_column):
        return False
    if _is_numeric_column_name(target_column):
        return True
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



def _case_end_index(text: str, start: int) -> int | None:
    """Index just past the ``END`` closing the ``CASE`` that begins at ``start``."""
    n = len(text)
    i = start + 4
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
        elif (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            if re.match(r"(?is)CASE\b", text[i:i + 5]):
                depth += 1
                i += 4
                continue
            if re.match(r"(?is)END\b", text[i:i + 4]):
                depth -= 1
                if depth == 0:
                    return i + 3
                i += 3
                continue
        i += 1
    return None


def _parenthesize_top_level_case(text: str) -> str:
    """Wrap every top-level ``CASE ... END`` that is not the whole expression
    in parentheses (``X * CASE WHEN a=b THEN 1 ELSE c/100 END`` ->
    ``X * (CASE ... END)``).

    The comparison / AND-OR / arithmetic splitters only track parentheses, so
    an operator inside an un-parenthesised CASE (the ``/`` or ``=`` above)
    would otherwise be split at the wrong level and corrupt the expression.
    """
    if not re.search(r"(?i)\bCASE\b", text):
        return text
    out: list[str] = []
    n = len(text)
    i = 0
    depth = 0
    in_single = False
    while i < n:
        ch = text[i]
        if in_single:
            out.append(ch)
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    out.append(text[i + 1])
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif (
            depth == 0
            and re.match(r"(?is)CASE\b", text[i:i + 5])
            and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"))
        ):
            end = _case_end_index(text, i)
            if end is not None:
                segment = text[i:end]
                if i == 0 and end == n:
                    return text
                out.append(f"({segment})")
                i = end
                continue
        out.append(ch)
        i += 1
    return "".join(out)


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


def _try_parse_sql_column_ref(text: str, default_entity: str) -> dict[str, Any] | None:
    """Parse ``Entity.[Spaced Name]``, ``[Spaced Name]``, or ``Entity.Col`` refs."""
    raw = (text or "").strip()
    if not raw:
        return None
    ent_bracket = re.match(
        r"^(?P<ent>[#A-Za-z_][#A-Za-z0-9_]*)\.\[(?P<col>[^\]]+)\]\s*$",
        raw,
    )
    if ent_bracket:
        return _column_ref(
            ent_bracket.group("ent"),
            bare_ident(ent_bracket.group("col")),
        )
    only_bracket = re.match(r"^\[(?P<col>[^\]]+)\]\s*$", raw)
    if only_bracket:
        return _column_ref(default_entity, bare_ident(only_bracket.group("col")))
    return None


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


def _expression_tokens(text):
    return re.finditer(r"'(?:''|[^'])*'|\[(?:[^\]])*\]|[A-Za-z_][A-Za-z0-9_]*|[()+*/-]", text)


def _split_logical(text, operator):
    depth = 0
    between = False
    cuts = []
    for token in _expression_tokens(text):
        word = token.group().upper()
        if word == "(": depth += 1
        elif word == ")": depth -= 1
        elif depth == 0:
            if word == "BETWEEN": between = True
            elif word == "AND" and between: between = False
            elif word == operator: cuts.append((token.start(), token.end()))
    parts = []; start = 0
    for a, b in cuts:
        parts.append(text[start:a]); start = b
    return parts + [text[start:]]


def _split_arithmetic(text):
    depth = 0; choices = []
    for token in _expression_tokens(text):
        word = token.group()
        if word == "(": depth += 1
        elif word == ")": depth -= 1
        elif depth == 0 and word in {"+", "-", "*", "/"}:
            left = text[:token.start()].rstrip()
            if left and left[-1] not in "+-*/(<>=,":
                choices.append((token.start(), word))
    for operators in ({"+", "-"}, {"*", "/"}):
        matches = [(i, op) for i, op in choices if op in operators]
        if matches:
            i, op = matches[-1]
            return text[:i], op, text[i+1:]
    return None


def _literal_from_sql_token(token: str) -> dict[str, Any]:
    text = token.strip()
    if text.upper() in {"NULL", "NONE"}:
        return {"type": "LITERAL", "value_type": "NULL", "value": None}
    if re.match(r"^@[A-Za-z_][A-Za-z0-9_]*$", text):
        return {"type": "VARIABLE_REF", "name": text}
    if _is_single_quoted_string_literal(text) and text[0] != '"':
        inner = text[2:-1] if text.upper().startswith("N'") else text[1:-1]
        inner = inner.replace("''", "'")
        return {"type": "LITERAL", "value_type": "STRING", "value": inner}
    if _is_single_quoted_string_literal(text) and text[0] == '"':
        return {"type": "LITERAL", "value_type": "STRING", "value": text[1:-1]}
    if re.match(r"^-?\d+(\.\d+)?$", text):
        number: Any = float(text) if "." in text else int(text)
        return {"type": "LITERAL", "value_type": "NUMBER", "value": number}
    # Bare word → string literal (e.g. STANDARD without quotes in some dialects)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text):
        return {"type": "LITERAL", "value_type": "STRING", "value": text}
    return {"type": "FUNCTION_CALL", "function_name": "__UNSUPPORTED_SQL__",
            "arguments": [], "_validation_error": f"Untranslated SQL expression: {text}"}


def _try_parse_percent_wildcard_concat(
    pattern_sql: str,
    *,
    default_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """``'%' + expr + '%'`` → substring needle expression for CONTAINS."""
    match = re.match(
        r"(?is)^'%'\s*\+\s*(?P<mid>.+?)\s*\+\s*'%'\s*$",
        (pattern_sql or "").strip(),
    )
    if not match:
        return None
    return parse_sql_expression_to_ast(
        match.group("mid").strip(),
        default_entity=default_entity,
        target_column=target_column,
    )


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
        wildcard_mid = _try_parse_percent_wildcard_concat(
            pattern_sql, default_entity=default_entity, target_column=target_column
        )
        if wildcard_mid is not None:
            lhs_node = parse_sql_expression_to_ast(
                lhs_sql, default_entity=default_entity, target_column=target_column
            )
            operator = "DOESNOTCONTAINS" if negate else "CONTAINS"
            return {
                "type": "MEMBERSHIP_OP",
                "operator": operator,
                "column": lhs_node,
                "values": [wildcard_mid],
            }
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
    pattern = re.compile(rf"\b(?:{keyword_pattern})\b", re.I | re.S)
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
        if ch == "(" or ch == "[":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")" or ch == "]":
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
