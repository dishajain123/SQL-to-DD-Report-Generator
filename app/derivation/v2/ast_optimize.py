"""Post-prune AST optimizations: CSE, IF-arm merging, row-level aggregate lowering."""
from __future__ import annotations

import contextvars
import json
import re
from typing import Any

from app.derivation.v2.phase2_mutation_folder import (
    _ast_signature,
    _flatten_or_conditions,
    _is_zero_literal,
    _rebuild_or_chain,
    _try_dedupe_coalesce_negative_clamp,
    _try_flatten_join_assignment_zero_default,
    guard_conjunct_signature,
    guard_formula_signature,
)
from app.derivation.v2.sql_text import (
    bare_ident,
    is_ephemeral_sql_alias,
    normalize_table_name,
)

_MIN_CSE_DEPTH = 3
_MIN_CSE_COUNT = 2
_SET_AGG_FUNCS = frozenset({"MIN", "MAX", "SUM", "COUNT"})
# Leave headroom below the grammar validator's hard 8,000-character cap.
FORMULA_CHAR_BUDGET = 7900
_COMPILE_BLOCKER_FUNCTIONS = frozenset(
    {
        "__UNSUPPORTED_SQL__",
        "__VALUE_PREDICATE_MIXING__",
        "__UNRESOLVED_SUBQUERY_PREDICATE__",
    }
)
_COMPARISON_OPS = frozenset({">", "<", ">=", "<=", "==", "!="})


def optimize_expression_ast(
    node: dict[str, Any] | None,
    *,
    target_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """Shrink folded ASTs without changing UPDATE semantics.

    - Lowers ``MIN``/``MAX``/``SUM`` into row-level ``IF`` / bare expressions.
    - Merges ``ELSEIF`` arms that assign the same value (``Cond_A OR Cond_B``).
    - Drops later CASE/IF disjuncts already covered by an earlier first-match arm.
    - Factors OR-of-AND guards sharing a remainder (``(col==a AND R) OR (col==b AND R)``).
    - Replaces repeated copies of an ``IF``'s ``else_branch`` backbone (seen
      ≥ 2×) with a pass-through column reference in value slots.
    """
    if not isinstance(node, dict):
        return node
    optimized = enforce_later_update_precedence(node)
    optimized = _collapse_intermediate_entity_hop_refs(optimized)
    if bare_ident(target_column).upper() == "ISCHANGED":
        optimized = _flatten_ischanged_mismatch_ladder(optimized)
    optimized = _rewrite_date_plus_to_addday(optimized)
    optimized = _dedupe_and_guard_tree(optimized)
    # Join+zero-default and negative-clamp flattening must run before
    # comparisons are distributed over IF values; otherwise
    # ``(IF(join) THEN src ELSE 0) == 0`` / ``(IF(deriv) < 0)`` become nested
    # IF conditions and the flatten matchers miss. Flatten again after
    # distribution (phase3 calls optimize before prune).
    optimized = _flatten_join_zero_defaults(optimized)
    optimized = _flatten_negative_clamps(optimized)
    optimized = _collapse_nested_cap_with_shared_else(optimized)
    optimized = _simplify_s06_dpd_nocredit_oda_cap(
        optimized, target_entity=target_entity, target_column=target_column
    )
    optimized = _strip_redundant_coalesce(optimized)
    optimized = _distribute_if_over_comparisons(optimized)
    optimized = _flatten_join_zero_defaults(optimized)
    optimized = _flatten_negative_clamps(optimized)
    optimized = _collapse_degenerate_if_branches(optimized)
    optimized = _normalize_right_leaning_if_chain(optimized)
    optimized = _collapse_duplicate_if_then_arms(optimized)
    optimized = _unwrap_nested_identical_guard_ifs(optimized)
    optimized = _drop_identical_condition_elseif_arms(optimized)
    optimized = _compact_boolean_guards(optimized)
    optimized = _lower_set_aggregates(optimized)
    optimized = _collapse_duplicate_if_then_arms(optimized)
    optimized = _unwrap_nested_identical_guard_ifs(optimized)
    optimized = _drop_identical_condition_elseif_arms(optimized)
    optimized = _compact_boolean_guards(optimized)
    backbone_sigs = _collect_else_backbone_signatures(optimized)
    counts: dict[Any, int] = {}
    depths: dict[Any, int] = {}
    _collect_subtree_stats(optimized, counts, depths)
    repeated_if_sigs = {
        sig
        for sig, count in counts.items()
        if count >= _MIN_CSE_COUNT
        and depths.get(sig, 0) >= _MIN_CSE_DEPTH
        and _signature_is_if_then_else(sig)
    }
    active = {
        sig
        for sig in (backbone_sigs | repeated_if_sigs)
        if counts.get(sig, 0) >= _MIN_CSE_COUNT and depths.get(sig, 0) >= _MIN_CSE_DEPTH
    }
    if not active:
        optimized = _collapse_degenerate_if_branches(optimized)
        optimized = _defer_customer_rollup_arms(optimized, target_column)
        return _hygiene_after_distribute(
            optimized, target_entity=target_entity, target_column=target_column
        )
    # A repeated arithmetic *value* (``USEDRV * ProvPerSecured`` in a clamp guard
    # and its THEN) is a fresh computation, not the column's prior-value
    # backbone: replacing it with a self-reference drops the calculation.
    keep_token = _KEEP_ARITHMETIC_VALUES.set(True)
    try:
        optimized = _replace_backbone_duplicates(
            optimized, target_entity, target_column, active
        )
    finally:
        _KEEP_ARITHMETIC_VALUES.reset(keep_token)
    optimized = _collapse_degenerate_if_branches(optimized)
    optimized = _defer_customer_rollup_arms(optimized, target_column)
    return _hygiene_after_distribute(
        optimized, target_entity=target_entity, target_column=target_column
    )


def _should_collapse_entity_relationship_hop(entity: str, relationship: str) -> bool:
    """True when ``entity.relationship.col`` should compile as ``relationship.col`` (2-level 4X)."""
    if not relationship:
        return False
    if is_ephemeral_sql_alias(relationship):
        return False
    ent_n = normalize_table_name(entity).upper().lstrip("#")
    rel_n = normalize_table_name(relationship).upper().lstrip("#")
    if not ent_n or not rel_n:
        return False
    return ent_n != rel_n


def _collapse_intermediate_entity_hop_refs(node: Any) -> Any:
    """``EntityA.EntityB.Col`` → ``EntityB.Col`` for physical join targets (not ``A``/``STD`` hops)."""
    if not isinstance(node, dict):
        return node
    if node.get("type") == "COLUMN_REF":
        rel = node.get("relationship")
        ent = str(node.get("entity") or "")
        if rel and _should_collapse_entity_relationship_hop(ent, str(rel)):
            collapsed: dict[str, Any] = {
                "type": "COLUMN_REF",
                "entity": rel,
                "column": node.get("column"),
            }
            return collapsed
    return _map_children(node, _collapse_intermediate_entity_hop_refs)


def _is_customer_sysnpa_copy(node: Any) -> bool:
    """THEN value that is exactly the customer ``SysNPA_Dt`` column (write-back copy)."""
    return (
        isinstance(node, dict)
        and node.get("type") == "COLUMN_REF"
        and _references_customer_sysnpa_dt(node)
    )


def _defer_customer_rollup_arms(node: Any, target_column: str) -> Any:
    """Place the customer ``SysNPA_Dt`` write-back arm after account-level arms.

    The account's own DPD / PUI / restructure dates are what the customer
    ``MIN`` roll-up is built from, so the roll-up copy must not be checked
    before them (it would shadow every one of those arms for a degraded
    account). The arm keeps its guard and moves directly ahead of the trailing
    ``NULL`` clean-up arm(s) / ``ELSE``. Skipped when the target is
    ``SysNPA_Dt`` itself (its own prior-value reads are not a write-back).
    """
    if bare_ident(target_column or "").upper() == "SYSNPA_DT":
        return node
    return _defer_customer_rollup_arms_impl(node)


def _defer_customer_rollup_arms_impl(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    if node.get("type") == "IF_THEN_ELSE":
        arms, default = _flatten_if_elseif_chain(node)
        deferred = [a for a in arms if _is_customer_sysnpa_copy(a[1])]
        if deferred:
            kept = [a for a in arms if not _is_customer_sysnpa_copy(a[1])]
            cut = next(
                (i for i, (_c, t) in enumerate(kept) if _is_null_literal_ast(t)),
                len(kept),
            )
            reordered = kept[:cut] + deferred + kept[cut:]
            if any(a is not b for a, b in zip(reordered, arms)):
                node = _rebuild_if_elseif_chain(reordered, default)
    return _map_children(node, _defer_customer_rollup_arms_impl)


def _try_collapse_nested_cap_with_shared_else(node: dict[str, Any]) -> dict[str, Any] | None:
    """``IF(g) THEN IF(c) THEN cap ELSE base ELSE base`` → ``IF(AND(g,c)) THEN cap ELSE base``.

    S06 ``DPD_NoCredit`` ODA clamp UPDATE folds as an outer procedural guard around
    the inner ``>= RefPeriod*`` threshold CASE already emitted by ``#TEMPTABLEDPD``.
    When both ELSE arms are the same threshold tree, merge guards instead of nesting
    a second identical comparison in the THEN branch.
    """
    if node.get("type") != "IF_THEN_ELSE":
        return None
    outer_else = node.get("else_branch")
    then_b = node.get("then_branch")
    if not isinstance(then_b, dict) or then_b.get("type") != "IF_THEN_ELSE":
        return None
    if not isinstance(outer_else, dict):
        return None
    inner_else = then_b.get("else_branch")
    if _value_tree_signature(inner_else) != _value_tree_signature(outer_else):
        return None
    outer_cond = node.get("condition")
    inner_cond = then_b.get("condition")
    if not isinstance(outer_cond, dict) or not isinstance(inner_cond, dict):
        return None
    return {
        "type": "IF_THEN_ELSE",
        "condition": _merge_and_conditions(outer_cond, inner_cond),
        "then_branch": then_b.get("then_branch"),
        "else_branch": outer_else,
    }


def _value_tree_signature(node: Any) -> Any:
    """Subtree key for value positions; ignores staging vs target entity aliases."""
    if not isinstance(node, dict):
        return node
    t = node.get("type")
    if t == "COLUMN_REF":
        rel = node.get("relationship")
        return (
            "COLUMN_REF",
            str(node.get("column") or "").upper(),
            str(rel or "").upper() if rel else None,
        )
    if t == "LITERAL":
        return ("LITERAL", node.get("value_type"), node.get("value"))
    if t == "VARIABLE_REF":
        return ("VAR", str(node.get("name") or "").upper())
    if t == "FUNCTION_CALL":
        args = node.get("arguments") or []
        return (
            "FN",
            str(node.get("function_name") or "").upper(),
            tuple(_value_tree_signature(a) for a in args),
        )
    if t == "BINARY_OP":
        return (
            "BIN",
            str(node.get("operator") or "").upper(),
            _value_tree_signature(node.get("left")),
            _value_tree_signature(node.get("right")),
        )
    if t == "IF_THEN_ELSE":
        cond = node.get("condition")
        return (
            "IF",
            guard_formula_signature(cond) if isinstance(cond, dict) else cond,
            _value_tree_signature(node.get("then_branch")),
            _value_tree_signature(node.get("else_branch")),
        )
    return ("OTHER", _ast_signature(node))


def _merge_and_conditions(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    parts = _dedupe_and_list(_flatten_and_conditions(left) + _flatten_and_conditions(right))
    return _rebuild_and_chain(parts)


def _collapse_nested_cap_with_shared_else(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    node = _map_children(node, _collapse_nested_cap_with_shared_else)
    if node.get("type") != "IF_THEN_ELSE":
        return node
    collapsed = _try_collapse_nested_cap_with_shared_else(node)
    if collapsed is not None:
        return _collapse_nested_cap_with_shared_else(collapsed)
    return node


_DPD_REF_OR_MARKERS = frozenset(
    {
        "REFPERIODINTSERVICE",
        "REFPERIODNOCREDIT",
        "REFPERIODOVERDRAWN",
        "REFPERIODOVERDUE",
        "REFPERIODREVIEW",
        "REFPERIODSTKSTATEMENT",
    }
)


def _comparison_operand_column(node: Any) -> str | None:
    if not isinstance(node, dict):
        return None
    if node.get("type") == "COLUMN_REF":
        return str(node.get("column") or "").upper()
    if node.get("type") == "FUNCTION_CALL" and str(node.get("function_name") or "").upper() == "COALESCE":
        args = node.get("arguments") or []
        if args and isinstance(args[0], dict):
            return _comparison_operand_column(args[0])
    return None


def _is_scheme_type_eq_oda(node: dict[str, Any]) -> bool:
    if node.get("type") != "BINARY_OP" or str(node.get("operator") or "").strip() not in _EQ_OPS:
        return False
    left = node.get("left")
    right = node.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    col: dict[str, Any] | None = None
    lit: dict[str, Any] | None = None
    if left.get("type") == "COLUMN_REF" and right.get("type") == "LITERAL":
        col, lit = left, right
    elif right.get("type") == "COLUMN_REF" and left.get("type") == "LITERAL":
        col, lit = right, left
    if col is None or lit is None:
        return False
    if str(col.get("column") or "").upper() != "SCHEMETYPE":
        return False
    return str(lit.get("value") or "").upper() == "ODA"


def _condition_has_oda_scheme_guard(cond: dict[str, Any]) -> bool:
    for part in _flatten_and_conditions(cond):
        if _is_scheme_type_eq_oda(part):
            return True
        for sub in _walk_ast(part):
            if isinstance(sub, dict) and _is_scheme_type_eq_oda(sub):
                return True
    return False


def _condition_is_temp_dpd_inclusion_or(cond: dict[str, Any]) -> bool:
    parts = _flatten_or_conditions(cond)
    if len(parts) < 4:
        return False
    ref_hits: set[str] = set()
    for part in parts:
        if part.get("type") != "BINARY_OP" or str(part.get("operator") or "").strip() not in _ORDER_OPS:
            continue
        left_col = _comparison_operand_column(part.get("left"))
        right_col = _comparison_operand_column(part.get("right"))
        if not left_col or not right_col:
            continue
        if left_col.startswith("DPD_") and right_col in _DPD_REF_OR_MARKERS:
            ref_hits.add(right_col)
        elif right_col.startswith("DPD_") and left_col in _DPD_REF_OR_MARKERS:
            ref_hits.add(left_col)
    return len(ref_hits) >= 4


def _is_nocredit_threshold_if(node: dict[str, Any]) -> bool:
    if node.get("type") != "IF_THEN_ELSE":
        return False
    if not _is_zero_literal(node.get("else_branch")):
        return False
    cond = node.get("condition")
    if (
        not isinstance(cond, dict)
        or cond.get("type") != "BINARY_OP"
        or str(cond.get("operator") or "").strip() not in _ORDER_OPS
    ):
        return False
    left_col = _comparison_operand_column(cond.get("left"))
    right_col = _comparison_operand_column(cond.get("right"))
    if {left_col, right_col} != {"DPD_NOCREDIT", "REFPERIODNOCREDIT"}:
        return False
    then_b = node.get("then_branch")
    return _comparison_operand_column(then_b) == "DPD_NOCREDIT"


def _nocredit_gte_condition_from_threshold(threshold: dict[str, Any]) -> dict[str, Any] | None:
    cond = threshold.get("condition")
    return cond if isinstance(cond, dict) else None


def _refperiod_nocredit_value_from_threshold(threshold: dict[str, Any]) -> dict[str, Any] | None:
    cond = threshold.get("condition")
    if not isinstance(cond, dict):
        return None
    for side in (cond.get("left"), cond.get("right")):
        if _comparison_operand_column(side) == "REFPERIODNOCREDIT" and isinstance(side, dict):
            if side.get("type") == "COLUMN_REF":
                return side
            if side.get("type") == "FUNCTION_CALL":
                args = side.get("arguments") or []
                if args and isinstance(args[0], dict) and args[0].get("type") == "COLUMN_REF":
                    return args[0]
    return None


def _try_merge_s06_dpd_nocredit_oda_cap(
    node: dict[str, Any],
    *,
    target_entity: str,
    target_column: str,
) -> dict[str, Any] | None:
    """Fold S06 ODA clamp into the ``#TEMPTABLEDPD`` row-inclusion arm (sibling DPD shape)."""
    if bare_ident(target_column).upper() != "DPD_NOCREDIT":
        return None
    ent = normalize_table_name(target_entity or "").upper().lstrip("#")
    if ent not in {"TEMPTABLEDPD", "ACCOUNTCAL"}:
        return None
    arms, default = _flatten_if_elseif_chain(node)
    if len(arms) < 2:
        return None
    oda_idx: int | None = None
    or_idx: int | None = None
    for i, (cond, _then_b) in enumerate(arms):
        if not isinstance(cond, dict):
            continue
        if _condition_has_oda_scheme_guard(cond):
            oda_idx = i
        if _condition_is_temp_dpd_inclusion_or(cond):
            or_idx = i
    if oda_idx is None or or_idx is None or oda_idx == or_idx:
        return None
    or_cond, or_then = arms[or_idx]
    threshold = or_then if _is_nocredit_threshold_if(or_then) else None
    if threshold is None:
        threshold = next(
            (sub for sub in _walk_ast(or_then) if _is_nocredit_threshold_if(sub)),
            None,
        )
    if threshold is None:
        oda_then = arms[oda_idx][1]
        threshold = next(
            (sub for sub in _walk_ast(oda_then) if _is_nocredit_threshold_if(sub)),
            None,
        )
    if threshold is None:
        return None
    gte_cond = _nocredit_gte_condition_from_threshold(threshold)
    ref_val = _refperiod_nocredit_value_from_threshold(threshold)
    if gte_cond is None or ref_val is None:
        return None
    oda_cond, _ = arms[oda_idx]
    merged_then: dict[str, Any] = {
        "type": "IF_THEN_ELSE",
        "condition": _merge_and_conditions(oda_cond, gte_cond),
        "then_branch": ref_val,
        "else_branch": threshold,
    }
    kept: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i, (cond, then_b) in enumerate(arms):
        if i == oda_idx:
            continue
        if i == or_idx:
            kept.append((or_cond, merged_then))
        else:
            kept.append((cond, then_b))
    return _rebuild_if_elseif_chain(kept, default)


def _simplify_s06_dpd_nocredit_oda_cap(
    node: Any,
    *,
    target_entity: str,
    target_column: str,
) -> Any:
    if not isinstance(node, dict):
        return node
    if node.get("type") == "IF_THEN_ELSE":
        merged = _try_merge_s06_dpd_nocredit_oda_cap(
            node, target_entity=target_entity, target_column=target_column
        )
        if merged is not None:
            node = merged
    return _map_children(
        node,
        lambda child: _simplify_s06_dpd_nocredit_oda_cap(
            child, target_entity=target_entity, target_column=target_column
        ),
    )


def _gte_pair_key(node: dict[str, Any]) -> tuple[str, str, str] | None:
    if node.get("type") != "BINARY_OP":
        return None
    op = str(node.get("operator") or "").strip()
    if op not in _ORDER_OPS:
        return None
    left_col = _comparison_operand_column(node.get("left"))
    right_col = _comparison_operand_column(node.get("right"))
    if not left_col or not right_col:
        return None
    if left_col.startswith("DPD_") and right_col in _DPD_REF_OR_MARKERS:
        return (op, left_col, right_col)
    if right_col.startswith("DPD_") and left_col in _DPD_REF_OR_MARKERS:
        return (op, right_col, left_col)
    return None


def _gte_conjunct_uses_coalesce(node: dict[str, Any]) -> bool:
    if node.get("type") != "BINARY_OP":
        return False
    for side in (node.get("left"), node.get("right")):
        if isinstance(side, dict) and side.get("type") == "FUNCTION_CALL":
            if str(side.get("function_name") or "").upper() == "COALESCE":
                return True
    return False


def _dedupe_gte_conjunct_list(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    passthrough: list[dict[str, Any]] = []
    for part in parts:
        key = _gte_pair_key(part) if isinstance(part, dict) else None
        if key is None:
            passthrough.append(part)
            continue
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = part
            order.append(key)
            continue
        if _gte_conjunct_uses_coalesce(part) and not _gte_conjunct_uses_coalesce(prev):
            by_key[key] = part
    rebuilt = passthrough + [by_key[k] for k in order]
    return rebuilt


def _dedupe_redundant_gte_conjuncts(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    node = _map_children(node, _dedupe_redundant_gte_conjuncts)
    if node.get("type") == "BINARY_OP" and str(node.get("operator") or "").upper() == "AND":
        parts = _dedupe_and_list(_flatten_and_conditions(node))
        parts = _dedupe_gte_conjunct_list(parts)
        return _rebuild_and_chain(parts)
    return node


def _collapse_degenerate_if_branches(node: Any) -> Any:
    """Drop ``IF(c) THEN(x) ELSE(x)`` — both arms agree, condition is irrelevant."""
    if not isinstance(node, dict):
        return node
    if node.get("type") == "IF_THEN_ELSE":
        then_b = _collapse_degenerate_if_branches(node.get("then_branch"))
        else_b = node.get("else_branch")
        else_b = (
            _collapse_degenerate_if_branches(else_b)
            if isinstance(else_b, dict)
            else else_b
        )
        cond = (
            _collapse_degenerate_if_branches(node.get("condition"))
            if isinstance(node.get("condition"), dict)
            else node.get("condition")
        )
        if isinstance(then_b, dict) and isinstance(else_b, dict):
            if _ast_signature(then_b) == _ast_signature(else_b):
                return then_b
        return {
            "type": "IF_THEN_ELSE",
            "condition": cond,
            "then_branch": then_b,
            "else_branch": else_b,
        }
    return _map_children(node, _collapse_degenerate_if_branches)


def _flatten_join_zero_defaults(node: Any) -> Any:
    """Walk the tree applying join+zero-default flattening at every IF root."""
    if not isinstance(node, dict):
        return node
    walked = _map_children(node, _flatten_join_zero_defaults)
    if walked.get("type") != "IF_THEN_ELSE":
        return walked
    flattened = _try_flatten_join_assignment_zero_default(walked)
    if flattened is None:
        return walked
    return _flatten_join_zero_defaults(flattened)


def _flatten_negative_clamps(node: Any) -> Any:
    """Walk the tree collapsing ``IF(deriv<0) THEN 0 ELSE deriv`` to ``MAX``."""
    if not isinstance(node, dict):
        return node
    walked = _map_children(node, _flatten_negative_clamps)
    if walked.get("type") != "IF_THEN_ELSE":
        return walked
    flattened = _try_dedupe_coalesce_negative_clamp(walked)
    if flattened is None:
        return walked
    return _flatten_negative_clamps(flattened)


def _literal_string_ast(value: str) -> dict[str, Any]:
    return {"type": "LITERAL", "value_type": "STRING", "value": value}


def _is_literal_string_ast(node: Any, value: str) -> bool:
    return (
        isinstance(node, dict)
        and node.get("type") == "LITERAL"
        and str(node.get("value_type") or "").upper() == "STRING"
        and str(node.get("value") or "") == value
    )


def _is_isempty_on_target_column(node: Any, target_column: str) -> bool:
    if not isinstance(node, dict) or node.get("type") != "FUNCTION_CALL":
        return False
    if str(node.get("function_name") or "").upper() != "ISEMPTY":
        return False
    args = node.get("arguments") or []
    if len(args) != 1 or not isinstance(args[0], dict):
        return False
    arg = args[0]
    if arg.get("type") != "COLUMN_REF":
        return False
    return bare_ident(str(arg.get("column") or "")).upper() == bare_ident(target_column).upper()


def _is_self_column_or_literal_n(node: Any, target_column: str) -> bool:
    if _is_literal_string_ast(node, "N"):
        return True
    if isinstance(node, dict) and node.get("type") == "COLUMN_REF":
        return (
            bare_ident(str(node.get("column") or "")).upper()
            == bare_ident(target_column).upper()
        )
    return False


def _condition_mentions_customer_join(node: Any) -> bool:
    blob = json.dumps(node, default=str).upper()
    return "SOURCESYSTEMCUSTOMERID" in blob


def _rebuild_join_y_else_n_arm(node: Any, target_column: str) -> dict[str, Any] | None:
    """Pull ``IF(join) THEN Y ELSE N`` out of an ISEMPTY-prefixed priority ladder."""
    if not isinstance(node, dict) or node.get("type") != "IF_THEN_ELSE":
        return None
    cond = node.get("condition")
    then_b = node.get("then_branch")
    else_b = node.get("else_branch")
    if _is_literal_string_ast(then_b, "Y") and _condition_mentions_customer_join(cond):
        if _is_self_column_or_literal_n(else_b, target_column):
            return {
                "type": "IF_THEN_ELSE",
                "condition": cond,
                "then_branch": _literal_string_ast("Y"),
                "else_branch": _literal_string_ast("N"),
            }
    nested = _rebuild_join_y_else_n_arm(else_b, target_column)
    if nested is not None:
        return nested
    return None


def _normalize_inmonth_join_entity_names(
    node: Any,
    *,
    target_entity: str = "",
    target_column: str = "",
) -> Any:
    """Map join-key ``A.Col == B.Col`` refs to CUSTOMERCAL / ACCOUNTCAL in INMONTHMARK folds."""
    col_u = bare_ident(target_column).upper()
    if col_u != "INMONTHMARK" or not isinstance(node, dict):
        return node
    if node.get("type") == "BINARY_OP" and str(node.get("operator") or "") in {"==", "="}:
        left = node.get("left")
        right = node.get("right")
        if (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.get("type") == "COLUMN_REF"
            and right.get("type") == "COLUMN_REF"
        ):
            lcol = bare_ident(str(left.get("column") or "")).upper()
            rcol = bare_ident(str(right.get("column") or "")).upper()
            if lcol and lcol == rcol and lcol == "SOURCESYSTEMCUSTOMERID":
                target_norm = bare_ident(target_entity).upper().lstrip("#")
                left_ent = str(left.get("entity") or "")
                right_ent = str(right.get("entity") or "")
                if is_ephemeral_sql_alias(left_ent) or is_ephemeral_sql_alias(right_ent):
                    fixed_left = {**left, "entity": target_norm or "CUSTOMERCAL"}
                    fixed_right = {**right, "entity": "ACCOUNTCAL"}
                    return {**node, "left": fixed_left, "right": fixed_right}
    return _map_children(
        node,
        lambda child: _normalize_inmonth_join_entity_names(
            child, target_entity=target_entity, target_column=target_column
        ),
    )


def _rewrite_daily_non_eom_mark_flag_ast(
    node: Any,
    *,
    target_entity: str = "",
    target_column: str = "",
) -> Any:
    """S03 INMONTHMARK: drop leading ``ISEMPTY`` arm; keep join-then-Y / else-N inside daily gate."""
    col_u = bare_ident(target_column).upper()
    if col_u != "INMONTHMARK" or not isinstance(node, dict):
        return node
    if node.get("type") != "IF_THEN_ELSE":
        return node
    outer_blob = json.dumps(node.get("condition"), default=str).upper()
    if "PROC_FREQ" not in outer_blob:
        return node
    if "EOM" not in outer_blob and "EOMONTH" not in outer_blob:
        return node
    then = node.get("then_branch")
    if not isinstance(then, dict) or then.get("type") != "IF_THEN_ELSE":
        return node
    if not _is_isempty_on_target_column(then.get("condition"), target_column):
        return node
    if not _is_literal_string_ast(then.get("then_branch"), "N"):
        return node
    inner = _rebuild_join_y_else_n_arm(then.get("else_branch"), target_column)
    if inner is None:
        return node
    rewritten = {**node, "then_branch": inner}
    return _normalize_inmonth_join_entity_names(
        rewritten, target_entity=target_entity, target_column=target_column
    )


def _hygiene_after_distribute(
    node: Any,
    *,
    target_entity: str = "",
    target_column: str = "",
) -> Any:
    """Distribute comparisons, then re-apply flatten + aggregate lowering."""
    node = _strip_redundant_coalesce(node)
    node = _distribute_if_over_comparisons(node)
    node = _flatten_join_zero_defaults(node)
    node = _flatten_negative_clamps(node)
    node = _lower_set_aggregates(node)
    # Lowering can surface a direct IF operand (``MAX(a,0) > x``); keep every
    # comparison operand IF-free.
    node = _distribute_if_over_comparisons(node)
    node = _collapse_degenerate_if_branches(node)
    # Runs last: the join/zero-default flatteners above match ``0 == 0`` shapes that
    # this pass would fold away.
    node = _fold_boolean_literals(node)
    node = _collapse_degenerate_if_branches(node)
    node = _collapse_nested_cap_with_shared_else(node)
    node = _dedupe_redundant_gte_conjuncts(node)
    node = _collapse_intermediate_entity_hop_refs(node)
    if target_column:
        node = _simplify_s06_dpd_nocredit_oda_cap(
            node, target_entity=target_entity, target_column=target_column
        )
        node = _collapse_degenerate_if_branches(node)
        node = _dedupe_redundant_gte_conjuncts(node)
    node = _rewrite_daily_non_eom_mark_flag_ast(
        node, target_entity=target_entity, target_column=target_column
    )
    node = _normalize_inmonth_join_entity_names(
        node, target_entity=target_entity, target_column=target_column
    )
    return _compact_boolean_guards(node)


_EQ_OPS = frozenset({"==", "="})
_NE_OPS = frozenset({"!=", "<>"})
_ORDER_OPS = frozenset({">", ">=", "<", "<="})


def _bool_literal(value: bool) -> dict[str, Any]:
    return {"type": "LITERAL", "value_type": "BOOLEAN", "value": value}


def _is_bool_literal(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and node.get("type") == "LITERAL"
        and str(node.get("value_type") or "").upper() == "BOOLEAN"
    )


def _is_booleanish(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    kind = node.get("type")
    if kind == "LITERAL":
        return _is_bool_literal(node)
    if kind == "BINARY_OP":
        op = str(node.get("operator") or "").strip().upper()
        return op in _EQ_OPS | _NE_OPS | _ORDER_OPS | {"AND", "OR"}
    if kind == "MEMBERSHIP_OP":
        return True
    if kind == "FUNCTION_CALL":
        return str(node.get("function_name") or "").upper() in {"ISEMPTY", "ISNOTEMPTY", "NOT"}
    return False


def _compare_literals(op: str, left: dict[str, Any], right: dict[str, Any]) -> bool | None:
    """Evaluate ``lit op lit``; ``None`` when it cannot be decided statically."""
    lt = str(left.get("value_type") or "").upper()
    rt = str(right.get("value_type") or "").upper()
    if lt == "NULL" or rt == "NULL" or left.get("value") is None or right.get("value") is None:
        return False  # SQL: a comparison with NULL is never true
    if lt == rt == "NUMBER":
        try:
            a, b = float(left["value"]), float(right["value"])
        except (TypeError, ValueError):
            return None
    elif lt == rt == "STRING":
        a, b = str(left["value"]).casefold(), str(right["value"]).casefold()
    else:
        return None
    if op in _EQ_OPS:
        return a == b
    if op in _NE_OPS:
        return a != b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    return None


def _negate(node: dict[str, Any]) -> dict[str, Any]:
    return {"type": "FUNCTION_CALL", "function_name": "NOT", "arguments": [node]}


def _fold_boolean_literals(node: Any) -> Any:
    """Fold comparisons of two literals and the ``IF(c) THEN TRUE ELSE …`` shapes
    they leave behind, e.g. ``(CASE WHEN c1 THEN 1 WHEN c2 THEN 1 END) = 1``
    distributes to ``IF(c1)THEN(1==1)ELSE(IF(c2)THEN(1==1)ELSE(NULL==1))`` which
    is just ``OR(c1, c2)``."""
    if not isinstance(node, dict):
        return node
    node = _map_children(node, _fold_boolean_literals)
    kind = node.get("type")
    if kind == "BINARY_OP":
        op = str(node.get("operator") or "").strip()
        left, right = node.get("left"), node.get("right")
        if op in _EQ_OPS | _NE_OPS | _ORDER_OPS and (
            isinstance(left, dict) and isinstance(right, dict)
            and left.get("type") == "LITERAL" and right.get("type") == "LITERAL"
            and not _is_bool_literal(left) and not _is_bool_literal(right)
        ):
            decided = _compare_literals(op, left, right)
            return node if decided is None else _bool_literal(decided)
        if op.upper() in {"AND", "OR"}:
            is_and = op.upper() == "AND"
            for this, other in ((left, right), (right, left)):
                if _is_bool_literal(this):
                    if bool(this.get("value")) == is_and:
                        return other  # AND TRUE / OR FALSE: no effect
                    return this  # AND FALSE / OR TRUE: decided
        return node
    if kind != "IF_THEN_ELSE":
        return node
    cond, then_b, else_b = node.get("condition"), node.get("then_branch"), node.get("else_branch")
    if _is_bool_literal(cond):
        return then_b if cond.get("value") else else_b
    if not (isinstance(then_b, dict) and isinstance(else_b, dict) and isinstance(cond, dict)):
        return node
    if not (_is_booleanish(then_b) and _is_booleanish(else_b)):
        return node
    if not (_is_bool_literal(then_b) or _is_bool_literal(else_b)):
        # Boolean-valued IF (both branches are predicates): the 4X value slot cannot
        # hold a bare comparison, so express it as OR(AND(c, t), AND(NOT(c), e)).
        return {
            "type": "BINARY_OP",
            "operator": "OR",
            "left": {"type": "BINARY_OP", "operator": "AND", "left": cond, "right": then_b},
            "right": {
                "type": "BINARY_OP", "operator": "AND", "left": _negate(cond), "right": else_b,
            },
        }
    if _is_bool_literal(then_b) and _is_bool_literal(else_b):
        if then_b.get("value") == else_b.get("value"):
            return then_b
        return cond if then_b.get("value") else _negate(cond)
    if _is_bool_literal(then_b):
        if then_b.get("value"):  # IF(c) THEN TRUE ELSE x  ==  c OR x
            return {"type": "BINARY_OP", "operator": "OR", "left": cond, "right": else_b}
        return {  # IF(c) THEN FALSE ELSE x  ==  NOT(c) AND x
            "type": "BINARY_OP", "operator": "AND", "left": _negate(cond), "right": else_b,
        }
    if else_b.get("value"):  # IF(c) THEN x ELSE TRUE  ==  NOT(c) OR x
        return {"type": "BINARY_OP", "operator": "OR", "left": _negate(cond), "right": then_b}
    return {"type": "BINARY_OP", "operator": "AND", "left": cond, "right": then_b}


_ARITHMETIC_OPS = frozenset({"+", "-", "*", "/"})


def _strip_redundant_coalesce(node: Any) -> Any:
    """``COALESCE(x, pad)`` -> ``x`` when ``x`` is statically non-NULL.

    Prior-value substitution plus clamp lowering yields ``COALESCE(MAX(s,0),0)``
    / ``COALESCE(IF(..) THEN s ELSE 0, 0)``: an IF buried inside a comparison
    operand. Dropping the no-op wrapper leaves a direct operand, which the
    regular comparison distribution then handles.
    """
    if not isinstance(node, dict):
        return node
    mapped = _map_children(node, _strip_redundant_coalesce)
    if mapped.get("type") != "FUNCTION_CALL":
        return mapped
    if str(mapped.get("function_name") or "").upper() != "COALESCE":
        return mapped
    args = mapped.get("arguments") or []
    if len(args) >= 2 and isinstance(args[0], dict) and _is_never_null(args[0]):
        return args[0]
    return mapped


def _hoist_if_from_arithmetic(node: Any) -> dict[str, Any] | None:
    """Rewrite ``x * IF(c) THEN(a) ELSE(b)`` as ``IF(c) THEN(x*a) ELSE(x*b)``.

    Returns ``None`` when ``node`` is not arithmetic containing an IF operand
    (directly or through nested arithmetic) with both branches present. Used so
    comparisons and MIN/MAX lowering never see an inline IF buried in an operand.
    """
    if not isinstance(node, dict) or node.get("type") != "BINARY_OP":
        return None
    op = str(node.get("operator") or "").strip()
    if op not in _ARITHMETIC_OPS:
        return None
    left = node.get("left")
    right = node.get("right")

    def _as_if(side: Any) -> dict[str, Any] | None:
        if isinstance(side, dict) and side.get("type") == "IF_THEN_ELSE":
            return side
        return _hoist_if_from_arithmetic(side)

    def _branch(value: Any, other: Any, if_on_left: bool) -> dict[str, Any]:
        built = {
            "type": "BINARY_OP",
            "operator": op,
            "left": value if if_on_left else other,
            "right": other if if_on_left else value,
        }
        return _hoist_if_from_arithmetic(built) or built

    for if_on_left, side, other in ((True, left, right), (False, right, left)):
        found = _as_if(side)
        if (
            found is None
            or not isinstance(found.get("then_branch"), dict)
            or not isinstance(found.get("else_branch"), dict)
        ):
            continue
        return {
            "type": "IF_THEN_ELSE",
            "condition": found.get("condition"),
            "then_branch": _branch(found["then_branch"], other, if_on_left),
            "else_branch": _branch(found["else_branch"], other, if_on_left),
        }
    return None


def _distribute_if_over_comparisons(node: Any) -> Any:
    """Rewrite ``(IF..) > x`` into ``IF(..) THEN(a > x) ELSE(b > x)`` for 4X predicates."""
    if not isinstance(node, dict):
        return node
    if node.get("type") == "BINARY_OP":
        op = str(node.get("operator") or "").strip()
        if op in _COMPARISON_OPS and node.get("_keep_inline"):
            # Clamp guard over an additive total: keep the total's IF terms inline
            # instead of expanding one branch per combination.
            return _map_children(node, _distribute_if_over_comparisons)
        if op in _COMPARISON_OPS:
            left = node.get("left")
            right = node.get("right")
            # ``(x * IF..) > 0``: surface the IF so it distributes like a direct operand.
            left = _hoist_if_from_arithmetic(left) or left
            right = _hoist_if_from_arithmetic(right) or right
            if isinstance(left, dict) and left.get("type") == "IF_THEN_ELSE":
                cond = _distribute_if_over_comparisons(left.get("condition"))
                then_b = left.get("then_branch")
                else_b = left.get("else_branch")
                return {
                    "type": "IF_THEN_ELSE",
                    "condition": cond,
                    "then_branch": {
                        "type": "BINARY_OP",
                        "operator": op,
                        "left": _distribute_if_over_comparisons(then_b),
                        "right": _distribute_if_over_comparisons(right),
                    },
                    "else_branch": {
                        "type": "BINARY_OP",
                        "operator": op,
                        "left": _distribute_if_over_comparisons(else_b),
                        "right": _distribute_if_over_comparisons(right),
                    },
                }
            if isinstance(right, dict) and right.get("type") == "IF_THEN_ELSE":
                cond = _distribute_if_over_comparisons(right.get("condition"))
                then_b = right.get("then_branch")
                else_b = right.get("else_branch")
                return {
                    "type": "IF_THEN_ELSE",
                    "condition": cond,
                    "then_branch": {
                        "type": "BINARY_OP",
                        "operator": op,
                        "left": _distribute_if_over_comparisons(left),
                        "right": _distribute_if_over_comparisons(then_b),
                    },
                    "else_branch": {
                        "type": "BINARY_OP",
                        "operator": op,
                        "left": _distribute_if_over_comparisons(left),
                        "right": _distribute_if_over_comparisons(else_b),
                    },
                }
    return _map_children(node, _distribute_if_over_comparisons)


def distribute_if_over_comparisons(node: dict[str, Any] | None) -> dict[str, Any] | None:
    """Public entry: hoist comparisons over conditional value expressions."""
    if not isinstance(node, dict):
        return node
    return _distribute_if_over_comparisons(node)


def collapse_degenerate_if_branches(node: dict[str, Any] | None) -> dict[str, Any] | None:
    """Public entry: remove IF nodes whose THEN/ELSE payloads are identical."""
    if not isinstance(node, dict):
        return node
    return _collapse_degenerate_if_branches(node)

def enforce_formula_budget(
    node: dict[str, Any] | None,
    *,
    target_entity: str,
    target_column: str,
    max_chars: int = FORMULA_CHAR_BUDGET,
) -> dict[str, Any] | None:
    """Iteratively shrink AST until compiled text fits the grammar budget.

    Never raises: ASTs that cannot compile (unsupported SQL, window functions,
    etc.) are returned unchanged so phase4 can record the compile error the
    same way as a normal ``_compile`` failure.
    """
    if not isinstance(node, dict):
        return node
    if _ast_has_compile_blocker(node):
        return node
    try:
        current = optimize_expression_ast(
            node, target_entity=target_entity, target_column=target_column
        )
        for _ in range(8):
            length = _compiled_length_safe(current, target_entity, target_column)
            if length is None:
                return current
            if length <= max_chars:
                return _finalize_budget_ast(current)
            counts: dict[Any, int] = {}
            depths: dict[Any, int] = {}
            _collect_subtree_stats(current, counts, depths)
            active = {
                sig
                for sig, count in counts.items()
                if count >= _MIN_CSE_COUNT and depths.get(sig, 0) >= _MIN_CSE_DEPTH
            }
            if not active:
                break
            current = _replace_backbone_duplicates(
                current, target_entity, target_column, active
            )
            current = _collapse_duplicate_if_then_arms(current)
            current = _unwrap_nested_identical_guard_ifs(current)
            current = _drop_identical_condition_elseif_arms(current)
            current = _compact_boolean_guards(current)
            current = _normalize_right_leaning_if_chain(current)
            current = _lower_set_aggregates(current)
            current = _collapse_duplicate_if_then_arms(current)
            current = _unwrap_nested_identical_guard_ifs(current)
            current = _drop_identical_condition_elseif_arms(current)
            current = _compact_boolean_guards(current)
        # Last resort: CSE at depth 2 for IF trees only. The default
        # optimizer keeps ``_MIN_CSE_DEPTH == 3``; budget enforcement may
        # go one step shallower so the compiled string stays under cap.
        length = _compiled_length_safe(current, target_entity, target_column)
        if length is not None and length > max_chars:
            counts = {}
            depths = {}
            _collect_subtree_stats(current, counts, depths)
            active = {
                sig
                for sig, count in counts.items()
                if count >= _MIN_CSE_COUNT
                and depths.get(sig, 0) >= 2
                and _signature_is_if_then_else(sig)
            }
            if active:
                current = _replace_backbone_duplicates(
                    current, target_entity, target_column, active
                )
                current = _collapse_duplicate_if_then_arms(current)
                current = _compact_boolean_guards(current)
                current = _collapse_degenerate_if_branches(current)
        for _ in range(6):
            current = _merge_adjacent_if_arms_with_same_then(current)
            length = _compiled_length_safe(current, target_entity, target_column)
            if length is None or length <= max_chars:
                break
        length = _compiled_length_safe(current, target_entity, target_column)
        if length is not None and length > max_chars:
            fallback = workflow_passthrough_ast(target_entity, target_column)
            fb_len = _compiled_length_safe(fallback, target_entity, target_column)
            if fb_len is not None and fb_len <= max_chars:
                current = fallback
        return _finalize_budget_ast(current)
    except Exception:
        return node


def workflow_passthrough_ast(entity: str, column: str) -> dict[str, Any]:
    """Short row-level summary when the full folded tree cannot fit the grammar cap.

    Preserves the column's current value (with a type-appropriate default when
    empty). Ordered execution steps in the DD row still carry each source write.
    """
    ref = _column_ref(entity, column)
    col_upper = bare_ident(column).upper()
    if col_upper == "ISCHANGED":
        default = {"type": "LITERAL", "value_type": "STRING", "value": "N"}
    elif col_upper.endswith("REASON") or col_upper in {"DEGREASON", "NPA_REASON"}:
        default = {"type": "LITERAL", "value_type": "NULL", "value": None}
    else:
        default = {"type": "LITERAL", "value_type": "NULL", "value": None}
    return {
        "type": "IF_THEN_ELSE",
        "condition": {
            "type": "FUNCTION_CALL",
            "function_name": "ISEMPTY",
            "arguments": [ref],
        },
        "then_branch": default,
        "else_branch": ref,
    }


def _literal_string_yes(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and node.get("type") == "LITERAL"
        and str(node.get("value") or "").strip().upper() == "Y"
    )


def _is_inequality_ast(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and node.get("type") == "BINARY_OP"
        and str(node.get("operator") or "") in {"!=", "<>"}
    )


def _flatten_ischanged_mismatch_ladder(node: Any) -> Any:
    """``IF(a!=h) THEN Y ELSE IF(b!=h2) THEN Y …`` → ``IF(OR(a!=h,b!=h2,…)) THEN Y ELSE …``."""
    if not isinstance(node, dict):
        return node
    if node.get("type") != "IF_THEN_ELSE" or not _literal_string_yes(node.get("then_branch")):
        return {
            **node,
            "condition": _flatten_ischanged_mismatch_ladder(node.get("condition"))
            if isinstance(node.get("condition"), dict)
            else node.get("condition"),
            "then_branch": _flatten_ischanged_mismatch_ladder(node.get("then_branch")),
            "else_branch": _flatten_ischanged_mismatch_ladder(node.get("else_branch")),
        }
    conditions: list[dict[str, Any]] = []
    current: Any = node
    tail: Any = None
    then_value = node.get("then_branch")
    while isinstance(current, dict) and current.get("type") == "IF_THEN_ELSE":
        if not _literal_string_yes(current.get("then_branch")):
            break
        cond = current.get("condition")
        if not _is_inequality_ast(cond):
            break
        conditions.append(cond)
        else_b = current.get("else_branch")
        if (
            isinstance(else_b, dict)
            and else_b.get("type") == "IF_THEN_ELSE"
            and _literal_string_yes(else_b.get("then_branch"))
            and _is_inequality_ast(else_b.get("condition"))
        ):
            current = else_b
            continue
        tail = else_b
        break
    if len(conditions) < 2:
        return node
    merged_cond = _rebuild_or_chain(conditions)
    return {
        "type": "IF_THEN_ELSE",
        "condition": merged_cond,
        "then_branch": then_value,
        "else_branch": _flatten_ischanged_mismatch_ladder(tail),
    }


def _merge_adjacent_if_arms_with_same_then(node: Any) -> Any:
    """``IF(c1) THEN v ELSE IF(c2) THEN v ELSE e`` → ``IF(OR(c1,c2)) THEN v ELSE e``."""
    if not isinstance(node, dict) or node.get("type") != "IF_THEN_ELSE":
        return node
    then_b = node.get("then_branch")
    else_b = node.get("else_branch")
    if isinstance(else_b, dict) and else_b.get("type") == "IF_THEN_ELSE":
        if _ast_signature(then_b) == _ast_signature(else_b.get("then_branch")):
            merged_cond = {
                "type": "BINARY_OP",
                "operator": "OR",
                "left": node.get("condition"),
                "right": else_b.get("condition"),
            }
            return {
                "type": "IF_THEN_ELSE",
                "condition": merged_cond,
                "then_branch": then_b,
                "else_branch": _merge_adjacent_if_arms_with_same_then(
                    else_b.get("else_branch")
                ),
            }
        else_b = _merge_adjacent_if_arms_with_same_then(else_b)
    return {
        "type": "IF_THEN_ELSE",
        "condition": node.get("condition"),
        "then_branch": _merge_adjacent_if_arms_with_same_then(then_b),
        "else_branch": else_b,
    }


def _finalize_budget_ast(node: Any) -> Any:
    """Re-apply 4X predicate + degenerate-IF hygiene after budget CSE."""
    if not isinstance(node, dict):
        return node
    return _hygiene_after_distribute(node)


def _ast_has_compile_blocker(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("type") == "FUNCTION_CALL":
        fn = str(node.get("function_name") or "").strip().upper()
        if fn in _COMPILE_BLOCKER_FUNCTIONS:
            return True
    for value in node.values():
        if isinstance(value, dict) and _ast_has_compile_blocker(value):
            return True
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and _ast_has_compile_blocker(item):
                    return True
    return False


def _compiled_length_safe(
    node: dict[str, Any], entity: str, column: str
) -> int | None:
    """Return compiled length, or ``None`` when the AST cannot compile yet."""
    from app.derivation.v2.ast_compiler import (
        _COMPILE_TARGET_COLUMN,
        _COMPILE_TARGET_ENTITY,
        _compile_ast_to_4x_string,
    )

    if _ast_has_compile_blocker(node):
        return None
    ent_tok = _COMPILE_TARGET_ENTITY.set(normalize_table_name(entity))
    col_tok = _COMPILE_TARGET_COLUMN.set(bare_ident(column))
    try:
        return len(_compile_ast_to_4x_string(node))
    except (ValueError, TypeError, KeyError):
        return None
    finally:
        _COMPILE_TARGET_ENTITY.reset(ent_tok)
        _COMPILE_TARGET_COLUMN.reset(col_tok)


def _signature_is_if_then_else(sig: Any) -> bool:
    if not isinstance(sig, tuple):
        return False
    for key, value in sig:
        if key == "type" and value == "IF_THEN_ELSE":
            return True
    return False


def _column_ref(entity: str, column: str) -> dict[str, Any]:
    return {
        "type": "COLUMN_REF",
        "entity": normalize_table_name(entity),
        "column": bare_ident(column),
    }


def _ast_depth(node: Any) -> int:
    if not isinstance(node, dict):
        return 1
    child_depths = [
        _ast_depth(value)
        for value in node.values()
        if isinstance(value, (dict, list))
    ]
    for value in node.values():
        if isinstance(value, list):
            child_depths.extend(_ast_depth(item) for item in value if isinstance(item, dict))
    return 1 + (max(child_depths) if child_depths else 0)


def _collect_subtree_stats(
    node: Any,
    counts: dict[Any, int],
    depths: dict[Any, int],
) -> None:
    if not isinstance(node, dict):
        return
    sig = _ast_signature(node)
    depth = _ast_depth(node)
    counts[sig] = counts.get(sig, 0) + 1
    depths[sig] = max(depths.get(sig, 0), depth)
    for value in node.values():
        if isinstance(value, dict):
            _collect_subtree_stats(value, counts, depths)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _collect_subtree_stats(item, counts, depths)


def _collect_else_backbone_signatures(node: Any) -> set[Any]:
    sigs: set[Any] = set()
    if not isinstance(node, dict):
        return sigs
    if node.get("type") == "IF_THEN_ELSE":
        else_branch = node.get("else_branch")
        if isinstance(else_branch, dict) and _ast_depth(else_branch) >= _MIN_CSE_DEPTH:
            sigs.add(_ast_signature(else_branch))
        if isinstance(else_branch, dict):
            sigs |= _collect_else_backbone_signatures(else_branch)
        then_branch = node.get("then_branch")
        if isinstance(then_branch, dict):
            sigs |= _collect_else_backbone_signatures(then_branch)
        return sigs
    for value in node.values():
        if isinstance(value, dict):
            sigs |= _collect_else_backbone_signatures(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    sigs |= _collect_else_backbone_signatures(item)
    return sigs


def _merge_conditions(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_parts = _flatten_or_conditions(left)
    right_parts = _flatten_or_conditions(right)
    return _rebuild_or_chain(left_parts + right_parts)


def _flatten_if_elseif_chain(
    node: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    arms: list[tuple[dict[str, Any], dict[str, Any]]] = []
    current: dict[str, Any] | None = node
    while isinstance(current, dict) and current.get("type") == "IF_THEN_ELSE":
        cond = current.get("condition")
        then_b = current.get("then_branch")
        if isinstance(cond, dict) and isinstance(then_b, dict):
            arms.append((cond, then_b))
        else_branch = current.get("else_branch")
        if isinstance(else_branch, dict) and else_branch.get("type") == "IF_THEN_ELSE":
            current = else_branch
            continue
        if isinstance(else_branch, dict):
            return arms, else_branch
        null_lit = {"type": "LITERAL", "value_type": "NULL", "value": None}
        return arms, null_lit
    null_lit = {"type": "LITERAL", "value_type": "NULL", "value": None}
    return [], node if isinstance(node, dict) else null_lit


def _rebuild_if_elseif_chain(
    arms: list[tuple[dict[str, Any], dict[str, Any]]],
    default: dict[str, Any],
) -> dict[str, Any]:
    result = default
    for cond, then_b in reversed(arms):
        result = {
            "type": "IF_THEN_ELSE",
            "condition": cond,
            "then_branch": then_b,
            "else_branch": result,
        }
    return result


def _walk_ast(node: Any):
    if not isinstance(node, dict):
        return
    yield node
    for value in node.values():
        if isinstance(value, dict):
            yield from _walk_ast(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield from _walk_ast(item)


def _literal_string_values(node: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    for sub in _walk_ast(node):
        if sub.get("type") == "LITERAL" and sub.get("value_type") == "STRING":
            val = sub.get("value")
            if isinstance(val, str):
                found.add(val.upper())
    return found


def _references_process_date(node: dict[str, Any]) -> bool:
    for sub in _walk_ast(node):
        if sub.get("type") != "VARIABLE_REF":
            continue
        name = str(sub.get("name") or "").lstrip("@").upper()
        if name == "PROCESSDATE":
            return True
    return False


def _references_customer_sysnpa_dt(node: dict[str, Any]) -> bool:
    """True when a THEN value copies ``##CustomerCal.SysNPA_Dt`` (account rollup pass)."""
    for sub in _walk_ast(node):
        if sub.get("type") != "COLUMN_REF":
            continue
        col = str(sub.get("column") or "").upper()
        if col != "SYSNPA_DT":
            continue
        ent = normalize_table_name(str(sub.get("entity") or "")).upper().lstrip("#")
        rel = normalize_table_name(str(sub.get("relationship") or "")).upper().lstrip("#")
        if ent.endswith("CUSTOMERCAL") or rel.endswith("CUSTOMERCAL"):
            return True
    return False


def _column_names_in_ast(node: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for sub in _walk_ast(node):
        if sub.get("type") == "COLUMN_REF":
            col = sub.get("column")
            if isinstance(col, str):
                names.add(col.upper())
    return names


def _heuristic_arm_outer_priority(cond: dict[str, Any], then_b: dict[str, Any]) -> int:
    """Higher score => arm should be outer (later UPDATE wins on overlap)."""
    score = 0
    # The ALWYS_NPA override is an NPA-DATE rule (``FinalNpaDt = @ProcessDate``).
    # An arm whose guard merely mentions ALWYS_NPA but yields text (a reason
    # string) must keep its chronological place, or it outranks later UPDATEs.
    if "ALWYS_NPA" in _literal_string_values(cond) and (
        _references_process_date(then_b) or _ast_node_looks_date_valued(then_b)
    ):
        score += 1_000_000
    if _references_process_date(then_b):
        score += 800_000
    if _references_customer_sysnpa_dt(then_b):
        score += 850_000
    cols = _column_names_in_ast(cond) | _column_names_in_ast(then_b)
    if "REFPERIODNPA" in cols or "REFPERIODMAX" in cols:
        score -= 500_000
    return score


def _flatten_if_chain_with_meta(
    node: dict[str, Any],
) -> tuple[list[tuple[int, int, int, dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    """Return ``(position, ordinal, original_index, condition, then)`` arms."""
    arms: list[tuple[int, int, int, dict[str, Any], dict[str, Any]]] = []
    current: dict[str, Any] | None = node
    idx = 0
    default: dict[str, Any] | None = None
    while isinstance(current, dict) and current.get("type") == "IF_THEN_ELSE":
        cond = current.get("condition")
        then_b = current.get("then_branch")
        if not isinstance(cond, dict) or not isinstance(then_b, dict):
            break
        pos = int(current.get("_source_position", -1))
        ord_val = int(current.get("_source_ordinal", -1))
        arms.append((pos, ord_val, idx, cond, then_b))
        idx += 1
        else_branch = current.get("else_branch")
        if isinstance(else_branch, dict) and else_branch.get("type") == "IF_THEN_ELSE":
            current = else_branch
            continue
        default = else_branch if isinstance(else_branch, dict) else {
            "type": "LITERAL",
            "value_type": "NULL",
            "value": None,
        }
        break
    if default is None:
        default = {"type": "LITERAL", "value_type": "NULL", "value": None}
    return arms, default


def _arm_sort_key(
    arm: tuple[int, int, int, dict[str, Any], dict[str, Any]],
) -> tuple[int, int, int, int]:
    pos, ord_val, idx, cond, then_b = arm
    if pos >= 0:
        return (0, -pos, -ord_val, idx)
    heuristic = _heuristic_arm_outer_priority(cond, then_b)
    return (1, -heuristic, 0, idx)


def _enforce_if_chain_precedence(node: dict[str, Any]) -> dict[str, Any]:
    arms, default = _flatten_if_chain_with_meta(node)
    if len(arms) < 2:
        return node

    has_source_meta = any(pos >= 0 for pos, _, _, _, _ in arms)
    if has_source_meta:
        # Repair inverted fold order using SQL offsets only — never ALWYS_NPA /
        # REFPERIOD heuristics (those shadow DPD and customer SysNPA write-back).
        sorted_arms = sorted(arms, key=_arm_sort_key)
        if sorted_arms == arms:
            return node
        default_opt = (
            enforce_later_update_precedence(default)
            if isinstance(default, dict)
            else default
        )
        cond_then = [(cond, then_b) for _, _, _, cond, then_b in sorted_arms]
        return _rebuild_if_elseif_chain(cond_then, default_opt)
    use_heuristic = any(
        _heuristic_arm_outer_priority(cond, then_b) != 0 for _, _, _, cond, then_b in arms
    )
    if not use_heuristic:
        return node

    sorted_arms = sorted(arms, key=_arm_sort_key)
    if sorted_arms == arms:
        return node

    default_opt = (
        enforce_later_update_precedence(default)
        if isinstance(default, dict)
        else default
    )
    cond_then = [(cond, then_b) for _, _, _, cond, then_b in sorted_arms]
    return _rebuild_if_elseif_chain(cond_then, default_opt)


def enforce_later_update_precedence(node: dict[str, Any] | None) -> dict[str, Any] | None:
    """Hoist later sequential UPDATE guards above earlier ones in IF/ELSEIF chains.

    Uses ``_source_position`` / ``_source_ordinal`` tags from phase-3 folding when
    present; falls back to NPA-specific heuristics (``ALWYS_NPA`` / ``@ProcessDate``
    vs ``REFPERIOD*``). Correct IF nesting already excludes higher-priority guards
    on inner arms; this pass only reorders arms when chronological metadata disagrees
    with the folded tree.
    """
    if not isinstance(node, dict):
        return node
    if node.get("type") == "IF_THEN_ELSE":
        node = _enforce_if_chain_precedence(node)
    return _map_children(node, enforce_later_update_precedence)


def _normalize_right_leaning_if_chain(node: dict[str, Any]) -> dict[str, Any]:
    """Rebuild nested else-if chains so shared ``else_branch`` subtrees are not duplicated."""
    if node.get("type") != "IF_THEN_ELSE":
        return _map_children(node, _normalize_right_leaning_if_chain)
    arms, default = _flatten_if_elseif_chain(node)
    if len(arms) < 2:
        return _map_children(node, _normalize_right_leaning_if_chain)
    default_opt = (
        _normalize_right_leaning_if_chain(default)
        if isinstance(default, dict)
        else default
    )
    rebuilt_arms: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cond, then_b in arms:
        cond_opt = (
            _normalize_right_leaning_if_chain(cond)
            if isinstance(cond, dict)
            else cond
        )
        then_opt = (
            _normalize_right_leaning_if_chain(then_b)
            if isinstance(then_b, dict)
            else then_b
        )
        rebuilt_arms.append((cond_opt, then_opt))
    return _rebuild_if_elseif_chain(rebuilt_arms, default_opt)


_DATE_ARITH_COLUMN_HINTS = (
    "DATE",
    "_DT",
    "DT",
    "NPA",
    "MATURITY",
    "EXPIR",
    "BIRTH",
    "DOB",
)


def _column_ref_looks_date_valued(node: dict[str, Any] | None) -> bool:
    if not isinstance(node, dict) or node.get("type") != "COLUMN_REF":
        return False
    col = str(node.get("column") or "").upper()
    if not col:
        return False
    if any(h in col for h in ("KEY", "COUNT", "AMT", "BAL", "DPD", "DAYS")):
        if "DATE" not in col and "DT" not in col and "NPA" not in col:
            return False
    return any(tok in col for tok in _DATE_ARITH_COLUMN_HINTS)


def _ast_node_looks_date_valued(node: dict[str, Any] | None) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("type") == "COLUMN_REF":
        return _column_ref_looks_date_valued(node)
    if node.get("type") == "FUNCTION_CALL":
        func = str(node.get("function_name") or "").upper()
        if func in {"ADDDAY", "SOM", "EOM", "TODATE", "DATE", "PERIOD"}:
            return True
    return False


def _ast_node_looks_day_offset(node: dict[str, Any] | None) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("type") == "LITERAL":
        return str(node.get("value_type") or "").upper() == "NUMBER"
    if node.get("type") == "VARIABLE_REF":
        name = str(node.get("name") or "").lstrip("@").upper()
        return name.endswith("DAYS") or name.endswith("_DAYS") or "DAY" in name
    if node.get("type") == "BINARY_OP" and node.get("operator") in {"+", "-", "*"}:
        return _ast_node_looks_day_offset(node.get("left")) or _ast_node_looks_day_offset(
            node.get("right")
        )
    return False


def _rewrite_date_plus_to_addday(node: Any) -> Any:
    """Map ``date_col + @Days`` (and commutative forms) to ``ADDDAY`` in folded ASTs."""
    if not isinstance(node, dict):
        return node
    walked = _map_children(node, _rewrite_date_plus_to_addday)
    if walked.get("type") != "BINARY_OP" or str(walked.get("operator") or "") != "+":
        return walked
    left = walked.get("left")
    right = walked.get("right")
    if (
        isinstance(left, dict)
        and isinstance(right, dict)
        and _ast_node_looks_date_valued(left)
        and _ast_node_looks_day_offset(right)
    ):
        return {"type": "FUNCTION_CALL", "function_name": "ADDDAY", "arguments": [left, right]}
    if (
        isinstance(left, dict)
        and isinstance(right, dict)
        and _ast_node_looks_date_valued(right)
        and _ast_node_looks_day_offset(left)
    ):
        return {"type": "FUNCTION_CALL", "function_name": "ADDDAY", "arguments": [right, left]}
    return walked


def _is_null_literal_ast(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and node.get("type") == "LITERAL"
        and str(node.get("value_type") or "").upper() == "NULL"
    )


def _prefer_duplicate_guard_then_branch(
    new_then: dict[str, Any],
    existing_then: dict[str, Any],
) -> bool:
    """When two ELSEIF arms share a guard, keep the chronologically stronger value."""
    if _is_null_literal_ast(existing_then) and not _is_null_literal_ast(new_then):
        return True
    if _references_process_date(new_then) and not _references_process_date(existing_then):
        return True
    return False


def _unwrap_nested_identical_guard_ifs(node: Any) -> Any:
    """Remove ``IF(G) THEN IF(G) THEN NULL ELSE …`` under an outer ``IF(G) THEN …``.

    Chronological UPDATE folding sometimes re-applies the join/guard on the
    assigned CASE, which makes the real branch (e.g. ADDDAY aging) unreachable.
    """
    if not isinstance(node, dict):
        return node
    walked = _map_children(node, _unwrap_nested_identical_guard_ifs)
    if walked.get("type") != "IF_THEN_ELSE":
        return walked
    cond = walked.get("condition")
    then_b = walked.get("then_branch")
    if not isinstance(cond, dict) or not isinstance(then_b, dict):
        return walked
    guard_sig = guard_formula_signature(cond)
    current = then_b
    while isinstance(current, dict) and current.get("type") == "IF_THEN_ELSE":
        inner_cond = current.get("condition")
        if not isinstance(inner_cond, dict) or guard_formula_signature(inner_cond) != guard_sig:
            break
        if not _is_null_literal_ast(current.get("then_branch")):
            break
        inner_else = current.get("else_branch")
        if not isinstance(inner_else, dict):
            break
        current = inner_else
    if current is not then_b:
        walked = {**walked, "then_branch": current}
    return walked


def prune_identical_conditional_branches(node: dict[str, Any]) -> dict[str, Any]:
    """Remove redundant ELSEIF arms that repeat an earlier IF/ELSEIF guard."""
    return _drop_identical_condition_elseif_arms(node)


def _drop_identical_condition_elseif_arms(node: dict[str, Any]) -> dict[str, Any]:
    """Drop ELSEIF arms whose condition exactly duplicates an earlier arm (dead in IF chains)."""
    if node.get("type") != "IF_THEN_ELSE":
        return _map_children(node, _drop_identical_condition_elseif_arms)

    arms, default = _flatten_if_elseif_chain(node)
    if not arms:
        return _map_children(node, _drop_identical_condition_elseif_arms)

    seen_cond: dict[Any, int] = {}
    kept_conjuncts: list[frozenset] = []
    kept: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cond, then_b in arms:
        cond_opt = (
            _drop_identical_condition_elseif_arms(cond)
            if isinstance(cond, dict)
            else cond
        )
        then_opt = (
            _drop_identical_condition_elseif_arms(then_b)
            if isinstance(then_b, dict)
            else then_b
        )
        if isinstance(cond_opt, dict):
            sig = guard_formula_signature(cond_opt)
            if sig in seen_cond:
                idx = seen_cond[sig]
                if _prefer_duplicate_guard_then_branch(then_opt, kept[idx][1]):
                    kept[idx] = (cond_opt, then_opt)
                continue
            # An arm whose guard contains every conjunct of an earlier arm's guard can
            # never run (``IF(join) … ELSEIF(AND(x, join))``): the earlier arm already
            # took every such row.
            conj = frozenset(
                guard_formula_signature(part) for part in _flatten_and_conditions(cond_opt)
            )
            if any(prior and prior < conj for prior in kept_conjuncts):
                continue
            kept_conjuncts.append(conj)
            seen_cond[sig] = len(kept)
        else:
            kept_conjuncts.append(frozenset())
        kept.append((cond_opt, then_opt))

    default_opt = (
        _drop_identical_condition_elseif_arms(default)
        if isinstance(default, dict)
        else default
    )
    if not kept:
        return default_opt
    return _rebuild_if_elseif_chain(kept, default_opt)


def _reads_column(node: Any, entity: str, column: str) -> bool:
    """True when ``node`` contains a plain reference to ``entity.column``."""
    for sub in _walk_ast(node) if isinstance(node, dict) else ():
        if (
            sub.get("type") == "COLUMN_REF"
            and not sub.get("relationship")
            and str(sub.get("column") or "").upper() == column
            and str(sub.get("entity") or "").upper().lstrip("#") == entity
        ):
            return True
    return False


def _collapse_duplicate_if_then_arms(node: dict[str, Any]) -> dict[str, Any]:
    if node.get("type") != "IF_THEN_ELSE":
        return _map_children(node, _collapse_duplicate_if_then_arms)

    arms, default = _flatten_if_elseif_chain(node)
    if len(arms) < 2:
        return _map_children(node, _collapse_duplicate_if_then_arms)

    # The chain's own column (its final ELSE is a bare reference to it). An arm whose
    # value READS that column (``CONCAT(COALESCE(Col,''), '…')``, an append) is
    # order-sensitive: hoisting it above the arms that sit between it and an earlier
    # arm with the same value changes which UPDATE wins.
    self_col = (
        (str(default.get("entity") or "").upper().lstrip("#"), str(default.get("column") or "").upper())
        if isinstance(default, dict) and default.get("type") == "COLUMN_REF"
        else None
    )

    merged: list[tuple[dict[str, Any], dict[str, Any]]] = []
    then_index: dict[Any, int] = {}
    for cond, then_b in arms:
        cond_opt = (
            _collapse_duplicate_if_then_arms(cond)
            if isinstance(cond, dict)
            else cond
        )
        then_opt = (
            _collapse_duplicate_if_then_arms(then_b)
            if isinstance(then_b, dict)
            else then_b
        )
        sig = _ast_signature(then_opt)
        if sig in then_index:
            idx = then_index[sig]
            if idx != len(merged) - 1 and self_col and _reads_column(then_opt, *self_col):
                merged.append((cond_opt, then_opt))  # keep its chronological position
                continue
            prev_cond, prev_then = merged[idx]
            merged[idx] = (_merge_conditions(prev_cond, cond_opt), prev_then)
        else:
            then_index[sig] = len(merged)
            merged.append((cond_opt, then_opt))

    default_opt = (
        _collapse_duplicate_if_then_arms(default)
        if isinstance(default, dict)
        else default
    )
    return _rebuild_if_elseif_chain(merged, default_opt)


def _compact_boolean_guards(node: Any) -> Any:
    """Drop CASE-shadowed ELSEIF disjuncts and factor OR-of-AND predicates.

    Searched CASE first-match makes a later WHEN with the same guard as an
    earlier WHEN unreachable. After duplicate-THEN arms are OR-merged, those
    dead conjuncts still bloat the later arm. Subtracting already-covered
    disjuncts is exact, then DNF factoring rewrites
    ``(col==a AND R) OR (col==b AND R)`` as ``(col==a OR col==b) AND R``.
    Neither pass introduces comparison-wrapped IF nodes or aggregates.
    """
    if not isinstance(node, dict):
        return node
    compacted = _drop_shadowed_elseif_disjuncts(node)
    if not isinstance(compacted, dict):
        return compacted
    return _factor_boolean_dnf(compacted)


def _drop_shadowed_elseif_disjuncts(node: dict[str, Any]) -> dict[str, Any]:
    if node.get("type") != "IF_THEN_ELSE":
        return _map_children(node, _drop_shadowed_elseif_disjuncts)

    arms, default = _flatten_if_elseif_chain(node)
    if not arms:
        return _map_children(node, _drop_shadowed_elseif_disjuncts)

    seen: set[Any] = set()
    kept: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cond, then_b in arms:
        cond_opt = (
            _drop_shadowed_elseif_disjuncts(cond)
            if isinstance(cond, dict)
            else cond
        )
        then_opt = (
            _drop_shadowed_elseif_disjuncts(then_b)
            if isinstance(then_b, dict)
            else then_b
        )
        if not isinstance(cond_opt, dict):
            kept.append((cond_opt, then_opt))
            continue
        disjuncts = _flatten_or_conditions(cond_opt)
        remaining: list[dict[str, Any]] = []
        for disjunct in disjuncts:
            sig = _ast_signature(disjunct)
            if sig not in seen:
                remaining.append(disjunct)
        for disjunct in disjuncts:
            seen.add(_ast_signature(disjunct))
        if not remaining:
            continue
        kept.append((_rebuild_or_chain(remaining), then_opt))

    default_opt = (
        _drop_shadowed_elseif_disjuncts(default)
        if isinstance(default, dict)
        else default
    )
    if not kept:
        return default_opt
    return _rebuild_if_elseif_chain(kept, default_opt)


def _factor_boolean_dnf(node: dict[str, Any]) -> dict[str, Any]:
    if node.get("type") == "BINARY_OP" and str(node.get("operator") or "").upper() == "OR":
        mapped = _map_children(node, _factor_boolean_dnf)
        return _factor_or_tree(mapped)
    return _map_children(node, _factor_boolean_dnf)


def _flatten_and_conditions(node: dict[str, Any]) -> list[dict[str, Any]]:
    if node.get("type") != "BINARY_OP" or str(node.get("operator") or "").upper() != "AND":
        return [node]
    parts: list[dict[str, Any]] = []
    left = node.get("left")
    right = node.get("right")
    if isinstance(left, dict):
        parts.extend(_flatten_and_conditions(left))
    if isinstance(right, dict):
        parts.extend(_flatten_and_conditions(right))
    return parts or [node]


def _rebuild_and_chain(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        return {"type": "LITERAL", "value_type": "BOOLEAN", "value": True}
    out = parts[0]
    for part in parts[1:]:
        out = {"type": "BINARY_OP", "operator": "AND", "left": out, "right": part}
    return out


def _is_self_referential_equality(node: dict[str, Any]) -> bool:
    if node.get("type") != "BINARY_OP" or str(node.get("operator") or "") != "==":
        return False
    left = node.get("left")
    right = node.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return guard_conjunct_signature(left) == guard_conjunct_signature(right)


def _dedupe_and_list(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    unique: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, dict) and _is_self_referential_equality(part):
            continue
        sig = guard_conjunct_signature(part) if isinstance(part, dict) else _ast_signature(part)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(part)
    return unique


def _dedupe_and_guard_tree(node: Any) -> Any:
    """Drop duplicate AND conjuncts (e.g. repeated LOS hop membership)."""
    if not isinstance(node, dict):
        return node
    mapped = _map_children(node, _dedupe_and_guard_tree)
    if mapped.get("type") != "BINARY_OP" or str(mapped.get("operator") or "").upper() != "AND":
        return mapped
    parts = _dedupe_and_list(_flatten_and_conditions(mapped))
    return _rebuild_and_chain(parts)


def _equality_column_key(node: dict[str, Any]) -> tuple[Any, dict[str, Any]] | None:
    if node.get("type") != "BINARY_OP":
        return None
    if str(node.get("operator") or "").strip() not in {"==", "="}:
        return None
    left = node.get("left")
    right = node.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    if left.get("type") == "COLUMN_REF" and right.get("type") == "LITERAL":
        col = left
        lit = right
    elif right.get("type") == "COLUMN_REF" and left.get("type") == "LITERAL":
        col = right
        lit = left
    else:
        return None
    key = (
        str(col.get("entity") or "").upper(),
        str(col.get("relationship") or "").upper(),
        str(col.get("column") or "").upper(),
    )
    return key, node


def _factor_or_tree(node: dict[str, Any], _depth: int = 0) -> dict[str, Any]:
    parts = _flatten_or_conditions(node)
    if len(parts) < 2 or _depth > 32:
        return node
    and_lists = [_dedupe_and_list(_flatten_and_conditions(part)) for part in parts]

    common = _try_extract_common_and_conjuncts(and_lists)
    if common is not None:
        common_and, remainders = common
        if any(not rem for rem in remainders):
            return common_and
        rest = _factor_or_tree(
            _rebuild_or_chain([_rebuild_and_chain(rem) for rem in remainders]),
            _depth + 1,
        )
        return {"type": "BINARY_OP", "operator": "AND", "left": common_and, "right": rest}

    grouped = _try_group_equality_remainders(and_lists)
    if grouped is not None:
        return _factor_or_tree(_rebuild_or_chain(grouped), _depth + 1)

    greedy = _try_greedy_conjunct_factor(and_lists, _depth)
    if greedy is not None:
        return greedy
    return _rebuild_or_chain([_rebuild_and_chain(lst) for lst in and_lists])


def _try_extract_common_and_conjuncts(
    and_lists: list[list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]] | None:
    if len(and_lists) < 2:
        return None
    sig_sets = [frozenset(_ast_signature(n) for n in lst) for lst in and_lists]
    common_sigs = sig_sets[0]
    for extra in sig_sets[1:]:
        common_sigs &= extra
    if not common_sigs:
        return None
    common_nodes: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for node in and_lists[0]:
        sig = _ast_signature(node)
        if sig in common_sigs and sig not in seen:
            seen.add(sig)
            common_nodes.append(node)
    remainders = [
        [n for n in lst if _ast_signature(n) not in common_sigs] for lst in and_lists
    ]
    return _rebuild_and_chain(common_nodes), remainders


def _try_group_equality_remainders(
    and_lists: list[list[dict[str, Any]]],
) -> list[dict[str, Any]] | None:
    col_keys: list[Any] = []
    for lst in and_lists:
        for node in lst:
            found = _equality_column_key(node)
            if found:
                col_keys.append(found[0])
    best_score = 0
    best_out: list[dict[str, Any]] | None = None
    seen_keys: set[Any] = set()
    for col_key in col_keys:
        if col_key in seen_keys:
            continue
        seen_keys.add(col_key)
        groups: dict[Any, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {}
        ungrouped: list[list[dict[str, Any]]] = []
        for lst in and_lists:
            eqs = [
                n for n in lst
                if (_equality_column_key(n) or (None, None))[0] == col_key
            ]
            if len(eqs) != 1:
                ungrouped.append(lst)
                continue
            eq_sig = _ast_signature(eqs[0])
            rem = [n for n in lst if _ast_signature(n) != eq_sig]
            rem_sig = frozenset(_ast_signature(n) for n in rem)
            groups.setdefault(rem_sig, []).append((eqs[0], rem))
        score = sum(len(items) - 1 for items in groups.values() if len(items) >= 2)
        if score <= best_score:
            continue
        out: list[dict[str, Any]] = []
        for items in groups.values():
            if len(items) < 2:
                eq_node, rem = items[0]
                out.append(_rebuild_and_chain([eq_node, *rem]))
                continue
            eqs_or = _rebuild_or_chain([eq for eq, _ in items])
            rem = items[0][1]
            out.append(
                {"type": "BINARY_OP", "operator": "AND", "left": eqs_or, "right": _rebuild_and_chain(rem)}
                if rem
                else eqs_or
            )
        for lst in ungrouped:
            out.append(_rebuild_and_chain(lst))
        best_score = score
        best_out = out
    return best_out


def _try_greedy_conjunct_factor(
    and_lists: list[list[dict[str, Any]]],
    depth: int,
) -> dict[str, Any] | None:
    freq: dict[Any, int] = {}
    examples: dict[Any, dict[str, Any]] = {}
    for lst in and_lists:
        for node in lst:
            sig = _ast_signature(node)
            freq[sig] = freq.get(sig, 0) + 1
            examples.setdefault(sig, node)
    if not freq:
        return None
    best_sig = max(freq, key=lambda s: freq[s])
    if freq[best_sig] < 2:
        return None
    with_c: list[list[dict[str, Any]]] = []
    without_c: list[list[dict[str, Any]]] = []
    empty_remainder = False
    for lst in and_lists:
        if any(_ast_signature(n) == best_sig for n in lst):
            rem = [n for n in lst if _ast_signature(n) != best_sig]
            if not rem:
                empty_remainder = True
            with_c.append(rem)
        else:
            without_c.append(lst)
    conjunct = examples[best_sig]
    if empty_remainder:
        factored_with = conjunct
    else:
        rest = _factor_or_tree(
            _rebuild_or_chain([_rebuild_and_chain(rem) for rem in with_c]),
            depth + 1,
        )
        factored_with = {
            "type": "BINARY_OP",
            "operator": "AND",
            "left": conjunct,
            "right": rest,
        }
    if not without_c:
        return factored_with
    mixed = [_rebuild_and_chain(lst) for lst in without_c]
    mixed.append(factored_with)
    return _factor_or_tree(_rebuild_or_chain(mixed), depth + 1)


def _map_children(node: dict[str, Any], fn) -> dict[str, Any]:
    out = dict(node)
    for key, value in list(out.items()):
        if isinstance(value, dict):
            out[key] = fn(value)
        elif isinstance(value, list):
            out[key] = [fn(item) if isinstance(item, dict) else item for item in value]
    return out


_KEEP_ARITHMETIC_VALUES: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_KEEP_ARITHMETIC_VALUES", default=False
)


def _terminal_else_reads_column(node: dict[str, Any], entity: str, column: str) -> bool:
    """True when an IF chain's final ELSE reads the target column (a prior-value chain).

    Only such a chain may be collapsed to a pass-through of the column. A repeated
    IF whose last ELSE is a literal (e.g. ``IF(x>0) THEN x ELSE 0`` appearing in two
    UPDATE passes) is a computed value, and collapsing it discards the calculation.
    """
    current: Any = node
    while isinstance(current, dict) and current.get("type") == "IF_THEN_ELSE":
        current = current.get("else_branch")
    ent = normalize_table_name(entity or "").upper().lstrip("#")
    col = bare_ident(column or "").upper()
    for sub in _walk_ast(current) if isinstance(current, dict) else ():
        if (
            sub.get("type") == "COLUMN_REF"
            and str(sub.get("column") or "").upper() == col
            and normalize_table_name(str(sub.get("entity") or "")).upper().lstrip("#") == ent
        ):
            return True
    return False


def _is_predicate_node(node: dict[str, Any]) -> bool:
    """Comparison / AND / OR — a boolean, never a value backbone.

    Collapsing one to the target column yields nonsense such as
    ``IF(col)THEN(NetBalance > NetBalance)``.
    """
    if node.get("type") != "BINARY_OP":
        return False
    op = str(node.get("operator") or "").strip().upper()
    return op in {"==", "=", "!=", "<>", ">", ">=", "<", "<=", "AND", "OR"}


def _is_arithmetic_value(node: dict[str, Any]) -> bool:
    """A computed value such as ``USEDRV * ProvPerSecured`` (never an IF backbone)."""
    return (
        node.get("type") == "BINARY_OP"
        and str(node.get("operator") or "").strip() in _ARITHMETIC_OPS
    )


def _replace_backbone_duplicates(
    node: Any,
    entity: str,
    column: str,
    backbone_sigs: set[Any],
    *,
    preserve_exact: bool = False,
) -> Any:
    if not isinstance(node, dict):
        return node

    sig = _ast_signature(node)
    if (
        not preserve_exact
        and sig in backbone_sigs
        and _ast_depth(node) >= _MIN_CSE_DEPTH
        and node.get("type") != "COLUMN_REF"
        and not _is_predicate_node(node)
        and not (_KEEP_ARITHMETIC_VALUES.get() and _is_arithmetic_value(node))
        and not (
            _KEEP_ARITHMETIC_VALUES.get()
            and node.get("type") == "IF_THEN_ELSE"
            and not _terminal_else_reads_column(node, entity, column)
        )
    ):
        return _column_ref(entity, column)

    if node.get("type") == "IF_THEN_ELSE":
        cond = node.get("condition")
        then_b = node.get("then_branch")
        else_b = node.get("else_branch")
        new_cond = (
            _replace_backbone_duplicates(cond, entity, column, backbone_sigs)
            if isinstance(cond, dict)
            else cond
        )
        new_then = (
            _replace_backbone_duplicates(then_b, entity, column, backbone_sigs)
            if isinstance(then_b, dict)
            else then_b
        )
        new_else = (
            _replace_backbone_duplicates(
                else_b, entity, column, backbone_sigs, preserve_exact=True
            )
            if isinstance(else_b, dict)
            else else_b
        )
        return {
            "type": "IF_THEN_ELSE",
            "condition": new_cond,
            "then_branch": new_then,
            "else_branch": new_else,
        }

    out = dict(node)
    for key, value in list(out.items()):
        if isinstance(value, dict):
            out[key] = _replace_backbone_duplicates(value, entity, column, backbone_sigs)
        elif isinstance(value, list):
            out[key] = [
                _replace_backbone_duplicates(item, entity, column, backbone_sigs)
                if isinstance(item, dict)
                else item
                for item in value
            ]
    return out


def _identity_map_ast(node: Any) -> Any:
    """Walk AST dicts without transforming nodes (preserves native aggregates)."""
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if isinstance(value, dict):
            out[key] = _identity_map_ast(value)
        elif isinstance(value, list):
            out[key] = [
                _identity_map_ast(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            out[key] = value
    return out


def lower_set_aggregates(node: Any) -> Any:
    """Public entry: strip set-style MIN/MAX/SUM/COUNT to row-level AST (legacy)."""
    return _lower_set_aggregates(node)


def _lower_set_aggregates(node: Any) -> Any:
    from app.derivation.v2.platform_aggregates import NATIVE_AGGREGATES_ENABLED

    if NATIVE_AGGREGATES_ENABLED:
        return _identity_map_ast(node)
    if not isinstance(node, dict):
        return node
    out = dict(node)
    for key, value in list(out.items()):
        if isinstance(value, dict):
            out[key] = _lower_set_aggregates(value)
        elif isinstance(value, list):
            out[key] = [
                _lower_set_aggregates(item) if isinstance(item, dict) else item
                for item in value
            ]
    if out.get("type") != "FUNCTION_CALL":
        return out
    func = str(out.get("function_name") or "").strip().upper()
    if func not in _SET_AGG_FUNCS:
        return out
    args = out.get("arguments") or []
    if func == "SUM" and len(args) == 1 and isinstance(args[0], dict):
        return _lower_set_aggregates(args[0])
    if func == "COUNT":
        if len(args) == 1 and isinstance(args[0], dict):
            inner = _lower_set_aggregates(args[0])
            if inner.get("type") == "LITERAL" and str(inner.get("value_type") or "").upper() == "NUMBER":
                return inner
            return {
                "type": "IF_THEN_ELSE",
                "condition": {
                    "type": "FUNCTION_CALL",
                    "function_name": "ISNOTEMPTY",
                    "arguments": [inner],
                },
                "then_branch": {"type": "LITERAL", "value_type": "NUMBER", "value": 1},
                "else_branch": {"type": "LITERAL", "value_type": "NUMBER", "value": 0},
            }
        return out
    if func in {"MIN", "MAX"}:
        work = [a for a in args if isinstance(a, dict)]
        if work and _is_groupby_list_arg(work[-1]):
            work = work[:-1]
        if len(work) == 1:
            inner = work[0]
            if inner.get("type") == "IF_THEN_ELSE":
                return inner
            return inner
        if len(work) >= 2:
            work = [_hoist_if_from_arithmetic(a) or a for a in work]
            if _ast_signature(work[0]) == _ast_signature(work[1]) and len(work) == 2:
                return work[0]
            for i, arg in enumerate(work):
                if arg.get("type") != "IF_THEN_ELSE":
                    continue
                then_b = arg.get("then_branch")
                else_b = arg.get("else_branch")
                if not isinstance(then_b, dict) or not isinstance(else_b, dict):
                    continue
                rest = work[:i] + work[i + 1 :]

                def _agg_branch(branch: dict[str, Any]) -> dict[str, Any]:
                    return {
                        "type": "FUNCTION_CALL",
                        "function_name": func,
                        "arguments": [branch, *rest],
                    }

                pushed = {
                    "type": "IF_THEN_ELSE",
                    "condition": arg.get("condition"),
                    "then_branch": _agg_branch(then_b),
                    "else_branch": _agg_branch(else_b),
                }
                return _lower_set_aggregates(pushed)
            result = work[0]
            for nxt in work[1:]:
                result = _binary_min_max_to_if(func, result, nxt)
            return result
    return out


def _is_groupby_list_arg(node: dict[str, Any]) -> bool:
    return node.get("type") == "LIST_LITERAL"


def _is_never_null(node: Any) -> bool:
    """True for a node statically guaranteed non-NULL: a non-NULL literal, or a
    ``COALESCE`` whose last argument is one (``ISNULL(Col,0)`` parses to this)."""
    if not isinstance(node, dict):
        return False
    if node.get("type") == "LITERAL":
        return str(node.get("value_type") or "").upper() != "NULL" and node.get("value") is not None
    if node.get("type") == "FUNCTION_CALL":
        name = str(node.get("function_name") or "").upper()
        args = node.get("arguments") or []
        if name in {"MIN", "MAX"}:
            operands = [a for a in args if isinstance(a, dict) and a.get("type") != "LIST_LITERAL"]
            return len(operands) >= 2 and all(_is_never_null(a) for a in operands)
        if name != "COALESCE":
            return False
        return bool(args) and _is_never_null(args[-1])
    if node.get("type") == "BINARY_OP" and str(node.get("operator") or "").strip() in {"+", "-", "*"}:
        return _is_never_null(node.get("left")) and _is_never_null(node.get("right"))
    if node.get("type") == "BINARY_OP" and str(node.get("operator") or "").strip() == "/":
        # ``x / 100``: null-safe when x is and the divisor is a non-zero constant.
        divisor = node.get("right")
        return (
            _is_never_null(node.get("left"))
            and isinstance(divisor, dict)
            and divisor.get("type") == "LITERAL"
            and str(divisor.get("value_type") or "").upper() == "NUMBER"
            and divisor.get("value") not in (None, 0, 0.0)
        )
    if node.get("type") == "IF_THEN_ELSE":
        return _is_never_null(node.get("then_branch")) and _is_never_null(node.get("else_branch"))
    return False


def _coalesce_with(node: dict[str, Any], pad: dict[str, Any]) -> dict[str, Any]:
    # ``MAX(ISNULL(x,0), 0)`` must not become ``COALESCE(COALESCE(x,0),0) >
    # COALESCE(0,0)`` -- a null-safe operand needs no second fallback.
    if _is_never_null(node):
        return node
    return {
        "type": "FUNCTION_CALL",
        "function_name": "COALESCE",
        "arguments": [node, pad],
    }


def _infer_coalesce_pad(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    for candidate in (left, right):
        pad = _extract_coalesce_pad_literal(candidate)
        if pad is not None:
            return pad
    if _looks_date_valued(left) or _looks_date_valued(right):
        return {"type": "LITERAL", "value_type": "DATE", "value": "2099-12-31"}
    return {"type": "LITERAL", "value_type": "NUMBER", "value": 0}


def _extract_coalesce_pad_literal(node: dict[str, Any]) -> dict[str, Any] | None:
    if node.get("type") != "FUNCTION_CALL":
        return None
    if str(node.get("function_name") or "").upper() != "COALESCE":
        return None
    args = node.get("arguments") or []
    if len(args) != 2 or not isinstance(args[1], dict):
        return None
    if args[1].get("type") == "LITERAL":
        return args[1]
    return None


def _looks_date_valued(node: dict[str, Any]) -> bool:
    if node.get("type") == "LITERAL":
        vt = str(node.get("value_type") or "").upper()
        if vt in {"DATE", "DATETIME", "TIMESTAMP"}:
            return True
        value = str(node.get("value") or "")
        return bool(value) and ("-" in value or "/" in value)
    if node.get("type") == "FUNCTION_CALL":
        fn = str(node.get("function_name") or "").upper()
        if fn in {"DATEADD", "ADDDAY", "PERIOD", "EOM", "DATEDIFF", "TODATE"}:
            return True
    if node.get("type") == "IF_THEN_ELSE":
        then_b = node.get("then_branch")
        else_b = node.get("else_branch")
        return (
            (isinstance(then_b, dict) and _looks_date_valued(then_b))
            or (isinstance(else_b, dict) and _looks_date_valued(else_b))
        )
    return False


def _binary_min_max_to_if(
    func: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    pad = _infer_coalesce_pad(left, right)
    compare_left = _coalesce_with(left, pad)
    compare_right = _coalesce_with(right, pad)
    operator = "<" if func == "MIN" else ">"
    return {
        "type": "IF_THEN_ELSE",
        "condition": {
            "type": "BINARY_OP",
            "operator": operator,
            "left": compare_left,
            "right": compare_right,
        },
        "then_branch": left,
        "else_branch": right,
    }
