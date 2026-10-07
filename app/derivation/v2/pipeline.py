"""Derivation Engine v2 — main orchestration entrypoint.

Runs the 4-phase AST pipeline per written column and returns ``DDRow``
objects compatible with the existing report / export / review stack.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from app.derivation.v2.ast_compiler import (
    DATA_TYPE_STRING,
    column_type_key,
    compile_ast_to_4x_string,
    data_type_from_column_name,
    infer_value_data_type,
    sql_type_to_data_type,
)
from app.derivation.v2.execution_steps import build_execution_steps, is_identity_write
from app.derivation.v2.phase1_lineage import LineageMap, build_lineage_map
from app.derivation.v2.phase2_mutation_folder import (
    MutationPass,
    MutationSourceIndex,
    _written_columns_by_table,
    fold_column_mutations,
)
from app.derivation.v2.phase3_ast_generator import generate_ast
from app.derivation.derivation_option import format_expression_syntax
from app.derivation.v2.phase4_metadata import build_metadata, metadata_to_dd_row
from app.derivation.v2.semantic_checks import mutation_semantic_errors
from app.derivation.v2.sql_text import extract_declared_column_types, is_staging_derivation_entity, normalize_table_name
from app.models.core import (
    CanonicalModel,
    DDRow,
    ExecutionStep,
    LineageChain,
    SQLObject,
    StructuralInfo,
)
from app.utils.config import settings
from app.utils.entity_name_map import resolve_entity_name
from app.utils.identity import canonical_logical_name
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_dd_rows_for_chains(
    chains: list[LineageChain],
    canonical_models: list[CanonicalModel],
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo],
    llm_client: Any,
    function_reference: str = "",
    entity_name_map: dict[str, str] | None = None,
    timekey_map: dict[int, date] | None = None,
    rag_store: Any = None,
) -> list[DDRow]:
    """Batch DD generation across all chains (v2 AST pipeline).

    Signature mirrors the legacy engine so the orchestrator can swap
    imports without changing call sites. ``function_reference`` /
    ``rag_store`` are accepted for compatibility and unused in v2.
    """
    del function_reference, rag_store  # v2 does not use RAG/prompt references

    if len(chains) != len(canonical_models):
        raise ValueError("Every lineage chain must have a canonical model")
    jobs: list[tuple] = []
    for chain, model in zip(chains, canonical_models):
        jobs.extend(
            _build_column_jobs(
                chain=chain,
                canonical_model=model,
                objects=objects,
                structural_infos=structural_infos,
                llm_client=llm_client,
                entity_name_map=entity_name_map,
                timekey_map=timekey_map,
            )
        )

    if not jobs:
        return []

    max_workers = max(1, min(settings.dd_generation_max_workers, len(jobs)))
    results: list[tuple[DDRow, _DerivedColumn | None]] = []
    if max_workers == 1:
        for job in jobs:
            results.extend(_run_column_job(*job))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_column_job, *job) for job in jobs]
            for fut in futures:
                results.extend(fut.result())

    _resolve_copied_data_types(results)
    rows = [row for row, _ in results]
    rows.sort(key=_execution_sort_key(chains))
    return rows


def generate_dd_rows(
    chain: LineageChain,
    canonical_model: CanonicalModel,
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo],
    llm_client: Any,
    function_reference: str = "",
    entity_name_map: dict[str, str] | None = None,
    timekey_map: dict[int, date] | None = None,
    rag_store: Any = None,
) -> list[DDRow]:
    """Single-chain convenience wrapper."""
    return generate_dd_rows_for_chains(
        chains=[chain],
        canonical_models=[canonical_model],
        objects=objects,
        structural_infos=structural_infos,
        llm_client=llm_client,
        function_reference=function_reference,
        entity_name_map=entity_name_map,
        timekey_map=timekey_map,
        rag_store=rag_store,
    )


def generate_for_sql(
    sql_text: str,
    target_entity: str,
    target_column: str,
    *,
    entity_map: dict[str, str] | None = None,
    llm_client: Any | None = None,
    source_chain_id: str = "v2-direct",
    source_object_ids: list[str] | None = None,
    business_summary: str = "",
    timekey_map: dict[int, date] | None = None,
) -> tuple[DDRow, dict[str, Any]]:
    """Run phases 1–4 for one entity.column against raw SQL (CLI / tests)."""
    lineage = build_lineage_map(sql_text, entity_map)
    entity = resolve_entity_name(target_entity, entity_map) or target_entity
    derived = _derive_column(
        sql_text,
        entity,
        target_column,
        lineage,
        entity_map,
        fold_entity=target_entity,
        llm_client=llm_client,
        business_summary=business_summary,
        timekey_map=timekey_map,
        statement_label="",
    )
    row = metadata_to_dd_row(
        derived.meta,
        source_chain_id=source_chain_id,
        source_object_ids=list(source_object_ids or []),
        source_statement_refs=[
            f"stmt #{m.statement_index} (ordinal={m.ordinal})" for m in derived.mutations
        ],
        source_statement_sql=[m.raw_sql for m in derived.mutations],
        data_type=derived.data_type,
        ast=derived.ast,
    )
    derived.apply_to(row)
    debug = {
        "lineage": lineage.as_dict(),
        "mutations": [m.as_dict() for m in derived.mutations],
        "ast": derived.ast,
        "formula": derived.formula,
        "exception_handler_formula": derived.exception_formula,
        "metadata": derived.meta.as_platform_dict(),
    }
    return row, debug


@dataclass
class _DerivedColumn:
    """Everything phases 2–4 produce for one column, before it becomes a DDRow."""

    mutations: list[MutationPass]
    ast: dict[str, Any]
    formula: str
    exception_formula: str
    meta: Any
    data_type: str
    declared_type: str | None
    scalar_types: dict[str, str]
    execution_order: int | None
    steps: list[ExecutionStep]
    advisories: list[str]
    workflow_gates: list[str]

    def apply_to(self, row: DDRow) -> None:
        row.execution_order = self.execution_order
        row.execution_steps = list(self.steps)
        row.workflow_gates = list(self.workflow_gates)
        row.exception_handler_expression = format_expression_syntax(
            self.exception_formula or ""
        )
        row.advisory_notes = list(dict.fromkeys([*row.advisory_notes, *self.advisories]))


def _compile(ast: dict[str, Any], *, entity: str, column: str) -> tuple[str, str | None]:
    try:
        return (
            compile_ast_to_4x_string(
                ast, target_entity=entity, target_column=column
            ),
            None,
        )
    except Exception as exc:
        return "", str(exc)


def _derive_column(
    sql_text: str,
    entity: str,
    column: str,
    lineage: LineageMap,
    entity_map: dict[str, str] | None,
    *,
    fold_entity: str | None = None,
    llm_client: Any = None,
    business_summary: str = "",
    timekey_map: dict[int, date] | None = None,
    statement_label: str = "",
    source_index: MutationSourceIndex | None = None,
) -> _DerivedColumn:
    mutations = fold_column_mutations(
        sql_text, fold_entity or entity, column, lineage, entity_map, source_index=source_index
    )
    # The CATCH handler only runs when the main path fails, so its writes are
    # never folded into the main formula as if they were later UPDATEs.
    # A column written only inside CATCH keeps that handler as its formula.
    main = [m for m in mutations if not m.is_exception_handler]
    handler = [m for m in mutations if m.is_exception_handler]
    primary = main or handler

    ast = generate_ast(primary, target_entity=entity, target_column=column, llm_client=llm_client)
    formula, compile_error = _compile(ast, entity=entity, column=column)
    exception_formula = ""
    handler_ast: dict[str, Any] | None = None
    if main and handler:
        handler_ast = generate_ast(handler, target_entity=entity, target_column=column)
        exception_formula, _ = _compile(handler_ast, entity=entity, column=column)

    meta = build_metadata(
        target_entity=entity,
        target_column=column,
        formula=formula,
        ast=ast,
        source_sql=sql_text,
        mutation_count=len(primary),
        timekey_map=timekey_map,
        business_summary=business_summary,
        mutation_sql_fragments=[m.raw_sql for m in mutations if m.raw_sql],
        mutation_dependency_refs=[ref for m in mutations for ref in (m.dependency_refs or [])],
    )
    meta.validation_errors.extend(mutation_semantic_errors(primary, sql_text))
    if compile_error:
        meta.validation_errors.append(f"AST compile error: {compile_error}")
        meta.confidence = min(meta.confidence, 0.2)

    steps, advisories = build_execution_steps(
        mutations,
        sql_text=sql_text,
        target_entity=entity,
        target_column=column,
        statement_label=statement_label,
    )
    if handler and not main:
        advisories.append(
            f"{entity}.{column} is written only inside the BEGIN CATCH exception handler; "
            "this formula applies only when the procedure fails."
        )

    gates: list[str] = []
    for m in primary:
        if m.workflow_gate:
            label = f"{m.workflow_gate} := IF {m.workflow_gate_condition}"
            if label not in gates:
                gates.append(label)
    if gates:
        advisories.append(
            f"Procedural context: {entity}.{column} is written inside procedure-wide "
            f"branches ({'; '.join(gates)}). SQL Server evaluates each gate once for the "
            "whole table and runs only the first branch that holds; the formula applies "
            "each branch's condition row by row instead. Rows that satisfy a later "
            "branch's condition while an earlier gate holds for the run can differ, so "
            "exact behaviour needs a platform workflow step that evaluates the gate."
        )

    if is_staging_derivation_entity(entity):
        advisories.append(
            f"{entity}.{column} is written on an intermediate staging/backup object "
            "(temp table, CTE alias, or *_BKUP); it is not a core Data Dictionary "
            "export target — formula completeness is best-effort."
        )

    declared_types = extract_declared_column_types(sql_text)
    declared = _declared_column_type(declared_types, fold_entity or entity, entity, column=column)
    data_type = (
        sql_type_to_data_type(declared)
        or infer_value_data_type(
            ast, target_entity=entity, target_column=column,
            scalar_types=declared_types.get("@"),
        )
        or (handler_ast and infer_value_data_type(
            handler_ast, target_entity=entity, target_column=column,
            scalar_types=declared_types.get("@"),
        ))
        or data_type_from_column_name(column)
        or DATA_TYPE_STRING
    )

    return _DerivedColumn(
        mutations=mutations,
        ast=ast,
        formula=formula,
        exception_formula=exception_formula,
        meta=meta,
        data_type=data_type,
        declared_type=declared,
        scalar_types=declared_types.get("@", {}),
        execution_order=_execution_order(
            [m for m in mutations if not is_identity_write(m, entity, column)] or mutations,
            column,
        ),
        steps=steps,
        advisories=advisories,
        workflow_gates=gates,
    )


def _declared_column_type(
    declared: dict[str, dict[str, str]], *entity_names: str, column: str
) -> str | None:
    for name in entity_names:
        table = normalize_table_name(name or "").upper()
        for key in (table, f"#{table.lstrip('#')}", table.lstrip("#")):
            found = declared.get(key, {}).get(column.upper())
            if found:
                return found
    return None


def _execution_order(mutations: list[MutationPass], column: str) -> int | None:
    """Offset of the column's first assignment (statement start + column offset).

    Columns set in the same statement keep their SET / column-list order.
    """
    if not mutations:
        return None
    first = min(mutations, key=lambda m: m.source_position)
    match = re.search(rf"(?i)\b{re.escape(column)}\b", first.raw_sql or "")
    return first.source_position + (match.start() if match else 0)


def _build_column_jobs(
    *,
    chain: LineageChain,
    canonical_model: CanonicalModel,
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo],
    llm_client: Any,
    entity_name_map: dict[str, str] | None,
    timekey_map: dict[int, date] | None,
) -> list[tuple]:
    jobs: list[tuple] = []
    # Cache lineage per object SQL.
    lineage_by_oid: dict[str, LineageMap] = {}

    for oid in chain.order or chain.object_ids:
        obj = objects.get(oid)
        info = structural_infos.get(oid)
        if obj is None or info is None:
            continue
        if oid not in lineage_by_oid:
            lineage_by_oid[oid] = build_lineage_map(obj.raw_sql, entity_name_map)

        source_index = MutationSourceIndex.build(obj.raw_sql)
        columns_by_table = dict(info.columns_written_by_table or {})
        # sqlglot occasionally omits SET targets on complex UPDATEs; the regex
        # mutation index still sees columns like AccountCal.FinalAssetClassAlt_Key.
        for table_key, col_names in _written_columns_by_table(
            source_index, lineage_by_oid[oid], entity_name_map
        ).items():
            display_table = table_key
            for existing in columns_by_table:
                if normalize_table_name(existing).upper() == table_key.upper():
                    display_table = existing
                    break
            bucket = columns_by_table.setdefault(display_table, [])
            upper_seen = {c.upper() for c in bucket}
            for col in sorted(col_names):
                if col.upper() not in upper_seen:
                    bucket.append(col)
                    upper_seen.add(col.upper())
        # The same logical column is often written under case-variant spellings
        # (``Asset_Norm`` / ``ASSET_NORM``, ``##AccountCal`` / ``##ACCOUNTCAL``).
        # Mutation folding is case-insensitive, so one job covers all spellings;
        # a second job would emit a duplicate DD row for the same field.
        seen_targets: set[tuple[str, str]] = set()
        for table, columns in columns_by_table.items():
            entity = resolve_entity_name(table, entity_name_map) or canonical_logical_name(
                table
            )
            # ``#X`` staging a load of permanent ``X``: keep the hash so the two do
            # not share (and overwrite) one set of DD rows.
            norm_table = normalize_table_name(str(table))
            if (
                norm_table.startswith("#")
                and not norm_table.startswith("##")
                and entity.lstrip("#").upper() in lineage_by_oid[oid].hash_collisions
            ):
                entity = "#" + entity.lstrip("#")
            # Skip obvious run-status / audit sinks that are not derivation targets.
            if _is_non_derivation_table(entity):
                continue
            for column in columns:
                target_key = (canonical_logical_name(entity), canonical_logical_name(column))
                if target_key in seen_targets:
                    continue
                seen_targets.add(target_key)
                jobs.append(
                    (
                        obj,
                        info,
                        lineage_by_oid[oid],
                        entity,
                        column,
                        chain,
                        canonical_model,
                        llm_client,
                        entity_name_map,
                        timekey_map,
                        source_index,
                    )
                )
    return jobs


def _run_column_job(
    obj: SQLObject,
    info: StructuralInfo,
    lineage: LineageMap,
    entity: str,
    column: str,
    chain: LineageChain,
    canonical_model: CanonicalModel,
    llm_client: Any,
    entity_name_map: dict[str, str] | None,
    timekey_map: dict[int, date] | None,
    source_index: MutationSourceIndex | None = None,
) -> list[tuple[DDRow, _DerivedColumn | None]]:
    try:
        derived = _derive_column(
            obj.raw_sql,
            entity,
            column,
            lineage,
            entity_name_map,
            llm_client=llm_client,
            business_summary=canonical_model.business_summary or "",
            timekey_map=timekey_map,
            statement_label=f"{obj.source_file} ",
            source_index=source_index,
        )
        row = metadata_to_dd_row(
            derived.meta,
            source_chain_id=chain.chain_id,
            source_object_ids=[obj.object_id],
            source_statement_refs=[
                f"{obj.source_file} stmt #{m.statement_index} (ordinal={m.ordinal})"
                for m in derived.mutations
            ],
            source_statement_sql=[m.raw_sql for m in derived.mutations],
            data_type=derived.data_type,
            ast=derived.ast,
        )
        derived.apply_to(row)
        return [(row, derived)]
    except Exception as exc:  # pragma: no cover - never crash the whole job
        logger.exception(
            "v2 derivation failed for %s.%s in %s: %s",
            entity,
            column,
            obj.object_id,
            exc,
        )
        meta = build_metadata(target_entity=entity, target_column=column,
                              formula="", source_sql=obj.raw_sql)
        meta.validation_errors.append(f"Generation failed: {exc}")
        return [(metadata_to_dd_row(meta, source_chain_id=chain.chain_id,
                                    source_object_ids=[obj.object_id],
                                    data_type=data_type_from_column_name(column)
                                    or DATA_TYPE_STRING), None)]


def _is_non_derivation_table(entity: str) -> bool:
    bare = canonical_logical_name(entity).upper()
    return bare in {
        "RUNSTATUS",
        "SYSDAYMATRIX",
        "JOB_LOG",
        "ERROR_LOG",
        # Per-step execution audit (INSERT … ORIGINAL_LOGIN(), 'RUNNING', GETDATE();
        # UPDATE … EndTime = GETDATE()); its values are runtime metadata, not DD rules.
        "PROCESSMONITOR",
    }


def _resolve_copied_data_types(results: list[tuple[DDRow, _DerivedColumn | None]]) -> None:
    """Let a column that copies another DD column inherit that column's type.

    ``SET S.DpdBucket = A.DpdBucket`` carries no literal to infer from; once
    every row has a first-pass type, re-infer with the other rows' types
    known. Declared (CREATE TABLE) types are never overridden.
    """
    for _ in range(2):  # a second pass settles copy-of-a-copy chains
        known = {
            column_type_key(row.entity_name, row.column_name): row.data_type
            for row, _ in results
        }
        for row, derived in results:
            if derived is None or derived.declared_type:
                continue
            others = {k: v for k, v in known.items()
                      if k != column_type_key(row.entity_name, row.column_name)}
            inferred = infer_value_data_type(
                derived.ast,
                target_entity=row.entity_name,
                target_column=row.column_name,
                known_types=others,
                scalar_types=derived.scalar_types,
            )
            if inferred:
                row.data_type = inferred


def _execution_sort_key(chains: list[LineageChain]):
    """Rows in chain order, then object order, then SQL execution order."""
    chain_rank = {chain.chain_id: i for i, chain in enumerate(chains)}
    object_rank: dict[str, int] = {}
    for chain in chains:
        for i, oid in enumerate(chain.order or chain.object_ids):
            object_rank.setdefault(oid, i)

    def key(row: DDRow) -> tuple:
        oid = row.source_object_ids[0] if row.source_object_ids else ""
        order = row.execution_order if row.execution_order is not None else float("inf")
        return (chain_rank.get(row.source_chain_id, len(chain_rank)),
                object_rank.get(oid, len(object_rank)), order,
                row.entity_name.upper(), row.column_name.upper())

    return key
