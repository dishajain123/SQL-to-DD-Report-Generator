"""Derivation Engine v2 — main orchestration entrypoint.

Runs the 4-phase AST pipeline per written column and returns ``DDRow``
objects compatible with the existing report / export / review stack.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, Optional

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.phase1_lineage import LineageMap, build_lineage_map
from app.derivation.v2.phase2_mutation_folder import MutationPass, fold_column_mutations
from app.derivation.v2.phase3_ast_generator import generate_ast
from app.derivation.v2.phase4_metadata import build_metadata, metadata_to_dd_row
from app.models.core import (
    CanonicalModel,
    DDRow,
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
    if max_workers == 1:
        rows: list[DDRow] = []
        for job in jobs:
            rows.extend(_run_column_job(*job))
        return rows

    results: list[DDRow] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_column_job, *job) for job in jobs]
        for fut in futures:
            results.extend(fut.result())
    return results


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
    mutations = fold_column_mutations(
        sql_text, target_entity, target_column, lineage, entity_map
    )
    ast = generate_ast(
        mutations,
        target_entity=resolve_entity_name(target_entity, entity_map) or target_entity,
        target_column=target_column,
        llm_client=llm_client,
    )
    try:
        formula = compile_ast_to_4x_string(ast)
    except Exception as exc:
        logger.exception("AST compile failed for %s.%s", target_entity, target_column)
        formula = ""
        compile_error = str(exc)
    else:
        compile_error = None

    entity = resolve_entity_name(target_entity, entity_map) or target_entity
    mutation_deps: list[str] = []
    for m in mutations:
        mutation_deps.extend(m.dependency_refs or [])
    meta = build_metadata(
        target_entity=entity,
        target_column=target_column,
        formula=formula,
        ast=ast,
        source_sql=sql_text,
        mutation_count=len(mutations),
        timekey_map=timekey_map,
        business_summary=business_summary,
        mutation_sql_fragments=[m.raw_sql for m in mutations if m.raw_sql],
        mutation_dependency_refs=mutation_deps,
    )
    if compile_error:
        meta.validation_errors.append(f"AST compile error: {compile_error}")
        meta.confidence = min(meta.confidence, 0.2)

    row = metadata_to_dd_row(
        meta,
        source_chain_id=source_chain_id,
        source_object_ids=list(source_object_ids or []),
        source_statement_refs=[
            f"stmt #{m.statement_index} (ordinal={m.ordinal})" for m in mutations
        ],
        source_statement_sql=[m.raw_sql for m in mutations],
    )
    debug = {
        "lineage": lineage.as_dict(),
        "mutations": [m.as_dict() for m in mutations],
        "ast": ast,
        "formula": formula,
        "metadata": meta.as_platform_dict(),
    }
    return row, debug


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

        columns_by_table = info.columns_written_by_table or {}
        for table, columns in columns_by_table.items():
            entity = resolve_entity_name(table, entity_name_map) or canonical_logical_name(
                table
            )
            # Skip obvious run-status / audit sinks that are not derivation targets.
            if _is_non_derivation_table(entity):
                continue
            for column in columns:
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
) -> list[DDRow]:
    try:
        mutations = fold_column_mutations(
            obj.raw_sql, entity, column, lineage, entity_name_map
        )
        if not mutations:
            # No UPDATE sites found for this column — skip silently.
            return []

        ast = generate_ast(
            mutations,
            target_entity=entity,
            target_column=column,
            llm_client=llm_client,
        )
        try:
            formula = compile_ast_to_4x_string(ast)
            compile_error = None
        except Exception as exc:
            formula = ""
            compile_error = str(exc)

        meta = build_metadata(
            target_entity=entity,
            target_column=column,
            formula=formula,
            ast=ast,
            source_sql=obj.raw_sql,
            mutation_count=len(mutations),
            timekey_map=timekey_map,
            business_summary=canonical_model.business_summary or "",
            mutation_sql_fragments=[m.raw_sql for m in mutations if m.raw_sql],
            mutation_dependency_refs=[
                ref for m in mutations for ref in (m.dependency_refs or [])
            ],
        )
        if compile_error:
            meta.validation_errors.append(f"AST compile error: {compile_error}")
            meta.confidence = min(meta.confidence, 0.2)

        row = metadata_to_dd_row(
            meta,
            source_chain_id=chain.chain_id,
            source_object_ids=list(chain.object_ids),
            source_statement_refs=[
                f"{obj.source_file} stmt #{m.statement_index} (ordinal={m.ordinal})"
                for m in mutations
            ],
            source_statement_sql=[m.raw_sql for m in mutations],
            data_type=_guess_data_type(column, formula),
        )
        return [row]
    except Exception as exc:  # pragma: no cover - never crash the whole job
        logger.exception(
            "v2 derivation failed for %s.%s in %s: %s",
            entity,
            column,
            obj.object_id,
            exc,
        )
        return []


def _is_non_derivation_table(entity: str) -> bool:
    bare = canonical_logical_name(entity).upper()
    return bare in {
        "RUNSTATUS",
        "SYSDAYMATRIX",
        "JOB_LOG",
        "ERROR_LOG",
    }


def _guess_data_type(column: str, formula: str) -> str:
    name = (column or "").upper()
    if any(tok in name for tok in ("DATE", "DT", "TIME")):
        return "Date"
    if any(tok in name for tok in ("FLAG", "FLG", "YN")):
        return "String"
    if any(tok in name for tok in ("AMT", "AMOUNT", "PCT", "PERCENT", "BAL", "DPD")):
        return "Decimal"
    if formula and any(ch.isdigit() for ch in formula) and "IF(" not in formula.upper():
        return "Decimal"
    return "String"
