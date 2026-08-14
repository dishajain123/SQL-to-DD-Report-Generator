"""Architecture step 12: Canonical Understanding Model.

Built per lineage chain (not per isolated object) — the technical and
business reasoning cover the whole chain of linked procedures together.
"""
from __future__ import annotations

from app.derivation.llm_client import LLMClient
from app.models.core import CanonicalModel, LineageChain, SQLObject, StructuralInfo


def build_canonical_model(
    chain: LineageChain,
    job_id: str,
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo],
    llm_client: LLMClient,
) -> CanonicalModel:
    ordered_objects = [objects[oid] for oid in chain.order]
    sql_snippets = [f"-- Object: {o.name}\n{o.raw_sql}" for o in ordered_objects]

    technical_summary = llm_client.technical_reasoning(sql_snippets)
    business_summary = llm_client.business_reasoning(technical_summary)

    evidence = []
    for oid in chain.order:
        info = structural_infos[oid]
        evidence.append(objects[oid].name)
        evidence.extend(info.tables_read)
        evidence.extend(info.tables_written)

    avg_confidence = sum(structural_infos[oid].confidence for oid in chain.order) / len(chain.order)
    if chain.order_confidence == "low":
        avg_confidence = min(avg_confidence, 0.6)

    return CanonicalModel(
        chain_id=chain.chain_id,
        job_id=job_id,
        object_ids=chain.order,
        technical_summary=technical_summary,
        business_summary=business_summary,
        derived_rules=[],
        evidence=sorted(set(evidence)),
        confidence=round(avg_confidence, 3),
    )
