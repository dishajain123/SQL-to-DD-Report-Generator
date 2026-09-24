"""Architecture step 12: Canonical Understanding Model.

Built per lineage chain (not per isolated object) — the technical and
business reasoning cover the whole chain of linked procedures together.
"""
from __future__ import annotations

from app.derivation.llm_client import LLMClient
from app.models.core import CanonicalModel, GlossaryTerm, LineageChain, SQLObject, StructuralInfo


def _fallback_technical_summary(
    ordered_objects: list[SQLObject],
    structural_infos: dict[str, StructuralInfo],
    chain: LineageChain,
) -> str:
    names = ", ".join(obj.name for obj in ordered_objects) or "the procedure"
    tables: list[str] = []
    for object_id in chain.order:
        info = structural_infos.get(object_id)
        if info is not None:
            tables.extend(info.tables_written or [])
    written = ", ".join(sorted(set(tables))[:8])
    if written:
        return f"{names} writes {written}. Column conditions are taken from the procedure SQL."
    return f"{names} derives its target columns from the procedure SQL."


def _fallback_business_summary(ordered_objects: list[SQLObject]) -> str:
    names = ", ".join(obj.name for obj in ordered_objects) or "This procedure"
    return (
        f"{names} applies the conditions encoded in the source SQL. "
        "The narrative model did not respond, so the report and the DD export "
        "use those SQL conditions directly."
    )


def build_canonical_model(
    chain: LineageChain,
    job_id: str,
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo],
    llm_client: LLMClient,
) -> CanonicalModel:
    ordered_objects = [objects[oid] for oid in chain.order]
    # Narrative model calls were blocking the API process itself: the status
    # endpoint then stopped answering and the UI aborted before any CSV or
    # Excel was written. Column conditions come from the SQL, so the summary
    # is filled locally and the export is not gated on a model response.
    technical_summary = _fallback_technical_summary(ordered_objects, structural_infos, chain)
    business_summary = _fallback_business_summary(ordered_objects)
    glossary_terms: list[GlossaryTerm] = []

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
        glossary_terms=glossary_terms,
        derived_rules=[],
        evidence=sorted(set(evidence)),
        confidence=round(avg_confidence, 3),
    )
