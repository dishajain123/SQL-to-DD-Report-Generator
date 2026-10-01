"""Fail closed when a procedure's complete write semantics are unverified."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256

from app.guardrails.dd_row_coverage import mark_ledger_coverage
from app.models.core import DDStatus, ReviewState
from app.parsing.coverage_ledger import build_coverage_ledger


def enforce_completeness(rows, objects, structural_infos, entity_name_map=None, structural_errors=None):
    """Return an auditable inventory and withhold ACTIVE on incomplete objects.

    A valid formula is not proof of SQL equivalence. A missing write may
    affect downstream columns, so an unresolved object gates all its rows.
    Keep formulas and source fragments intact for review.
    """
    evidence = {}
    for oid, obj in objects.items():
        info = structural_infos.get(oid)
        blockers = list((structural_errors or {}).get(oid, []))
        ledger = None
        if info is None:
            blockers.append("Structural analysis is missing for this object")
        else:
            ledger = build_coverage_ledger(info, source_sql=obj.raw_sql)
            mark_ledger_coverage(ledger, rows, entity_name_map)
            blockers.extend(ledger.blockers)
        object_rows = [row for row in rows if oid in row.source_object_ids]
        for row in object_rows:
            blockers.extend(f"{row.entity_name}.{row.column_name}: {e}" for e in row.validation_errors)
            # These notes explicitly describe unproven semantics (including
            # dates and execution ordering); they cannot certify correctness.
            blockers.extend(f"{row.entity_name}.{row.column_name}: {e}" for e in row.advisory_notes)
        blockers = sorted(set(blockers))
        if blockers:
            reason = "Source completeness is unverified; see completeness.json for all unresolved writes and semantics"
            for row in object_rows:
                row.status = DDStatus.PENDING_REVIEW
                if row.review_state != ReviewState.UNSUPPORTED:
                    row.review_state = ReviewState.NEEDS_REVIEW
                if reason not in row.validation_errors:
                    row.validation_errors.append(reason)
        evidence[oid] = {
            "source_file": obj.source_file,
            "source_sha256": sha256(obj.raw_sql.encode("utf-8")).hexdigest(),
            "source_characters": len(obj.raw_sql),
            "source_lines": len(obj.raw_sql.splitlines()),
            "ready": not blockers and bool(ledger and ledger.ready_to_present),
            "blockers": blockers,
            "writes": [asdict(entry) for entry in ledger.entries] if ledger else [],
        }
    return {"ready": bool(evidence) and all(e["ready"] for e in evidence.values()), "objects": evidence}
