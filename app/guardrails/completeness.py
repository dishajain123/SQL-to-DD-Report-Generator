"""Fail closed when a procedure's complete write semantics are unverified."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256

from app.guardrails.dd_row_coverage import mark_ledger_coverage
from app.models.core import DDStatus, ReviewState
from app.parsing.coverage_ledger import build_coverage_ledger

COMPLETENESS_GATE_REASON = (
    "Source completeness is unverified; see completeness.json for all unresolved writes and semantics"
)


def enforce_completeness(rows, objects, structural_infos, entity_name_map=None, structural_errors=None):
    """Return an auditable inventory and withhold ACTIVE on incomplete objects.

    A valid formula is not proof of SQL equivalence. A missing write may
    affect downstream columns, so unresolved *object-level* inventory gates
    every row for that procedure.

    Row ``validation_errors`` gate only that row. ``advisory_notes`` are
    informational (MERGE workflow, synthetic dates, same-procedure deps) and
    must not demote unrelated rows or the whole procedure.
    """
    evidence = {}
    for oid, obj in objects.items():
        info = structural_infos.get(oid)
        object_blockers = list((structural_errors or {}).get(oid, []))
        ledger = None
        if info is None:
            object_blockers.append("Structural analysis is missing for this object")
        else:
            ledger = build_coverage_ledger(info, source_sql=obj.raw_sql)
            mark_ledger_coverage(ledger, rows, entity_name_map)
            object_blockers.extend(ledger.blockers)

        object_rows = [row for row in rows if oid in row.source_object_ids]

        gating_blockers = sorted(set(object_blockers))
        advisories: list[str] = []
        for row in object_rows:
            gating_blockers.extend(
                f"{row.entity_name}.{row.column_name}: {e}" for e in row.validation_errors
            )
            advisories.extend(
                f"{row.entity_name}.{row.column_name}: {e}" for e in row.advisory_notes
            )
        gating_blockers = sorted(set(gating_blockers))
        advisories = sorted(set(advisories))

        if object_blockers:
            reason = COMPLETENESS_GATE_REASON
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
            "ready": (
                not object_blockers
                and bool(ledger and ledger.ready_to_present)
                and all(not row.validation_errors for row in object_rows)
            ),
            "blockers": gating_blockers,
            "advisories": advisories,
            "writes": [asdict(entry) for entry in ledger.entries] if ledger else [],
        }
    return {"ready": bool(evidence) and all(e["ready"] for e in evidence.values()), "objects": evidence}
