#!/usr/bin/env python3
"""Audit DD formulas for all PRO_SPs_Sequenced SQL files (offline)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from app.derivation.v2.pipeline import generate_dd_rows_for_chains
from app.derivation.v2.sql_text import is_staging_derivation_entity
from app.grammar.validator import validate_expression
from app.models.core import CanonicalModel, LineageChain
from app.parsing.dialect import detect_dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object
from app.parsing.write_inventory_scan import read_sql_file
from app.utils.entity_name_map import build_entity_name_map_for_tables

ROOT = Path(__file__).resolve().parents[1]
SP_DIR = ROOT / "samples" / "sql" / "PRO_SPs_Sequenced"
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)
_CHAR_BUDGET = 8000
_THEN_RE = re.compile(r"THEN\s*\(", re.I)

# Columns checked for duplicate-tree / over-nesting heuristics.
_KEY_COLUMNS = frozenset(
    {
        "FINALNPADT",
        "DEGREASON",
        "SYSNPA_DT",
        "NPA_REASON",
        "SYSASSETCLASSALT_KEY",
        "FINALASSETCLASSALT_KEY",
        "ASSET_NORM",
        "DPD_INTSERVICE",
    }
)

# Sentinels that should not repeat many times after CSE (NPA / asset-class folds).
_DUP_SENTINELS = (
    "ALWYS_NPA",
    "REFPERIODNPA",
    "FLGPROCESSING",
    "1900-01-01",
    "ALWYS_STD",
    "CONDI_STD",
)


def _normalize_entity(entity: str) -> str:
    return (entity or "").strip().strip('"').upper()


def _normalize_column(column: str) -> str:
    return (column or "").strip().strip('"').upper()


def _is_staging_entity(entity: str) -> bool:
    return is_staging_derivation_entity(entity)


def _empty_reason(row) -> str:
    parts: list[str] = []
    for err in row.validation_errors or []:
        parts.append(str(err).strip())
    for note in row.advisory_notes or []:
        parts.append(str(note).strip())
    if parts:
        return "; ".join(dict.fromkeys(parts))
    return "empty formula (expansion limit, unsupported fold, or no assignment extracted)"


def _repeated_substring_hits(expr: str, min_len: int = 48, min_count: int = 3) -> list[str]:
    """Find coarse repeated chunks (same text appears min_count+ times)."""
    text = expr
    if len(text) < min_len * min_count:
        return []
    hits: list[str] = []
    # Slide windows at a few anchor lengths to keep this O(n)-ish.
    step = max(8, min_len // 4)
    for width in (min_len, min_len + 24, min_len + 48):
        counts: Counter[str] = Counter()
        for i in range(0, len(text) - width + 1, step):
            chunk = text[i : i + width]
            if chunk.count("(") < 2:
                continue
            counts[chunk] += 1
        for chunk, count in counts.most_common(5):
            if count >= min_count:
                preview = chunk[:60].replace("\n", " ")
                hits.append(f"chunk×{count} @{width}: {preview}…")
    return hits[:5]


def _duplicate_heuristic(entity: str, column: str, expr: str) -> list[str]:
    col = _normalize_column(column)
    if col not in _KEY_COLUMNS:
        return []
    flags: list[str] = []
    then_count = len(_THEN_RE.findall(expr))
    if then_count >= 12:
        flags.append(f"THEN( count={then_count} (possible un-pruned nesting)")
    for sentinel in _DUP_SENTINELS:
        count = expr.count(sentinel)
        if count >= 3:
            flags.append(f"{sentinel} repeated {count}×")
    for hit in _repeated_substring_hits(expr):
        flags.append(hit)
    return flags


def _issue_line(entity: str, column: str, message: str) -> str:
    return f"{entity}.{column}: {message}"


def audit_file(path: Path) -> dict:
    sql = read_sql_file(path)
    object_list = split_objects(sql, path.name, detect_dialect(sql))
    for index, obj in enumerate(object_list):
        obj.object_id = f"{path.stem}:{index}"
    objects = {o.object_id: o for o in object_list}
    infos = {oid: analyze_object(obj) for oid, obj in objects.items()}
    mapping = build_entity_name_map_for_tables(
        [t for info in infos.values() for t in info.tables_read + info.tables_written]
    )
    chains = [LineageChain(chain_id=path.stem, object_ids=list(objects), order=list(objects))]
    model = CanonicalModel(
        chain_id=path.stem,
        job_id=path.stem,
        object_ids=list(objects),
        technical_summary="audit",
        business_summary="",
    )
    rows = generate_dd_rows_for_chains(
        chains, [model], objects, infos, None, entity_name_map=mapping
    )

    issues: list[str] = []
    staging_issues: list[str] = []
    duplicate_flags: list[str] = []
    over_budget: list[str] = []

    with_formula = 0
    agg_hits = 0
    invalid = 0
    empty = 0
    empty_business = 0
    empty_staging = 0
    over_budget_count = 0
    duplicate_heuristic_count = 0

    for row in rows:
        entity = row.entity_name or ""
        column = row.column_name or ""
        staging = _is_staging_entity(entity)
        expr = (row.display_derivation_expression or "").strip()

        if not expr:
            empty += 1
            reason = _empty_reason(row)
            line = _issue_line(entity, column, f"empty — {reason}")
            if staging:
                empty_staging += 1
                staging_issues.append(line)
            else:
                empty_business += 1
                issues.append(line)
            continue

        with_formula += 1
        expr_len = len(expr)

        if expr_len > _CHAR_BUDGET:
            over_budget_count += 1
            msg = f"length {expr_len} exceeds {_CHAR_BUDGET} characters"
            line = _issue_line(entity, column, msg)
            over_budget.append(line)
            if staging:
                staging_issues.append(line)
            else:
                issues.append(line)

        if _AGG_RE.search(expr):
            agg_hits += 1
            line = _issue_line(entity, column, "aggregate MIN/MAX/SUM/COUNT in formula")
            if staging:
                staging_issues.append(line)
            else:
                issues.append(line)

        dup = _duplicate_heuristic(entity, column, expr)
        if dup:
            duplicate_heuristic_count += 1
            detail = "; ".join(dup)
            line = _issue_line(entity, column, f"duplicate-tree heuristic — {detail}")
            duplicate_flags.append(line)
            if not staging:
                issues.append(line)

        if row.validation_errors:
            invalid += 1
            for err in row.validation_errors[:3]:
                line = _issue_line(entity, column, str(err))
                if staging:
                    staging_issues.append(line)
                else:
                    issues.append(line)
        else:
            vr = validate_expression(expr)
            if not vr.valid:
                invalid += 1
                line = _issue_line(entity, column, f"grammar {vr.error}")
                if staging:
                    staging_issues.append(line)
                else:
                    issues.append(line)

        if row.advisory_notes and not staging:
            for note in row.advisory_notes[:2]:
                text = str(note).strip()
                if text and "workflow" in text.lower():
                    issues.append(_issue_line(entity, column, f"advisory — {text}"))

    # De-dupe issue lines while preserving order.
    issues = list(dict.fromkeys(issues))
    staging_issues = list(dict.fromkeys(staging_issues))

    return {
        "file": path.name,
        "rows": len(rows),
        "with_formula": with_formula,
        "empty_expressions": empty,
        "empty_expressions_business": empty_business,
        "empty_expressions_staging": empty_staging,
        "aggregate_formulas": agg_hits,
        "invalid_or_errors": invalid,
        "over_budget_formulas": over_budget_count,
        "duplicate_heuristic_flags": duplicate_heuristic_count,
        "issues": issues[:40],
        "staging_objects": {
            "empty_count": empty_staging,
            "issue_count": len(staging_issues),
            "issues": staging_issues[:30],
        },
        "over_budget": over_budget[:20],
        "duplicate_heuristic": duplicate_flags[:20],
    }


def _rollup_summary(results: list[dict]) -> dict:
    return {
        "files": len(results),
        "total_rows": sum(r["rows"] for r in results),
        "total_with_formula": sum(r["with_formula"] for r in results),
        "total_empty_expressions": sum(r["empty_expressions"] for r in results),
        "total_empty_business": sum(r["empty_expressions_business"] for r in results),
        "total_empty_staging": sum(r["empty_expressions_staging"] for r in results),
        "files_with_any_aggregate": sum(1 for r in results if r["aggregate_formulas"]),
        "files_with_business_issues": sum(
            1 for r in results if r["issues"] or r["empty_expressions_business"]
        ),
        "files_with_any_invalid": sum(
            1 for r in results if r["invalid_or_errors"] or r["aggregate_formulas"]
        ),
        "total_aggregate_formulas": sum(r["aggregate_formulas"] for r in results),
        "total_over_budget_formulas": sum(r["over_budget_formulas"] for r in results),
        "total_duplicate_heuristic_flags": sum(r["duplicate_heuristic_flags"] for r in results),
    }


def main() -> None:
    paths = sorted(SP_DIR.glob("*.sql"))
    results = [audit_file(p) for p in paths]
    summary = _rollup_summary(results)
    out = ROOT / "output" / "audit_pro_sp_sequenced.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "files": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    print("\n--- Business column gaps (non-staging) ---")
    for r in results:
        if not r["issues"] and not r["empty_expressions_business"]:
            continue
        print(
            f"\n{r['file']}: empty_business={r['empty_expressions_business']} "
            f"over_budget={r['over_budget_formulas']} dup_flags={r['duplicate_heuristic_flags']}"
        )
        for line in r["issues"][:12]:
            print(f"  - {line}")

    print("\n--- Staging / temp / backup (informational) ---")
    for r in results:
        st = r["staging_objects"]
        if st["empty_count"] or st["issues"]:
            print(f"\n{r['file']}: staging_empty={st['empty_count']}")
            for line in st["issues"][:6]:
                print(f"  - {line}")


if __name__ == "__main__":
    main()
