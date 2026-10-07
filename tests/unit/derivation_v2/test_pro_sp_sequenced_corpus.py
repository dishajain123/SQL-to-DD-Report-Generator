"""PRO_SPs_Sequenced: business-table formulas must be valid row-level 4X."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_dd_rows_for_chains
from app.derivation.v2.sql_text import is_staging_derivation_entity
from app.grammar.validator import validate_expression
from app.models.core import CanonicalModel, LineageChain
from app.parsing.dialect import detect_dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object
from app.parsing.write_inventory_scan import read_sql_file
from app.utils.entity_name_map import build_entity_name_map_for_tables

_REPO = Path(__file__).resolve().parents[3]
SP_DIR = _REPO / "samples/sql/PRO_SPs_Sequenced"
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)
_CHAR_CAP = 8000
_CORE_ENTITIES = frozenset({"ACCOUNTCAL", "CUSTOMERCAL", "COBORROWERCAL"})


def _is_staging_entity(entity: str) -> bool:
    return is_staging_derivation_entity(entity)


def _rows_for_sql(path: Path):
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
        technical_summary="corpus test",
        business_summary="",
    )
    return generate_dd_rows_for_chains(
        chains, [model], objects, infos, None, entity_name_map=mapping
    )


@pytest.fixture(scope="module")
def all_sp_paths() -> list[Path]:
    return sorted(SP_DIR.glob("*.sql"))


@pytest.mark.slow
def test_pro_sp_sequenced_no_set_aggregates_in_any_formula(all_sp_paths: list[Path]):
    failures: list[str] = []
    for path in all_sp_paths:
        for row in _rows_for_sql(path):
            expr = (row.display_derivation_expression or "").strip()
            if not expr or not _AGG_RE.search(expr):
                continue
            failures.append(f"{path.name}: {row.entity_name}.{row.column_name}")
    assert not failures, "MIN/MAX/SUM/COUNT in formula:\n" + "\n".join(failures[:30])


@pytest.mark.slow
def test_pro_sp_sequenced_core_business_formulas_when_present(all_sp_paths: list[Path]):
    failures: list[str] = []
    empty_core: list[str] = []
    for path in all_sp_paths:
        if path.name.startswith("00_MASTER"):
            continue
        for row in _rows_for_sql(path):
            ent = (row.entity_name or "").upper().replace("##", "")
            if ent not in _CORE_ENTITIES or _is_staging_entity(row.entity_name or ""):
                continue
            expr = (row.display_derivation_expression or "").strip()
            if not expr:
                empty_core.append(f"{path.name}: {row.entity_name}.{row.column_name}")
                continue
            if len(expr) > _CHAR_CAP:
                failures.append(
                    f"{path.name}: {row.entity_name}.{row.column_name}: "
                    f"length {len(expr)} > {_CHAR_CAP}"
                )
            vr = validate_expression(expr)
            if not vr.valid:
                failures.append(
                    f"{path.name}: {row.entity_name}.{row.column_name}: {vr.error}"
                )
            if row.validation_errors:
                failures.append(
                    f"{path.name}: {row.entity_name}.{row.column_name}: "
                    f"{'; '.join(row.validation_errors[:2])}"
                )
    assert not failures, "\n".join(failures[:40])


# Golden rows: must produce a non-empty, valid formula after optimization.
_GOLDEN_NON_EMPTY: list[tuple[str, str, str]] = [
    (
        "07_S02_PRO.DPD_Calculation.StoredProcedure.sql",
        "AccountCal",
        "DPD_IntService",
    ),
    (
        "11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql",
        "AccountCal",
        "FinalNpaDt",
    ),
    (
        "01_S00_PRO.InsertDataforAssetClassficationRBL.StoredProcedure.sql",
        "CustomerCal",
        "DegReason",
    ),
    (
        "01_S00_PRO.InsertDataforAssetClassficationRBL.StoredProcedure.sql",
        "AccountCal",
        "DegReason",
    ),
    (
        "21_S13_PRO.Marking_NPA_Reason_NPAAccount.StoredProcedure.sql",
        "AccountCal",
        "NPA_Reason",
    ),
]


@pytest.mark.slow
def test_pro_sp_sequenced_golden_columns_non_empty():
    failures: list[str] = []
    for filename, entity, column in _GOLDEN_NON_EMPTY:
        path = SP_DIR / filename
        rows = _rows_for_sql(path)
        match = [
            r
            for r in rows
            if (r.entity_name or "").upper().replace("##", "") == entity.upper()
            and (r.column_name or "").upper() == column.upper()
        ]
        if not match:
            failures.append(f"{filename}: missing row {entity}.{column}")
            continue
        row = match[0]
        expr = (row.display_derivation_expression or "").strip()
        if not expr:
            failures.append(f"{filename}: {entity}.{column} empty")
            continue
        if len(expr) > _CHAR_CAP:
            failures.append(f"{filename}: {entity}.{column} len={len(expr)}")
        if not validate_expression(expr).valid:
            failures.append(f"{filename}: {entity}.{column} grammar invalid")
        if _AGG_RE.search(expr):
            failures.append(f"{filename}: {entity}.{column} has aggregate")
    assert not failures, "\n".join(failures)
