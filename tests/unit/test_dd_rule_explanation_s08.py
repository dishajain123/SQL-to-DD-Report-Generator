"""Rule explanations for PRO NPA_Erosion_Aging (S08)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.report.dd_rule_explanation import build_rule_explanation

_REPO = Path(__file__).resolve().parents[2]
_S08 = _REPO / "samples/sql/PRO_SPs_Sequenced/13_S08_PRO.NPA_Erosion_Aging.StoredProcedure.sql"


def _sql() -> str:
    return _S08.read_text(encoding="utf-8", errors="replace")


def _formula(entity: str, column: str) -> str:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    if not formula:
        pytest.skip(f"no formula for {entity}.{column}")
    return formula


def test_s08_cte_netbalance_explanation():
    formula = _formula("CTE_CustomerWiseBalance", "NetBalance")
    text = build_rule_explanation("CTE_CustomerWiseBalance", "NetBalance", formula)
    assert "SUM(PrincOutStd)" in text or "PrincOutStd" in text
    assert "SecApp" in text
    assert "nested conditional" not in text.lower()
    assert "..." not in text


def test_s08_flg_erosion_explanation():
    formula = _formula("CustomerCal", "FlgErosion")
    text = build_rule_explanation("CustomerCal", "FlgErosion", formula)
    assert "MoveToLoss" in text
    assert "MoveToDB1" in text
    assert "sets to 'N'" in text or "sets to 'N'." in text
    assert len(text) > 80
    assert "..." not in text


def test_s08_erosion_dt_explanation():
    formula = _formula("CustomerCal", "ErosionDt")
    text = build_rule_explanation("CustomerCal", "ErosionDt", formula)
    assert "@PROCESSDATE" in text
    assert "nested conditional" not in text.lower()


def test_s08_sys_asset_class_alt_key_explanation():
    formula = _formula("CustomerCal", "SysAssetClassAlt_Key")
    text = build_rule_explanation("CustomerCal", "SysAssetClassAlt_Key", formula)
    assert "Security Erosion" in text
    assert "NPA Aging" in text
    assert "Manual Overrides" not in text
    assert "period('month'" not in text.lower()


def test_s08_dbt_dt_explanation():
    formula = _formula("CustomerCal", "DbtDt")
    text = build_rule_explanation("CustomerCal", "DbtDt", formula)
    assert "Security Erosion" in text
    assert "SUB_Months" in text or "Sub-Standard" in text
    assert "Manual Overrides" not in text
    assert "..." not in text
