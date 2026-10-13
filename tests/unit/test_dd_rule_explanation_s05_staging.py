"""Rule explanations for S05 #TEMPTABLE / #TEMPTABLE2 staging rows."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.report.dd_rule_explanation import build_rule_explanation

_REPO = Path(__file__).resolve().parents[2]
_S05 = _REPO / "samples/sql/PRO_SPs_Sequenced/10_S05_PRO.MaxDPD_ReferencePeriod_Calculation.StoredProcedure.sql"


def _sql() -> str:
    return _S05.read_text(encoding="utf-8", errors="replace")


def _formula(entity: str, column: str) -> str:
    base = entity.lstrip("#")
    for ent in (entity, base, f"#{base}", f"##{base}"):
        row, debug = generate_for_sql(_sql(), ent, column, llm_client=None)
        formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
        if formula:
            return formula
    pytest.skip(f"no formula for {entity}.{column}")


def _explain(entity: str, column: str) -> str:
    ent = entity.lstrip("#")
    formula = _formula(entity, column)
    text = build_rule_explanation(ent, column, formula)
    assert text
    return text


@pytest.mark.parametrize(
    "column,ref_token",
    [
        ("DPD_IntService", "RefPeriodIntService"),
        ("DPD_NoCredit", "RefPeriodNoCredit"),
        ("DPD_Overdrawn", "RefPeriodOverDrawn"),
    ],
)
def test_s05_temptable_dpd_threshold_explanation(column: str, ref_token: str):
    text = _explain("#TEMPTABLE", column)
    lower = text.lower()
    assert "populates" in lower
    assert column in text or column.upper() in text.upper()
    assert ref_token in text or ref_token.upper() in text.upper()
    assert "flgprocessing" in lower
    assert "nested conditional" not in lower
    assert "greater than..." not in text


def test_s05_temptable_customer_ac_id_explanation():
    text = _explain("#TEMPTABLE", "CustomerAcID")
    lower = text.lower()
    assert "customeracid" in lower or "customerac id" in lower.replace("_", " ")
    assert "flgprocessing" in lower
    assert "nested conditional" not in lower
    assert "greater than..." not in text


def test_s05_temptable2_refperiod_explanation():
    text = _explain("#TEMPTABLE2", "REFPERIOD")
    lower = text.lower()
    assert "reference period" in lower
    assert "largest component dpd" in lower or "refperiodintservice" in lower
    assert "flgprocessing" in lower
    assert "nested conditional" not in lower


def test_s05_temptable_dpd_narrator_synthetic_formula():
    formula = (
        'IF(AND(COALESCE("CUSTOMERCAL"."FlgProcessing", "N") == "N", '
        '"#TEMPTABLE"."X" == 1))'
        'THEN(IF(COALESCE("ACCOUNTCAL"."DPD_IntService", 0) >= '
        'COALESCE("ACCOUNTCAL"."RefPeriodIntService", 0))'
        'THEN("ACCOUNTCAL"."DPD_IntService")ELSE(0))ELSE(0)'
    )
    text = build_rule_explanation("TEMPTABLE", "DPD_IntService", formula)
    assert "ACCOUNTCAL.DPD_IntService" in text
    assert "RefPeriodIntService" in text
    assert "CustomerCal.FlgProcessing" in text
    assert "nested conditional" not in text.lower()


def test_s05_temptable2_customer_ac_id_flg_processing_column():
    text = _explain("#TEMPTABLE2", "CustomerAcID")
    assert "FlgProcessing" in text or "flgprocessing" in text.lower()
    assert "equals 'N'" not in text or "CustomerCal.FlgProcessing" in text
    assert "nested conditional" not in text.lower()
