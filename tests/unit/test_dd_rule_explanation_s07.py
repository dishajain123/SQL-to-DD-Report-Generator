"""Rule Explanation regressions for PRO Update_AssetClass (S07)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.report.dd_rule_explanation import build_rule_explanation

_REPO = Path(__file__).resolve().parents[2]
_S07 = _REPO / "samples/sql/PRO_SPs_Sequenced/12_S07_PRO.Update_AssetClass.StoredProcedure.sql"


def _sql() -> str:
    return _S07.read_text(encoding="utf-8", errors="replace")


def _formula(entity: str, column: str) -> str:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    assert formula, getattr(row, "validation_errors", None)
    return formula


@pytest.mark.parametrize(
    "entity,column,needles",
    [
        (
            "##CustomerCal",
            "SysAssetClassAlt_Key",
            ("aging", "SUB", "870", "LOS", "tree("),
        ),
        (
            "##AccountCal",
            "FinalAssetClassAlt_Key",
            ("SysAssetClassAlt_Key", "NORMAL", "retains existing", "tree("),
        ),
        (
            "##CustomerCal",
            "DbtDt",
            ("DbtDt", "NULL", "ADDDAY", "tree("),
        ),
        (
            "##CustomerCal",
            "DegDate",
            ("ProcessDate", "FlgDeg", "tree("),
        ),
        (
            "ACLRUNNINGPROCESSSTATUS",
            "COMPLETED",
            ("RUNNINGPROCESSNAME", "Update_AssetClass", "retains existing COMPLETED", "tree("),
        ),
        (
            "ACLRUNNINGPROCESSSTATUS",
            "COUNT",
            ("Increments COUNT", "RUNNINGPROCESSNAME", "Update_AssetClass", "retains existing COUNT", "tree("),
        ),
        (
            "##CustomerCal",
            "LossDt",
            ("customer-wise balance", "NULL", "FlgDeg", "tree("),
        ),
    ],
)
def test_s07_rule_explanation_from_pipeline(entity: str, column: str, needles: tuple[str, ...]):
    formula = _formula(entity, column)
    ent = entity.lstrip("#")
    text = build_rule_explanation(ent, column, formula)
    assert text
    lower = text.lower()
    for needle in needles:
        if needle == "tree(":
            assert needle not in lower
        elif needle in ("aging", "retains existing"):
            assert needle in lower
        else:
            assert needle in text or needle.upper() in text.upper()
