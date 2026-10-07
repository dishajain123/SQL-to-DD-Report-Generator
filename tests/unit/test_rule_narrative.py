"""Business narrative helpers for Markdown reports."""
from __future__ import annotations

import json
from pathlib import Path

from app.report.report_generator import generate_report
from app.report.rule_narrative import (
    assert_no_forbidden_report_phrases,
    brief_formula_summary,
    brief_row_condition_summary,
    build_stakeholder_rule_explanation,
)
from app.models.core import (
    CanonicalModel,
    ColumnType,
    DDRow,
    DDStatus,
    DerivationOption,
    ExecutionStep,
    Intent,
    JobPlan,
    ObjectType,
    SQLObject,
    StructuralInfo,
    Dialect,
)

_PROJECT = Path(__file__).resolve().parents[2]
_S14 = (
    _PROJECT
    / "samples/sql/PRO_SPs_Sequenced/22_S14_PRO.UPDATE_NPA_TYPE.StoredProcedure.sql"
)
_DD062 = _PROJECT / "output/062_job-2f94b4fbc3/dd_rows.json"
_JOB062_FORMULA = json.loads(_DD062.read_text(encoding="utf-8"))[0]["display_derivation_expression"]


def test_npa_type_stakeholder_summary_lists_regular_sticky_multiple():
    if not _DD062.is_file():
        from app.derivation.v2.pipeline import generate_for_sql

        sql = _S14.read_text(encoding="utf-8")
        _, debug = generate_for_sql(sql, "AccountCal", "NpaType", llm_client=None)
        formula = debug.get("formula") or ""
    else:
        formula = _JOB062_FORMULA
    text = build_stakeholder_rule_explanation("AccountCal", "NpaType", formula)
    assert "REGULAR" in text
    assert "STICKY" in text
    assert "MULTIPLE" in text
    assert "VisionPLUS" in text
    assert "narrative model" not in text.lower()
    assert "could not be rendered" not in text.lower()


def test_brief_formula_summary_for_npa_type_is_short():
    if _DD062.is_file():
        step_value = json.loads(_DD062.read_text(encoding="utf-8"))[0]["execution_steps"][0][
            "assigned_value"
        ]
    else:
        step_value = (
            "IF(AND("
            'OR(AND("AccountCal"."CD" == 5, "AccountCal"."DPD_MAX" >= 90, "AccountCal"."DPD_MAX" <= 119)), '
            'OR("AccountCal"."DimAssetClass"."ASSETCLASSSHORTNAME" == "SUB"))'
            ')THEN("REGULAR")ELSE(NULL)'
        )
    summary = brief_formula_summary(step_value, column_name="NpaType")
    assert len(summary) < 300
    assert "REGULAR" in summary
    assert "THEN(" not in summary
    assert "..." not in summary


def test_brief_row_condition_uses_full_asset_class_key_label():
    cond = (
        'AND("AccountCal"."DimSourceDB"."SourceName" == "VisionPLUS", '
        '"AccountCal"."DimAssetClass"."AssetClassGroup" == "NPA", '
        '"AccountCal"."FinalAssetClassAlt_Key" > 1)'
    )
    summary = brief_row_condition_summary(cond)
    assert "FinalAssetClassAlt_Key" in summary
    assert "..." not in summary


def test_s14_report_markdown_has_no_placeholder_fallbacks(tmp_path):
    from app.derivation.v2.pipeline import generate_for_sql
    from app.parsing.structural_analysis import analyze_object

    sql = _S14.read_text(encoding="utf-8")
    row, debug = generate_for_sql(sql, "AccountCal", "NpaType", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    assert formula

    obj = SQLObject(
        object_id="s14",
        name="PRO.UPDATE_NPA_TYPE",
        object_type=ObjectType.PROCEDURE,
        dialect=Dialect.SQLSERVER,
        raw_sql=sql,
        source_file=_S14.name,
    )
    info = analyze_object(obj)
    row.display_derivation_expression = formula
    row.source_object_ids = ["s14"]
    if getattr(row, "execution_steps", None):
        inner_assign = row.execution_steps[0].assigned_value
    elif _DD062.is_file():
        inner_assign = json.loads(_DD062.read_text(encoding="utf-8"))[0]["execution_steps"][0][
            "assigned_value"
        ]
    else:
        inner_assign = formula
    row.execution_steps = [
        ExecutionStep(
            step=1,
            source_line=36,
            row_condition='AND("AccountCal"."DimSourceDB"."SourceName" == "VisionPLUS", "AccountCal"."DimAssetClass"."AssetClassGroup" == "NPA")',
            join_conditions=[
                "JOIN DimSourceDB ON ##AccountCal.SourceAlt_Key=DimSourceDB.SourceAlt_Key",
                "JOIN DimAssetClass ON ##AccountCal.FinalAssetClassAlt_Key=DimAssetClass.AssetClassAlt_Key",
            ],
            assigned_value=inner_assign,
        )
    ]

    model = CanonicalModel(
        chain_id="c1",
        job_id="j1",
        object_ids=["s14"],
        technical_summary="UPDATE_NPA_TYPE writes NpaType on AccountCal.",
        business_summary="Classifies NPA account types from CD and DPD.",
        evidence=["PRO.UPDATE_NPA_TYPE"],
    )
    plan = JobPlan(job_id="j1", intent=Intent.GENERATE_DD, company="x", platform="4X")
    report_path = generate_report(
        plan,
        [model],
        [row],
        tmp_path / "report.md",
        objects={"s14": obj},
        structural_infos={"s14": info},
    )
    text = report_path.read_text(encoding="utf-8")
    assert_no_forbidden_report_phrases(text)
    assert "- **REGULAR:**" in text
    assert "\n- **STICKY:**" in text
    assert "\n- **MULTIPLE:**" in text
    assert formula in text
    assert "Execution sequence:" in text
    assert len([line for line in text.splitlines() if "THEN(\"REGULAR\")" in line and "|" in line]) == 0
