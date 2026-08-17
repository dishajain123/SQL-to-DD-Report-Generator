from datetime import date

from app.models.core import CanonicalModel, ColumnType, DDRow, DDStatus, DerivationOption, GlossaryTerm, Intent, JobPlan
from app.report.report_generator import generate_report


def _row(**overrides) -> DDRow:
    defaults = dict(
        entity_name="FCT_NPA_PRODUCT",
        column_name="REFPERIODMAX",
        column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression='IF(ISEMPTY("FCT_NPA_PRODUCT"."REFPERIODMAX"))THEN(0)ELSE("FCT_NPA_PRODUCT"."REFPERIODMAX")',
        effective_start_date=date(2026, 1, 1),
        status=DDStatus.ACTIVE,
        data_type="number",
        source_chain_id="chain-1",
        business_meaning="Keeps the rolling reference period value aligned with the latest applicable period.",
        validation_errors=[],
    )
    defaults.update(overrides)
    return DDRow(**defaults)


def test_report_uses_required_structure_and_rule_ids(tmp_path):
    job_plan = JobPlan(job_id="job-1", intent=Intent.GENERATE_DD, company="Acme", platform="4X")
    model = CanonicalModel(
        chain_id="chain-1",
        job_id="job-1",
        object_ids=["obj-1"],
        technical_summary="technical summary",
        business_summary="business summary. Additional detail is ignored for the top summary.",
        glossary_terms=[GlossaryTerm(term="DPD", definition="days past due")],
        evidence=["PRO.SampleProc"],
    )
    rows = [
        _row(),
        _row(
            column_name="REFPeriodMax",
            display_derivation_expression='IF(ISEMPTY("FCT_NPA_PRODUCT"."REFPeriodMax"))THEN(0)ELSE("FCT_NPA_PRODUCT"."REFPeriodMax")',
            effective_start_date=date(2026, 3, 1),
        ),
    ]

    out = generate_report(
        job_plan,
        [model],
        rows,
        tmp_path / "report.md",
        objects={"obj-1": type("Obj", (), {"name": "PRO.SampleProc"})()},
    )
    text = out.read_text()

    assert text.startswith("# DD Automation Report — PRO.SampleProc")
    assert "> **What this process does, in one line:** business summary." in text
    assert "## How to Read a Condition" in text
    assert "## Glossary" in text
    assert "## 1. Business Flow / Process Overview" in text
    assert "## 2. Column-Level Derivations & DD Conditions" in text
    assert "## 3. Business Rules / Logic Explanation" in text
    assert "## 3. Process Control & Traceability" not in text
    assert "| Column | Business Meaning | Depends On | Rule ID | Platform Formula | Effective Dates |" in text
    assert "BR-001" in text
    assert "BR-002" not in text  # only one logical rule group because the formulas differ only by case
    assert text.count("BR-001") >= 2
    assert "IF(ISEMPTY(\"FCT_NPA_PRODUCT\".\"REFPERIODMAX\"))THEN(0)ELSE(\"FCT_NPA_PRODUCT\".\"REFPERIODMAX\")" in text
    assert "- BR-001 REFPERIODMAX:" in text
    assert "Keeps the rolling reference period value aligned with the latest applicable period." in text
    assert "Process:" not in text
    assert "Company:" not in text
    assert "Platform:" not in text
    assert "Intent:" not in text
    assert "Source files:" not in text
    assert "Special Cases" not in text
    assert "Period-Specific Rules" not in text
    assert "Aggregation / Max Logic" not in text
    assert "This rule preserves the period-specific reference amount" not in text


def test_report_separates_technical_housekeeping_columns(tmp_path):
    job_plan = JobPlan(job_id="job-3", intent=Intent.GENERATE_DD, company="Acme", platform="4X")
    model = CanonicalModel(
        chain_id="chain-3",
        job_id="job-3",
        object_ids=["obj-3"],
        technical_summary="technical summary",
        business_summary="business summary.",
        evidence=["PRO.SampleProc"],
    )
    row = _row(entity_name="ACLRUNNINGPROCESSSTATUS", column_name="ERRORDATE")

    out = generate_report(
        job_plan,
        [model],
        [row],
        tmp_path / "report.md",
        objects={"obj-3": type("Obj", (), {"name": "PRO.SampleProc"})()},
    )
    text = out.read_text()

    assert "technical housekeeping, not business logic" in text
    assert "excluded from the business-rules count" in text


def test_report_marks_pending_review_items_without_altering_formula(tmp_path):
    job_plan = JobPlan(job_id="job-2", intent=Intent.GENERATE_DD, company="Acme", platform="4X")
    model = CanonicalModel(
        chain_id="chain-2",
        job_id="job-2",
        object_ids=["obj-2"],
        technical_summary="technical summary",
        business_summary="business summary",
        evidence=["PRO.SampleProc"],
    )
    row = _row(
        status=DDStatus.PENDING_REVIEW,
        validation_errors=["Grammar validation failed: Unexpected end-of-input"],
    )

    out = generate_report(
        job_plan,
        [model],
        [row],
        tmp_path / "report.md",
        objects={"obj-2": type("Obj", (), {"name": "PRO.SampleProc"})()},
    )
    text = out.read_text()

    assert "PENDING_REVIEW" in text
    assert 'IF(ISEMPTY("FCT_NPA_PRODUCT"."REFPERIODMAX"))THEN(0)ELSE("FCT_NPA_PRODUCT"."REFPERIODMAX")' in text
    assert 'PENDING_REVIEW' not in text.split('Platform Formula')[1].split('|')[0]
