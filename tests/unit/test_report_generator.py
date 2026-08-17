from datetime import date

from app.models.core import CanonicalModel, ColumnType, DDRow, DDStatus, DerivationOption, Intent, JobPlan
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
        business_summary="business summary",
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
    assert "## 1. Process Overview" in text
    assert "## 2. How to Read a DD Condition" in text
    assert "## 3. Business Logic" in text
    assert "## 4. Column-Level Derivations & DD Conditions" in text
    assert "## 5. Process Control & Traceability" in text
    assert "## 6. Business Rules / Logic Explanation" in text
    assert "| Rule ID | Column | Business Logic | Special Cases | Effective Dates |" in text
    assert "| Rule ID | Entity | Column | Effective Dates | Period Logic |" in text
    assert "| Column | Business Meaning | Depends On | Rule ID | Platform Formula | Effective Dates |" in text
    assert "BR-001" in text
    assert "BR-002" not in text  # only one logical rule group because the formulas differ only by case
    assert text.count("BR-001") >= 2
    assert "IF(ISEMPTY(\"FCT_NPA_PRODUCT\".\"REFPERIODMAX\"))THEN(0)ELSE(\"FCT_NPA_PRODUCT\".\"REFPERIODMAX\")" in text
    assert "This rule preserves the period-specific reference amount" in text


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
    assert "Unexpected end-of-input" in text
    assert 'IF(ISEMPTY("FCT_NPA_PRODUCT"."REFPERIODMAX"))THEN(0)ELSE("FCT_NPA_PRODUCT"."REFPERIODMAX")' in text
    assert 'PENDING_REVIEW' not in text.split('Platform Formula')[1].split('|')[0]
