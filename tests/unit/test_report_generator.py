from datetime import date

from app.models.core import CanonicalModel, ColumnType, DDRow, DDStatus, DerivationOption, GlossaryTerm, Intent, JobPlan
from app.report.condition_explainer import explain_expression
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
    assert "## 1. Process Overview" in text
    assert "### What the Source SQL Does" in text
    assert "technical summary" in text
    assert "### What It Means for the Business" in text
    assert "## 2. Rule Summary" in text
    assert "## 3. Detailed Business Rules & DD Conditions" in text
    # Superseded structure must be fully gone, not just renamed.
    assert "## 3. Process Control & Traceability" not in text
    assert "## 4. Business Rules / Logic Explanation" not in text
    assert "Special Cases" not in text
    assert "Period-Specific Rules" not in text
    assert "Aggregation / Max Logic" not in text

    assert "BR-001" in text
    assert "BR-002" not in text  # only one logical rule group because the formulas differ only by case
    # Rule Summary table row + the detail card heading + its anchor.
    assert "[BR-001](#br-001-refperiodmax)" in text
    assert '<a id="br-001-refperiodmax"></a>' in text
    assert "#### BR-001 — REFPERIODMAX" in text
    assert text.count("BR-001") >= 2
    assert 'IF(ISEMPTY("FCT_NPA_PRODUCT"."REFPERIODMAX"))THEN(0)ELSE("FCT_NPA_PRODUCT"."REFPERIODMAX")' in text
    assert "**Platform Condition:**" in text
    assert "**Human-Readable Explanation:**" in text
    assert "the result is 0" in text
    assert "the refperiodmax field in fct npa product" in text
    assert "**Purpose:**" not in text


def test_platform_condition_is_preserved_and_explanation_is_separate(tmp_path):
    """The machine-readable condition must remain byte-for-byte stable,
    while the human-readable explanation is allowed to rephrase the logic
    as long as it preserves the same meaning."""
    job_plan = JobPlan(job_id="job-4", intent=Intent.GENERATE_DD, company="Acme", platform="4X")
    model = CanonicalModel(
        chain_id="chain-4",
        job_id="job-4",
        object_ids=["obj-4"],
        technical_summary="technical summary",
        business_summary="business summary",
        evidence=["PRO.SampleProc"],
    )
    expression = (
        'IF("A"."X">1 OR "A"."Y"=="Y")'
        'THEN(IF(ISNOTEMPTY("B"."Z"))THEN("B"."Z")ELSE("B"."W"))'
        'ELSEIF("A"."X"<=0)THEN(0)'
        'ELSE(NULL)'
    )
    row = _row(display_derivation_expression=expression)

    out = generate_report(
        job_plan,
        [model],
        [row],
        tmp_path / "report.md",
        objects={"obj-4": type("Obj", (), {"name": "PRO.SampleProc"})()},
    )
    text = out.read_text()

    platform_condition = text.split("**Platform Condition:**")[1].split("**Human-Readable Explanation:**")[0]
    platform_condition = platform_condition.split("```text")[1].split("```")[0]
    explanation = text.split("**Human-Readable Explanation:**")[1].split("\n---")[0]

    def squash(s: str) -> str:
        return "".join(s.split())

    assert squash(platform_condition) == squash(expression)

    for fragment in [
        "at least one of the following is true",
        "the x field in a is greater than 1",
        "the y field in a equals yes",
        "another rule applies",
        "the z field in b has a value",
        "the result is the z field in b",
        "the result is the w field in b",
        "the result is no value",
    ]:
        assert fragment in explanation, f"missing fragment: {fragment!r}"

    assert explanation.strip().count("\n") >= 1


def test_condition_explainer_covers_simple_and_complex_cases():
    simple = explain_expression('IF(ISEMPTY("A"."B"))THEN(0)ELSE("A"."B")')
    complex_expr = (
        'IF("A"."X">1 OR "A"."Y"=="Y")'
        'THEN(IF(ISNOTEMPTY("B"."Z"))THEN("B"."Z")ELSE("B"."W"))'
        'ELSEIF("A"."X"<=0)THEN(0)'
        'ELSE(NULL)'
    )
    complex_explanation = explain_expression(complex_expr)

    assert simple == "If the b field in a is blank or missing, the result is 0. Otherwise, the result is the b field in a."
    assert complex_explanation is not None
    assert "at least one of the following is true" in complex_explanation
    assert "another rule applies" in complex_explanation
    assert "the result is no value" in complex_explanation


def test_condition_explainer_covers_nested_boolean_ranges_and_functions():
    expression = (
        'IF(AND('
        'NOT(ISEMPTY("A"."X")),'
        '("A"."Y" BETWEEN [1,30]),'
        '("A"."Z" IN ["Y","N"])'
        '))THEN('
        'IF(LOWER(TRIM("A"."CODE"))=="abc")THEN(REPLACE("A"."CODE","-"," "))ELSE(SUBSTR("A"."CODE",1,3))'
        ')ELSE('
        'IF(LEN(SUBSTR("A"."CODE",1,3))>0 OR "A"."DATE" BETWEEN [SOM("A"."DATE"),EOM("A"."DATE")])'
        'THEN(DATE("2026-01-01"))ELSE(PERIOD("A"."P"))'
        ')'
    )

    explanation = explain_expression(expression)

    assert explanation is not None
    for fragment in [
        "all of the following are true",
        "it is not true that",
        "is between 1 and 30",
        "is one of yes or no",
        "lowercased value",
        "trimmed value",
        "replaced by",
        "substring of the code field in a starting at 1 with length 3",
        "length of",
        "start of the month",
        "end of the month",
        "date value for",
        "period value for",
    ]:
        assert fragment in explanation


def test_condition_explainer_uses_safe_fallback_for_unknown_function():
    explanation = explain_expression('IF(BOGUSFUNC("A"."X"))THEN(1)ELSE(0)')

    assert explanation is not None
    assert "result of the function call" in explanation
    assert "bogusfunc" not in explanation.lower()


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
    assert "(technical)" in text  # Rule Summary table marks it too


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
    assert "Grammar validation failed: Unexpected end-of-input" in text
    # The Rule Summary table's own row must carry the flag, but the
    # Business Meaning table column itself is gone (moved into the card),
    # so this checks the summary line specifically.
    assert "PENDING_REVIEW: Keeps the rolling reference period" in text


def test_report_renders_cleanup_null_conditions_with_human_readable_explanation(tmp_path):
    job_plan = JobPlan(job_id="job-6", intent=Intent.GENERATE_DD, company="Acme", platform="4X")
    model = CanonicalModel(
        chain_id="chain-6",
        job_id="job-6",
        object_ids=["obj-6"],
        technical_summary="technical summary",
        business_summary="business summary",
        evidence=["PRO.SampleProc"],
    )
    row = _row(
        column_name="LASTCRDATE",
        display_derivation_expression="NULL",
        business_meaning="Clears the last credit date as part of a cleanup reset.",
    )

    out = generate_report(
        job_plan,
        [model],
        [row],
        tmp_path / "report.md",
        objects={"obj-6": type("Obj", (), {"name": "PRO.SampleProc"})()},
    )
    text = out.read_text()

    assert "**Platform Condition:**" in text
    assert "```text\nNULL\n```" in text
    assert "**Human-Readable Explanation:**" in text
    assert "The result is no value." in text


def test_tables_read_and_written_reflect_structural_info(tmp_path):
    from app.models.core import ObjectType, SQLObject, StructuralInfo, Dialect

    job_plan = JobPlan(job_id="job-5", intent=Intent.GENERATE_DD, company="Acme", platform="4X")
    model = CanonicalModel(
        chain_id="chain-5",
        job_id="job-5",
        object_ids=["obj-5"],
        technical_summary="technical summary",
        business_summary="business summary",
        evidence=["PRO.SampleProc"],
    )
    obj = SQLObject(
        object_id="obj-5",
        name="PRO.SampleProc",
        object_type=ObjectType.PROCEDURE,
        dialect=Dialect.ORACLE,
        raw_sql="BEGIN NULL; END;",
        source_file="sample.sql",
    )
    info = StructuralInfo(
        object_id="obj-5",
        tables_read=["SRC_TABLE"],
        tables_written=["TGT_TABLE"],
        columns_written_by_table={"TGT_TABLE": ["COL_A", "COL_B"]},
    )
    row = _row(entity_name="TGT_TABLE", column_name="COL_A")

    out = generate_report(
        job_plan,
        [model],
        [row],
        tmp_path / "report.md",
        objects={"obj-5": obj},
        structural_infos={"obj-5": info},
    )
    text = out.read_text()

    assert "### Tables Read" in text
    assert "SRC_TABLE" in text
    assert "### Tables Written" in text
    assert "TGT_TABLE" in text
    assert "COL_A, COL_B" in text
