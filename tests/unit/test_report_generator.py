from datetime import date

from app.models.core import (
    CanonicalModel,
    ColumnType,
    DDRow,
    DDStatus,
    DerivationOption,
    Dialect,
    GlossaryTerm,
    Intent,
    JobPlan,
    ObjectType,
    SQLObject,
)
from app.report.condition_explainer import explain_expression
from app.report.report_generator import _extract_dependencies, generate_report


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
    assert "- If the reference period max is blank or missing:" in text
    assert "- Return 0." in text
    assert "- Otherwise:" in text
    assert "- Return the reference period max." in text
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
        "- If at least one of the following is true: the x field is greater than 1 or the y field equals yes:",
        "- If the z field has a value:",
        "- Return the z field.",
        "- Otherwise, if the x field is less than or equal to 0:",
        "- Return 0.",
        "- Otherwise:",
        "- Leave the field blank.",
    ]:
        assert fragment in explanation, f"missing fragment: {fragment!r}"

    assert explanation.strip().count("\n") >= 1


def test_condition_explainer_renders_min_wrapped_conditional_completely():
    """Regression test: SQL's MIN(CASE WHEN ... END) pattern arrives as
    MIN(IF(...)THEN(...)ELSE(...)) -- a single conditional argument, not
    a flat list of values. The old code rendered only the condition and
    silently dropped the THEN/ELSE branches ("Return the lowest of If X
    is true."), which is incomplete to the point of misrepresenting what
    is being compared."""
    explanation = explain_expression(
        'MIN(IF(COALESCE("A"."FinalNpaDt","2099-12-31")>COALESCE("B"."DegDate","2099-12-31"))'
        'THEN("B"."DegDate")ELSE("A"."FinalNpaDt"))'
    )

    assert explanation is not None
    assert "the lowest value from" in explanation
    assert "the deg date" in explanation
    assert "otherwise the final npa dt" in explanation


def test_condition_explainer_renders_addday():
    """Regression test: ADDDAY is a documented platform function but had
    no handler, so it silently fell back to the meaningless "the result
    of the function call"."""
    explanation = explain_expression('ADDDAY("A"."BUSINESS_DATE",-30)')

    assert explanation is not None
    assert "the result of the function call" not in explanation
    assert "adding" in explanation and "days to" in explanation


def test_condition_explainer_covers_simple_and_complex_cases():
    simple = explain_expression('IF(ISEMPTY("A"."B"))THEN(0)ELSE("A"."B")')
    complex_expr = (
        'IF("A"."X">1 OR "A"."Y"=="Y")'
        'THEN(IF(ISNOTEMPTY("B"."Z"))THEN("B"."Z")ELSE("B"."W"))'
        'ELSEIF("A"."X"<=0)THEN(0)'
        'ELSE(NULL)'
    )
    complex_explanation = explain_expression(complex_expr)

    assert simple == "- If the b field is blank or missing:\n- Return 0.\n- Otherwise:\n- Return the b field."
    assert complex_explanation is not None
    assert "- If at least one of the following is true" in complex_explanation
    assert "- Otherwise, if the x field is less than or equal to 0:" in complex_explanation
    assert "- Leave the field blank." in complex_explanation


def test_condition_explainer_formats_coalesce_naturally():
    explanation = explain_expression(
        'IF(COALESCE("A"."BALANCE",0)>0 AND COALESCE("A"."FLGPROCESSING","N")=="N")'
        'THEN(COALESCE("ACCOUNTCAL"."var"."BUSINESS_DATE",NULL)-COALESCE("DPD"."DPD_MAX",0)+1)'
        'ELSE(COALESCE("A"."RefPeriodOverDrawn",0))'
    )

    assert explanation is not None
    assert "- If all of the following are true: the balance is greater than 0, treating blank as 0 and the flag processing equals no, treating blank as no:" in explanation
    assert "the business date minus the DPD max plus 1" in explanation
    assert "- Return the reference overdrawn period, treating blank as 0." in explanation


def test_condition_explainer_renders_flg_as_flag():
    explanation = explain_expression('IF("A"."FLGSMA"=="Y")THEN(1)ELSE(0)')

    assert explanation is not None
    assert "the flag SMA field equals yes" in explanation


def test_condition_explainer_does_not_carve_words_out_of_unrelated_identifiers():
    """Regression test for a bug where the all-caps identifier splitter
    matched a dictionary word (e.g. "NO", "ID") purely because it happens
    to be a substring of an unrelated word -- "ASSET_NORM" -> "asset no
    rm", "NORMAL" -> "no rmal", "MSME_COVID" -> "msme cov ID". These are
    not cosmetic: reading "no" into a word that has nothing to do with
    negation misrepresents the underlying condition. A split must only be
    accepted when EVERY resulting fragment is a real recognized word --
    a length check on the leftover is not sufficient, since "RMAL" and
    "COV" are both long enough to slip past a naive threshold while still
    being meaningless."""
    from app.report.condition_explainer import _render_name_text

    assert _render_name_text("ASSET_NORM") == "asset norm"
    assert _render_name_text("NORMAL") == "normal"
    assert _render_name_text("MSME_COVID") == "msme covid"
    assert _render_name_text("COVID_OTR_RF") == "covid otr rf"

    explanation = explain_expression(
        'IF(COALESCE("AccountCal"."ASSET_NORM","NORMAL")!="ALWYS_STD" '
        'AND COALESCE("AccountCal"."FlgDeg","N")=="Y")'
        'THEN("PRO"."PUI_CAL"."Asset_Norm")ELSE(NULL)'
    )
    assert explanation is not None
    assert "asset norm" in explanation
    assert "treating blank as normal" in explanation
    assert "no rm" not in explanation
    assert "cov id" not in explanation.lower()


def test_condition_explainer_keeps_existing_all_caps_splits_working():
    """Splits that are genuinely two real words must keep working after
    the full-coverage fix -- this is what stops the fix from being
    over-corrected into never splitting anything."""
    from app.report.condition_explainer import _render_name_text

    assert _render_name_text("FLGDEG") == "flag deg"
    assert _render_name_text("FLGPROCESSING") == "flag processing"
    assert _render_name_text("FLGSMA") == "flag SMA"
    assert _render_name_text("DPD_NOCREDIT") == "DPD no credit"


def test_condition_explainer_handles_coalesce_comparisons_and_dates():
    explanation = explain_expression(
        'IF(COALESCE("A"."FLAG","N")=="Y" OR COALESCE("A"."AMOUNT",0)>=10)THEN(DATE("2026-01-01"))ELSE(NULL)'
    )

    assert explanation is not None
    assert "the flag equals yes, treating blank as no" in explanation
    assert "the amount is greater than or equal to 10, treating blank as 0" in explanation
    assert "the date value for 2026-01-01" in explanation


def test_report_dependency_extraction_omits_literals_and_constants():
    expression = (
        'IF("A"."FLAG"=="Y" AND COALESCE("A"."BALANCE",0)>0 AND "B"."STATUS"=="ACTIVE")'
        'THEN("A"."TARGET")ELSE(NULL)'
    )

    assert _extract_dependencies(expression) == [
        "A.FLAG",
        "A.BALANCE",
        "B.STATUS",
        "A.TARGET",
    ]


def test_report_dependency_extraction_preserves_exact_table_casing():
    expression = (
        'IF(ISNOTEMPTY("ACCOUNTCAL"."AccountEntityID") AND "ACCOUNTCAL"."Status"=="ACTIVE")'
        'THEN("ACCOUNTCAL"."AccountEntityID")ELSE(NULL)'
    )

    assert _extract_dependencies(expression) == [
        "ACCOUNTCAL.AccountEntityID",
        "ACCOUNTCAL.Status",
    ]


def test_report_resolves_aliases_to_source_table_names(tmp_path):
    job_plan = JobPlan(job_id="job-5", intent=Intent.GENERATE_DD, company="Acme", platform="4X")
    model = CanonicalModel(
        chain_id="chain-5",
        job_id="job-5",
        object_ids=["obj-5"],
        technical_summary="technical summary",
        business_summary="business summary",
        evidence=["PRO.SampleProc"],
    )
    source_object = SQLObject(
        object_id="obj-5",
        name="PRO.SampleProc",
        object_type=ObjectType.PROCEDURE,
        dialect=Dialect.ORACLE,
        raw_sql="SELECT a.AccountEntityID FROM ACCOUNTCAL a",
        source_file="sample.sql",
    )
    row = _row(
        display_derivation_expression='IF(ISNOTEMPTY("a"."AccountEntityID"))THEN("a"."AccountEntityID")ELSE(NULL)',
        source_object_ids=["obj-5"],
    )

    out = generate_report(
        job_plan,
        [model],
        [row],
        tmp_path / "report.md",
        objects={"obj-5": source_object},
    )
    text = out.read_text()

    assert '"ACCOUNTCAL"."AccountEntityID"' in text
    assert "a.AccountEntityID" not in text
    assert "Depends On" in text
    assert "ACCOUNTCAL.AccountEntityID" in text


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
        "- If all of the following are true:",
        "It is not true that the x field is blank or missing",
        "the y field is between 1 and 30",
        "the z field is one of yes or no",
        "the lowercased value of the trimmed value of the code field",
        "the value of the code field with - replaced by a space",
        "the substring of the code field starting at 1 with length 3",
        "the length of the substring of the code field starting at 1 with length 3",
        "the start of the month for the date field",
        "the end of the month for the date field",
        "the date value for 2026-01-01",
        "the period value for the p field",
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


def test_report_marks_pending_review_items_without_showing_unsafe_formula(tmp_path):
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
        display_derivation_expression="",
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
    assert "(pending review — no formula was accepted)" in text
    assert 'IF(ISEMPTY("FCT_NPA_PRODUCT"."REFPERIODMAX"))THEN(0)ELSE("FCT_NPA_PRODUCT"."REFPERIODMAX")' not in text
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
    assert "- Leave the field blank." in text


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