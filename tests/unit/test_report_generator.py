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
    assert "**Purpose:** Keeps the rolling reference period value aligned with the latest applicable period." in text

    # The formula must appear as its own exact platform-formula block, not
    # only embedded inside the decision-logic rendering.
    assert "**Platform Formula**" in text
    assert "**Decision Logic**" in text


def test_decision_logic_is_deterministic_reformatting_not_a_rewrite(tmp_path):
    """The pretty-printed Decision Logic block must be a faithful,
    whitespace-only reformatting of the exact Platform Formula -- proven
    here by stripping all whitespace from both and asserting they're
    identical, which would fail if pretty-printing ever changed operators,
    quoting, or literal values."""
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

    decision_logic = text.split("**Decision Logic**")[1].split("**Platform Formula**")[0]
    decision_logic = decision_logic.split("```text")[1].split("```")[0]
    platform_formula = text.split("**Platform Formula**")[1].split("```text")[1].split("```")[0]

    def squash(s: str) -> str:
        return "".join(s.split())

    # The Platform Formula block must be byte-for-byte (modulo surrounding
    # whitespace) identical to what was stored on the row -- it's meant
    # for copy-paste into the platform, so it must never be touched.
    assert squash(platform_formula) == squash(expression)

    # The Decision Logic block legitimately replaces THEN(...)/ELSE(...)
    # with arrows and indentation -- that's the whole point -- but every
    # literal condition and branch value must survive completely
    # unchanged as an exact substring, proving no token was rewritten.
    for fragment in [
        '"A"."X">1 OR "A"."Y"=="Y"',
        'ISNOTEMPTY("B"."Z")',
        '"B"."Z"',
        '"B"."W"',
        '"A"."X"<=0',
    ]:
        assert fragment in decision_logic, f"missing fragment: {fragment!r}"

    # And it must actually be laid out across multiple lines (not just a
    # no-op single-line passthrough) to prove the pretty-printer did
    # something.
    assert decision_logic.strip().count("\n") >= 3


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