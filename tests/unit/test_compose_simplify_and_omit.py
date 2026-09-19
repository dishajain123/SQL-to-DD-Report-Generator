"""Regression: composed IF nesting must not keep dead/tautology branches."""
from __future__ import annotations

from app.derivation.dd_generation_engine import (
    _should_omit_passthrough_dd_row,
    _simplify_composed_expression,
)


def test_collapse_tautology_else_branches():
    raw = (
        'IF(ISNOTEMPTY("A"."Prev"))THEN("Y")ELSE('
        'IF("A"."Due" >= ADDDAY("A"."var"."BUSINESS_DATE", -3))THEN("N")ELSE("N"))'
    )
    assert _simplify_composed_expression(raw) == (
        'IF(ISNOTEMPTY("A"."Prev"))THEN("Y")ELSE("N")'
    )


def test_sample_08_restructure_eligible_keeps_nested_chain_and_scheme_else():
    """Sample 08 must keep the nested STANDARD/SMA eligibility CASE.

    NOT_ASSESSED is a later override; SCHEME_CLOSED is the procedural ELSE.
    The CASE outcomes must not disappear even when the IF date compare is
    a source anomaly (unreachable).
    """
    from pathlib import Path

    from app.derivation.dd_generation_engine import (
        _assignment_sites,
        _compose_simple_assignment_expression,
    )
    from app.grammar.validator import validate_expression
    from app.parsing.dialect import detect_dialect
    from app.parsing.object_splitter import split_objects
    from app.parsing.structural_analysis import analyze_object

    sql = Path("samples/sql/08_Loan_Restructuring_Eligibility.sql").read_text()
    obj = split_objects(sql, "08.sql", detect_dialect(sql))[0]
    info = analyze_object(obj)
    sites = _assignment_sites(info, "RestructureEligible", target_table="LoanAccountCal")
    composed = _compose_simple_assignment_expression(
        sites, "LoanAccountCal", "RestructureEligible", procedure_sql=sql
    )
    assert composed is not None
    assert validate_expression(composed).valid
    assert "NOT_ASSESSED" in composed
    assert "SCHEME_CLOSED" in composed
    assert "STANDARD" in composed and "SMA" in composed
    assert "PriorRestructureCount" in composed
    assert "OverdueDays" in composed
    assert 'THEN("Y")' in composed
    assert 'ELSE("N")' in composed


def test_sample_09_review_reason_drops_noop_else_and_embeds_case_in_concat():
    """Sample 09 quarter IF is always true; ELSE `SET col = col` is a no-op.

    Executable path: CASE reason, then append `_QUARTER_END` when ratio < 0.6.
    """
    from pathlib import Path

    from app.derivation.dd_generation_engine import (
        _assignment_sites,
        _compose_simple_assignment_expression,
    )
    from app.grammar.validator import validate_expression
    from app.parsing.dialect import detect_dialect
    from app.parsing.object_splitter import split_objects
    from app.parsing.structural_analysis import analyze_object

    sql = Path("samples/sql/09_Provision_Coverage_Merge.sql").read_text()
    obj = split_objects(sql, "09.sql", detect_dialect(sql))[0]
    info = analyze_object(obj)
    sites = _assignment_sites(info, "ReviewReason", target_table="#ProvisionCoverage")
    composed = _compose_simple_assignment_expression(
        sites, "ProvisionCoverage", "ReviewReason", procedure_sql=sql
    )
    assert composed is not None
    assert validate_expression(composed).valid
    assert "_QUARTER_END" in composed
    assert "SEVERE_UNDERCOVER" in composed
    assert 'CONCAT(IF(ISEMPTY("ProvisionCoverage"."CoverageRatio"))' in composed
    # Dead ELSE no-op must not wrap as an outer self-read.
    assert 'CoverageRatio" < 0.5)THEN("ProvisionCoverage"."ReviewReason")' not in composed

    reason_sites = _assignment_sites(info, "Reason", target_table="CollectionsQueue")
    reason = _compose_simple_assignment_expression(
        reason_sites, "CollectionsQueue", "Reason", procedure_sql=sql
    )
    assert reason is not None
    assert '"ProvisionCoverage"."ReviewReason" CONTAINS' in reason or (
        '."ReviewReason" CONTAINS' in reason
    )


def test_sample_10_notifycount_keeps_feeschedule_projection():
    from pathlib import Path

    from app.derivation.dd_generation_engine import (
        _assignment_sites,
        _compose_simple_assignment_expression,
    )
    from app.grammar.validator import validate_expression
    from app.parsing.dialect import detect_dialect
    from app.parsing.object_splitter import split_objects
    from app.parsing.structural_analysis import analyze_object

    sql = Path("samples/sql/10_Overdue_Account_Late_Fee_Assessment.sql").read_text()
    obj = split_objects(sql, "10.sql", detect_dialect(sql))[0]
    info = analyze_object(obj)
    composed = _compose_simple_assignment_expression(
        _assignment_sites(info, "NotifyCount", target_table="LoanAccountCal"),
        "LoanAccountCal",
        "NotifyCount",
        procedure_sql=sql,
    )
    assert composed and validate_expression(composed).valid
    assert "FeeSchedule" in composed and "NotifyCount" in composed


def test_sample_15_eligible_day_one_pending_and_watch_days_inlined():
    from pathlib import Path

    from app.derivation.dd_generation_engine import (
        _assignment_sites,
        _compose_simple_assignment_expression,
    )
    from app.grammar.validator import validate_expression
    from app.parsing.dialect import detect_dialect
    from app.parsing.object_splitter import split_objects
    from app.parsing.structural_analysis import analyze_object

    sql = Path("samples/sql/15_NPA_Upgrade_Watchlist_Merge.sql").read_text()
    obj = split_objects(sql, "15.sql", detect_dialect(sql))[0]
    info = analyze_object(obj)
    composed = _compose_simple_assignment_expression(
        _assignment_sites(info, "EligibleForUpgrade", target_table="#WatchlistStaging"),
        "WatchlistStaging",
        "EligibleForUpgrade",
        procedure_sql=sql,
    )
    assert composed and validate_expression(composed).valid
    assert 'DATEPART("d"' in composed
    assert "PENDING_APPROVAL" in composed
    assert "365" in composed and "545" in composed
    assert "StandardWatchPeriodDays" not in composed


def test_sample_16_cover_amount_exclusive_if_elseif_else():
    from pathlib import Path

    from app.derivation.dd_generation_engine import (
        _assignment_sites,
        _compose_simple_assignment_expression,
    )
    from app.grammar.validator import validate_expression
    from app.parsing.dialect import detect_dialect
    from app.parsing.object_splitter import split_objects
    from app.parsing.structural_analysis import analyze_object

    sql = Path("samples/sql/16_Guarantee_Cover_Appropriation.sql").read_text()
    obj = split_objects(sql, "16.sql", detect_dialect(sql))[0]
    info = analyze_object(obj)
    composed = _compose_simple_assignment_expression(
        _assignment_sites(info, "CoverAppropriatedAmount", target_table="LoanAccountCal"),
        "LoanAccountCal",
        "CoverAppropriatedAmount",
        procedure_sql=sql,
    )
    assert composed and validate_expression(composed).valid
    assert "ELSEIF" in composed
    assert "GuaranteeFund" in composed
    # Exclusive IF/ELSEIF/ELSE — not a WHERE-matched outer THEN(0) wipe.
    assert "ELSEIF" in composed and "THEN(0)" in composed


def test_strip_dead_isempty_under_isnotempty_elseif_chain():
    raw = (
        'IF(ISNOTEMPTY("A"."DpdDays"))THEN('
        'IF(ISEMPTY("A"."DpdDays"))THEN("NOT_APPLICABLE")'
        'ELSEIF("A"."DpdDays" == 0)THEN("CURRENT")'
        'ELSEIF("A"."DpdDays" BETWEEN [1,30])THEN("BUCKET_1_30")'
        'ELSE("BUCKET_90_PLUS")'
        ')ELSE("A"."DpdBucket")'
    )
    out = _simplify_composed_expression(raw)
    assert 'ISEMPTY("A"."DpdDays")' not in out
    assert 'THEN("CURRENT")' in out
    assert 'IF(("A"."DpdDays"' not in out  # no double-paren rewrite bug
    assert out.startswith('IF(ISNOTEMPTY("A"."DpdDays"))THEN(IF("A"."DpdDays" == 0)')


def test_omit_temp_and_queue_passthrough_copies_but_keep_case_reason():
    assert _should_omit_passthrough_dd_row(
        target_table="#DpdStaging",
        entity_name="DpdStaging",
        expression='IF("A"."BucketWorsened" == "Y")THEN("A"."AccountId")ELSE(NULL)',
    )
    assert not _should_omit_passthrough_dd_row(
        target_table="#DpdStaging",
        entity_name="DpdStaging",
        expression=(
            'IF(ISNOTEMPTY("DpdStaging"."AdjustedPenalty"))THEN('
            'IF("DpdStaging"."FacilityType" IN ["CC","OD"])THEN(('
            '"DpdStaging"."AdjustedPenalty" * 1.10))ELSE("DpdStaging"."AdjustedPenalty"))ELSE(NULL)'
        ),
    )
    assert _should_omit_passthrough_dd_row(
        target_table="CollectionsQueue",
        entity_name="CollectionsQueue",
        expression='IF("A"."BucketWorsened" == "Y")THEN("A"."AccountId")ELSE(NULL)',
    )
    assert not _should_omit_passthrough_dd_row(
        target_table="CollectionsQueue",
        entity_name="CollectionsQueue",
        expression=(
            'IF("A"."BucketWorsened" == "Y")THEN('
            'IF("A"."DpdBucket" IN ["BUCKET_61_90"])THEN("SEVERE")ELSE("MILD"))ELSE(NULL)'
        ),
    )
    assert _should_omit_passthrough_dd_row(
        target_table="AccountStatusAuditLog",
        entity_name="AccountStatusAuditLog",
        expression='"ClosureDecisions"."AccountId"',
    )
    assert _should_omit_passthrough_dd_row(
        target_table="AccountStatusAuditLog",
        entity_name="AccountStatusAuditLog",
        expression='"AccountStatusAuditLog"."var"."BUSINESS_DATE"',
    )
