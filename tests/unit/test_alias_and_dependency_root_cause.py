"""Regression tests for the two root causes traced from the SMA_MARKING
report: (1) the shared alias resolver silently failing on every SQL Server
statement because it passed the wrong sqlglot dialect name, and (2) the
Depends On extractor misclassifying quoted literals as source columns
because it relied on a hardcoded word list instead of the actual parsed
source SQL. Test fragments are drawn directly from report.md so this
exercises the real regression, not a simplified stand-in for it.
"""
from app.models.core import Dialect
from app.report.report_generator import _extract_dependencies, _is_dependency_literal
from app.utils.sql_aliases import (
    collect_known_reference_names,
    collect_table_aliases,
    resolve_aliases_in_expression,
)

# 1. `FROM TableName A` -> output uses TableName, never A. This must work
#    for T-SQL, which is where the dialect-name bug lived.
def test_sqlserver_from_alias_resolves_to_real_table_name():
    sql = (
        "SELECT A.AccountEntityID, A.CustomerEntityID, "
        "CASE WHEN ISNULL(A.DPD_Overdrawn,0) > 30 THEN 1 ELSE 0 END AS DPD_FLAG "
        "INTO #DPD FROM PRO.AccountCal A "
        "WHERE ISNULL(A.DPD_Overdrawn,0) > 30 OR ISNULL(A.DPD_Overdue,0) > 0"
    )
    alias_map = collect_table_aliases(sql, Dialect.SQLSERVER)
    assert alias_map, "alias map must not be empty for a T-SQL FROM clause"
    assert alias_map["A"] == ("PRO", "AccountCal")

    formula = 'COALESCE("A"."DPD_Overdrawn", 0) > 30 OR COALESCE("A"."DPD_Overdue", 0) > 0'
    resolved = resolve_aliases_in_expression(formula, alias_map, quote_replacements=True)
    assert '"A"' not in resolved
    assert '"PRO"."AccountCal"."DPD_Overdrawn"' in resolved
    assert '"PRO"."AccountCal"."DPD_Overdue"' in resolved


# 2. Multiple aliases in one statement.
def test_multiple_aliases_all_resolve():
    sql = (
        "UPDATE PRO.AccountCal SET A.SMA_CLASS = B.SMA_CLASS "
        "FROM PRO.AccountCal A INNER JOIN PRO.CustomerCal B "
        "ON A.CustomerEntityID = B.CustomerEntityID"
    )
    alias_map = collect_table_aliases(sql, Dialect.SQLSERVER)
    assert alias_map["A"] == ("PRO", "AccountCal")
    assert alias_map["B"] == ("PRO", "CustomerCal")


# 3. Nested queries / joins with aliases -- an alias reused for a genuinely
#    different table across separate statements in the same object must be
#    dropped as ambiguous rather than guessed, but aliases that are
#    consistent within their own statement still resolve.
def test_ambiguous_alias_across_statements_is_dropped_not_guessed():
    sql = (
        "UPDATE PRO.AccountCal_Stg SET A.DPD_NoCredit = 0 FROM PRO.AccountCal_Stg A "
        "WHERE ISNULL(A.DPD_NoCredit,0) < 0;\n"
        "UPDATE A SET A.DPD_Overdrawn = DATEDIFF(DAY, A.DebitSinceDt, GETDATE()) "
        "FROM PRO.AccountCal A;\n"
    )
    alias_map = collect_table_aliases(sql, Dialect.SQLSERVER)
    assert "A" not in alias_map, "an alias mapping to two different tables must not be guessed"


# 4. Mixed alias + unqualified columns -- unqualified columns must be left
#    alone (they don't need resolving, and must never be invented as
#    dependencies from thin air).
def test_unqualified_column_untouched_by_resolver():
    alias_map = {"A": ("PRO", "AccountCal")}
    formula = 'IF(COALESCE(DPD_IntService, 0) < 0)THEN(0)ELSE("A"."DPD_Overdrawn")'
    resolved = resolve_aliases_in_expression(formula, alias_map, quote_replacements=True)
    assert "DPD_IntService" in resolved
    assert '"A"' not in resolved


# 5. Exact source table casing is preserved, not normalized.
def test_exact_source_casing_preserved():
    sql = "SELECT A.DPD_Overdrawn FROM PRO.AccountCal A"
    alias_map = collect_table_aliases(sql, Dialect.SQLSERVER)
    assert alias_map["A"] == ("PRO", "AccountCal")  # not PRO.ACCOUNTCAL / pro.accountcal


# 6/7/8/9. String constants, numeric constants, dates, and COALESCE must
# never appear in Depends On. This is the literal ASSET_NORM<>'ALWYS_STD'
# condition quoted verbatim in report.md's Key Conditions table.
def test_string_constant_from_real_report_excluded_from_depends_on():
    sql = (
        "UPDATE PRO.AccountCal SET SMA_CLASS = 'SMA_0' "
        "FROM PRO.AccountCal A INNER JOIN PRO.CustomerCal B ON A.CustomerEntityID = B.CustomerEntityID "
        "WHERE ISNULL(B.FLGPROCESSING,'N')='N' AND ISNULL(A.FINALASSETCLASSALT_KEY,1)=1 "
        "AND ISNULL(A.BALANCE,0)>0 AND A.ASSET_NORM<>'ALWYS_STD' "
        "AND ISNULL(DPD.DPD_MAX,0)>0"
    )
    known = collect_known_reference_names(sql, Dialect.SQLSERVER)
    formula = (
        'COALESCE("ACCOUNTCAL"."FINALASSETCLASSALT_KEY",1)==1 AND COALESCE("ACCOUNTCAL"."BALANCE",0)>0 '
        'AND "ACCOUNTCAL"."ASSET_NORM"!="ALWYS_STD" AND COALESCE("ACCOUNTCAL"."DPD_MAX",0)>0'
    )
    deps = _extract_dependencies(formula, known)
    assert "ALWYS_STD" not in deps
    assert any("ASSET_NORM" in d for d in deps)
    assert any("FINALASSETCLASSALT_KEY" in d for d in deps)


def test_numeric_and_date_constants_excluded_from_depends_on():
    formula = 'IF("ACCOUNTCAL"."DPD_MAX">30)THEN("2024-01-01")ELSE(0)'
    deps = _extract_dependencies(formula, frozenset({"DPD_MAX"}))
    assert "30" not in deps
    assert "0" not in deps
    assert "2024-01-01" not in deps
    assert any("DPD_MAX" in d for d in deps)


def test_coalesce_wrapped_columns_are_dependencies_not_the_function_name():
    formula = 'COALESCE("ACCOUNTCAL"."RefPeriodOverDrawn", 0) > 0'
    deps = _extract_dependencies(formula, frozenset({"REFPERIODOVERDRAWN"}))
    assert "COALESCE" not in [d.upper() for d in deps]
    assert any("RefPeriodOverDrawn" in d for d in deps)


# A single-part bare token that genuinely is a real column/parameter (per
# the source SQL) must still be classified as a reference, not a literal --
# the fix must not overcorrect into treating everything as a literal.
def test_bare_known_parameter_is_not_misclassified_as_literal():
    assert _is_dependency_literal("p_TIMEKEY", frozenset({"P_TIMEKEY"})) is False
    assert _is_dependency_literal("ALWYS_STD", frozenset({"P_TIMEKEY"})) is True


# Without any known_names available at all (no source SQL resolvable for
# the row), fall back to the small curated literal list rather than
# silently treating every bare token as a real dependency.
def test_no_known_names_falls_back_to_curated_literal_list():
    assert _is_dependency_literal("Y", frozenset()) is True
    assert _is_dependency_literal("SOME_RANDOM_TOKEN", frozenset()) is False