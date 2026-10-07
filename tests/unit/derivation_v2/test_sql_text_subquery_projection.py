"""Subquery → row predicate projection helpers."""
from app.derivation.v2.sql_text import (
    exists_subquery_to_row_predicate,
    in_subquery_to_row_predicate,
)


def test_qualify_bare_columns_preserves_string_literals():
    from app.derivation.v2.sql_text import _qualify_bare_columns

    pred = _qualify_bare_columns(
        "AssetClassShortName='LOS' AND EffectiveFromTimeKey<=@TIMEKEY",
        "DimAssetClass",
    )
    assert "AssetClassShortName='LOS'" in pred.replace(" ", "")
    assert ".LOS'" not in pred


def test_in_subquery_bracketed_account_no_column():
    pred, deps = in_subquery_to_row_predicate(
        "B.[Account No]",
        "SELECT [Account No] FROM Manual_NPA",
    )
    assert pred is not None
    assert "Manual_NPA" in pred
    assert "Account No" in pred or "Account" in pred
    assert deps


def test_exists_subquery_qualifies_where_columns():
    pred, deps = exists_subquery_to_row_predicate(
        "EXISTS(SELECT 1 FROM PRO.ACCOUNTCAL_Hist WHERE EffectiveFromTimeKey>@TIMEKEY)"
    )
    assert pred is not None
    assert "ACCOUNTCAL_Hist" in pred
    assert "EffectiveFromTimeKey" in pred
    assert deps


def test_exists_subquery_rewrites_inner_alias_keeps_outer_correlation():
    pred, deps = exists_subquery_to_row_predicate(
        "EXISTS(SELECT 1 FROM ExceptionFinalStatusType A "
        "WHERE A.EffectiveFromTimeKey<=@TIMEKEY AND A.EffectiveToTimeKey>=@TIMEKEY "
        "and acl.CustomerAcID=a.ACID AND A.StatusType='Charge Off')"
    )
    assert pred is not None
    assert not pred.strip().upper().startswith("EXISTS")
    assert "ExceptionFinalStatusType" in pred
    assert "StatusType" in pred
    assert "acl.CustomerAcID" in pred or "ACL.CustomerAcID" in pred
    assert "ExceptionFinalStatusType.acl" not in pred
    assert deps


def test_session_hash_temps_are_staging_global_temps_are_not():
    from app.derivation.v2.sql_text import is_staging_derivation_entity

    assert is_staging_derivation_entity("#LINKACEXP")
    assert is_staging_derivation_entity("#BILL_OVERDUE_SCF_FINAL")
    assert is_staging_derivation_entity("#MOC_DATA")
    assert is_staging_derivation_entity("TEMPTABLEDPD")
    assert is_staging_derivation_entity("CTE_NPA_UCIFID")
    assert is_staging_derivation_entity("AccountCal_BKUP")
    assert not is_staging_derivation_entity("##ACCOUNTCAL")
    assert not is_staging_derivation_entity("##CUSTOMERCAL")
    assert not is_staging_derivation_entity("ACCOUNTCAL")
    assert not is_staging_derivation_entity("CustomerCal")
