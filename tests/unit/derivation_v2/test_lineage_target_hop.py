"""Unit tests for target→relationship hop collapse rules."""
from app.derivation.v2.sql_text import (
    lineage_keeps_target_hop,
    should_collapse_target_join_hop,
)


def test_sibling_cal_interface_keeps_three_part_hop():
    assert lineage_keeps_target_hop("AccountCal", "CustomerCal")
    assert lineage_keeps_target_hop("##AccountCal", "##CustomerCal")
    assert not lineage_keeps_target_hop("CustomerCal", "CustomerBasicDetail")


def test_joined_physical_and_temp_tables_collapse_under_accountcal():
    assert not lineage_keeps_target_hop("AccountCal", "PUI_CAL")
    assert not lineage_keeps_target_hop("AccountCal", "TEMPTABLENPA")
    assert not lineage_keeps_target_hop("AccountCal", "C")
    assert should_collapse_target_join_hop("AccountCal", "PUI_CAL", "AccountCal")
    assert should_collapse_target_join_hop("AccountCal", "TEMPTABLENPA", "AccountCal")
    assert not should_collapse_target_join_hop("AccountCal", "C", "CustomerCal")
    assert not should_collapse_target_join_hop("AccountCal", "CustomerCal", "AccountCal")
