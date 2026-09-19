"""Entity-name map: procedure table names by default; optional env renames."""
from __future__ import annotations

from app.utils.entity_name_map import (
    build_entity_name_map_for_tables,
    merge_entity_overrides,
    resolve_entity_name,
)


def test_default_map_keeps_procedure_table_names():
    mapping = merge_entity_overrides({})
    assert resolve_entity_name("LoanAccountCal", mapping) == "LoanAccountCal"
    assert resolve_entity_name("PRO.AccountCal", mapping) == "AccountCal"
    assert resolve_entity_name("#ReconciliationResults", mapping) == "ReconciliationResults"


def test_build_map_indexes_hash_and_bare_spellings():
    mapping = build_entity_name_map_for_tables(
        ["#DpdStaging", "LoanAccountCal"],
        merge_entity_overrides({}),
    )
    assert mapping["LoanAccountCal"] == "LoanAccountCal"
    assert mapping["#DpdStaging"] == "DpdStaging"
    assert mapping["DpdStaging"] == "DpdStaging"
    assert resolve_entity_name("#LoanAccountCal", mapping) == "LoanAccountCal"


def test_explicit_overrides_rename_when_configured():
    mapping = merge_entity_overrides({"LoanAccountCal": "CUSTOM_LOAN"})
    assert resolve_entity_name("LoanAccountCal", mapping) == "CUSTOM_LOAN"
