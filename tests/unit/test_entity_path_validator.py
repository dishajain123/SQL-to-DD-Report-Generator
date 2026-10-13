"""Entity path validation for compiled 4X (DimAssetClass hops vs SQL aliases)."""
from __future__ import annotations

from app.grammar.entity_path_validator import validate_entity_paths_in_expression


def test_dim_asset_class_relationship_hops_are_valid():
    for hop in ("SUB", "DB1", "DB2", "DB3", "LOS"):
        formula = f'"CUSTOMERCAL"."{hop}"."AssetClassAlt_Key"'
        assert validate_entity_paths_in_expression(formula) == []


def test_statement_alias_still_rejected_in_relationship_hop():
    formula = '"CUSTOMERCAL"."A"."SourceSystemCustomerID"'
    errors = validate_entity_paths_in_expression(formula)
    assert errors
    assert "subquery alias" in errors[0].lower()
