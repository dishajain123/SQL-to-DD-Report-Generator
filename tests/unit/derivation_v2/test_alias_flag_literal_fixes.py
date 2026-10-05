"""Derived-table alias masking, string-flag defaults, literal protection."""
from __future__ import annotations

from app.derivation.v2.phase2_mutation_folder import (
    DerivedTable,
    _mask_derived_tables,
    _resolve_expression_tables,
)
from app.derivation.v2.phase3_ast_generator import _is_string_flag_column


def test_mask_derived_table_replaces_subquery_with_token():
    from_clause = (
        "##CustomerCal A INNER JOIN ("
        "SELECT B.RefCustomerId, MIN(B.FinalNpaDt) FinalNpaDt FROM ##AccountCal B "
        "GROUP BY B.RefCustomerId) C ON A.RefCustomerId = C.RefCustomerId"
    )
    masked, sources = _mask_derived_tables(from_clause)
    assert "SELECT" not in masked.upper()
    assert "__DERIVED_0__ C ON" in " ".join(masked.split())
    assert sources["__DERIVED_0__"].upper().startswith("SELECT")


def test_mask_derived_table_alias_glued_to_closing_paren():
    masked, _ = _mask_derived_tables(
        "##CustomerCal D Inner Join (SELECT A.X FROM PRO.T A)c ON D.X =c.X"
    )
    assert "__DERIVED_0__ c ON" in " ".join(masked.split())


def test_derived_table_survives_copy():
    import copy

    dt = DerivedTable("##AccountCal", {"X": "MIN(B.X)"}, {"B": "##AccountCal"})
    clone = copy.deepcopy(dt)
    assert str(clone) == "##AccountCal"
    assert clone.projections == {"X": "MIN(B.X)"}


def test_string_flag_columns_detected():
    for name in ("FlgDeg", "FlgProcessing", "Flg_Deg", "DegFlag"):
        assert _is_string_flag_column(name)
    for name in ("FinalNpaDt", "SysNPA_Dt", "DPD_Overdue"):
        assert not _is_string_flag_column(name)


def test_quote_expression_parameters_round_trips_display_form():
    from app.derivation.derivation_option import (
        format_expression_syntax,
        quote_expression_parameters,
    )

    grammar = 'IF("D"."T" <= @timekey)THEN(ADDDAY("X"."Y", 1, @ProcessDate))ELSE("a@b")'
    display = format_expression_syntax(grammar.replace("@timekey", '"@timekey"'))
    assert "@timekey" in display and '"@timekey"' not in display
    assert quote_expression_parameters(display) == grammar.replace("@timekey", '"@timekey"').replace(
        "@ProcessDate", '"@ProcessDate"'
    )
    assert '"a@b"' in quote_expression_parameters(display)


def test_single_quoted_literals_are_not_alias_rewritten():
    class _Lineage:
        def resolve_column(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("literal text must not be resolved")

    out = _resolve_expression_tables(
        "'D.FDSEC'", {"D": "DimProduct"}, _Lineage(), None, "AccountCal"
    )
    assert out == "'D.FDSEC'"
