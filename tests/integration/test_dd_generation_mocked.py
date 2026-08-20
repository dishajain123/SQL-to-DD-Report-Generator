from app.derivation.canonical_model import build_canonical_model
from app.derivation.dd_generator import generate_dd_rows
from app.lineage.dependency_graph import build_graph, find_chains
from app.models.core import DDStatus, Dialect, DerivationOption
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object


def _build_chain(dpd_sql, maxdpd_sql, npa_sql):
    files = {"dpd.sql": dpd_sql, "maxdpd.sql": maxdpd_sql, "npa.sql": npa_sql}
    objects_list = []
    infos = {}
    for fname, text in files.items():
        for obj in split_objects(text, fname, Dialect.ORACLE):
            objects_list.append(obj)
            infos[obj.object_id] = analyze_object(obj)

    graph = build_graph(objects_list, infos)
    chains = find_chains(graph, "job-int-1", objects_list)
    objects = {o.object_id: o for o in objects_list}
    return chains[0], objects, infos


def test_canonical_model_covers_whole_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql, mock_llm_client):
    chain, objects, infos = _build_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql)

    model = build_canonical_model(chain, "job-int-1", objects, infos, mock_llm_client)

    assert model.chain_id == chain.chain_id
    assert set(model.object_ids) == set(chain.order)
    assert model.technical_summary
    assert model.business_summary
    assert "AccountCal_Stg" in model.evidence


def test_dd_generation_produces_valid_rows(dpd_calculation_sql, maxdpd_sql, npa_date_sql, mock_llm_client, function_reference):
    chain, objects, infos = _build_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql)
    model = build_canonical_model(chain, "job-int-1", objects, infos, mock_llm_client)

    rows = generate_dd_rows(
        chain=chain, canonical_model=model, objects=objects, structural_infos=infos,
        llm_client=mock_llm_client, function_reference=function_reference,
        entity_name_map={"AccountCal_Stg": "FCT_NPA_PRODUCT"},
    )

    assert len(rows) > 0
    # DPD_Calculation has 4 distinct TIMEKEY thresholds -> its columns should
    # produce multiple dated rows, not one.
    dpd_obj_id = next(oid for oid, o in objects.items() if o.name == "DPD_Calculation")
    dpd_rows = [r for r in rows if dpd_obj_id in r.source_object_ids]
    assert len(dpd_rows) > len(infos[dpd_obj_id].columns_written)

    for row in rows:
        assert row.entity_name
        assert row.column_name
        assert row.status in (DDStatus.ACTIVE, DDStatus.PENDING_REVIEW)
    assert any(row.advisory_notes for row in rows)


def test_dd_generation_flags_grammar_failures_for_review(
    dpd_calculation_sql, maxdpd_sql, npa_date_sql, broken_llm_client, function_reference
):
    chain, objects, infos = _build_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql)
    model = build_canonical_model(chain, "job-int-2", objects, infos, broken_llm_client)

    rows = generate_dd_rows(
        chain=chain, canonical_model=model, objects=objects, structural_infos=infos,
        llm_client=broken_llm_client, function_reference=function_reference,
    )

    assert len(rows) > 0
    assert any(r.status == DDStatus.PENDING_REVIEW for r in rows)
    assert any(r.status == DDStatus.ACTIVE for r in rows)
    assert any(r.validation_errors for r in rows)


def test_dd_generation_rejects_truncated_expressions_instead_of_exporting_them(
    dpd_calculation_sql, maxdpd_sql, npa_date_sql, function_reference
):
    class TruncatedLLMClient:
        def technical_reasoning(self, sql_snippets: list[str]) -> str:
            return "technical"

        def business_reasoning(self, technical_summary: str) -> str:
            return "business"

        def generate_formula_expression(
            self,
            technical_summary,
            business_summary,
            source_sql,
            function_reference,
            column_name="",
            entity_name="",
            relevant_sql="",
            rag_context="",
        ) -> str:
            return 'IF(ISNOTEMPTY("A"."X"))THEN(("A"."Y") +'

        def retry_with_error(self, previous_expression, error, context) -> str:
            return 'IF(ISNOTEMPTY("A"."X"))THEN(("A"."Y") +'

    chain, objects, infos = _build_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql)
    model = build_canonical_model(chain, "job-int-trunc", objects, infos, TruncatedLLMClient())

    rows = generate_dd_rows(
        chain=chain,
        canonical_model=model,
        objects=objects,
        structural_infos=infos,
        llm_client=TruncatedLLMClient(),
        function_reference=function_reference,
    )

    assert rows
    assert any(row.display_derivation_expression == "" for row in rows)
    assert any(
        any("Unexpected end-of-input" in error for error in row.validation_errors)
        for row in rows
    )


def test_sqlserver_procedure_end_to_end_generates_dd_rows(sma_marking_sql, mock_llm_client, function_reference):
    objects_list = []
    infos = {}
    for obj in split_objects(sma_marking_sql, "PRO.SMA_MARKING_12122023.StoredProcedure.sql", Dialect.SQLSERVER):
        objects_list.append(obj)
        infos[obj.object_id] = analyze_object(obj)

    graph = build_graph(objects_list, infos)
    chains = find_chains(graph, "job-sma-1", objects_list)
    objects = {o.object_id: o for o in objects_list}

    model = build_canonical_model(chains[0], "job-sma-1", objects, infos, mock_llm_client)
    rows = generate_dd_rows(
        chain=chains[0],
        canonical_model=model,
        objects=objects,
        structural_infos=infos,
        llm_client=mock_llm_client,
        function_reference=function_reference,
    )

    assert rows
    assert any(row.column_name for row in rows)
    assert any(row.status in (DDStatus.ACTIVE, DDStatus.PENDING_REVIEW) for row in rows)
    assert any(row.status == DDStatus.ACTIVE for row in rows)


def test_dd_generation_normalizes_legacy_comma_style_if(
    dpd_calculation_sql, maxdpd_sql, npa_date_sql, mock_llm_client, function_reference
):
    class LegacyIfLLMClient:
        def technical_reasoning(self, sql_snippets: list[str]) -> str:
            return mock_llm_client.technical_reasoning(sql_snippets)

        def business_reasoning(self, technical_summary: str) -> str:
            return mock_llm_client.business_reasoning(technical_summary)

        def generate_formula_expression(
            self,
            technical_summary,
            business_summary,
            source_sql,
            function_reference,
            column_name="",
            entity_name="",
            relevant_sql="",
            rag_context="",
        ) -> str:
            return 'IF(p_TIMEKEY > 26267, 1, 0)'

        def retry_with_error(self, previous_expression, error, context) -> str:
            return 'IF(p_TIMEKEY > 26267, 1, 0)'

    chain, objects, infos = _build_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql)
    model = build_canonical_model(chain, "job-int-3", objects, infos, mock_llm_client)

    rows = generate_dd_rows(
        chain=chain,
        canonical_model=model,
        objects=objects,
        structural_infos=infos,
        llm_client=LegacyIfLLMClient(),
        function_reference=function_reference,
    )

    assert len(rows) > 0
    # The comma-style IF must always be normalized into valid 4X syntax,
    # regardless of what semantic validation later decides about the
    # column's completeness -- these are two independent guarantees. A row
    # whose effective period falls entirely on one side of the p_TIMEKEY
    # threshold is legitimately period-pruned down to just the selected
    # branch's literal (see app/derivation/period_pruning.py), and some
    # columns are represented without IF at all (for example as a direct
    # reference or COALESCE). Only expressions that actually contain IF
    # must therefore be checked for THEN/ELSE normalization.
    assert all("THEN(" in row.display_derivation_expression or "IF(" not in row.display_derivation_expression for row in rows)
    assert all("IF(p_TIMEKEY > 26267, 1, 0)" not in row.display_derivation_expression for row in rows)
    assert all("Grammar validation failed" not in " ".join(row.validation_errors) for row in rows)

    # Columns whose source has no override/exception assignment site should
    # pass cleanly on this simple (always-valid) expression; the fixed mock
    # output naturally can't reflect a column-specific override, so those
    # columns may legitimately be flagged by semantic validation for
    # review instead -- that's the new, more correct behavior, not a bug.
    simple_columns = [r for r in rows if not any("Semantic validation" in e for e in r.validation_errors)]
    assert simple_columns, "expected at least some columns with no override/exception source to pass cleanly"


def test_decision_table_json_llm_output_is_not_sent_through_formula_parser(
    sma_marking_sql, monkeypatch, function_reference
):
    class DecisionTableLLMClient:
        def technical_reasoning(self, sql_snippets: list[str]) -> str:
            return "technical"

        def business_reasoning(self, technical_summary: str) -> str:
            return "business"

        def generate_formula_expression(
            self,
            technical_summary,
            business_summary,
            source_sql,
            function_reference,
            column_name="",
            entity_name="",
            relevant_sql="",
            rag_context="",
            ) -> str:
            return """```json
            {"decisionTable": {"rules": [{"when": "high", "then": "approve"}]}}
            ```"""

        def retry_with_error(self, previous_expression, error, context) -> str:
            return self.generate_formula_expression(None, None, None, None)

    monkeypatch.setattr(
        "app.derivation.dd_generation_engine._compose_simple_assignment_expression",
        lambda *args, **kwargs: None,
    )

    objects = []
    infos = {}
    for obj in split_objects(sma_marking_sql, "PRO.SMA_MARKING_12122023.StoredProcedure.sql", Dialect.SQLSERVER):
        objects.append(obj)
        infos[obj.object_id] = analyze_object(obj)

    graph = build_graph(objects, infos)
    chains = find_chains(graph, "job-decision-table", objects)
    model = build_canonical_model(
        chains[0],
        "job-decision-table",
        {o.object_id: o for o in objects},
        infos,
        DecisionTableLLMClient(),
    )
    rows = generate_dd_rows(
        chain=chains[0],
        canonical_model=model,
        objects={o.object_id: o for o in objects},
        structural_infos=infos,
        llm_client=DecisionTableLLMClient(),
        function_reference=function_reference,
        entity_name_map={"AccountCal_Stg": "FCT_NPA_PRODUCT"},
    )

    assert rows
    assert any(row.derivation_option == DerivationOption.DECISION_TABLE for row in rows)
    assert any(row.decision_table_json for row in rows)
    assert all("Grammar validation failed" not in " ".join(row.validation_errors) for row in rows)
