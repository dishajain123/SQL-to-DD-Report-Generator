from app.derivation.canonical_model import build_canonical_model
from app.derivation.dd_generator import _format_assignment_context, _normalize_expression, generate_dd_rows
from app.lineage.dependency_graph import build_graph, find_chains
from app.models.core import DDStatus, Dialect, SmartChunk
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
    assert all(r.status == DDStatus.PENDING_REVIEW for r in rows)
    assert all(r.validation_errors for r in rows)


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
    # branch's literal (see app/derivation/period_pruning.py), so "THEN("
    # is only required to survive on rows that weren't pruned to a bare
    # literal.
    assert all(
        "THEN(" in row.display_derivation_expression or row.display_derivation_expression in ("1", "0")
        for row in rows
    )
    assert all("," not in row.display_derivation_expression.split("THEN(", 1)[0] for row in rows)
    assert all("Grammar validation failed" not in " ".join(row.validation_errors) for row in rows)

    # Columns whose source has no override/exception assignment site should
    # pass cleanly on this simple (always-valid) expression; the fixed mock
    # output naturally can't reflect a column-specific override, so those
    # columns may legitimately be flagged by semantic validation for
    # review instead -- that's the new, more correct behavior, not a bug.
    simple_columns = [r for r in rows if not any("Semantic validation" in e for e in r.validation_errors)]
    assert simple_columns, "expected at least some columns with no override/exception source to pass cleanly"


def test_dd_generation_retries_after_validation_failure(
    dpd_calculation_sql, maxdpd_sql, npa_date_sql, function_reference
):
    class RepairingLLMClient:
        def __init__(self) -> None:
            self.retry_calls = 0

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
            return 'IF(ERROR_OCCURRED)THEN(SQLERRM)ELSE(NULL)'

        def retry_with_error(self, previous_expression, error, context) -> str:
            self.retry_calls += 1
            return 'IF(ISNOTEMPTY("ACLRUNNINGPROCESSSTATUS"."RUNNINGPROCESSNAME"))THEN(0)ELSE(1)'

    chain, objects, infos = _build_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql)
    client = RepairingLLMClient()
    model = build_canonical_model(chain, "job-int-4", objects, infos, client)

    rows = generate_dd_rows(
        chain=chain,
        canonical_model=model,
        objects=objects,
        structural_infos=infos,
        llm_client=client,
        function_reference=function_reference,
    )

    assert client.retry_calls >= 1
    assert any(not row.validation_errors for row in rows)


def test_dd_generator_normalizes_single_row_in_subquery_membership():
    expr = (
        'IF(FLGDEG=="Y" AND ISNOTEMPTY(RestructureTypeAlt_Key) '
        'AND RestructureTypeAlt_Key IN ["DimParameter"."ParameterAlt_Key" '
        'WHERE "DimParameter"."DimParameterName"=="TypeofRestructuring"])'
        'THEN("N")ELSE(NULL)'
    )
    normalized = _normalize_expression(expr)
    assert ' IN [' not in normalized
    assert 'RestructureTypeAlt_Key == "DimParameter"."ParameterAlt_Key"' in normalized


def test_assignment_context_is_ordered_and_labeled():
    chunks = [
        SmartChunk(
            chunk_id="chunk-1",
            object_id="obj-1",
            chunk_index=0,
            chunk_kind="MERGE",
            statement_indices=[1],
            raw_sql="MERGE INTO T SET X = 0",
            columns_written=["X"],
        ),
        SmartChunk(
            chunk_id="chunk-2",
            object_id="obj-1",
            chunk_index=1,
            chunk_kind="UPDATE",
            statement_indices=[2, 3],
            raw_sql="UPDATE T SET X = Y WHERE X IS NULL",
            columns_written=["X"],
        ),
    ]
    info = type("Info", (), {"smart_chunks": chunks})()

    context = _format_assignment_context(info, "X")

    assert "[Assignment 1 | role=MERGE | kind=MERGE | statements=1 | columns=X]" in context
    assert "[Assignment 2 | role=SEQUENTIAL_FIXUP | kind=UPDATE | statements=2,3 | columns=X]" in context
    assert context.index("MERGE INTO T SET X = 0") < context.index("UPDATE T SET X = Y WHERE X IS NULL")


def test_assignment_context_marks_merge_and_fixup_roles():
    chunks = [
        SmartChunk(
            chunk_id="chunk-1",
            object_id="obj-1",
            chunk_index=0,
            chunk_kind="MERGE",
            statement_indices=[1],
            raw_sql=(
                "MERGE INTO PRO.AdvAcRestructureCal B USING (SELECT B.ROWID AS RID, "
                "CASE WHEN B.PreRestructureNPA_Date IS NOT NULL THEN B.PreRestructureNPA_Date "
                "ELSE B.RestructureDt END AS NEW_DEGDATE "
                "FROM PRO.AccountCal_Stg A INNER JOIN PRO.AdvAcRestructureCal B ON A.AccountEntityID=B.AccountEntityId "
                "WHERE (A.FINALASSETCLASSALT_KEY>1 OR A.FlgDeg='Y') "
                "AND (CASE WHEN NVL(B.SP_ExpiryDate,DATE '1900-01-01')>=NVL(B.SP_ExpiryExtendedDate,DATE '1900-01-01') "
                "THEN B.SP_ExpiryDate ELSE B.SP_ExpiryExtendedDate END) > v_ProcessDate) SRC "
                "ON (B.ROWID=SRC.RID) WHEN MATCHED THEN UPDATE SET B.DEGDATE=SRC.NEW_DEGDATE"
            ),
            columns_written=["DEGDATE"],
        ),
        SmartChunk(
            chunk_id="chunk-2",
            object_id="obj-1",
            chunk_index=1,
            chunk_kind="UPDATE",
            statement_indices=[2],
            raw_sql="UPDATE PRO.AccountCal_Stg A SET A.REFPeriodMax=0;",
            columns_written=["REFPeriodMax"],
        ),
        SmartChunk(
            chunk_id="chunk-3",
            object_id="obj-1",
            chunk_index=2,
            chunk_kind="UPDATE",
            statement_indices=[3],
            raw_sql=(
                "UPDATE PRO.AccountCal_Stg SET REFPeriodMax=RefPeriodNoCredit "
                "WHERE REFPeriodMax IS NULL AND DPD_FinMaxType='RefPeriodNoCredit';"
            ),
            columns_written=["REFPeriodMax"],
        ),
    ]
    info = type("Info", (), {"smart_chunks": chunks})()

    context = _format_assignment_context(info, "DEGDATE") + "\n\n" + _format_assignment_context(info, "REFPeriodMax")

    assert "role=MERGE_USING_CASE_VALUE" in context
    assert "Treat the USING subquery as the value source" in context
    assert "Preserve the CASE branch choice inside the USING subquery" in context
    assert "role=INITIAL_RESET" in context
    assert "initialization/reset stage" in context
    assert "role=SEQUENTIAL_FIXUP" in context
    assert "later row-scoping or fix-up guard" in context
    assert "sequential override or backfill" in context


def test_assignment_context_keeps_exception_handler_bridge_statements():
    chunks = [
        SmartChunk(
            chunk_id="chunk-1",
            object_id="obj-1",
            chunk_index=0,
            chunk_kind="UPDATE",
            statement_indices=[30],
            raw_sql=(
                "UPDATE PRO.ACLRUNNINGPROCESSSTATUS SET COMPLETED='Y', "
                "ERRORDESCRIPTION=NULL WHERE RUNNINGPROCESSNAME='NPA_Date_Calculation';"
            ),
            columns_written=["ERRORDESCRIPTION"],
        ),
        SmartChunk(
            chunk_id="chunk-2",
            object_id="obj-1",
            chunk_index=1,
            chunk_kind="CONTROL_FLOW",
            statement_indices=[31],
            raw_sql="EXCEPTION",
            columns_written=[],
        ),
        SmartChunk(
            chunk_id="chunk-3",
            object_id="obj-1",
            chunk_index=2,
            chunk_kind="CONTROL_FLOW",
            statement_indices=[32],
            raw_sql="WHEN OTHERS THEN",
            columns_written=[],
        ),
        SmartChunk(
            chunk_id="chunk-4",
            object_id="obj-1",
            chunk_index=3,
            chunk_kind="SEQUENTIAL",
            statement_indices=[33],
            raw_sql="v_error := SQLERRM;",
            columns_written=[],
        ),
        SmartChunk(
            chunk_id="chunk-5",
            object_id="obj-1",
            chunk_index=4,
            chunk_kind="UPDATE",
            statement_indices=[34],
            raw_sql=(
                "UPDATE PRO.ACLRUNNINGPROCESSSTATUS SET COMPLETED='N', "
                "ERRORDESCRIPTION=v_error WHERE RUNNINGPROCESSNAME='NPA_Date_Calculation';"
            ),
            columns_written=["ERRORDESCRIPTION"],
        ),
    ]
    info = type("Info", (), {"smart_chunks": chunks})()

    context = _format_assignment_context(info, "ERRORDESCRIPTION")

    assert "EXCEPTION" in context
    assert "WHEN OTHERS THEN" in context
    assert "v_error := SQLERRM" in context
    assert context.index("COMPLETED='Y'") < context.index("v_error := SQLERRM") < context.index("COMPLETED='N'")


def test_normalize_bundled_business_date_variable_reference():
    expr = 'IF(ISNOTEMPTY("FCT_NPA_PRODUCT"."var_BUSINESS_DATE"))THEN("FCT_NPA_PRODUCT"."var_BUSINESS_DATE")ELSE(NULL)'
    normalized = _normalize_expression(expr)
    assert '"FCT_NPA_PRODUCT"."var"."BUSINESS_DATE"' in normalized
    assert 'var_BUSINESS_DATE' not in normalized


def test_real_branch_heavy_columns_receive_structured_assignment_context(
    dpd_calculation_sql, maxdpd_sql, npa_date_sql, mock_llm_client, function_reference
):
    class CaptureLLMClient:
        def __init__(self) -> None:
            self.calls = []

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
            self.calls.append(
                {
                    "column_name": column_name,
                    "entity_name": entity_name,
                    "source_sql": source_sql,
                    "relevant_sql": relevant_sql,
                }
            )
            return mock_llm_client.generate_formula_expression(
                technical_summary,
                business_summary,
                source_sql,
                function_reference,
                column_name=column_name,
                entity_name=entity_name,
                relevant_sql=relevant_sql,
                rag_context=rag_context,
            )

        def retry_with_error(self, previous_expression, error, context) -> str:
            return mock_llm_client.retry_with_error(previous_expression, error, context)

    chain, objects, infos = _build_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql)
    model = build_canonical_model(chain, "job-int-5", objects, infos, mock_llm_client)
    client = CaptureLLMClient()

    rows = generate_dd_rows(
        chain=chain,
        canonical_model=model,
        objects=objects,
        structural_infos=infos,
        llm_client=client,
        function_reference=function_reference,
        entity_name_map={"AccountCal_Stg": "FCT_NPA_PRODUCT", "AdvAcRestructureCal": "FCT_NPA_PRODUCT"},
    )

    assert len(rows) > 0
    degdate_call = next(call for call in client.calls if call["column_name"] == "DEGDATE")
    refperiod_call = next(call for call in client.calls if call["column_name"] == "REFPERIODMAX")

    assert "[Assignment 1 | role=MERGE_USING_CASE_VALUE" in degdate_call["source_sql"]
    assert "Treat the USING subquery as the value source" in degdate_call["source_sql"]
    assert "[Ordered write sequence]" in refperiod_call["source_sql"]
    assert refperiod_call["column_name"] == "REFPERIODMAX"
    ordered_text = refperiod_call["source_sql"]
    assert ordered_text.index("SET REFPeriodMax=RefPeriodNoCredit") < ordered_text.index("SET REFPeriodMax=RefPeriodOverdue") < ordered_text.index("SET REFPeriodMax=RefPeriodOverDrawn") < ordered_text.index("SET REFPeriodMax=RefPeriodStkStatement") < ordered_text.index("SET REFPeriodMax=RefPeriodReview")
