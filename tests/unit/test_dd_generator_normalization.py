from app.derivation.canonical_model import build_canonical_model
from app.derivation.dd_generator import (
    _build_source_statement_refs,
    _extract_aggregate_info,
    _extract_variable_trace,
    _format_assignment_context,
    _normalize_expression,
    generate_dd_rows,
)
from app.grammar.validator import validate_expression
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
    assert any(r.status == DDStatus.PENDING_REVIEW for r in rows)
    assert any(r.validation_errors for r in rows)
    assert any(r.status == DDStatus.ACTIVE for r in rows)


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
    assert all(
        "IF(p_TIMEKEY > 26267, 1, 0)" not in row.display_derivation_expression
        for row in rows
    )
    assert all("Grammar validation failed" not in " ".join(row.validation_errors) for row in rows)

    # Columns whose source has no override/exception assignment site should
    # pass cleanly on this simple (always-valid) expression; the fixed mock
    # output naturally can't reflect a column-specific override, so those
    # columns may legitimately be flagged by semantic validation for
    # review instead -- that's the new, more correct behavior, not a bug.
    simple_columns = [r for r in rows if not any("Semantic validation" in e for e in r.validation_errors)]
    assert simple_columns, "expected at least some columns with no override/exception source to pass cleanly"


def test_dd_generation_deterministically_derives_cleanup_rows(
    dpd_calculation_sql, maxdpd_sql, npa_date_sql, function_reference
):
    class HallucinatingLLMClient:
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
            return 'IF(Scheme=="Y")THEN(NULL)ELSE(NULL)'

        def retry_with_error(self, previous_expression, error, context) -> str:
            return 'IF(Scheme=="Y")THEN(NULL)ELSE(NULL)'

    chain, objects, infos = _build_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql)
    client = HallucinatingLLMClient()
    model = build_canonical_model(chain, "job-int-cleanup", objects, infos, client)

    rows = generate_dd_rows(
        chain=chain,
        canonical_model=model,
        objects=objects,
        structural_infos=infos,
        llm_client=client,
        function_reference=function_reference,
    )

    rows_by_column = {row.column_name: row for row in rows}
    for column in ["INTNOTSERVICEDDT", "LASTCRDATE", "OVERDUESINCEDT", "DEBITSINCEDT"]:
        assert column in rows_by_column
        assert rows_by_column[column].status == DDStatus.ACTIVE
        assert 'IF(Scheme=="Y")THEN(NULL)ELSE(NULL)' not in rows_by_column[column].display_derivation_expression

    assert 'ELSE("AccountCal_Stg"."LastCrDate")' in rows_by_column["LASTCRDATE"].display_derivation_expression


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


def test_dd_generator_rewrites_misused_isnotempty_with_default_argument():
    expr = 'IF(ISNOTEMPTY("A"."ASSET_NORM","NORMAL")<>"ALWYS_STD")THEN(1)ELSE(0)'
    normalized = _normalize_expression(expr)
    assert 'COALESCE("A"."ASSET_NORM","NORMAL")!="ALWYS_STD"' in normalized.replace(" ", "")


def test_dd_generator_rewrites_string_concatenation_to_concat():
    expr = 'IF(ISNOTEMPTY("A"."DEGREASON"))THEN("A"."DEGREASON" + "," + "B"."DEFAULT_REASON")ELSE("B"."DEFAULT_REASON")'
    normalized = _normalize_expression(expr)
    assert 'CONCAT(' in normalized
    assert '+' not in normalized.replace('+ 1', '')
    assert validate_expression(normalized).valid


def test_dd_generator_repairs_extra_closing_parens_before_then():
    expr = 'IF(ISNOTEMPTY("A"."X") AND ("A"."Y">0))))THEN(1)ELSE(0)'
    normalized = _normalize_expression(expr)
    assert normalized == 'IF(ISNOTEMPTY("A"."X") AND ("A"."Y">0))THEN(1)ELSE(0)'


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


def test_assignment_context_labels_exception_handler_as_mutually_exclusive():
    """Regression test for a real, confirmed defect: the normal-flow
    completion UPDATE and the EXCEPTION-handler UPDATE for the same
    column were both labeled with the same generic role
    (SEQUENTIAL_ASSIGNMENT), and the ordered-write-sequence overview even
    described the exception site with "later stages stay later" -- giving
    the model no signal that the two are mutually exclusive alternate
    paths rather than sequential steps, which led to a generated
    expression that tested the identical guard condition twice (see
    app/guardrails/semantic_validation.py::check_redundant_nested_condition,
    added to catch this class of defect from the output side; this test
    covers the same defect from the input/context side)."""
    chunks = [
        SmartChunk(
            chunk_id="chunk-1",
            object_id="obj-1",
            chunk_index=0,
            chunk_kind="UPDATE",
            statement_indices=[30],
            raw_sql=(
                "UPDATE PRO.ACLRUNNINGPROCESSSTATUS SET COMPLETED='Y', "
                "ERRORDATE=NULL WHERE RUNNINGPROCESSNAME='NPA_Date_Calculation';"
            ),
            columns_written=["ERRORDATE"],
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
            chunk_kind="UPDATE",
            statement_indices=[33],
            raw_sql=(
                "UPDATE PRO.ACLRUNNINGPROCESSSTATUS SET COMPLETED='N', "
                "ERRORDATE=SYSDATE WHERE RUNNINGPROCESSNAME='NPA_Date_Calculation';"
            ),
            columns_written=["ERRORDATE"],
        ),
    ]
    info = type("Info", (), {"smart_chunks": chunks})()

    context = _format_assignment_context(info, "ERRORDATE")

    # The two sites must receive genuinely different role labels -- the
    # normal-flow completion UPDATE stays whatever the deterministic
    # UPDATE heuristics classify it as, but the exception-handler site
    # must be labeled EXCEPTION_HANDLER, not the same generic label.
    assert "role=EXCEPTION_HANDLER" in context
    normal_flow_role = context.split("[Assignment 1")[1].split("role=")[1].split("|")[0].strip()
    assert normal_flow_role != "EXCEPTION_HANDLER"

    # The ordered-write-sequence overview must describe the exception
    # site as a mutually exclusive alternate path, not as a later
    # sequential stage.
    assert "mutually exclusive alternate path" in context

    # And the explicit decomposition hint telling the model not to reuse
    # the same guard condition for both sites must be present.
    assert "never" in context and "same repeated condition" in context


def test_normalize_bundled_business_date_variable_reference():
    expr = 'IF(ISNOTEMPTY("FCT_NPA_PRODUCT"."var_BUSINESS_DATE"))THEN("FCT_NPA_PRODUCT"."var_BUSINESS_DATE")ELSE(NULL)'
    normalized = _normalize_expression(expr)
    assert '"FCT_NPA_PRODUCT"."var"."BUSINESS_DATE"' in normalized
    assert 'var_BUSINESS_DATE' not in normalized


def test_normalize_fused_alias_column_reference_against_source_text():
    expr = "IF(DIMPRODUCT_ProductAlt_Key == PUI_CAL_DEFAULT_REASON)THEN(1)ELSE(0)"
    source_text = (
        "SELECT DP.ProductAlt_Key, B.DEFAULT_REASON, A.AccountEntityId "
        "FROM PRO.PUI_CAL B INNER JOIN PRO.AccountCal_Stg A ON A.AccountEntityID=B.AccountEntityId "
        "LEFT JOIN DimProduct DP ON A.ProductAlt_Key=DP.ProductAlt_Key"
    )
    normalized = _normalize_expression(expr, source_text)
    assert "ProductAlt_Key" in normalized
    assert "DEFAULT_REASON" in normalized
    assert "DIMPRODUCT_ProductAlt_Key" not in normalized
    assert "PUI_CAL_DEFAULT_REASON" not in normalized


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


def test_extract_aggregate_info_detects_grouped_aggregate():
    """A cross-row rollup (aggregate function combined with GROUP BY) must
    be detected -- this is the shape a per-row Formula Expression cannot
    literally reproduce (e.g. the real MAX(DPD)...GROUP BY AccountEntityID
    pattern in PRO_DPD_Calculation_StoredProcedure_2.sql)."""
    raw_sql = (
        "SELECT AccountEntityID, MAX(DPD) DPD_MaxFin FROM (...) GROUP BY AccountEntityID"
    )
    info = _extract_aggregate_info(raw_sql)
    assert len(info) == 1
    assert "MAX" in info[0]
    assert "AccountEntityID" in info[0]


def test_extract_aggregate_info_detects_multiple_aggregate_functions():
    raw_sql = "SELECT CustomerEntityId, MIN(NPA_DATE) NPA_DATE, LISTAGG(DEFAULT_REASON, ', ') FROM x GROUP BY CustomerEntityId"
    info = _extract_aggregate_info(raw_sql)
    assert len(info) == 1
    assert "MIN" in info[0]
    assert "LISTAGG" in info[0]


def test_extract_aggregate_info_ignores_scalar_aggregate_without_group_by():
    """A same-row scalar function call like MAX(a, b) -- picking the
    larger of two values already on the same row -- is not a cross-row
    rollup and must not be flagged just because the function name
    matches."""
    raw_sql = "UPDATE T SET X = MAX(A, B) WHERE Y = 1"
    assert _extract_aggregate_info(raw_sql) == []


def test_extract_variable_trace_finds_variables_used_downstream():
    raw_sql = "EXCEPTION\nWHEN OTHERS THEN\nv_error := SQLERRM;\nUPDATE T SET ERRORDESCRIPTION = v_error WHERE X = 1;"
    trace = _extract_variable_trace(raw_sql)
    assert len(trace) == 1
    assert "v_error := SQLERRM" in trace[0]
    assert "used later" in trace[0]


def test_extract_variable_trace_ignores_variables_never_referenced_again():
    """A local assignment that is never used by anything downstream in
    this same write site's text should not be surfaced -- only variables
    that actually feed the final assignment are relevant to a reviewer."""
    raw_sql = "v_unused := 1;\nUPDATE T SET X = 5 WHERE Y = 1;"
    assert _extract_variable_trace(raw_sql) == []


def test_assignment_context_shows_aggregate_summary_and_hint_for_real_dpd_max(dpd_calculation_sql):
    objects = split_objects(dpd_calculation_sql, "dpd.sql", Dialect.ORACLE)
    infos = {obj.object_id: analyze_object(obj) for obj in objects}
    info = next(iter(infos.values()))

    context = _format_assignment_context(info, "DPD_MaxFin")

    assert "aggregates MAX" in context
    assert "AccountEntityID" in context
    assert "cross-row aggregate" in context
    assert "cannot itself perform a GROUP BY" in context


def test_build_source_statement_refs_traces_real_errordate_to_both_sites(dpd_calculation_sql):
    objects = split_objects(dpd_calculation_sql, "dpd.sql", Dialect.ORACLE)
    infos = {obj.object_id: analyze_object(obj) for obj in objects}
    obj = objects[0]
    info = infos[obj.object_id]

    refs = _build_source_statement_refs(obj, info, "ERRORDATE")

    assert len(refs) == 2
    assert any("role=EXCEPTION_HANDLER" in ref for ref in refs)
    assert any("role=EXCEPTION_HANDLER" not in ref for ref in refs)
    assert all(ref.startswith("dpd.sql stmt #") for ref in refs)


def test_dd_rows_carry_source_statement_refs_end_to_end(
    dpd_calculation_sql, maxdpd_sql, npa_date_sql, mock_llm_client, function_reference
):
    """The traceability breadcrumbs must survive all the way through
    DD row generation -- not just be computable in isolation -- since a
    row with an empty source_statement_refs list is useless to a
    reviewer even if the underlying helper functions work correctly on
    their own."""
    chain, objects, infos = _build_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql)
    model = build_canonical_model(chain, "job-int-refs", objects, infos, mock_llm_client)

    rows = generate_dd_rows(
        chain=chain,
        canonical_model=model,
        objects=objects,
        structural_infos=infos,
        llm_client=mock_llm_client,
        function_reference=function_reference,
        entity_name_map={"AccountCal_Stg": "FCT_NPA_PRODUCT"},
    )

    assert len(rows) > 0
    assert all(row.source_statement_refs for row in rows), "every row should have at least one traced source statement"
    assert all(ref.count("stmt #") for row in rows for ref in row.source_statement_refs)