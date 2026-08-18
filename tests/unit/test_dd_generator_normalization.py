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


def test_normalize_ternary_operator_fixes_real_degdate_defect():
    """Regression test for a real, confirmed defect: a generated DEGDATE
    expression used a `condition ? true : false` ternary, which the 4X
    grammar has no support for at all -- it was correctly rejected by
    grammar validation with "No terminal matches '?'". The normalizer
    must deterministically rewrite this to the grammar's real
    IF(...)THEN(...)ELSE(...) form and the result must actually validate."""
    from app.derivation.dd_generator import _normalize_expression
    from app.grammar.validator import validate_expression

    buggy = (
        'IF(("AdvAcRestructureCal"."FINALASSETCLASSALT_KEY">1 OR "AdvAcRestructureCal"."FlgDeg"=="Y") '
        'AND (IF(ISNOTEMPTY("AdvAcRestructureCal"."SP_ExpiryDate") AND '
        'ISNOTEMPTY("AdvAcRestructureCal"."SP_ExpiryExtendedDate"))'
        'THEN(("AdvAcRestructureCal"."SP_ExpiryDate">="AdvAcRestructureCal"."SP_ExpiryExtendedDate") '
        '? "AdvAcRestructureCal"."SP_ExpiryDate" : "AdvAcRestructureCal"."SP_ExpiryExtendedDate")'
        'ELSE(COALESCE("AdvAcRestructureCal"."SP_ExpiryDate","AdvAcRestructureCal"."SP_ExpiryExtendedDate")) '
        '> "AdvAcRestructureCal"."var"."BUSINESS_DATE"))'
        'THEN(IF(ISNOTEMPTY("AdvAcRestructureCal"."PreRestructureNPA_Date"))'
        'THEN("AdvAcRestructureCal"."PreRestructureNPA_Date")ELSE("AdvAcRestructureCal"."RestructureDt"))'
        'ELSE("AdvAcRestructureCal"."DEGDATE")'
    )
    assert not validate_expression(buggy).valid  # confirm it's really broken to start

    fixed = _normalize_expression(buggy)
    assert "?" not in fixed
    result = validate_expression(fixed)
    assert result.valid, result.error


def test_normalize_ternary_operator_bails_out_when_condition_not_parenthesized():
    """A ternary whose condition isn't a clean, fully-parenthesized group
    immediately before the '?' must be left completely untouched rather
    than guessed at -- the '?' stays as a literal character, so grammar
    validation still rejects it exactly as before (no regression, no
    silent wrong rewrite)."""
    from app.derivation.dd_generator import _normalize_expression

    ambiguous = 'IF(X)THEN(A ? B : C)ELSE(D)'
    result = _normalize_expression(ambiguous)
    assert "?" in result


def test_normalize_ternary_operator_handles_simple_standalone_case():
    from app.derivation.dd_generator import _normalize_ternary_operator

    assert _normalize_ternary_operator('(X>1) ? A : B') == 'IF(X>1)THEN(A)ELSE(B)'


def test_compose_simple_assignment_expression_enforces_last_write_wins():
    """Item 'Track sequential operations': proves the deterministic
    composer already correctly implements last-write-wins precedence --
    a later stage's guard is checked FIRST in the composed expression
    (wrapping everything accumulated so far as its ELSE), so if a later
    write's condition is satisfied it always wins over an earlier one,
    matching how sequential SQL UPDATE/MERGE statements actually behave.
    This was not new work this session -- this test exists to make the
    already-correct behavior explicit and regression-proof."""
    from app.derivation.dd_generator import _AssignmentSite, _compose_simple_assignment_expression

    sites = [
        _AssignmentSite(kind="UPDATE", statement_indices=[1], raw_sql="UPDATE T SET X = 'A' WHERE COND1 = 'Y'", columns_written=["X"]),
        _AssignmentSite(kind="UPDATE", statement_indices=[2], raw_sql="UPDATE T SET X = 'B' WHERE COND2 = 'Y'", columns_written=["X"]),
    ]
    result = _compose_simple_assignment_expression(sites, "T", "X")
    assert result is not None
    # The LATER stage's condition (COND2) must be the outermost IF -- it
    # gets checked first, so it wins even if COND1 also happens to be true.
    assert result.index("COND2") < result.index("COND1")


def test_undeterminable_exception_sites_catches_real_completed_and_errordate_defect(dpd_calculation_sql):
    """Regression test for the real, confirmed root cause behind the
    ERRORDATE/COMPLETED exception-flow defect: both the normal-flow
    UPDATE and the EXCEPTION-handler UPDATE for ACLRUNNINGPROCESSSTATUS
    share the exact same row-scoping WHERE clause (both restricted to
    one process's status row), so neither the LLM nor the deterministic
    composer has any real data-driven way to distinguish "an exception
    occurred" from "normal completion" -- this must be detected and the
    exception-handler write excluded, not guessed at."""
    from app.derivation.dd_generator import _assignment_sites, undeterminable_exception_sites

    objects = split_objects(dpd_calculation_sql, "dpd.sql", Dialect.ORACLE)
    info = analyze_object(objects[0])

    for column in ["ERRORDATE", "ERRORDESCRIPTION", "COMPLETED", "COUNT"]:
        sites = _assignment_sites(info, column)
        undeterminable = undeterminable_exception_sites(sites)
        assert len(undeterminable) == 1, f"{column}: expected exactly one undeterminable exception site"


def test_where_guard_extraction_ignores_comment_text(dpd_calculation_sql):
    """Regression test for a real bug found while verifying the fix
    above: the source SQL has a `--` comment directly above the real
    UPDATE statement whose own free text happens to contain the word
    "where" (`--Update BANDAUDITSTATUS ... where BandName='...'`) -- an
    earlier version of the guard extractor matched the comment's "where"
    instead of the real WHERE clause several lines later, producing a
    guard that could never match anything and silently defeating the
    whole detection."""
    from app.derivation.dd_generator import _assignment_sites, _extract_where_guard_text

    objects = split_objects(dpd_calculation_sql, "dpd.sql", Dialect.ORACLE)
    info = analyze_object(objects[0])
    sites = _assignment_sites(info, "ERRORDATE")
    guards = [_extract_where_guard_text(s.raw_sql) for s in sites]
    assert len(guards) == 2
    assert guards[0] == guards[1]
    assert "BANDNAME" not in (guards[0] or "")


def test_dd_generation_deterministically_excludes_exception_handler_write(
    dpd_calculation_sql, maxdpd_sql, npa_date_sql, mock_llm_client, function_reference
):
    """End-to-end: the composed/generated expression for a column with an
    undeterminable exception-handler write must (a) never contain the
    fabricated duplicate-condition shape, (b) be forced to
    PENDING_REVIEW, and (c) carry a clear, specific reason explaining the
    representational gap -- not silently produce a plausible-looking but
    wrong formula."""
    chain, objects, infos = _build_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql)
    model = build_canonical_model(chain, "job-exc-exclusion", objects, infos, mock_llm_client)

    rows = generate_dd_rows(
        chain=chain,
        canonical_model=model,
        objects=objects,
        structural_infos=infos,
        llm_client=mock_llm_client,
        function_reference=function_reference,
        entity_name_map={"AccountCal_Stg": "FCT_NPA_PRODUCT"},
    )

    errordate_row = next(r for r in rows if r.column_name.upper() == "ERRORDATE")
    assert errordate_row.status.value == "PENDING_REVIEW"
    assert any("exception handler" in e for e in errordate_row.validation_errors)
    # Must not contain the old fabricated duplicate-condition bug shape.
    expr_upper = errordate_row.display_derivation_expression.upper()
    assert expr_upper.count("RUNNINGPROCESSNAME") <= 1


def test_auto_parenthesize_null_check_or_fixes_real_br011_defect():
    """Regression test for a real, confirmed defect: the source's single
    atomic comparison `NVL(A.FlgProcessing,'N')='N'` was correctly
    expanded to `ISEMPTY(FlgProcessing) OR FlgProcessing=="N"`, but lost
    its grouping when combined with a preceding AND, silently changing
    `A AND (B OR C)` into `(A AND B) OR C`. This must now be auto-
    corrected deterministically, and the ambiguous-grouping guardrail
    must go silent on the corrected result."""
    from app.derivation.dd_generator import _normalize_expression
    from app.grammar.validator import validate_expression
    from app.guardrails.semantic_validation import check_ambiguous_boolean_grouping

    buggy = (
        'IF("CustomerCal_Stg"."REFCUSTOMERID"=="C"."REFCUSTOMERID" AND '
        'ISEMPTY("CustomerCal_Stg"."FlgProcessing") OR '
        '"CustomerCal_Stg"."FlgProcessing"=="N")THEN("C"."FinalNpaDt")ELSE(NULL)'
    )
    assert check_ambiguous_boolean_grouping(buggy)  # confirm it's really broken to start

    fixed = _normalize_expression(buggy)
    assert '(ISEMPTY("CustomerCal_Stg"."FlgProcessing") OR "CustomerCal_Stg"."FlgProcessing"=="N")' in fixed
    assert validate_expression(fixed).valid
    assert check_ambiguous_boolean_grouping(fixed) == []


def test_auto_parenthesize_null_check_or_does_not_touch_mismatched_columns():
    """Must not wrap an OR just because it sits next to an AND+ISEMPTY --
    only when the OR's other side actually compares the SAME column the
    ISEMPTY call checks."""
    from app.derivation.dd_generator import _normalize_expression

    mismatch = 'IF("A"."X"==1 AND ISEMPTY("A"."Y") OR "A"."Z"=="N")THEN(1)ELSE(0)'
    assert _normalize_expression(mismatch) == _normalize_expression(mismatch)  # no crash
    fixed = _normalize_expression(mismatch)
    assert '(ISEMPTY("A"."Y") OR "A"."Z"=="N")' not in fixed


def test_auto_parenthesize_null_check_or_leaves_clean_expressions_untouched():
    from app.derivation.dd_generator import _normalize_expression

    clean = 'IF(ISNOTEMPTY("A"."X") AND ISNOTEMPTY("A"."Y") AND "A"."Z"=="N")THEN("Y")ELSEIF(ISNOTEMPTY("A"."W"))THEN("Y")ELSE(NULL)'
    assert _normalize_expression(clean) == clean


def test_whole_procedure_variable_trace_finds_select_into_definition(dpd_calculation_sql):
    """Regression test for a real, high-value gap: v_ProcessDate -- the
    business date driving nearly every date calculation in
    PRO_DPD_Calculation_StoredProcedure_2.sql -- is defined via
    `SELECT "Date" INTO v_ProcessDate FROM SysDayMatrix WHERE ...`, not
    `:=`. A tracer that only recognizes `:=` would never find it at all."""
    from app.derivation.dd_generator import _assignment_sites, _whole_procedure_variable_trace

    objects = split_objects(dpd_calculation_sql, "dpd.sql", Dialect.ORACLE)
    info = analyze_object(objects[0])
    sites = _assignment_sites(info, "DPD_IntService")
    assert sites

    traces = []
    for site in sites:
        before_index = min(site.statement_indices)
        traces.extend(_whole_procedure_variable_trace(site.raw_sql, info.statements, before_index))

    assert any("SELECT" in line and "SysDayMatrix" in line for line in traces)


def test_whole_procedure_variable_trace_shows_both_mutually_exclusive_definitions(dpd_calculation_sql):
    """Regression test for a bug caught while building this: naively
    picking only the definition with the highest statement index hid the
    real SELECT...INTO definition behind its own
    `EXCEPTION WHEN NO_DATA_FOUND THEN v_ProcessDate := NULL;` fallback,
    since the fallback's statement index is numerically later even
    though the two are mutually exclusive alternate paths, not a
    sequential overwrite. Both must be shown."""
    from app.derivation.dd_generator import _assignment_sites, _whole_procedure_variable_trace

    objects = split_objects(dpd_calculation_sql, "dpd.sql", Dialect.ORACLE)
    info = analyze_object(objects[0])
    sites = _assignment_sites(info, "DPD_IntService")
    traces = []
    for site in sites:
        before_index = min(site.statement_indices)
        traces.extend(_whole_procedure_variable_trace(site.raw_sql, info.statements, before_index))

    assert any("SELECT" in line for line in traces)
    assert any(":= NULL" in line for line in traces)


def test_variable_dependency_chain_follows_multi_hop_references():
    """Synthetic multi-hop chain (A depends on B depends on C), matching
    the proposal's own example shape (DPD -> Reference Period -> ... ->
    FinalNpaDt), proving recursion actually follows references rather
    than stopping at the first hop."""
    from app.derivation.dd_generator import _trace_variable_dependency_chain
    from app.models.core import StatementInfo

    statements = [
        StatementInfo(statement_index=1, statement_type="OTHER", raw_text="v_c := 100;"),
        StatementInfo(statement_index=2, statement_type="OTHER", raw_text="v_b := v_c + 1;"),
        StatementInfo(statement_index=3, statement_type="OTHER", raw_text="v_a := v_b * 2;"),
    ]
    from app.derivation.dd_generator import _find_variable_definitions

    definitions = _find_variable_definitions(statements)
    lines = _trace_variable_dependency_chain(["v_a"], definitions, before_index=100)

    assert any("v_a" in line for line in lines)
    assert any("v_b" in line for line in lines)
    assert any("v_c" in line for line in lines)
    # v_a's own line must come before v_b's, which must come before v_c's --
    # proving this follows the dependency direction correctly.
    joined = "\n".join(lines)
    assert joined.index("v_a") < joined.index("v_b") < joined.index("v_c")


def test_variable_dependency_chain_terminates_on_cycle():
    """Two variables that reference each other must not cause infinite
    recursion -- the visited set must break the cycle."""
    from app.derivation.dd_generator import _find_variable_definitions, _trace_variable_dependency_chain
    from app.models.core import StatementInfo

    statements = [
        StatementInfo(statement_index=1, statement_type="OTHER", raw_text="v_a := v_b + 1;"),
        StatementInfo(statement_index=2, statement_type="OTHER", raw_text="v_b := v_a + 1;"),
    ]
    definitions = _find_variable_definitions(statements)
    # Must return promptly (no infinite loop) and not raise.
    lines = _trace_variable_dependency_chain(["v_a"], definitions, before_index=100)
    assert isinstance(lines, list)