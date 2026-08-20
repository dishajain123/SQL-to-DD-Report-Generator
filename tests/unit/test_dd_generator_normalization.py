import json

import sqlglot
import pytest

from app.derivation.canonical_model import build_canonical_model
from app.derivation.dd_generator import (
    _build_source_statement_refs,
    _collect_alias_resolution_inventory,
    _extract_aggregate_info,
    _collect_source_reference_inventory,
    _extract_variable_trace,
    _format_assignment_context,
    _interpret_llm_output,
    _ground_expression_to_source_references,
    _normalize_expression,
    _translate_case_to_4x,
    generate_dd_rows,
)
from app.grammar.validator import validate_expression
from app.lineage.dependency_graph import build_graph, find_chains
from app.models.core import DDStatus, Dialect, DerivationOption, SmartChunk
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object
from app.utils.sql_aliases import resolve_aliases_in_expression


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
    # branch's literal (see app/derivation/period_pruning.py), and some
    # columns are represented without IF at all (for example as a direct
    # reference or COALESCE). Only expressions that actually contain IF
    # must therefore be checked for THEN/ELSE normalization.
    assert all("THEN(" in row.display_derivation_expression or "IF(" not in row.display_derivation_expression for row in rows)
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


def test_sqlserver_select_into_seed_projection_is_deterministic(sma_marking_sql, mock_llm_client, function_reference):
    objects_list = []
    infos = {}
    for obj in split_objects(sma_marking_sql, "PRO.SMA_MARKING_12122023.StoredProcedure.sql", Dialect.SQLSERVER):
        objects_list.append(obj)
        infos[obj.object_id] = analyze_object(obj)

    graph = build_graph(objects_list, infos)
    chains = find_chains(graph, "job-sma-seed", objects_list)
    objects = {o.object_id: o for o in objects_list}

    model = build_canonical_model(chains[0], "job-sma-seed", objects, infos, mock_llm_client)
    rows = generate_dd_rows(
        chain=chains[0],
        canonical_model=model,
        objects=objects,
        structural_infos=infos,
        llm_client=mock_llm_client,
        function_reference=function_reference,
    )

    rows_by_column = {row.column_name.upper(): row for row in rows}
    for column in ["LASTCRDATE", "INTNOTSERVICEDDT", "OVERDUESINCEDT", "REVIEWDUEDT"]:
        assert column in rows_by_column
        assert rows_by_column[column].status == DDStatus.ACTIVE
        assert "FCT_NPA_PRODUCT" not in rows_by_column[column].display_derivation_expression
        assert '"A".' in rows_by_column[column].display_derivation_expression
        assert not rows_by_column[column].validation_errors

    assert 'COALESCE("A"."DPD_IntService", 0) >= COALESCE("A"."RefPeriodIntService", 0)' in rows_by_column["DPD_INTSERVICE"].display_derivation_expression
    assert 'COALESCE("A"."DPD_NoCredit", 0) >= COALESCE("A"."RefPeriodNoCredit", 0)' in rows_by_column["DPD_NOCREDIT"].display_derivation_expression

    assert rows_by_column["UCIF_ID"].status == DDStatus.ACTIVE
    assert "FCT_NPA_PRODUCT" not in rows_by_column["UCIF_ID"].display_derivation_expression
    assert rows_by_column["UCIF_ID"].validation_errors == []


def test_interpret_llm_output_routes_json_decision_tables_before_formula_parsing():
    raw_output = """```json
    {"decisionTable": {"rules": [{"when": "A", "then": "B"}]}}
    ```"""

    derivation_option, expression, decision_table_json, parse_errors = _interpret_llm_output(raw_output)

    assert derivation_option == DerivationOption.DECISION_TABLE
    assert expression is None
    assert json.loads(decision_table_json) == {"rules": [{"when": "A", "then": "B"}]}
    assert parse_errors == []


def test_simple_case_expression_translates_to_valid_if_else_chain():
    case_node = sqlglot.parse_one("CASE x WHEN 1 THEN 'A' WHEN 2 THEN 'B' ELSE 'C' END", read="oracle")
    translated = _translate_case_to_4x(case_node)

    assert translated == 'IF(x == 1)THEN("A")ELSEIF(x == 2)THEN("B")ELSE("C")'
    assert validate_expression(translated).valid


def test_semantic_validation_errors_blank_review_only_expressions(
    sma_marking_sql, function_reference
):
    class SemanticHallucinationLLMClient:
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
            return '"DPD_AGGREGATED"."ACCOUNTENTITYID"'

        def retry_with_error(self, previous_expression, error, context) -> str:
            return '"DPD_AGGREGATED"."ACCOUNTENTITYID"'

    objects_list = []
    infos = {}
    for obj in split_objects(sma_marking_sql, "PRO.SMA_MARKING_12122023.StoredProcedure.sql", Dialect.SQLSERVER):
        objects_list.append(obj)
        infos[obj.object_id] = analyze_object(obj)

    graph = build_graph(objects_list, infos)
    chains = find_chains(graph, "job-sma-hallucination", objects_list)
    objects = {o.object_id: o for o in objects_list}

    client = SemanticHallucinationLLMClient()
    model = build_canonical_model(chains[0], "job-sma-hallucination", objects, infos, client)
    rows = generate_dd_rows(
        chain=chains[0],
        canonical_model=model,
        objects=objects,
        structural_infos=infos,
        llm_client=client,
        function_reference=function_reference,
    )

    assert any(row.validation_errors for row in rows)
    assert all(row.display_derivation_expression == "" for row in rows if row.validation_errors)
    assert all("DPD_AGGREGATED" not in row.display_derivation_expression for row in rows)


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
    assert rows_by_column["LASTCRDATE"].display_derivation_expression == "NULL"


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


def test_grounded_source_references_rewrite_only_to_unique_source_qualifiers():
    inventory = _collect_source_reference_inventory(
        """
        SELECT A.OverDueSinceDt, A.CustomerEntityID, B.OtherColumn
        FROM AccountCal A
        JOIN CustomerCal B ON A.CustomerEntityID = B.CustomerEntityID
        """,
        Dialect.ORACLE,
        entity_name="TARGET_ENTITY",
    )

    expr = (
        'IF(ISNOTEMPTY("FCT_NPA_PRODUCT"."OverDueSinceDt"))'
        'THEN(DATEDIFF("FCT_NPA_PRODUCT"."var"."BUSINESS_DATE",'
        '"FCT_NPA_PRODUCT"."OverDueSinceDt","d")+1)ELSE(0)'
    )
    grounded = _ground_expression_to_source_references(expr, inventory, "TARGET_ENTITY")

    assert '"A"."OverDueSinceDt"' in grounded
    assert '"FCT_NPA_PRODUCT"."OverDueSinceDt"' not in grounded
    assert '"TARGET_ENTITY"."var"."BUSINESS_DATE"' in grounded


def test_grounded_source_references_do_not_guess_when_multiple_source_qualifiers_exist():
    inventory = _collect_source_reference_inventory(
        """
        SELECT A.SharedColumn, B.SharedColumn
        FROM AccountCal A
        JOIN CustomerCal B ON A.CustomerEntityID = B.CustomerEntityID
        """,
        Dialect.ORACLE,
        entity_name="FCT_NPA_PRODUCT",
    )

    expr = 'IF(ISNOTEMPTY("FCT_NPA_PRODUCT"."SharedColumn"))THEN("FCT_NPA_PRODUCT"."SharedColumn")ELSE(0)'
    grounded = _ground_expression_to_source_references(expr, inventory, "FCT_NPA_PRODUCT")

    assert grounded == expr


def test_alias_resolution_rewrites_final_platform_condition_to_source_table_name():
    inventory = _collect_alias_resolution_inventory(
        "SELECT a.AccountEntityID FROM ACCOUNTCAL a",
        Dialect.ORACLE,
    )

    expr = 'IF(ISNOTEMPTY("a"."AccountEntityID"))THEN("a"."AccountEntityID")ELSE(NULL)'
    resolved = resolve_aliases_in_expression(expr, inventory, quote_replacements=True)

    assert resolved == 'IF(ISNOTEMPTY("ACCOUNTCAL"."AccountEntityID"))THEN("ACCOUNTCAL"."AccountEntityID")ELSE(NULL)'


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


def test_translate_case_to_4x_produces_valid_grammar():
    """A simple, literal-outcome SQL CASE WHEN must translate
    deterministically to the exact 4X IF/ELSEIF/ELSE form and validate
    cleanly -- the same category of mechanical syntax fix as the ternary
    normalizer, just for a different SQL-only construct the platform
    grammar also has no direct equivalent for."""
    from app.derivation.dd_generator import _compose_simple_assignment_expression, _AssignmentSite
    from app.grammar.validator import validate_expression

    raw = (
        "UPDATE T SET RISK_BAND = CASE WHEN DPD > 90 THEN 'NPA' "
        "WHEN DPD > 30 THEN 'SMA' ELSE 'STANDARD' END WHERE ACCOUNT_TYPE = 'LOAN'"
    )
    site = _AssignmentSite(kind="UPDATE", statement_indices=[1], raw_sql=raw, columns_written=["RISK_BAND"])
    composed = _compose_simple_assignment_expression([site], "T", "RISK_BAND")

    assert composed is not None
    assert 'IF(DPD > 90)THEN("NPA")' in composed
    assert 'ELSEIF(DPD > 30)THEN("SMA")' in composed
    assert 'ELSE("STANDARD")' in composed
    assert validate_expression(composed).valid


def test_translate_case_to_4x_bails_out_on_complex_real_nested_case(dpd_calculation_sql):
    """The real DPD_IntService assignment has a doubly-nested CASE WHEN
    with arithmetic branch values
    (`(v_ProcessDate - A.IntNotServicedDt) + 2`), not literal outcomes --
    this must NOT be force-translated (the branch values aren't simple),
    it must fall back to the LLM path exactly as before, same as any
    other case genuinely too complex for the deterministic composer."""
    from app.derivation.dd_generator import _assignment_sites, _compose_simple_assignment_expression

    objects = split_objects(dpd_calculation_sql, "dpd.sql", Dialect.ORACLE)
    info = analyze_object(objects[0])
    sites = _assignment_sites(info, "DPD_IntService")
    composed = _compose_simple_assignment_expression(sites, "AccountCal_Stg", "DPD_IntService")
    assert composed is None


def test_translate_case_to_4x_now_handles_simple_arithmetic_branch_values():
    """As of the write-order-coverage extension, `A + B`-shaped simple
    arithmetic branch values are recognized as safely composable (see
    _is_safely_composable_value) -- this intentionally widens the
    boundary from an earlier version of this test, which asserted that
    exact shape was rejected. The result must be grammar-valid."""
    from app.derivation.dd_generator import _compose_simple_assignment_expression, _AssignmentSite
    from app.grammar.validator import validate_expression

    raw = "UPDATE T SET X = CASE WHEN A > 1 THEN A + B ELSE 0 END WHERE Y = 1"
    site = _AssignmentSite(kind="UPDATE", statement_indices=[1], raw_sql=raw, columns_written=["X"])
    composed = _compose_simple_assignment_expression([site], "T", "X")
    assert composed is not None
    assert validate_expression(composed).valid


def test_translate_case_to_4x_still_returns_none_for_genuinely_complex_branch_values():
    """A branch value the deterministic composer genuinely cannot safely
    represent (a subquery) must still bail to the LLM path -- the
    boundary widened for simple arithmetic, it did not disappear."""
    from app.derivation.dd_generator import _compose_simple_assignment_expression, _AssignmentSite

    raw = "UPDATE T SET X = CASE WHEN A > 1 THEN (SELECT MAX(Z) FROM W) ELSE 0 END WHERE Y = 1"
    site = _AssignmentSite(kind="UPDATE", statement_indices=[1], raw_sql=raw, columns_written=["X"])
    assert _compose_simple_assignment_expression([site], "T", "X") is None


def test_repair_missing_if_before_then_fixes_isolated_case():
    """A bare (cond)THEN(a)ELSE(b) construct -- missing its leading IF --
    sitting as the value inside an outer THEN(...) clause must be
    repaired to IF(cond)THEN(a)ELSE(b). Confirmed against a real
    generation defect (a DEGDATE expression) that produced exactly this
    shape while attempting the same "pick the greater of two values"
    construct that separately also produced a ternary defect elsewhere."""
    from app.derivation.dd_generator import _normalize_expression
    from app.grammar.validator import validate_expression

    buggy = "IF(X==1)THEN((A>=B)THEN(A)ELSE(B))ELSE(0)"
    fixed = _normalize_expression(buggy)
    assert fixed == "IF(X==1)THEN(IF(A>=B)THEN(A)ELSE(B))ELSE(0)"
    assert validate_expression(fixed).valid


def test_repair_missing_if_before_then_leaves_well_formed_expressions_untouched():
    from app.derivation.dd_generator import _repair_missing_if_before_then

    clean = "IF(X==1)THEN(A)ELSEIF(X==2)THEN(B)ELSE(C)"
    assert _repair_missing_if_before_then(clean) == clean


def test_repair_missing_if_before_then_handles_multiple_occurrences():
    from app.derivation.dd_generator import _repair_missing_if_before_then
    from app.grammar.validator import validate_expression

    buggy = "IF(X==1)THEN((A>=B)THEN(A)ELSE(B))ELSE((C>=D)THEN(C)ELSE(D))"
    fixed = _repair_missing_if_before_then(buggy)
    assert validate_expression(fixed).valid


def test_repair_missing_if_before_then_is_honest_about_its_limitation():
    """This function alone does NOT fully repair the exact real compound
    defect it was built from (missing IF combined with a separate extra
    unmatched paren elsewhere in the same expression) -- documenting that
    limitation as an executable test, not just a comment, so it can never
    silently start being relied on for more than it actually does."""
    from app.derivation.dd_generator import _repair_missing_if_before_then
    from app.report.formula_pretty_printer import _pretty_parser

    real_compound_defect = (
        'IF(("AdvAcRestructureCal"."FINALASSETCLASSALT_KEY">1 OR "AdvAcRestructureCal"."FlgDeg"=="Y") '
        'AND IF(ISNOTEMPTY("AdvAcRestructureCal"."SP_ExpiryDate") AND '
        'ISNOTEMPTY("AdvAcRestructureCal"."SP_ExpiryExtendedDate")))'
        'THEN(("AdvAcRestructureCal"."SP_ExpiryDate">="AdvAcRestructureCal"."SP_ExpiryExtendedDate")'
        'THEN("AdvAcRestructureCal"."SP_ExpiryDate")ELSE("AdvAcRestructureCal"."SP_ExpiryExtendedDate"))'
        'ELSE("AdvAcRestructureCal"."SP_ExpiryExtendedDate") > "AdvAcRestructureCal"."var"."BUSINESS_DATE")'
        'THEN(IF(ISNOTEMPTY("AdvAcRestructureCal"."PreRestructureNPA_Date"))'
        'THEN("AdvAcRestructureCal"."PreRestructureNPA_Date")ELSE("AdvAcRestructureCal"."RestructureDt"))'
        'ELSE("AdvAcRestructureCal"."DEGDATE")'
    )
    fixed = _repair_missing_if_before_then(real_compound_defect)
    with pytest.raises(Exception):
        _pretty_parser.parse(fixed)


# ---------------------------------------------------------------------------
# Write-order coverage extension: regression tests for multiple sequential
# writes to the same column, per the explicit requirement that later writes
# must correctly override earlier writes and the LLM must never decide this.
# ---------------------------------------------------------------------------


def test_multiple_unconditional_writes_preserve_sequential_overwrite_semantics():
    """UPDATE X = A; UPDATE X = B; UPDATE X = C -- with no WHERE guards at
    all -- must NOT become mutually-exclusive IF/ELSEIF branches (there is
    no condition to branch on); it must preserve "the last unconditional
    write wins" by making the final composed value literally just the
    last stage's value, since each unconditional write unconditionally
    replaces the accumulated expression so far."""
    from app.derivation.dd_generator import _AssignmentSite, _compose_simple_assignment_expression

    sites = [
        _AssignmentSite(kind="UPDATE", statement_indices=[1], raw_sql="UPDATE T SET X = 'A'", columns_written=["X"]),
        _AssignmentSite(kind="UPDATE", statement_indices=[2], raw_sql="UPDATE T SET X = 'B'", columns_written=["X"]),
        _AssignmentSite(kind="UPDATE", statement_indices=[3], raw_sql="UPDATE T SET X = 'C'", columns_written=["X"]),
    ]
    composed = _compose_simple_assignment_expression(sites, "T", "X")
    assert composed == '"C"'


def test_three_conditional_writes_check_latest_condition_first():
    """Three conditional writes to the same column: the LATEST write's
    condition must be the outermost check (evaluated first), falling
    through to progressively earlier writes, then the original column
    value -- proving order is preserved across more than two writes, not
    just the two-write case already covered elsewhere."""
    from app.derivation.dd_generator import _AssignmentSite, _compose_simple_assignment_expression
    from app.grammar.validator import validate_expression

    sites = [
        _AssignmentSite(kind="UPDATE", statement_indices=[1], raw_sql="UPDATE T SET X = 'A' WHERE COND1 = 'Y'", columns_written=["X"]),
        _AssignmentSite(kind="UPDATE", statement_indices=[2], raw_sql="UPDATE T SET X = 'B' WHERE COND2 = 'Y'", columns_written=["X"]),
        _AssignmentSite(kind="UPDATE", statement_indices=[3], raw_sql="UPDATE T SET X = 'C' WHERE COND3 = 'Y'", columns_written=["X"]),
    ]
    composed = _compose_simple_assignment_expression(sites, "T", "X")
    assert composed is not None
    assert validate_expression(composed).valid
    # COND3 (latest write) must be checked before COND2, which must be
    # checked before COND1 (earliest write) -- proving strict order
    # across all three, not just adjacent pairs.
    assert composed.index("COND3") < composed.index("COND2") < composed.index("COND1")


def test_mixed_conditional_and_unconditional_writes_preserve_order():
    """A conditional write followed by a later UNCONDITIONAL write: the
    unconditional write must win outright (it always executes last and
    always applies), completely replacing the earlier conditional logic
    -- not merged into an ELSE branch, since the source SQL's own
    unconditional UPDATE has no guard to preserve."""
    from app.derivation.dd_generator import _AssignmentSite, _compose_simple_assignment_expression

    sites = [
        _AssignmentSite(kind="UPDATE", statement_indices=[1], raw_sql="UPDATE T SET X = 'A' WHERE COND1 = 'Y'", columns_written=["X"]),
        _AssignmentSite(kind="UPDATE", statement_indices=[2], raw_sql="UPDATE T SET X = 'B'", columns_written=["X"]),
    ]
    composed = _compose_simple_assignment_expression(sites, "T", "X")
    # The later, unconditional write completely replaces everything
    # accumulated before it -- the final result is just its own value.
    assert composed == '"B"'


def test_reversed_source_order_produces_different_precedence():
    """Sanity check that order genuinely matters and isn't a no-op: the
    SAME two conditional writes in the OPPOSITE source order must
    produce a DIFFERENT composed expression (the later one always
    becomes the outer/first-checked condition)."""
    from app.derivation.dd_generator import _AssignmentSite, _compose_simple_assignment_expression

    forward = [
        _AssignmentSite(kind="UPDATE", statement_indices=[1], raw_sql="UPDATE T SET X = 'A' WHERE COND1 = 'Y'", columns_written=["X"]),
        _AssignmentSite(kind="UPDATE", statement_indices=[2], raw_sql="UPDATE T SET X = 'B' WHERE COND2 = 'Y'", columns_written=["X"]),
    ]
    reversed_order = [
        _AssignmentSite(kind="UPDATE", statement_indices=[1], raw_sql="UPDATE T SET X = 'B' WHERE COND2 = 'Y'", columns_written=["X"]),
        _AssignmentSite(kind="UPDATE", statement_indices=[2], raw_sql="UPDATE T SET X = 'A' WHERE COND1 = 'Y'", columns_written=["X"]),
    ]
    forward_composed = _compose_simple_assignment_expression(forward, "T", "X")
    reversed_composed = _compose_simple_assignment_expression(reversed_order, "T", "X")
    assert forward_composed != reversed_composed
    # In `forward`, COND2 (the later write) is checked first.
    assert forward_composed.index("COND2") < forward_composed.index("COND1")
    # In `reversed_order`, COND1 is now the later write and is checked first.
    assert reversed_composed.index("COND1") < reversed_composed.index("COND2")


# ---------------------------------------------------------------------------
# _is_safely_composable_value: the new arithmetic/function-wrapper coverage
# ---------------------------------------------------------------------------


def test_is_safely_composable_value_accepts_simple_arithmetic():
    from app.derivation.dd_generator import _is_safely_composable_value

    assert _is_safely_composable_value("(v_ProcessDate - A.LastCrDate) + 1")
    assert _is_safely_composable_value("A + B - C")
    assert _is_safely_composable_value("NVL(A.LastCrDate, 0) + 1")


def test_is_safely_composable_value_rejects_subqueries_and_unknown_functions():
    from app.derivation.dd_generator import _is_safely_composable_value

    assert not _is_safely_composable_value("(SELECT MAX(X) FROM Y)")
    assert not _is_safely_composable_value("SOME_UNKNOWN_FUNC(A, B)")
    assert not _is_safely_composable_value("A + (SELECT 1 FROM DUAL)")


def test_case_with_parenthesized_wrapper_is_now_recognized():
    """Regression test for a real bug found while extending coverage: a
    CASE expression wrapped in an extra layer of source parentheses
    (`(CASE WHEN ... END)`, a very common real-world style -- confirmed
    in PRO_DPD_Calculation_StoredProcedure_2.sql's DPD_IntService) was
    being silently missed entirely, because sqlglot parses the extra
    parens as their own exp.Paren node wrapping the exp.Case, and the
    isinstance check wasn't unwrapping it first."""
    from app.derivation.dd_generator import _parse_simple_assignment_stage

    raw = "UPDATE T SET X = (CASE WHEN A IS NOT NULL THEN (B - A) ELSE 0 END) WHERE 1=1"
    result = _parse_simple_assignment_stage(raw, "X")
    assert result is not None
    guard, value, _ = result
    assert "IF(" in value and "THEN(" in value and "ELSE(" in value


def test_wrap_bare_not_in_parens_fixes_real_is_not_null_translation():
    """Regression test for a real bug found while extending coverage:
    sqlglot re-serializes `X IS NOT NULL` as `NOT X IS NULL`, which the
    existing IS-NULL rewrite turns into the still-grammar-invalid
    `NOT ISEMPTY(X)` (4X requires NOT as a function call, NOT(...), with
    no bare-operator form at all). This was silently causing a
    deterministically-translatable real CASE expression (DPD_IntService
    in PRO_DPD_Calculation_StoredProcedure_2.sql) to fall back to the LLM
    for a reason that had nothing to do with the CASE translation logic
    itself -- and would affect ANY generated formula containing a bare
    NOT, not just this one deterministic-composer code path."""
    from app.derivation.dd_generator import _normalize_expression
    from app.grammar.validator import validate_expression

    normalized = _normalize_expression("NOT A.IntNotServicedDt IS NULL", "")
    assert normalized == 'NOT(ISEMPTY("A"."IntNotServicedDt"))'
    full = f'IF({normalized})THEN(1)ELSE(0)'
    assert validate_expression(full).valid


def test_real_dpd_overdrawn_now_composes_fully_deterministically(dpd_calculation_sql):
    """End-to-end proof on real SQL: DPD_Overdrawn -- which fell back to
    the LLM before this extension -- now composes fully deterministically
    end-to-end, and the result is grammar-valid."""
    from app.derivation.dd_generator import _assignment_sites, _compose_simple_assignment_expression
    from app.grammar.validator import validate_expression

    objects = split_objects(dpd_calculation_sql, "dpd.sql", Dialect.ORACLE)
    info = analyze_object(objects[0])
    sites = _assignment_sites(info, "DPD_Overdrawn")
    composed = _compose_simple_assignment_expression(sites, "AccountCal_Stg", "DPD_Overdrawn")
    assert composed is not None
    assert validate_expression(composed).valid
