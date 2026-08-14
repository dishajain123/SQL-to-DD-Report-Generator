"""LangGraph definition wiring the full pipeline together. Conditional
routing after Canonical Model building follows the Job Plan: DD Generation
only runs when the intent actually requires it (architecture step 3/12).
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.derivation.canonical_model import build_canonical_model
from app.derivation.dd_generator import generate_dd_rows
from app.derivation.llm_client import LLMClient
from app.guardrails.output_guardrails import check_dd_row
from app.guardrails.structural_guardrails import check_structural_info
from app.lineage.dependency_graph import build_graph, find_chains
from app.models.core import JobPlan, SQLObject
from app.parsing.dialect import detect_dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object
from app.report.excel_export import export_dd_rows
from app.report.report_generator import generate_report
from app.utils import db
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class PipelineState(TypedDict, total=False):
    job_plan: JobPlan
    uploaded_files: dict[str, str]  # filename -> raw content
    objects: dict[str, SQLObject]
    structural_infos: dict[str, Any]
    structural_errors: dict[str, list[str]]
    smart_chunks: dict[str, list[Any]]
    chains: list[Any]
    canonical_models: list[Any]
    dd_rows: list[Any]
    function_reference: str
    entity_name_map: dict[str, str]
    report_path: str
    excel_path: str
    existing_dd_excel_path: str


def node_split_and_parse(state: PipelineState) -> PipelineState:
    objects: dict[str, SQLObject] = {}
    for filename, content in state["uploaded_files"].items():
        dialect = detect_dialect(content)
        for obj in split_objects(content, filename, dialect):
            objects[obj.object_id] = obj
    state["objects"] = objects
    return state


def node_structural_analysis(state: PipelineState) -> PipelineState:
    structural_infos = {}
    structural_errors = {}
    for oid, obj in state["objects"].items():
        info = analyze_object(obj)
        structural_infos[oid] = info
        guardrail_result = check_structural_info(info)
        if not guardrail_result.passed:
            structural_errors[oid] = guardrail_result.errors
    state["structural_infos"] = structural_infos
    state["structural_errors"] = structural_errors
    return state


def node_smart_chunking(state: PipelineState) -> PipelineState:
    smart_chunks = {}
    for oid, info in state["structural_infos"].items():
        smart_chunks[oid] = info.smart_chunks
        db.log_audit(
            state["job_plan"].job_id,
            "smart_chunking",
            f"{len(info.smart_chunks)} chunk(s) derived for object {oid}",
        )
    state["smart_chunks"] = smart_chunks
    return state


def node_lineage(state: PipelineState) -> PipelineState:
    objects_list = list(state["objects"].values())
    graph = build_graph(objects_list, state["structural_infos"])
    chains = find_chains(graph, state["job_plan"].job_id, objects_list)
    state["chains"] = chains
    return state


def node_canonical_models(state: PipelineState, llm_client: LLMClient) -> PipelineState:
    models = []
    for chain in state["chains"]:
        model = build_canonical_model(
            chain=chain,
            job_id=state["job_plan"].job_id,
            objects=state["objects"],
            structural_infos=state["structural_infos"],
            llm_client=llm_client,
        )
        models.append(model)
        db.log_audit(state["job_plan"].job_id, "canonical_model", f"Built for chain {chain.chain_id}")
    state["canonical_models"] = models
    return state


def node_dd_generation(state: PipelineState, llm_client: LLMClient) -> PipelineState:
    all_rows = []
    for chain, model in zip(state["chains"], state["canonical_models"]):
        rows = generate_dd_rows(
            chain=chain,
            canonical_model=model,
            objects=state["objects"],
            structural_infos=state["structural_infos"],
            llm_client=llm_client,
            function_reference=state.get("function_reference", ""),
            entity_name_map=state.get("entity_name_map"),
        )
        for row in rows:
            guardrail_result = check_dd_row(row, model)
            if not guardrail_result.passed:
                row.validation_errors.extend(guardrail_result.errors)
                row.status = row.status.__class__.PENDING_REVIEW
            all_rows.append(row)

    state["dd_rows"] = all_rows
    for i, row in enumerate(all_rows):
        db.record_dd_row(state["job_plan"].job_id, row.source_chain_id, i, row.model_dump())
    return state


def node_skip_dd_generation(state: PipelineState) -> PipelineState:
    state["dd_rows"] = []
    return state


def node_report_and_export(state: PipelineState) -> PipelineState:
    job_plan = state["job_plan"]
    report_path = generate_report(
        job_plan, state["canonical_models"], state.get("dd_rows", []),
        output_path=f"output/{job_plan.job_id}/report.md",
        objects=state["objects"],
        structural_infos=state.get("structural_infos"),
    )
    state["report_path"] = str(report_path)

    if job_plan.include_dd_excel and state.get("dd_rows"):
        excel_output_path = f"output/{job_plan.job_id}/dd_export.xlsx"
        existing_path = state.get("existing_dd_excel_path")
        excel_path = export_dd_rows(state["dd_rows"], excel_output_path, existing_dd_path=existing_path)
        state["excel_path"] = str(excel_path)

    db.update_job_status(
        job_plan.job_id,
        "COMPLETED",
        report_path=state.get("report_path"),
        excel_path=state.get("excel_path"),
    )
    return state


def _requires_dd_generation(state: PipelineState) -> str:
    return "generate_dd" if state["job_plan"].requires_dd_generation else "skip_dd"


def build_pipeline(llm_client: LLMClient | None = None) -> Any:
    llm_client = llm_client or LLMClient()

    graph = StateGraph(PipelineState)
    graph.add_node("split_and_parse", node_split_and_parse)
    graph.add_node("structural_analysis", node_structural_analysis)
    graph.add_node("smart_chunking", node_smart_chunking)
    graph.add_node("lineage", node_lineage)
    graph.add_node("canonical_models", lambda state: node_canonical_models(state, llm_client))
    graph.add_node("generate_dd", lambda state: node_dd_generation(state, llm_client))
    graph.add_node("skip_dd", node_skip_dd_generation)
    graph.add_node("report_and_export", node_report_and_export)

    graph.add_edge(START, "split_and_parse")
    graph.add_edge("split_and_parse", "structural_analysis")
    graph.add_edge("structural_analysis", "smart_chunking")
    graph.add_edge("smart_chunking", "lineage")
    graph.add_edge("lineage", "canonical_models")
    graph.add_conditional_edges(
        "canonical_models",
        _requires_dd_generation,
        {"generate_dd": "generate_dd", "skip_dd": "skip_dd"},
    )
    graph.add_edge("generate_dd", "report_and_export")
    graph.add_edge("skip_dd", "report_and_export")
    graph.add_edge("report_and_export", END)

    return graph.compile()