"""Architecture step 9: Cross-Object Dependency & Lineage.

Builds a job-level graph linking objects that read tables another object in
the same job writes to (or that reads the same table another object wrote,
capturing the common ETL pattern of several procs mutating one staging
table in sequence). Weakly-connected components become lineage chains, each
topologically ordered so downstream steps process objects in write order.
"""
from __future__ import annotations

import uuid

import networkx as nx

from app.models.core import LineageChain, SQLObject, StructuralInfo


def build_graph(
    objects: list[SQLObject], structural_infos: dict[str, StructuralInfo]
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for obj in objects:
        graph.add_node(obj.object_id, name=obj.name)

    # Definite producer -> consumer edges: A writes a table B reads.
    for a in objects:
        info_a = structural_infos[a.object_id]
        for b in objects:
            if a.object_id == b.object_id:
                continue
            info_b = structural_infos[b.object_id]
            if set(info_a.tables_written) & set(info_b.tables_read):
                graph.add_edge(a.object_id, b.object_id, reason="read_after_write")

    # Explicit call references (EXEC/CALL).
    name_to_id = {o.name.lower(): o.object_id for o in objects}
    for a in objects:
        info_a = structural_infos[a.object_id]
        for called in info_a.called_objects:
            target_id = name_to_id.get(called.lower())
            if target_id and target_id != a.object_id:
                graph.add_edge(a.object_id, target_id, reason="explicit_call")

    # Weaker signal: two objects write the same table but no read/write
    # edge already links them. Order by position in the input list (upload
    # order) as a last-resort tiebreaker, flagged with lower confidence.
    for i, a in enumerate(objects):
        info_a = structural_infos[a.object_id]
        for b in objects[i + 1 :]:
            info_b = structural_infos[b.object_id]
            if set(info_a.tables_written) & set(info_b.tables_written):
                if not graph.has_edge(a.object_id, b.object_id) and not graph.has_edge(
                    b.object_id, a.object_id
                ):
                    graph.add_edge(a.object_id, b.object_id, reason="shared_write_target_weak")

    return graph


def find_chains(graph: nx.DiGraph, job_id: str, objects: list[SQLObject]) -> list[LineageChain]:
    """Group objects into lineage chains and order each chain.

    Ordering is a real topological sort when the graph is acyclic. Objects
    that mutually read/write the same staging table (a common self-referencing
    ETL pattern — see DPD_Calculation <-> NPA_Date_Calculation in the sample
    procs) can create a cycle that table-name-level heuristics cannot resolve
    on their own. In that case we fall back to the order the objects were
    submitted in and mark order_confidence="low" so Human Review (or a
    supplied job/batch sequence, if available) can correct it rather than the
    pipeline silently guessing.
    """
    submission_index = {o.object_id: i for i, o in enumerate(objects)}
    chains: list[LineageChain] = []
    for component in nx.weakly_connected_components(graph):
        subgraph = graph.subgraph(component)
        try:
            order = list(nx.topological_sort(subgraph))
            confidence = "high"
        except nx.NetworkXUnfeasible:
            order = sorted(component, key=lambda oid: submission_index.get(oid, 0))
            confidence = "low"
        chains.append(
            LineageChain(
                chain_id=f"chain-{uuid.uuid4().hex[:8]}",
                object_ids=list(component),
                order=order,
                order_confidence=confidence,
            )
        )
    return chains
