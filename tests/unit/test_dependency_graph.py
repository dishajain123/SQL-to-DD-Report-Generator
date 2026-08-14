from app.lineage.dependency_graph import build_graph, find_chains
from app.models.core import Dialect
from app.parsing.object_splitter import split_objects
from app.parsing.structural_analysis import analyze_object


def _build(dpd_sql, maxdpd_sql, npa_sql):
    files = {
        "dpd.sql": dpd_sql,
        "maxdpd.sql": maxdpd_sql,
        "npa.sql": npa_sql,
    }
    objects = []
    infos = {}
    for fname, text in files.items():
        for obj in split_objects(text, fname, Dialect.ORACLE):
            objects.append(obj)
            infos[obj.object_id] = analyze_object(obj)
    return objects, infos


def test_real_procs_form_a_single_connected_chain(dpd_calculation_sql, maxdpd_sql, npa_date_sql):
    objects, infos = _build(dpd_calculation_sql, maxdpd_sql, npa_date_sql)
    graph = build_graph(objects, infos)
    chains = find_chains(graph, "job-1", objects)

    assert len(chains) == 1
    id_to_name = {o.object_id: o.name for o in objects}
    names_in_order = [id_to_name[oid] for oid in chains[0].order]
    assert set(names_in_order) == {"DPD_Calculation", "MaxDPD_ReferencePeriod_Calculation", "NPA_Date_Calculation"}


def test_unrelated_objects_form_separate_chains():
    from app.models.core import ObjectType, SQLObject
    from app.parsing.structural_analysis import analyze_object as analyze

    obj_a = SQLObject(
        object_id="a", name="A", object_type=ObjectType.PROCEDURE, dialect=Dialect.ORACLE,
        raw_sql="UPDATE table_a SET x = 1", source_file="a.sql",
    )
    obj_b = SQLObject(
        object_id="b", name="B", object_type=ObjectType.PROCEDURE, dialect=Dialect.ORACLE,
        raw_sql="UPDATE table_b SET y = 1", source_file="b.sql",
    )
    infos = {"a": analyze(obj_a), "b": analyze(obj_b)}
    graph = build_graph([obj_a, obj_b], infos)
    chains = find_chains(graph, "job-2", [obj_a, obj_b])

    assert len(chains) == 2


def test_cyclic_dependency_falls_back_to_low_confidence_order(dpd_calculation_sql, npa_date_sql):
    # DPD_Calculation and NPA_Date_Calculation alone (without MaxDPD in the
    # middle) both read AND write AccountCal_Stg -- a real cycle.
    objects, infos = _build(dpd_calculation_sql, npa_date_sql, npa_date_sql)
    objects = objects[:1] + objects[1:2]  # dpd + first npa copy only
    infos = {o.object_id: infos[o.object_id] for o in objects}
    graph = build_graph(objects, infos)
    chains = find_chains(graph, "job-3", objects)

    assert len(chains) == 1
    assert chains[0].order_confidence == "low"
    # deterministic: falls back to submission order, not silently arbitrary
    id_to_name = {o.object_id: o.name for o in objects}
    assert [id_to_name[oid] for oid in chains[0].order] == ["DPD_Calculation", "NPA_Date_Calculation"]
