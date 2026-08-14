from app.derivation.canonical_model import build_canonical_model
from app.derivation.dd_generator import generate_dd_rows
from app.lineage.dependency_graph import build_graph, find_chains
from app.models.core import DDStatus, Dialect
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
