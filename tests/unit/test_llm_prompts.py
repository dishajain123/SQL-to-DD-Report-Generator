from typing import Optional

from app.derivation.llm_client import LLMClient, _load_prompts


def test_prompt_yaml_loads_required_keys():
    prompts = _load_prompts()

    assert "technical_reasoning_system" in prompts
    assert "business_reasoning_system" in prompts
    assert "dd_generation_system" in prompts
    assert "rule_explanation_system" in prompts
    assert "retry_with_error_system" in prompts
    assert "domain-specific abbreviation, status code, or business term" in prompts["business_reasoning_system"]


def test_llm_client_methods_use_expected_prompt_pairs(monkeypatch):
    prompts = {
        "technical_reasoning_system": "TECH_SYS",
        "technical_reasoning_user": "TECH_USER {sql_snippets}",
        "business_reasoning_system": "BUS_SYS",
        "business_reasoning_user": "BUS_USER {technical_summary}",
        "dd_generation_system": "DD_SYS",
        "dd_generation_user": "DD_USER {entity_name} {column_name} {relevant_sql} {rag_context}",
        "rule_explanation_system": "RULE_SYS",
        "rule_explanation_user": "RULE_USER {entity_name} {column_name} {formula}",
        "retry_with_error_system": "RETRY_SYS",
        "retry_with_error_user": "RETRY_USER {previous_expression} {error} {context}",
    }

    monkeypatch.setattr("app.derivation.llm_client._load_prompts", lambda: prompts)

    captured: list[tuple[str, str]] = []

    class Client(LLMClient):
        def _complete(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
            captured.append((system, user))
            if system == "BUS_SYS":
                return '{"business_summary":"ok","glossary_terms":[{"term":"DPD","definition":"days past due"}]}'
            if system == "RULE_SYS":
                return "why this matters"
            return "ok"

    client = Client(provider="openai", api_key="k", model="gpt-4.1", base_url="https://api.openai.com/v1")

    assert client.technical_reasoning(["SELECT 1;"]) == "ok"
    assert client.business_reasoning("technical") == "ok"
    assert client.business_reasoning_details("technical").glossary_terms[0].term == "DPD"
    assert (
        client.generate_formula_expression(
            technical_summary="technical",
            business_summary="business",
            source_sql="source",
            function_reference="reference",
            column_name="COL",
            entity_name="ENT",
            relevant_sql="relevant",
            rag_context="rag",
        )
        == "ok"
    )
    assert (
        client.rule_explanation(
            technical_summary="technical",
            business_summary="business",
            source_sql="source",
            function_reference="reference",
            column_name="COL",
            entity_name="ENT",
            relevant_sql="relevant",
            formula="formula",
        )
        == "why this matters"
    )
    assert client.retry_with_error("prev", "error", "context") == "ok"

    assert captured == [
        ("TECH_SYS", "TECH_USER SELECT 1;"),
        ("BUS_SYS", "BUS_USER technical"),
        ("BUS_SYS", "BUS_USER technical"),
        ("DD_SYS", "DD_USER ENT COL relevant rag"),
        ("RULE_SYS", "RULE_USER ENT COL formula"),
        ("RETRY_SYS", "RETRY_USER prev error context"),
    ]
