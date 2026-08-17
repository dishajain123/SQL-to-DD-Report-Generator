from typing import Optional

from app.derivation.llm_client import LLMClient, _load_prompts


def test_prompt_yaml_loads_required_keys():
    prompts = _load_prompts()

    assert "technical_reasoning_system" in prompts
    assert "business_reasoning_system" in prompts
    assert "dd_generation_system" in prompts
    assert "retry_with_error_system" in prompts


def test_llm_client_methods_use_expected_prompt_pairs(monkeypatch):
    prompts = {
        "technical_reasoning_system": "TECH_SYS",
        "technical_reasoning_user": "TECH_USER {sql_snippets}",
        "business_reasoning_system": "BUS_SYS",
        "business_reasoning_user": "BUS_USER {technical_summary}",
        "dd_generation_system": "DD_SYS",
        "dd_generation_user": "DD_USER {entity_name} {column_name} {relevant_sql} {rag_context}",
        "retry_with_error_system": "RETRY_SYS",
        "retry_with_error_user": "RETRY_USER {previous_expression} {error} {context}",
    }

    monkeypatch.setattr("app.derivation.llm_client._load_prompts", lambda: prompts)

    captured: list[tuple[str, str]] = []

    class Client(LLMClient):
        def _complete(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
            captured.append((system, user))
            return "ok"

    client = Client(provider="openai", api_key="k", model="gpt-4.1", base_url="https://api.openai.com/v1")

    assert client.technical_reasoning(["SELECT 1;"]) == "ok"
    assert client.business_reasoning("technical") == "ok"
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
    assert client.retry_with_error("prev", "error", "context") == "ok"

    assert captured == [
        ("TECH_SYS", "TECH_USER SELECT 1;"),
        ("BUS_SYS", "BUS_USER technical"),
        ("DD_SYS", "DD_USER ENT COL relevant rag"),
        ("RETRY_SYS", "RETRY_USER prev error context"),
    ]
