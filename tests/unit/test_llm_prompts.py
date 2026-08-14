from app.derivation.llm_client import _load_prompts


def test_prompt_yaml_loads_required_keys():
    prompts = _load_prompts()

    assert "technical_reasoning_system" in prompts
    assert "business_reasoning_system" in prompts
    assert "dd_generation_system" in prompts
    assert "retry_with_error_system" in prompts
