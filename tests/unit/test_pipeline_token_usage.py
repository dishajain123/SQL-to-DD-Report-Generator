import json

from app.derivation.llm_client import LLMClient
from app.orchestration.pipeline import _persist_llm_token_usage


def test_persist_llm_token_usage_writes_one_audit_entry_per_stage(monkeypatch):
    recorded: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "app.orchestration.pipeline.db.log_audit_bulk",
        lambda entries, db_path=None: recorded.extend(entries),
    )

    client = LLMClient(provider="openai", api_key="k", model="gpt-4.1", base_url="https://api.openai.com/v1")
    client.token_usage_log.extend(
        [
            {
                "stage": "technical_reasoning", "provider": "openai", "model": "gpt-4.1",
                "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            },
            {
                "stage": "dd_generation", "provider": "openai", "model": "gpt-4.1",
                "prompt_tokens": 300, "completion_tokens": 50, "total_tokens": 350,
            },
        ]
    )

    _persist_llm_token_usage("job-usage-1", client)

    assert len(recorded) == 2
    job_id, stage, detail = recorded[0]
    assert job_id == "job-usage-1"
    assert stage == "llm_tokens:technical_reasoning"
    assert json.loads(detail) == {
        "provider": "openai", "model": "gpt-4.1",
        "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
    }
    assert recorded[1][1] == "llm_tokens:dd_generation"

    # Draining leaves nothing behind for a later checkpoint to re-persist.
    assert client.token_usage_log == []


def test_persist_llm_token_usage_is_a_noop_when_nothing_was_recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.orchestration.pipeline.db.log_audit_bulk",
        lambda entries, db_path=None: calls.append(entries),
    )

    client = LLMClient(provider="openai", api_key="k", model="gpt-4.1", base_url="https://api.openai.com/v1")
    _persist_llm_token_usage("job-usage-2", client)

    assert calls == []
