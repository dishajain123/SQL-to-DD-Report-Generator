from __future__ import annotations

import io
import json
from urllib.error import HTTPError

from app.derivation.llm_client import LLMClient


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTransport:
    def __init__(self, *, reject_primary: bool = False, usage: dict[str, int] | None = None):
        self.calls: list[dict[str, object]] = []
        self.reject_primary = reject_primary
        self.usage = usage

    def __call__(self, req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        self.calls.append(
            {
                "url": req.full_url,
                "model": payload["model"],
                "max_tokens": payload["max_tokens"],
                "headers": dict(req.headers),
            }
        )
        if self.reject_primary and payload["model"] == "gpt-4.1":
            raise HTTPError(
                req.full_url,
                404,
                "not found",
                hdrs=None,
                fp=io.BytesIO(
                    json.dumps({"error": {"message": "The model gpt-4.1 is not available"}}).encode("utf-8")
                ),
            )
        response_payload: dict[str, object] = {
            "choices": [
                {
                    "message": {
                        "content": "fallback ok",
                    }
                }
            ]
        }
        if self.usage is not None:
            response_payload["usage"] = self.usage
        return _FakeHTTPResponse(response_payload)


def test_llm_client_uses_openai_endpoint_and_model():
    transport = _FakeTransport()
    client = LLMClient(
        provider="openai",
        api_key="test-key",
        model="gpt-4.1",
        base_url="https://api.openai.com/v1",
        transport=transport,
        temperature=0.2,
        max_new_tokens=128,
    )

    result = client.technical_reasoning(["SELECT 1;"])

    assert result == "fallback ok"
    assert transport.calls[0]["url"] == "https://api.openai.com/v1/chat/completions"
    assert transport.calls[0]["model"] == "gpt-4.1"
    assert transport.calls[0]["max_tokens"] == 768
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_llm_client_falls_back_when_primary_model_is_blocked():
    transport = _FakeTransport(reject_primary=True)
    client = LLMClient(
        provider="openai",
        api_key="test-key",
        model="gpt-4.1",
        base_url="https://api.openai.com/v1",
        transport=transport,
    )

    result = client.technical_reasoning(["SELECT 1;"])

    assert result == "fallback ok"
    assert [call["model"] for call in transport.calls] == ["gpt-4.1", "gpt-4o-mini"]


def test_llm_client_uses_configured_max_tokens_for_formula_generation():
    transport = _FakeTransport()
    client = LLMClient(
        provider="openai",
        api_key="test-key",
        model="gpt-4.1",
        base_url="https://api.openai.com/v1",
        transport=transport,
        max_new_tokens=512,
    )

    result = client.generate_formula_expression(
        technical_summary="tech",
        business_summary="biz",
        source_sql="select 1",
        function_reference="ref",
        column_name="COL",
        entity_name="ENT",
    )

    assert result == "fallback ok"
    assert transport.calls[0]["max_tokens"] == 512


def test_bedrock_converse_uses_model_and_caps_nova_lite_output():
    class FakeBedrock:
        def __init__(self):
            self.calls = []

        def converse(self, **kwargs):
            self.calls.append(kwargs)
            return {"output": {"message": {"content": [{"text": "  formula "}]}}}

    bedrock = FakeBedrock()
    client = LLMClient(
        provider="bedrock",
        model="bedrock/amazon.nova-lite-v1:0",
        bedrock_client=bedrock,
        max_new_tokens=8192,
    )
    assert client.generate_formula_expression("tech", "biz", "select 1", "ref") == "formula"
    assert bedrock.calls[0]["modelId"] == "amazon.nova-lite-v1:0"
    assert bedrock.calls[0]["inferenceConfig"] == {"maxTokens": 5000, "temperature": 0.0}
    assert bedrock.calls[0]["system"][0]["text"]
    assert bedrock.calls[0]["messages"][0]["role"] == "user"


def test_auto_provider_accepts_bedrock_model():
    client = LLMClient(provider="auto", model="bedrock/amazon.nova-lite-v1:0")
    assert client.provider == "bedrock"


def test_token_usage_is_captured_and_tagged_with_stage_for_openai_style_response():
    transport = _FakeTransport(usage={"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125})
    client = LLMClient(
        provider="openai",
        api_key="test-key",
        model="gpt-4.1",
        base_url="https://api.openai.com/v1",
        transport=transport,
    )

    client.technical_reasoning(["SELECT 1;"])

    usage = client.drain_token_usage()
    assert len(usage) == 1
    assert usage[0] == {
        "stage": "technical_reasoning",
        "provider": "openai",
        "model": "gpt-4.1",
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
    }
    # drain_token_usage() clears the log, so a second drain is empty.
    assert client.drain_token_usage() == []


def test_missing_usage_field_does_not_break_the_response_or_log_anything():
    transport = _FakeTransport()  # no `usage` key in the fake response payload
    client = LLMClient(
        provider="openai",
        api_key="test-key",
        model="gpt-4.1",
        base_url="https://api.openai.com/v1",
        transport=transport,
    )

    result = client.technical_reasoning(["SELECT 1;"])

    assert result == "fallback ok"
    assert client.drain_token_usage() == []


def test_bedrock_token_usage_is_captured_from_converse_response():
    class FakeBedrock:
        def converse(self, **kwargs):
            return {
                "output": {"message": {"content": [{"text": "formula"}]}},
                "usage": {"inputTokens": 40, "outputTokens": 10, "totalTokens": 50},
            }

    client = LLMClient(
        provider="bedrock",
        model="bedrock/amazon.nova-lite-v1:0",
        bedrock_client=FakeBedrock(),
        max_new_tokens=8192,
    )

    client.generate_formula_expression("tech", "biz", "select 1", "ref")

    usage = client.drain_token_usage()
    assert usage == [
        {
            "stage": "dd_generation",
            "provider": "bedrock",
            "model": "amazon.nova-lite-v1:0",
            "prompt_tokens": 40,
            "completion_tokens": 10,
            "total_tokens": 50,
        }
    ]
