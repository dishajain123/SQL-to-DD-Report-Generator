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
    def __init__(self, *, reject_primary: bool = False):
        self.calls: list[dict[str, object]] = []
        self.reject_primary = reject_primary

    def __call__(self, req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        self.calls.append(
            {
                "url": req.full_url,
                "model": payload["model"],
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
        return _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "fallback ok",
                        }
                    }
                ]
            }
        )


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
