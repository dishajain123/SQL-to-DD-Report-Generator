from __future__ import annotations

from groq import PermissionDeniedError

from app.derivation.llm_client import LLMClient


class _FakeResponse:
    status_code = 403
    headers = {}
    request = None

    def __init__(self, text: str = ""):
        self.text = text


class _FakeGroqClient:
    def __init__(self):
        self.calls: list[str] = []
        self.chat = self
        self.completions = self

    def create(self, *, model, messages, max_tokens, temperature):
        self.calls.append(model)
        if model == "llama-3.3-70b-versatile":
            raise PermissionDeniedError("blocked", response=_FakeResponse("blocked"), body={"error": "blocked"})

        return type(
            "FakeCompletion",
            (),
            {
                "choices": [
                    type(
                        "FakeChoice",
                        (),
                        {
                            "message": type("FakeMessage", (), {"content": "fallback ok"})(),
                        },
                    )()
                ]
            },
        )()


def test_llm_client_falls_back_when_primary_model_is_blocked(monkeypatch):
    fake_client = _FakeGroqClient()
    monkeypatch.setattr("app.derivation.llm_client.Groq", lambda api_key: fake_client)

    client = LLMClient(
        api_key="test-key",
        model="llama-3.3-70b-versatile",
        model_fallbacks="llama-3.1-8b-instant",
    )

    result = client.technical_reasoning(["SELECT 1;"])

    assert result == "fallback ok"
    assert fake_client.calls == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
