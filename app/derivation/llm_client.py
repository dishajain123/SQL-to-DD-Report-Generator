"""Architecture steps 11 (AI Understanding) and 13 (DD Generation).

This client is provider-agnostic: choose the backend from environment
variables only, then keep calling ``LLMClient()`` everywhere else.

Supported providers:
- OpenAI via `LLM_PROVIDER=openai`
- Groq via `LLM_PROVIDER=groq`

If `LLM_PROVIDER=auto` or unset, the client tries to infer the provider from
`LLM_MODEL_NAME` or `LLM_BASE_URL`. The rest of the app never needs to know
which provider is being used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional
from urllib import error, request

import yaml

from app.utils.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).with_name("prompts")


class _ModelRejectedError(RuntimeError):
    """Raised when a provider rejects a candidate model and fallback is okay."""


@lru_cache(maxsize=1)
def _load_prompts() -> dict[str, str]:
    prompt_specs = {
        "technical_reasoning": _PROMPTS_DIR / "technical_reasoning.yaml",
        "business_reasoning": _PROMPTS_DIR / "business_reasoning.yaml",
        "dd_generation": _PROMPTS_DIR / "dd_generation.yaml",
        "retry_with_error": _PROMPTS_DIR / "retry_with_error.yaml",
    }

    prompts: dict[str, str] = {}
    for prefix, path in prompt_specs.items():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        if not isinstance(loaded, dict):
            raise RuntimeError(f"Prompt file {path} must contain a YAML mapping")

        required_keys = {"system", "user"}
        missing = sorted(required_keys - loaded.keys())
        if missing:
            raise RuntimeError(f"Prompt file {path} is missing keys: {', '.join(missing)}")

        prompts[f"{prefix}_system"] = str(loaded["system"])
        prompts[f"{prefix}_user"] = str(loaded["user"])

    return prompts


def _render_prompt(template: str, **kwargs: object) -> str:
    return template.format(**kwargs)


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped.strip("`").strip()

    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return stripped.strip("`").strip()


def _parse_json_payload(raw_output: str) -> dict[str, object]:
    parsed = json.loads(_strip_markdown_fences(raw_output))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def _normalize_provider(raw_provider: str, model: str, base_url: str) -> str:
    provider = raw_provider.strip().lower()
    if provider in {"", "auto"}:
        model_hint = model.strip().lower()
        base_url_hint = base_url.strip().lower()
        if model_hint.startswith("gpt-") or "openai" in base_url_hint:
            return "openai"
        if model_hint.startswith("llama") or "groq" in base_url_hint:
            return "groq"
        return "openai"
    if provider not in {"openai", "groq"}:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{raw_provider}'. Expected 'auto', 'openai', or 'groq'."
        )
    return provider


def _provider_from_model(model: str) -> Optional[str]:
    lowered = model.strip().lower()
    if lowered.startswith("gpt-") or lowered.startswith("openai/"):
        return "openai"
    if lowered.startswith("llama") or lowered.startswith("groq/"):
        return "groq"
    return None


def _provider_from_api_key(api_key: str) -> Optional[str]:
    lowered = api_key.strip().lower()
    if lowered.startswith("gsk_"):
        return "groq"
    if lowered.startswith("sk-proj-") or lowered.startswith("sk-"):
        return "openai"
    return None


def _extract_error_message(raw_body: str, fallback: str) -> str:
    if not raw_body.strip():
        return fallback

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body.strip() or fallback

    if isinstance(parsed, dict):
        error_obj = parsed.get("error")
        if isinstance(error_obj, dict):
            for key in ("message", "type", "code"):
                value = error_obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(error_obj, str) and error_obj.strip():
            return error_obj.strip()

        detail = parsed.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()

    return raw_body.strip() or fallback


def _looks_like_rejected_model(status_code: Optional[int], message: str) -> bool:
    if status_code not in {400, 401, 403, 404}:
        return False

    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "model",
            "permission",
            "not found",
            "does not exist",
            "unsupported",
            "invalid model",
        )
    )


def _truncate_for_provider(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated to fit provider input limit]"


@dataclass
class LLMClient:
    """Provider-agnostic chat-completions client for the pipeline."""

    provider: str = settings.llm_provider
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_new_tokens: int = 1024
    max_input_chars: int = 12000
    transport: Optional[Callable[..., Any]] = None

    def __post_init__(self) -> None:
        self.model = self.model or settings.llm_model_name.strip()
        self.base_url = self.base_url or settings.llm_base_url.strip()
        self.api_key = self.api_key or settings.llm_api_key.strip()
        inferred_provider = _normalize_provider(self.provider, self.model, self.base_url)
        model_provider = _provider_from_model(self.model)
        key_provider = _provider_from_api_key(self.api_key)

        if self.provider in {"", "auto"}:
            if model_provider and key_provider and model_provider != key_provider:
                raise ValueError(
                    "LLM config mismatch: LLM_MODEL_NAME looks like "
                    f"{model_provider} but LLM_API_KEY looks like {key_provider}. "
                    "Make the model and API key belong to the same provider."
                )
            if key_provider:
                inferred_provider = key_provider
            elif model_provider:
                inferred_provider = model_provider

        self.provider = inferred_provider
        self.base_url = self.base_url or self._default_base_url()
        self.model = self.model or self._default_model()
        if self.transport is None:
            self.transport = request.urlopen

    def _default_base_url(self) -> str:
        if self.provider == "openai":
            return "https://api.openai.com/v1"
        return "https://api.groq.com/openai/v1"

    def _default_model(self) -> str:
        if self.provider == "openai":
            return "gpt-4.1"
        return "llama-3.3-70b-versatile"

    def _candidate_models(self) -> list[str]:
        fallback_models = ["gpt-4o-mini"] if self.provider == "openai" else ["llama-3.1-8b-instant", "openai/gpt-oss-20b"]
        candidates = [self.model, *fallback_models]
        seen: set[str] = set()
        ordered: list[str] = []
        for model in candidates:
            if model and model not in seen:
                seen.add(model)
                ordered.append(model)
        return ordered

    def _chat_completions_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if not base_url:
            raise RuntimeError(
                f"No base URL is configured for provider '{self.provider}'. "
                "Set LLM_BASE_URL or the provider-specific base URL in .env."
            )
        return base_url + "/chat/completions"

    def _complete_once(self, model: str, system: str, user: str, max_tokens: int) -> str:
        if not self.api_key:
            raise RuntimeError(
                "No API key is configured for the LLM provider. "
                "Set LLM_API_KEY in your .env file."
            )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": _truncate_for_provider(user, self.max_input_chars)},
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._chat_completions_url(),
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with self.transport(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8") if exc.fp else ""
            message = _extract_error_message(raw, exc.reason if isinstance(exc.reason, str) else str(exc.reason))
            if exc.code == 1010 or "1010" in message:
                raise RuntimeError(
                    f"{self.provider} returned Cloudflare error 1010 while calling model '{model}'. "
                    "This is an upstream access block, not a prompt or SQL bug. "
                    "If you want to continue without this blocker, switch .env to a provider/model "
                    "your account can reach (for example LLM_PROVIDER=openai with gpt-4.1), "
                    "or use a Groq account/network that is allowed to access the API."
                ) from exc
            if _looks_like_rejected_model(exc.code, message):
                raise _ModelRejectedError(message) from exc
            raise RuntimeError(
                f"{self.provider} chat completion failed for model '{model}': {message}"
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", str(exc))
            raise RuntimeError(f"Could not reach the {self.provider} API: {reason}") from exc

        parsed = _parse_json_payload(raw)
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"{self.provider} returned an unexpected response: missing choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError(f"{self.provider} returned an unexpected response: invalid choice")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"{self.provider} returned an unexpected response: missing message")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"{self.provider} returned an empty message content")
        return content.strip()

    def _complete_with_model(self, system: str, user: str, max_tokens: int = 1024) -> str:
        candidates = self._candidate_models()
        if not candidates:
            raise RuntimeError(
                f"No model is configured for provider '{self.provider}'. "
                "Set LLM_MODEL_NAME or the provider-specific model variable in .env."
            )

        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                return self._complete_once(candidate, system, user, max_tokens=max_tokens)
            except _ModelRejectedError as exc:
                last_error = exc
                continue

        raise RuntimeError(
            f"{self.provider} rejected the configured model(s): {', '.join(candidates)}. "
            f"Last error: {last_error}"
        )

    def _complete(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        return self._complete_with_model(system, user, max_tokens=max_tokens or self.max_new_tokens)

    def technical_reasoning(self, sql_snippets: list[str]) -> str:
        prompts = _load_prompts()
        user = _render_prompt(
            prompts["technical_reasoning_user"],
            sql_snippets="\n\n---\n\n".join(sql_snippets),
        )
        return self._complete(prompts["technical_reasoning_system"], user)

    def business_reasoning(self, technical_summary: str) -> str:
        prompts = _load_prompts()
        user = _render_prompt(prompts["business_reasoning_user"], technical_summary=technical_summary)
        return self._complete(prompts["business_reasoning_system"], user)

    def generate_formula_expression(
        self,
        technical_summary: str,
        business_summary: str,
        source_sql: str,
        function_reference: str,
        column_name: str = "",
        entity_name: str = "",
        relevant_sql: str = "",
        rag_context: str = "",
    ) -> str:
        prompts = _load_prompts()
        user = _render_prompt(
            prompts["dd_generation_user"],
            technical_summary=technical_summary,
            business_summary=business_summary,
            source_sql=source_sql,
            function_reference=function_reference,
            column_name=column_name,
            entity_name=entity_name,
            relevant_sql=relevant_sql.strip() or (
                "(No specific assignment statements were isolated automatically "
                "-- search the full source SQL below for every place this "
                "column is assigned.)"
            ),
            rag_context=rag_context.strip() or (
                "(No specific reference material was retrieved for this "
                "column -- rely on the full platform reference below.)"
            ),
        )
        return self._complete(prompts["dd_generation_system"], user, max_tokens=min(self.max_new_tokens, 512))

    def retry_with_error(self, previous_expression: str, error: str, context: str) -> str:
        prompts = _load_prompts()
        user = _render_prompt(
            prompts["retry_with_error_user"],
            previous_expression=previous_expression,
            error=error,
            context=context,
        )
        return self._complete(prompts["retry_with_error_system"], user, max_tokens=min(self.max_new_tokens, 512))