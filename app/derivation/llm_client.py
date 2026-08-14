"""Architecture steps 11 (AI Understanding) and 13 (DD Generation).

This client now uses Groq's OpenAI-compatible chat completions API by default
when `GROQ_API_KEY` is set. Prompt text lives in separate YAML assets so each
stage can be updated without editing Python code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from groq import APIConnectionError, APIStatusError, BadRequestError, Groq, NotFoundError, PermissionDeniedError, RateLimitError

import yaml

from app.utils.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).with_name("prompts")


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


def _parse_fallback_models(raw: str) -> list[str]:
    models = [item.strip() for item in raw.split(",")]
    return [model for model in models if model]


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


@dataclass
class LLMClient:
    """Thin wrapper around Groq's chat completions API."""

    api_key: str = settings.groq_api_key
    model: str = settings.groq_model
    model_fallbacks: str = settings.groq_model_fallbacks
    client: Groq | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = Groq(api_key=self.api_key) if self.api_key else None

    def _candidate_models(self) -> list[str]:
        candidates = [self.model, *_parse_fallback_models(self.model_fallbacks), "llama-3.1-8b-instant"]
        seen: set[str] = set()
        ordered: list[str] = []
        for model in candidates:
            if model and model not in seen:
                seen.add(model)
                ordered.append(model)
        return ordered

    def _is_retryable_model_error(self, exc: Exception) -> bool:
        if isinstance(exc, (PermissionDeniedError, NotFoundError)):
            return True
        if isinstance(exc, BadRequestError):
            text = str(exc).lower()
            return "model" in text or "permission" in text or "not found" in text
        if isinstance(exc, APIStatusError):
            return getattr(exc, "status_code", None) in {403, 404}
        return False

    def _complete_with_model(self, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set — cannot call the LLM. "
                "Set it in your environment or .env file."
            )
        if self.client is None:
            raise RuntimeError("Groq client is not initialized. Check GROQ_API_KEY.")

        last_error: Exception | None = None
        for candidate in self._candidate_models():
            try:
                response = self.client.chat.completions.create(
                    model=candidate,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("Groq returned an empty message content")
                return content.strip()
            except APIConnectionError as exc:
                raise RuntimeError(f"Could not reach Groq API: {exc}") from exc
            except RateLimitError as exc:
                raise RuntimeError(f"Groq rate limit hit while using model '{candidate}': {exc}") from exc
            except APIStatusError as exc:
                last_error = exc
                if not self._is_retryable_model_error(exc):
                    raise RuntimeError(f"Groq chat completion failed for model '{candidate}': {exc}") from exc
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_model_error(exc):
                    raise RuntimeError(f"Groq chat completion failed for model '{candidate}': {exc}") from exc

        raise RuntimeError(
            "Groq rejected the configured model(s): "
            f"{', '.join(self._candidate_models())}. "
            "This usually means the API key does not have access to the primary model. "
            f"Last error: {last_error}"
        )

    def _complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return self._complete_with_model(self.model, system, user, max_tokens=max_tokens)

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
        self, technical_summary: str, business_summary: str, source_sql: str, function_reference: str
    ) -> str:
        prompts = _load_prompts()
        user = _render_prompt(
            prompts["dd_generation_user"],
            technical_summary=technical_summary,
            business_summary=business_summary,
            source_sql=source_sql,
            function_reference=function_reference,
        )
        return self._complete(prompts["dd_generation_system"], user, max_tokens=512)

    def retry_with_error(self, previous_expression: str, error: str, context: str) -> str:
        prompts = _load_prompts()
        user = _render_prompt(
            prompts["retry_with_error_user"],
            previous_expression=previous_expression,
            error=error,
            context=context,
        )
        return self._complete(prompts["retry_with_error_system"], user, max_tokens=512)
