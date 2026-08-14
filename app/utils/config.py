"""Central configuration, loaded from environment variables / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    default_company_name: str = os.getenv("DEFAULT_COMPANY_NAME", "Acme Bank")
    default_platform_name: str = os.getenv("DEFAULT_PLATFORM_NAME", "4X")
    default_intent: str = os.getenv("DEFAULT_INTENT", "Generate DD")
    default_function_reference_path: str = os.getenv(
        "DEFAULT_FUNCTION_REFERENCE_PATH", "samples/platform_docs/4x_functions_operators.md"
    )
    default_entity_name_map_json: str = os.getenv("DEFAULT_ENTITY_NAME_MAP_JSON", "{}")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_model_fallbacks: str = os.getenv("GROQ_MODEL_FALLBACKS", "llama-3.1-8b-instant,openai/gpt-oss-20b")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", "dd_automation.db")
    output_dir: str = os.getenv("OUTPUT_DIR", "output")
    structural_confidence_threshold: float = float(
        os.getenv("STRUCTURAL_CONFIDENCE_THRESHOLD", "0.5")
    )
    output_guardrail_confidence_threshold: float = float(
        os.getenv("OUTPUT_GUARDRAIL_CONFIDENCE_THRESHOLD", "0.7")
    )


settings = Settings()
