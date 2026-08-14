"""Architecture step 2: Input Guardrails."""
from __future__ import annotations

from dataclasses import dataclass, field

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = (".sql",)

# Extremely blunt heuristic to catch prompt-injection-style content smuggled
# inside an uploaded "SQL" file — SQL files should not contain these.
_SUSPICIOUS_PATTERNS = (
    "ignore previous instructions",
    "disregard all prior",
    "system prompt",
)


@dataclass
class GuardrailResult:
    passed: bool
    errors: list[str] = field(default_factory=list)


def check_input_file(filename: str, content: str) -> GuardrailResult:
    errors = []

    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        errors.append(f"Unsupported file type: {filename}")

    size = len(content.encode("utf-8"))
    if size > MAX_FILE_SIZE_BYTES:
        errors.append(f"File exceeds max size ({size} > {MAX_FILE_SIZE_BYTES} bytes)")

    if not content.strip():
        errors.append("File is empty")

    lowered = content.lower()
    for pattern in _SUSPICIOUS_PATTERNS:
        if pattern in lowered:
            errors.append(f"Suspicious content detected: '{pattern}'")

    return GuardrailResult(passed=not errors, errors=errors)


def check_job_plan(company: str, platform: str) -> GuardrailResult:
    errors = []
    if not company or not company.strip():
        errors.append("Company is required")
    if not platform or not platform.strip():
        errors.append("Platform is required")
    return GuardrailResult(passed=not errors, errors=errors)
