"""Prepare SQL from uploads, bundled samples, or pasted text for job intake."""

from pathlib import Path
from typing import Iterable, Protocol

from app.guardrails.input_guardrails import check_input_file
from app.utils.text_encoding import decode_text_bytes


SAMPLES_SQL_DIR = Path(__file__).resolve().parents[2] / "samples" / "sql"
PASTED_SQL_NAME = "pasted_procedure.sql"


class SQLUpload(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def bundled_sample_names(sample_dir: Path = SAMPLES_SQL_DIR) -> list[str]:
    if not sample_dir.is_dir():
        return []
    return sorted(
        path.name for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".sql"
    )


def _validate_sql(filename: str, content: str) -> None:
    result = check_input_file(filename, content)
    if not result.passed:
        raise ValueError(f"{filename}: {'; '.join(result.errors)}")


def uploaded_sql_files(uploads: Iterable[SQLUpload]) -> dict[str, str]:
    files: dict[str, str] = {}
    for uploaded in uploads:
        filename = uploaded.name
        if filename in files:
            raise ValueError(f"Duplicate uploaded filename: {filename}")
        try:
            content = decode_text_bytes(uploaded.getvalue()).text
        except UnicodeDecodeError as exc:
            raise ValueError(f"{filename} could not be read as SQL text") from exc
        _validate_sql(filename, content)
        files[filename] = content
    return files


def bundled_sql_file(filename: str, sample_dir: Path = SAMPLES_SQL_DIR) -> dict[str, str]:
    if filename not in bundled_sample_names(sample_dir):
        raise ValueError("Select a SQL file from the bundled sample list.")
    try:
        content = decode_text_bytes((sample_dir / filename).read_bytes()).text
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read bundled sample {filename}: {exc}") from exc
    _validate_sql(filename, content)
    return {filename: content}


def pasted_sql_file(content: str) -> dict[str, str]:
    _validate_sql(PASTED_SQL_NAME, content)
    return {PASTED_SQL_NAME: content}
