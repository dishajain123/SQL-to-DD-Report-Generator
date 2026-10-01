"""Batch-run the DD pipeline over a directory (or .zip) of SQL procedures.

    python -m pipeline --input-dir PRO_SPs/ --output-dir dist/
    python -m pipeline --input-dir PRO_SPs.zip --output-dir dist/

Each file becomes one job, exactly as if it were submitted through the UI.
Files are decoded with the shared encoding detector (UTF-8 / UTF-8 BOM /
UTF-16 LE+BE with or without BOM / cp1252), so SSMS "Unicode" exports work.
A summary of every file — encoding, rows, genuine parse failures, errors — is
written to ``<output-dir>/batch_summary.md`` and ``batch_summary.json``.

The derivation path is deterministic (no model calls), so a batch run needs
no API key. Jobs use a private SQLite database under the output directory so
batch runs never mix with UI jobs.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterator

SQL_SUFFIXES = {".sql", ".prc", ".pks", ".pkb"}


@dataclass
class FileResult:
    file: str
    encoding: str = ""
    job_id: str = ""
    output_dir: str = ""
    rows: int = 0
    active: int = 0
    parse_failures: list[str] = field(default_factory=list)
    error: str = ""


def _iter_sources(input_path: Path) -> Iterator[tuple[str, bytes]]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            for member in sorted(archive.namelist()):
                if Path(member).suffix.lower() in SQL_SUFFIXES and not member.endswith("/"):
                    yield member, archive.read(member)
        return
    if input_path.is_file():
        yield input_path.name, input_path.read_bytes()
        return
    for path in sorted(p for p in input_path.rglob("*") if p.suffix.lower() in SQL_SUFFIXES):
        yield str(path.relative_to(input_path)), path.read_bytes()


def _configure_storage(output_dir: Path) -> None:
    from app.utils import db

    output_dir.mkdir(parents=True, exist_ok=True)
    db.settings = replace(
        db.settings,
        output_dir=str(output_dir),
        sqlite_db_path=str(output_dir / "batch_jobs.db"),
    )
    db.init_db()


def run_file(name: str, data: bytes) -> FileResult:
    from app.models.core import Intent, JobPlan
    from app.orchestration.pipeline import build_pipeline
    from app.utils import db
    from app.utils.text_encoding import decode_text_bytes

    result = FileResult(file=name)
    try:
        decoded = decode_text_bytes(data)
    except UnicodeDecodeError as exc:
        result.error = f"could not decode file: {exc}"
        return result
    result.encoding = decoded.encoding

    job_id = f"job-{uuid.uuid4().hex[:10]}"
    result.job_id = job_id
    db.record_job(job_id, "Batch", "4X", Intent.GENERATE_DD.value, "RUNNING")
    # Keep the original bytes as well as the pipeline's decoded SQL. No
    # encoding repair or whitespace normalization may overwrite the upload.
    source_dir = Path(db.get_job_output_dir(job_id)) / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "original.sql").write_bytes(data)
    try:
        state = build_pipeline().invoke(
            {
                "job_plan": JobPlan(job_id=job_id, intent=Intent.GENERATE_DD,
                                    company="Batch", platform="4X"),
                "uploaded_files": {Path(name).name: decoded.text},
                "function_reference": "",
                "entity_name_map": {},
                "timekey_map": {},
            }
        )
    except Exception as exc:  # one bad file must not stop the batch
        db.update_job_status(job_id, "FAILED")
        result.error = f"{type(exc).__name__}: {exc}"
        result.output_dir = str(db.get_job_output_dir(job_id))
        (Path(result.output_dir)).mkdir(parents=True, exist_ok=True)
        (Path(result.output_dir) / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return result

    result.output_dir = str(db.get_job_output_dir(job_id))
    rows = state.get("dd_rows") or []
    result.rows = len(rows)
    result.active = sum(1 for r in rows if r.status.value == "ACTIVE")
    for info in (state.get("structural_infos") or {}).values():
        result.parse_failures.extend(info.parse_failures)
    return result


def _write_summary(results: list[FileResult], output_dir: Path) -> Path:
    (output_dir / "batch_summary.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    failed = [r for r in results if r.error]
    unparsed = [r for r in results if r.parse_failures]
    lines = [
        "# Batch run summary",
        "",
        f"Files: {len(results)} · succeeded: {len(results) - len(failed)} · "
        f"failed: {len(failed)} · files with genuine parse failures: {len(unparsed)}",
        "",
        "| File | Encoding | Rows | Active | Parse failures | Error | Output |",
        "|------|----------|------|--------|----------------|-------|--------|",
    ]
    for r in results:
        cell = lambda text: " ".join(str(text).split()).replace("|", "\\|")
        lines.append(
            f"| {cell(r.file)} | {r.encoding or '—'} | {r.rows} | {r.active} | "
            f"{len(r.parse_failures)} | {cell(r.error) or '—'} | {cell(r.output_dir) or '—'} |"
        )
    if unparsed:
        lines.extend(["", "## Genuine parse failures", ""])
        for r in unparsed:
            for failure in r.parse_failures:
                lines.append(f"- `{r.file}` {failure}")
    path = output_dir / "batch_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline", description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="directory of SQL files (searched recursively), a .zip, or one file")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.input_dir.exists():
        parser.error(f"input not found: {args.input_dir}")
    output_dir = args.output_dir.resolve()
    _configure_storage(output_dir)

    results: list[FileResult] = []
    sources = list(_iter_sources(args.input_dir))
    for index, (name, data) in enumerate(sources, 1):
        print(f"[{index}/{len(sources)}] {name}", flush=True)
        result = run_file(name, data)
        status = f"ERROR {result.error}" if result.error else (
            f"{result.encoding} rows={result.rows} active={result.active} "
            f"parse_failures={len(result.parse_failures)}"
        )
        print(f"    {status}", flush=True)
        results.append(result)

    summary = _write_summary(results, output_dir)
    print(f"\nSummary: {summary}")
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
