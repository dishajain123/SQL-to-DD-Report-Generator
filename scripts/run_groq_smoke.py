#!/usr/bin/env python3
"""Run the Groq-backed reasoning steps end to end and print each output.

This is a lightweight verification script for:
1. Environment setup
2. Technical reasoning
3. Business reasoning
4. DD formula generation
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.derivation.llm_client import LLMClient
from app.utils.config import settings


DEFAULT_SQL_FILES = [
    Path("samples/sql/PRO_DPD_Calculation_StoredProcedure_2.sql"),
    Path("samples/sql/PRO_MaxDPD_ReferencePeriod_Calculation_StoredProcedure.sql"),
    Path("samples/sql/PRO_NPA_Date_Calculation_StoredProcedure_1.sql"),
]
DEFAULT_FUNCTION_REFERENCE = Path("samples/platform_docs/4x_functions_operators.md")


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def _print_section(title: str, content: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    print(content.strip())
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Groq-backed reasoning steps and print each output."
    )
    parser.add_argument(
        "--sql",
        dest="sql_files",
        action="append",
        type=Path,
        help="Path to a SQL file. Repeat to pass multiple files. Defaults to sample procs.",
    )
    parser.add_argument(
        "--function-reference",
        type=Path,
        default=DEFAULT_FUNCTION_REFERENCE,
        help="Path to the platform function/operator reference markdown file.",
    )
    args = parser.parse_args()

    sql_files = args.sql_files or DEFAULT_SQL_FILES
    sql_snippets = [_read_text(path) for path in sql_files]
    function_reference = _read_text(args.function_reference)

    print("Groq smoke test configuration")
    print(f"- GROQ_API_KEY set: {'yes' if settings.groq_api_key else 'no'}")
    print(f"- GROQ_MODEL: {settings.groq_model}")
    print(f"- SQL files: {', '.join(str(path) for path in sql_files)}")
    print(f"- Function reference: {args.function_reference}")

    if not settings.groq_api_key:
        print(
            "\nGROQ_API_KEY is missing. Add it to your .env file, then rerun this script."
        )
        return 1

    client = LLMClient()

    try:
        technical = client.technical_reasoning(sql_snippets)
        _print_section("STEP 1: Technical reasoning", technical)

        business = client.business_reasoning(technical)
        _print_section("STEP 2: Business reasoning", business)

        formula = client.generate_formula_expression(
            technical_summary=technical,
            business_summary=business,
            source_sql=sql_snippets[0],
            function_reference=function_reference,
        )
        _print_section("STEP 3: DD formula expression", formula)
    except RuntimeError as exc:
        print(f"\nGroq request failed: {exc}")
        print(
            "If you are running inside a restricted environment, the network may be blocked. "
            "Run this on your local machine with internet access to see the live Groq output."
        )
        return 1

    print("Groq smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
