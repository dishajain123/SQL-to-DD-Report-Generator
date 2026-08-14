"""Architecture step 13d: validate a generated Formula Expression string
against the platform's real grammar before it's accepted into the DD Model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lark import Lark, UnexpectedInput

_GRAMMAR_PATH = Path(__file__).parent / "fourx_grammar.lark"

_parser = Lark(_GRAMMAR_PATH.read_text(), parser="earley", start="start")


@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None


# Function names the grammar accepts syntactically as FUNC_NAME, cross-checked
# here against the actual documented library so an expression using a made-up
# function name (syntactically valid, semantically wrong) is still rejected.
KNOWN_FUNCTIONS = {
    "ISEMPTY", "ISNOTEMPTY", "MAX", "COALESCE",
    "SUBSTR", "LOWER", "UPPER", "LEN", "CONVERT", "REGEX", "CONCAT", "TRIM", "REPLACE",
    "SOM", "EOM", "SOY", "EOY", "SOFY", "EOFY", "DATEPART", "DATEDIFF", "TODATE",
    "ADDDAY", "PERIOD", "SOQ", "EOQ",
    "ROUND", "ABS", "FLOOR", "CEIL",
}


def validate_expression(expression: str) -> ValidationResult:
    try:
        tree = _parser.parse(expression)
    except UnexpectedInput as exc:
        return ValidationResult(valid=False, error=str(exc))
    except Exception as exc:  # any other Lark/grammar error
        return ValidationResult(valid=False, error=f"Grammar error: {exc}")

    unknown = _find_unknown_functions(tree)
    if unknown:
        return ValidationResult(
            valid=False,
            error=f"Unknown function(s) not in the 4X library: {', '.join(sorted(unknown))}",
        )
    return ValidationResult(valid=True)


def _find_unknown_functions(tree) -> set[str]:
    unknown = set()
    for node in tree.find_data("function_call"):
        func_name = str(node.children[0])
        if func_name.upper() not in KNOWN_FUNCTIONS:
            unknown.add(func_name)
    return unknown
