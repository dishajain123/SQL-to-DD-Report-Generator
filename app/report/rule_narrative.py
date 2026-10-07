"""Plain-English rule narratives for business-facing Markdown reports.

Deterministic only — no LLM. Uses the 4X grammar explainer plus targeted
templates for high-volume procedure patterns (e.g. S14 NpaType).
"""
from __future__ import annotations

import re

from app.report.condition_explainer import explain_expression

_FORBIDDEN_FALLBACK_PHRASES = (
    "narrative model did not respond",
    "could not be rendered safely in plain English",
)

_THEN_STRING_RE = re.compile(r'THEN\("([^"]*)"\)', re.IGNORECASE)
_COLUMN_SELF_ELSE_RE = re.compile(
    r'ELSE\("(?P<entity>[^"]+)"\."(?P<column>[^"]+)"\)\s*$',
    re.IGNORECASE,
)
_JOIN_TABLE_RE = re.compile(r"\bJOIN\s+([#A-Za-z_][A-Za-z0-9_#]*)", re.IGNORECASE)


def _balanced_close(text: str, open_paren: int) -> int:
    if open_paren < 0 or open_paren >= len(text) or text[open_paren] != "(":
        return -1
    depth = 0
    for i in range(open_paren, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top_level_if_then_else(formula: str) -> tuple[str, str, str] | None:
    """Return (condition, then_body, else_body) for a single top-level IF…THEN…ELSE."""
    s = (formula or "").strip()
    if not s.upper().startswith("IF("):
        return None
    cond_open = 2
    cond_close = _balanced_close(s, cond_open)
    if cond_close < 0:
        return None
    idx = cond_close + 1
    while idx < len(s) and s[idx].isspace():
        idx += 1
    if s[idx : idx + 5].upper() != "THEN(":
        return None
    then_open = idx + 4
    then_close = _balanced_close(s, then_open)
    if then_close < 0:
        return None
    idx = then_close + 1
    while idx < len(s) and s[idx].isspace():
        idx += 1
    if s[idx : idx + 5].upper() != "ELSE(":
        return None
    else_open = idx + 4
    else_close = _balanced_close(s, else_open)
    if else_close < 0 or else_close != len(s) - 1:
        return None
    return (
        s[cond_open + 1 : cond_close],
        s[then_open + 1 : then_close],
        s[else_open + 1 : else_close],
    )


def _explain_condition_snippet(condition: str) -> str:
    wrapped = f"IF({condition})THEN(\"yes\")ELSE(\"no\")"
    explained = explain_expression(wrapped)
    if not explained:
        return "the listed eligibility conditions are met"
    for line in explained.splitlines():
        stripped = line.strip().lstrip("- ").strip()
        if stripped.lower().startswith("if "):
            return stripped[3:].rstrip(":").strip()
        if stripped.lower().startswith("return "):
            continue
        if stripped:
            return stripped.rstrip(".")
    return "the listed eligibility conditions are met"


def _specialized_npa_type_narrative(entity: str, column: str, formula: str) -> str | None:
    col = (column or "").strip().upper()
    if col != "NPATYPE":
        return None
    upper = formula.upper()
    if 'THEN("REGULAR")' not in upper or 'THEN("STICKY")' not in upper or 'THEN("MULTIPLE")' not in upper:
        return None
    entity_label = entity or "AccountCal"
    lines = [
        f"This rule categorizes non-performing loan accounts into one of three NPA types on "
        f"**{entity_label}** based on **Cycle Days (CD)**, **Maximum Days Past Due (DPD_MAX)**, "
        f"and **Asset Class** (Substandard and Doubtful buckets):",
        "",
        "- **REGULAR:** Accounts with CD between 5 and 9 and DPD between 90 and 210+ days across "
        "Substandard (**SUB**) and Doubtful (**DB1**, **DB2**, **DB3**) asset classes.",
        "- **STICKY:** Accounts with CD between 2 and 4 and DPD between 1 and 89 days across the "
        "same asset classes.",
        "- **MULTIPLE:** Accounts with zero DPD (`DPD_MAX = 0`) and low CD (0 or 1) across those "
        "asset classes.",
        "",
        "If the account does not belong to the **VisionPLUS** source system, is not in the **NPA** "
        f"asset-class group, or does not match any branch above, the existing **{column}** value "
        "is left unchanged; otherwise unmatched branches clear the field.",
    ]
    return "\n".join(lines)


def _outcome_literals(formula: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _THEN_STRING_RE.finditer(formula or ""):
        value = match.group(1)
        key = value.upper()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _outcome_bullet_lines(outcomes: list[str], descriptions: dict[str, str] | None = None) -> list[str]:
    lines: list[str] = []
    for label in outcomes:
        detail = (descriptions or {}).get(label.upper(), "when matching business conditions apply.")
        lines.append(f"- **{label}:** {detail}")
    return lines


def _generic_conditional_summary(entity: str, column: str, formula: str) -> str:
    outcomes = _outcome_literals(formula)
    if outcomes:
        lines = [
            f"This rule sets **{column}** on **{entity or 'the target table'}** using these outcomes:",
            "",
            *_outcome_bullet_lines(
                outcomes,
                {o.upper(): "when the matching thresholds and joins in the Platform Condition are satisfied." for o in outcomes},
            ),
        ]
        return "\n".join(lines)
    return (
        f"This rule derives **{column}** on **{entity or 'the target table'}** using conditional "
        "logic from the source procedure. See the Platform Condition for the exact machine-readable rule."
    )


def executive_business_purpose_line(entity: str, column: str, formula: str) -> str:
    """Single-line summary for At a Glance / Business Rule Summary tables."""
    col = (column or "").strip().upper()
    if col == "NPATYPE" and 'THEN("REGULAR")' in (formula or "").upper():
        return (
            "Classifies NPA accounts into REGULAR, STICKY, or MULTIPLE from CD, DPD_MAX, "
            "and asset class (VisionPLUS / NPA scope)."
        )
    outcomes = _outcome_literals(formula or "")
    if outcomes:
        labels = ", ".join(outcomes[:5])
        suffix = f" (+{len(outcomes) - 5} more)" if len(outcomes) > 5 else ""
        return f"Sets {column} on {entity or 'target'} to {labels}{suffix} per conditional rules."
    return f"Derives {column} on {entity or 'target'} from conditional business logic in the source SQL."


def format_stakeholder_markdown(text: str) -> str:
    """Normalize prose for Markdown renderers (Streamlit, GitHub, etc.).

    Uses ``- **Label:**`` list syntax so each outcome renders on its own line.
    Unicode ``•`` bullets are converted to ``-`` because most Markdown parsers
    treat ``•`` as inline text and collapse multiple lines into one paragraph.
    """
    lines_out: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("• **") and ":**" in stripped:
            lines_out.append("- " + stripped[2:].strip())
            continue
        if stripped.startswith("• "):
            lines_out.append("- " + stripped[2:].strip())
            continue
        if stripped.startswith("- "):
            body = stripped[2:].strip()
            if body.startswith("Return "):
                value = body[7:].strip().strip('"').strip("'")
                if value and value.lower() not in {"no value", "null"}:
                    lines_out.append(f"- **{value}:** assigned when the branch condition is met.")
                    continue
        lines_out.append(line)
    return "\n".join(lines_out)


def build_stakeholder_rule_explanation(entity: str, column: str, formula: str) -> str:
    """Multi-line Markdown/plain text for the \"What this rule does\" section."""
    formula = (formula or "").strip()
    specialized = _specialized_npa_type_narrative(entity, column, formula)
    if specialized:
        return specialized

    parts = split_top_level_if_then_else(formula)
    if parts:
        guard, inner, else_body = parts
        self_else = _COLUMN_SELF_ELSE_RE.match(else_body.strip())
        guard_text = _explain_condition_snippet(guard)
        inner_explained = explain_expression(inner) if inner.strip().upper().startswith("IF(") else None
        lines: list[str] = []
        if self_else:
            lines.append(
                f"When {guard_text}, the procedure updates **{column}** using the decision "
                "branches below; otherwise the existing value is kept."
            )
        else:
            lines.append(f"When {guard_text}, the procedure applies the following logic:")
        lines.append("")
        if inner_explained:
            lines.extend(format_stakeholder_markdown(inner_explained).splitlines())
        else:
            lines.append(_generic_conditional_summary(entity, column, inner or formula))
        return format_stakeholder_markdown("\n".join(lines))

    explained = explain_expression(formula)
    if explained:
        intro = (
            f"This rule determines **{column}** on **{entity or 'the target table'}** as follows:"
        )
        return format_stakeholder_markdown(f"{intro}\n\n{explained}")

    return _generic_conditional_summary(entity, column, formula)


def brief_join_summary(join_conditions: list[str]) -> str:
    tables: list[str] = []
    blob = " ".join(join_conditions or [])
    for match in _JOIN_TABLE_RE.finditer(blob):
        name = match.group(1).lstrip("#")
        if name and name.upper() not in {t.upper() for t in tables}:
            tables.append(name)
    if not tables:
        return "—"
    scd = "@TIMEKEY" in blob.upper() or "EFFECTIVEFROMTIMEKEY" in blob.upper()
    joined = ", ".join(tables)
    return f"{joined} (SCD effective dates)" if scd else joined


def brief_formula_summary(expression: str, *, column_name: str = "") -> str:
    """Short label for execution-sequence table cells (not the full 4X formula)."""
    expr = (expression or "").strip()
    if not expr:
        return "—"

    col = (column_name or "").strip().upper()
    if col == "NPATYPE":
        return (
            "REGULAR / STICKY / MULTIPLE from CD, DPD_MAX, and asset class; else NULL in scope"
        )

    outcomes = _outcome_literals(expr)
    if outcomes:
        if len(outcomes) <= 5:
            return f"Assign: {', '.join(outcomes)} (conditional)"
        return f"Assign one of {len(outcomes)} coded outcomes (conditional)"

    if _COLUMN_SELF_ELSE_RE.search(expr):
        return "Conditional value; else keep existing column"

    if expr.upper().startswith("IF("):
        return "Conditional assignment (see Platform Condition)"

    one_line = " ".join(expr.split())
    if len(one_line) <= 160:
        return one_line
    return "Derived value (see Platform Condition)"


def brief_row_condition_summary(condition: str) -> str:
    text = (condition or "").strip()
    if not text:
        return "All rows in scope"
    upper = text.upper()
    tags: list[str] = []
    if "VISIONPLUS" in upper:
        tags.append("VisionPLUS source")
    if "ASSETCLASSGROUP" in upper and "NPA" in upper:
        tags.append("NPA asset-class group")
    if "FLGPROCESSING" in upper:
        tags.append("processing flag")
    if "BALANCE" in upper and ">" in text:
        tags.append("positive balance")
    if "FINALASSETCLASSALT_KEY" in upper:
        compact = text.replace(" ", "")
        if ">1" in compact:
            tags.append("FinalAssetClassAlt_Key > 1")
        elif "==1" in compact:
            tags.append("FinalAssetClassAlt_Key = 1")
    if tags:
        return "; ".join(tags)
    explained = _explain_condition_snippet(text)
    if explained and len(explained) <= 200:
        return explained
    return "Row filters per Platform Condition"


def assert_no_forbidden_report_phrases(text: str) -> None:
    lower = (text or "").lower()
    for phrase in _FORBIDDEN_FALLBACK_PHRASES:
        if phrase in lower:
            raise AssertionError(f"Report contains forbidden placeholder phrase: {phrase}")
