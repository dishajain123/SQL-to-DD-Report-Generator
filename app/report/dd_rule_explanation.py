"""Plain-English Rule Explanation cells for DD Excel/CSV export.

Deterministic AST walk via :func:`app.report.condition_explainer.explain_expression`
— no LLM. Output is capped at one or two sentences for spreadsheet review.
"""
from __future__ import annotations

import re

from app.report.condition_explainer import (
    build_dynamic_rule_sentence,
    explain_expression,
    explain_expression_for_export,
    format_explainer_lines_as_branches,
    sanitize_rule_explanation_text,
    try_entity_scoped_rule_explanation,
)
from app.report.rule_narrative import executive_business_purpose_line

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_DEFAULT_DATE_MARKERS = ("1900-01-01", "01/01/1900")
_SCHEME_PRODUCT_MARKERS = ("DIMPRODUCT", "AQUA_SCHEME", "SCHEMETYPE")
_ISEMPTY_THEN_RE = re.compile(
    r'ISEMPTY\("([^"]+)"\."([^"]+)"\)\)THEN\("([^"]+)"\."([^"]+)"\)',
    re.IGNORECASE,
)
_COALESCE_PAIR_RE = re.compile(
    r'COALESCE\("([^"]+)"\."([^"]+)"\s*,\s*"([^"]+)"\."([^"]+)"\)',
    re.IGNORECASE,
)
_STANDALONE_BATCH_PARAMS = frozenset(
    {
        "@timekey",
        "@processingdate",
        "@processdate",
        "@histtimekey",
    }
)
_QUALIFIED_COL_RE = re.compile(r'"([^"]+)"\."([^"]+)"', re.IGNORECASE)
_THEN_QUALIFIED_RE = re.compile(
    r'THEN\("([^"]+)"\."([^"]+)"\)',
    re.IGNORECASE,
)
_IDENTITY_COLUMN_SUFFIXES = (
    "ACCOUNTENTITYID",
    "CUSTOMERENTITYID",
    "CUSTOMERACID",
    "UCIFENTITYID",
    "PRODUCTALT_KEY",
    "SOURCEALT_KEY",
)
_DUMMY_ID_MARKERS = ("FORMO", "FORPM", "FORMF", "AAAAA1111A", "DUMMY", "INVALID")
_DATE_RESET_COLUMNS = frozenset(
    {
        "SRCNPA_DT",
        "DBTDT",
        "NPADT",
        "NPADATE",
        "SYSNPA_DT",
        "FINALNPADT",
        "INITIALNPADT",
    }
)


def build_rule_explanation(
    entity: str,
    column: str,
    expression: str,
    *,
    business_meaning: str = "",
) -> str:
    """Concise business summary for the ``Rule Explanation`` export column."""
    formula = (expression or "").strip()
    if not formula:
        return ""

    batch = _batch_parameter_explanation(formula)
    if batch:
        return _finalize_rule_explanation(batch)

    proc = _running_process_name_literal(formula)
    if proc and (entity or "").strip().upper().lstrip("#") == "ACLRUNNINGPROCESSSTATUS":
        acl = _acl_process_status_explanation(column, proc)
        if acl:
            return _finalize_rule_explanation(acl)

    entity_scoped = try_entity_scoped_rule_explanation(
        formula, target_column=column, target_entity=entity
    )
    if entity_scoped:
        return _limit_export_text(entity_scoped)

    try:
        dynamic = build_dynamic_rule_sentence(
            formula,
            target_column=column,
            target_entity=entity,
        )
    except Exception:
        dynamic = None
    if dynamic:
        return _limit_export_text(dynamic)

    patterned = _pattern_based_rule_explanation(entity, column, formula)
    if patterned:
        return _limit_export_text(patterned)

    export_walk = explain_expression_for_export(formula, target_column=column)
    if export_walk:
        return _limit_export_text(export_walk)

    explained = explain_expression(formula)
    if explained:
        compact = _compact_explanation(explained, column)
        if compact:
            return _limit_export_text(compact)

    meaning = (business_meaning or "").strip()
    if meaning and not _is_generic_business_meaning(meaning, column, entity):
        return _finalize_rule_explanation(_limit_sentences(meaning, 2))

    structured = _structured_formula_fallback(entity, column, formula)
    if structured:
        return _limit_export_text(structured)

    retry = explain_expression_for_export(formula, target_column=column)
    if retry:
        return _limit_export_text(retry)

    line = executive_business_purpose_line(entity, column, formula)
    if _is_generic_executive_line(line, column, entity):
        structured = _structured_formula_fallback(entity, column, formula, broad=True)
        if structured and not _is_vague_structured_line(structured):
            return _limit_export_text(structured)
        retry_walk = explain_expression_for_export(formula, target_column=column)
        if retry_walk:
            return _limit_export_text(retry_walk)
        if "IF(" in formula.upper():
            retry_dynamic = build_dynamic_rule_sentence(
                formula,
                target_column=column,
                target_entity=entity,
            )
            if retry_dynamic:
                return _limit_export_text(retry_dynamic)
    return _finalize_rule_explanation(line)


def _pattern_based_rule_explanation(entity: str, column: str, formula: str) -> str | None:
    """Banking-oriented summaries from recurring 4X formula shapes (offline heuristics)."""
    col = (column or "column").strip()

    batch = _batch_parameter_explanation(formula)
    if batch:
        return _finalize_rule_explanation(batch)

    if _matches_date_clearing_scheme_reset(formula, col):
        return (
            f"Resets {col} to NULL for Aqua Scheme ODA products "
            f"(DIMPRODUCT.Aqua_Scheme = 'Y' and SchemeType = 'ODA') or legacy placeholder "
            f"dates ('1900-01-01'); otherwise retains existing {col}."
        )

    asset_reset = _asset_class_date_reset_explanation(col, formula)
    if asset_reset:
        return asset_reset

    splcatg = _splcatg_alt_key_explanation(col, formula)
    if splcatg:
        return splcatg

    passthrough = _entity_key_passthrough_explanation(entity, col, formula)
    if passthrough:
        return passthrough

    identity = _identity_validation_explanation(col, formula)
    if identity:
        return identity

    if _matches_priority_classification_ladder(formula):
        return (
            "Evaluates account status against priority degradation rules "
            "(ALWAYS_NPA, PUI, Restructuring) to assign the appropriate asset classification."
        )

    if _matches_dpd_or_servicing_threshold(formula):
        if (column or "").upper() in {"DPD_MAX", "DPD_FINMAXTYPE", "REFPERIODMAX"}:
            return None
        return (
            f"Calculates {col} by evaluating days past due (DPD) and servicing timelines "
            f"against regulatory classification thresholds."
        )

    fallback = _coalesce_fallback_explanation(col, formula)
    if fallback:
        return fallback

    return None


def _batch_parameter_explanation(formula: str) -> str | None:
    """Only when the entire derivation is a standalone batch parameter."""
    clean = (formula or "").strip()
    if clean.lower() in _STANDALONE_BATCH_PARAMS:
        return "Assigns the active processing timekey / system batch execution date."
    return None


def _finalize_rule_explanation(text: str) -> str:
    """Sanitize text for the Rule Explanation export column (no length truncation)."""
    if not text:
        return ""
    return _limit_export_text(text)


def _limit_export_text(text: str) -> str:
    lines = [" ".join(part.split()) for part in (text or "").splitlines()]
    normalized = "\n".join(line for line in lines if line).strip()
    return sanitize_rule_explanation_text(normalized) or ""


def _is_vague_structured_line(text: str) -> bool:
    lowered = text.lower()
    return (
        "row conditions are met" in lowered
        or "joined source tables" in lowered
        or "ordered conditional rules from joined" in lowered
    )


def _asset_class_date_reset_explanation(column: str, formula: str) -> str | None:
    upper = formula.upper()
    compact = upper.replace(" ", "")
    if "THEN(NULL)" not in compact:
        return None
    col_u = (column or "").upper()
    # S07 Update_AssetClass: DbtDt / SysAssetClassAlt_Key use dedicated narrators.
    if col_u in {"DBTDT", "DEGDATE"} and "ADDDAY(" in upper and "SYSNPA" in upper:
        return None
    if col_u == "SYSASSETCLASSALT_KEY" and "ADDDAY(" in upper and '"SUB"' in formula:
        return None
    asset_guard = any(
        token in upper
        for token in (
            "ASSETCLASS",
            "FINALASSETCLASSALT_KEY",
            "SYSASSETCLASS",
            "ALWYS_STD",
            "ALWYS_NPA",
        )
    )
    date_col = col_u in _DATE_RESET_COLUMNS or col_u.endswith("DT") or "NPA" in col_u
    if not asset_guard and not date_col:
        return None
    return (
        f"Resets {column} to NULL for standard accounts; otherwise populates from "
        f"historical or source NPA records."
    )


def _splcatg_alt_key_explanation(column: str, formula: str) -> str | None:
    col_u = (column or "").upper()
    if "SPLCATG" not in col_u or "ALT_KEY" not in col_u:
        return None
    upper = formula.upper()
    if "EFFECTIVEFROMTIMEKEY" not in upper and "EFFECTIVETOTIMEKEY" not in upper:
        if "SPLCATG" not in upper:
            return None
    source = _dominant_source_entity(formula, exclude_entity="")
    source_label = source or "the linked customer detail table"
    return (
        f"Assigns the special category classification key from {source_label} when "
        f"effective dates are valid; defaults to 0."
    )


def _entity_key_passthrough_explanation(entity: str, column: str, formula: str) -> str | None:
    col_u = (column or "").upper()
    if not any(col_u.endswith(suffix) or col_u == suffix for suffix in _IDENTITY_COLUMN_SUFFIXES):
        return None
    then_hits = list(_THEN_QUALIFIED_RE.finditer(formula))
    if not then_hits:
        return None
    ent_u = (entity or "").upper().lstrip("#")
    source_ent = None
    for match in then_hits:
        ref_ent, ref_col = match.group(1), match.group(2)
        if ref_col.upper() == col_u and ref_ent.upper().lstrip("#") != ent_u:
            source_ent = ref_ent
            break
        if ref_ent.upper().lstrip("#") != ent_u:
            source_ent = ref_ent
    if not source_ent:
        source_ent = _dominant_source_entity(formula, exclude_entity=entity)
    if not source_ent and formula.upper().count("IF(") <= 1:
        return None
    source_label = source_ent or "the source entity"
    return (
        f"Populates {column} from source entity {source_label}, retaining the current "
        f"value if unassigned."
    )


def _identity_validation_explanation(column: str, formula: str) -> str | None:
    col_u = (column or "").upper()
    if col_u not in {"PANNO", "AADHARCARDNO", "AADHAR_NO", "AADHARNO"} and "AADHAR" not in col_u:
        if "PAN" not in col_u or col_u.endswith("COMPANY"):
            return None
    upper = formula.upper()
    if " IN [" not in upper and "NOTIN [" not in upper:
        if not any(marker in upper for marker in _DUMMY_ID_MARKERS):
            return None
    if any(marker in upper for marker in _DUMMY_ID_MARKERS) or " IN [" in upper:
        return (
            f"Validates and extracts {column} from customer relationship records, "
            f"filtering out default/dummy values."
        )
    return None


def _structured_formula_fallback(
    entity: str,
    column: str,
    formula: str,
    *,
    broad: bool = False,
) -> str | None:
    """Last-chance heuristics before the generic executive one-liner."""
    col = (column or "column").strip()
    upper = formula.upper()
    compact = upper.replace(" ", "")

    for fn in (
        _batch_parameter_explanation,
        lambda f: _splcatg_alt_key_explanation(col, f),
        lambda f: _asset_class_date_reset_explanation(col, f),
        lambda f: _identity_validation_explanation(col, f),
        lambda f: _entity_key_passthrough_explanation(entity, col, f),
    ):
        hit = fn(formula)
        if hit:
            return hit

    if "COALESCE" in upper and "COUNT" in upper and "+" in formula:
        proc = _running_process_name_literal(formula)
        if proc:
            return (
                f"Increments {col} by 1 when ACLRUNNINGPROCESSSTATUS.RUNNINGPROCESSNAME equals "
                f"'{proc}'; otherwise retains existing {col}."
            )
        return f"Increments {col} when the process completes successfully; otherwise retains the prior count."

    if 'COMPLETED' in upper and ('THEN("Y")' in compact or 'THEN("N")' in compact):
        proc = _running_process_name_literal(formula)
        if proc:
            return (
                f"Sets {col} to 'Y' when ACLRUNNINGPROCESSSTATUS.RUNNINGPROCESSNAME equals "
                f"'{proc}'; otherwise retains existing {col}."
            )
        return "Marks process completion status for the running batch step."

    if broad:
        if "MIN(" in upper or "MAX(" in upper:
            return f"Derives {col} using an aggregate across related account or customer records."
        if "ADDDAY(" in upper or "DATEDIFF(" in upper:
            return f"Computes {col} from calendar offsets against process or source dates."

    return None


def _dominant_source_entity(formula: str, *, exclude_entity: str) -> str | None:
    exclude = (exclude_entity or "").upper().lstrip("#")
    counts: dict[str, int] = {}
    for ent, _col in _QUALIFIED_COL_RE.findall(formula):
        key = ent.strip()
        norm = key.upper().lstrip("#")
        if exclude and norm == exclude:
            continue
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _is_generic_executive_line(text: str, column: str, entity: str) -> bool:
    return _is_generic_business_meaning(text, column, entity) or (
        "conditional business logic in the source SQL" in text
    )


def _matches_date_clearing_scheme_reset(formula: str, column: str) -> bool:
    upper = formula.upper()
    compact = upper.replace(" ", "")
    if not any(marker in upper for marker in _DEFAULT_DATE_MARKERS):
        return False
    if not any(marker in upper for marker in _SCHEME_PRODUCT_MARKERS) and "ODA" not in upper:
        return False
    if "THEN(NULL)" not in compact and "THEN(0)" not in compact:
        return False
    col_upper = (column or "").upper()
    if col_upper and col_upper not in upper:
        return False
    return "ELSE(" in compact


def _matches_priority_classification_ladder(formula: str) -> bool:
    upper = formula.upper()
    markers = sum(
        1
        for token in ("ALWYS_NPA", "PUI_CAL", "PUI", "RESTR_NPA", "RESTRUCTURE", "ADVACRESTRUCTURECAL")
        if token in upper
    )
    if markers < 2:
        return False
    return upper.count("IF(") >= 2 or "ELSEIF(" in upper.replace(" ", "")


def _matches_dpd_or_servicing_threshold(formula: str) -> bool:
    upper = formula.upper()
    if "DATEDIFF(" in upper:
        return True
    if "DPD_" in upper or ".DPD" in upper.replace(" ", ""):
        return True
    if "REFPERIOD" in upper and ("DPD" in upper or "INTSERVICE" in upper or "OVERDUE" in upper):
        return True
    return False


_RUNNING_PROCESS_LITERAL_RE = re.compile(
    r'RUNNINGPROCESSNAME"\s*==\s*"([^"]+)"',
    re.IGNORECASE,
)


def _running_process_name_literal(formula: str) -> str | None:
    match = _RUNNING_PROCESS_LITERAL_RE.search(formula or "")
    return match.group(1).strip() if match else None


def _acl_process_status_explanation(column: str, proc: str) -> str | None:
    col = (column or "").strip()
    col_u = col.upper()
    if col_u == "COMPLETED":
        return (
            f"Sets {col} to 'Y' when ACLRUNNINGPROCESSSTATUS.RUNNINGPROCESSNAME equals "
            f"'{proc}'; otherwise retains existing {col}."
        )
    if col_u == "COUNT":
        return (
            f"Increments {col} by 1 when ACLRUNNINGPROCESSSTATUS.RUNNINGPROCESSNAME equals "
            f"'{proc}'; otherwise retains existing {col}."
        )
    if col_u == "ERRORDATE":
        return (
            f"Resets {col} to NULL when ACLRUNNINGPROCESSSTATUS.RUNNINGPROCESSNAME equals "
            f"'{proc}'; otherwise retains existing {col}."
        )
    if col_u == "ERRORDESCRIPTION":
        return (
            f"Resets {col} to NULL when ACLRUNNINGPROCESSSTATUS.RUNNINGPROCESSNAME equals "
            f"'{proc}'; otherwise retains existing {col}."
        )
    return None


def _coalesce_fallback_explanation(column: str, formula: str) -> str | None:
    match = _ISEMPTY_THEN_RE.search(formula)
    if match:
        col_a = match.group(2)
        col_b = match.group(4)
        return (
            f"Populates {column} using {col_a}, defaulting to {col_b} when unassigned or null."
        )
    coalesce = _COALESCE_PAIR_RE.search(formula)
    if coalesce and formula.upper().count("IF(") <= 2:
        col_a = coalesce.group(2)
        col_b = coalesce.group(4)
        return (
            f"Populates {column} using {col_a}, defaulting to {col_b} when unassigned or null."
        )
    return None


def _is_generic_business_meaning(text: str, column: str, entity: str) -> bool:
    normalized = " ".join(text.split())
    col = (column or "column").strip()
    ent = (entity or "entity").strip()
    prefixes = (
        f"Derives {col} for {ent} based on execution logic.",
        f"Derives {col} for {ent} by evaluating conditional business logic",
        f"Derives {col} for {ent} by folding ",
        f"Derives {col} for {ent} using ",
        f"Derives {col} for {ent} based on a membership",
        f"Derives {col} for {ent} by copying ",
        f"Derives {col} on {ent} from conditional business logic in the source SQL.",
        f"Sets {col} on {ent} to ",
        f"Derives {col} on {ent} from conditional business logic",
    )
    return any(normalized.startswith(p) for p in prefixes)


def _limit_sentences(text: str, max_sentences: int) -> str:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(" ".join(text.split())) if p.strip()]
    if not parts:
        return ""
    return " ".join(parts[:max_sentences])


def _compact_explanation(explained: str, column: str = "") -> str:
    lines = _substantive_lines(explained)
    if not lines:
        return ""
    if len(lines) >= 3:
        return format_explainer_lines_as_branches(column, lines)
    if len(lines) == 1:
        return lines[0]

    first, second = lines[0], lines[1]
    first_lower = first.lower()
    if first_lower.startswith("if "):
        condition = first[3:].strip().rstrip(".")
        if second.lower().startswith(("return ", "leave ")):
            return f"When {condition}, {second[0].lower()}{second[1:]}"
        return f"{first.rstrip('.')}; {second}"

    if first_lower.startswith("otherwise, if "):
        condition = first[len("Otherwise, if ") :].strip().rstrip(".")
        if second.lower().startswith(("return ", "leave ")):
            return f"Otherwise when {condition}, {second[0].lower()}{second[1:]}"
    return _limit_sentences(f"{first} {second}", 2)


def _substantive_lines(explained: str) -> list[str]:
    out: list[str] = []
    for raw in explained.splitlines():
        text = _MARKDOWN_BOLD_RE.sub(r"\1", raw.strip())
        if text.startswith("- "):
            text = text[2:].strip()
        if not text or text.lower() == "otherwise:":
            continue
        if text.endswith(":") and (
            text.lower().startswith("if ")
            or text.lower().startswith("otherwise, if ")
        ):
            text = text[:-1].strip()
        out.append(text)
    return out
