"""Shared SQL-table → platform-entity mapping for demo and live runs.

Live Streamlit/API jobs and `scripts/generate_sample_dd_demo.py` must resolve
the same physical tables (AccountCal, LoanAccountCal, …) to the same 4X
entity names, or CSV/report output diverges between paths.
"""
from __future__ import annotations

import json
from typing import Iterable

from app.utils.config import settings

# No built-in SQL-table → invented platform-entity renames.
# Entity Name / Column Name in DD output come from the procedure identifiers
# (LoanAccountCal, RestructureRegister, …). Optional renames only via
# DEFAULT_ENTITY_NAME_MAP_JSON when the user explicitly configures them.
DEFAULT_ENTITY_OVERRIDES: dict[str, str] = {}


def _strip_temp_prefix(name: str) -> str:
    text = (name or "").strip()
    if text.startswith("##"):
        return text[2:]
    if text.startswith("#"):
        return text[1:]
    return text


def _bare_table(name: str) -> str:
    text = _strip_temp_prefix(name)
    if "." in text:
        text = text.split(".")[-1]
    return text.strip().strip('"').strip("[]")


def merge_entity_overrides(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return built-in overrides overlaid with optional caller/env entries."""
    merged = dict(DEFAULT_ENTITY_OVERRIDES)
    for key, value in (extra or {}).items():
        clean_key = (key or "").strip()
        clean_value = (value or "").strip()
        if clean_key and clean_value:
            merged[clean_key] = clean_value
    return merged


def load_configured_entity_overrides() -> dict[str, str]:
    """Built-in sample map + DEFAULT_ENTITY_NAME_MAP_JSON from the environment."""
    raw = (settings.default_entity_name_map_json or "").strip()
    extra: dict[str, str] = {}
    if raw and raw != "{}":
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("DEFAULT_ENTITY_NAME_MAP_JSON must be a JSON object.")
        for key, value in parsed.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("DEFAULT_ENTITY_NAME_MAP_JSON keys and values must both be strings.")
            clean_key = key.strip()
            clean_value = value.strip()
            if not clean_key or not clean_value:
                raise ValueError("DEFAULT_ENTITY_NAME_MAP_JSON keys and values cannot be blank.")
            extra[clean_key] = clean_value
    return merge_entity_overrides(extra)


def resolve_entity_name(table: str, entity_name_map: dict[str, str] | None = None) -> str:
    """Map a physical/temp table name to the platform entity used in DD rows."""
    mapping = entity_name_map or {}
    candidates = [
        table,
        _strip_temp_prefix(table),
        _bare_table(table),
    ]
    upper_index = {key.upper(): value for key, value in mapping.items()}
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in mapping:
            return _strip_temp_prefix(mapping[candidate])
        hit = upper_index.get(candidate.upper())
        if hit:
            return _strip_temp_prefix(hit)
        bare = _bare_table(candidate)
        hit = upper_index.get(bare.upper())
        if hit:
            return _strip_temp_prefix(hit)
    return _bare_table(table)


def build_entity_name_map_for_tables(
    tables: Iterable[str],
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Expand overrides into a lookup covering raw, bare, and #temp spellings."""
    base = merge_entity_overrides(overrides)
    mapping: dict[str, str] = {}
    for table in tables:
        if not table:
            continue
        resolved = resolve_entity_name(table, base)
        mapping[table] = resolved
        bare = _bare_table(table)
        mapping[bare] = resolved
        mapping[f"#{bare}"] = resolved
        mapping[f"##{bare}"] = resolved
        # Also index override keys so pipeline `.get(target_table)` works.
    for key, value in base.items():
        mapping.setdefault(key, _strip_temp_prefix(value))
        mapping.setdefault(_bare_table(key), _strip_temp_prefix(value))
    return mapping


def build_entity_name_map_for_info(info, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Build the entity map for one analyzed SQL object (demo + live parity)."""
    candidates = set(getattr(info, "tables_written", None) or [])
    base = merge_entity_overrides(overrides)
    upper_keys = {key.upper() for key in base}
    for table in set(getattr(info, "tables_read", None) or []):
        if table in candidates or _bare_table(table).upper() in upper_keys:
            candidates.add(table)
    return build_entity_name_map_for_tables(candidates, base)
