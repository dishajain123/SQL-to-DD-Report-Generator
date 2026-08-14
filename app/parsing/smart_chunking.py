"""Dependency-aware logical chunking for SQL objects.

This keeps control-flow blocks like IF/ELSE and CASE together while leaving
ordinary sequential statements as their own chunks. The goal is to preserve
semantic relationships without over-merging an entire procedure body into a
single analysis unit.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from app.models.core import SmartChunk, StatementInfo

_OPENERS = {"IF", "CASE", "LOOP"}
_BRANCH_MARKERS = {"ELSE", "ELSIF", "WHEN"}
_CLOSER_RE = re.compile(r"^END(?:\s+(IF|CASE|LOOP))?\b", re.IGNORECASE)


@dataclass
class _ChunkBuffer:
    statement_infos: list[StatementInfo] = field(default_factory=list)
    block_stack: list[str] = field(default_factory=list)
    saw_semantic_block: bool = False

    def append(self, info: StatementInfo) -> None:
        self.statement_infos.append(info)

    def clear(self) -> None:
        self.statement_infos.clear()
        self.block_stack.clear()
        self.saw_semantic_block = False

    def is_empty(self) -> bool:
        return not self.statement_infos


def build_smart_chunks(object_id: str, statements: list[StatementInfo]) -> list[SmartChunk]:
    """Group raw statements into dependency-aware chunks.

    Blocks opened by IF/CASE/LOOP are preserved as a unit. Sequential DML or
    SELECT statements outside those blocks remain separate so downstream
    lineage and reasoning do not lose statement-level granularity.
    """
    chunks: list[SmartChunk] = []
    buffer = _ChunkBuffer()

    for info in statements:
        keyword = _leading_keyword(info.raw_text)
        opener = keyword in _OPENERS
        closer = bool(_CLOSER_RE.match(_strip_leading_comments(info.raw_text)))
        branch_marker = keyword in _BRANCH_MARKERS

        if buffer.is_empty():
            buffer.append(info)
        else:
            buffer.append(info)

        if opener:
            buffer.block_stack.append(keyword)
            buffer.saw_semantic_block = True
        elif closer and buffer.block_stack:
            _pop_matching_block(buffer.block_stack, info.raw_text)

        # Inside a semantic block we keep everything together until the block
        # actually closes. Branch markers like ELSE/WHEN remain in the same
        # chunk so the guardrails can see the full decision context.
        if buffer.block_stack:
            continue

        # Outside a semantic block:
        # - a branch marker is kept with the current chunk if one exists
        # - a standalone opener/closer flushes as its own chunk
        # - ordinary statements are separate chunks by default
        if branch_marker and len(buffer.statement_infos) == 1:
            # A branch marker without a surrounding block is malformed, but
            # keep it visible rather than dropping it.
            _flush_buffer(chunks, object_id, buffer)
            continue

        if opener or closer or not buffer.saw_semantic_block:
            _flush_buffer(chunks, object_id, buffer)

    if not buffer.is_empty():
        _flush_buffer(chunks, object_id, buffer)

    return chunks


def _flush_buffer(chunks: list[SmartChunk], object_id: str, buffer: _ChunkBuffer) -> None:
    if not buffer.statement_infos:
        return

    statement_indices = [info.statement_index for info in buffer.statement_infos]
    raw_sql = "\n".join(info.raw_text.rstrip() for info in buffer.statement_infos).strip()
    tables_read = _sorted_unique(value for info in buffer.statement_infos for value in info.tables_read)
    tables_written = _sorted_unique(value for info in buffer.statement_infos for value in info.tables_written)
    columns_written = _sorted_unique(
        value for info in buffer.statement_infos for values in info.set_columns_by_table.values() for value in values
    )
    join_tables = _sorted_unique(value for info in buffer.statement_infos for value in info.join_tables)
    conditions = _dedupe_preserve_order(
        value for info in buffer.statement_infos for value in (*info.conditions, *info.join_conditions)
    )

    dependency_hints = _dedupe_preserve_order(
        [
            *[f"read:{table}" for table in tables_read],
            *[f"write:{table}" for table in tables_written],
            *[f"join:{table}" for table in join_tables],
            *[f"condition:{condition}" for condition in conditions],
        ]
    )

    dml_statements = [info for info in buffer.statement_infos if info.statement_type in {"SELECT", "UPDATE", "MERGE", "INSERT", "DELETE"}]
    parsed_ok_count = sum(1 for info in dml_statements if info.parsed_ok)
    confidence = parsed_ok_count / len(dml_statements) if dml_statements else 1.0

    chunk_kind = "CONTROL_FLOW_BLOCK" if buffer.saw_semantic_block else (buffer.statement_infos[0].statement_type or "SEQUENTIAL")
    contains_control_flow = any(info.statement_type == "CONTROL_FLOW" for info in buffer.statement_infos)

    chunks.append(
        SmartChunk(
            chunk_id=f"chunk-{uuid.uuid4().hex[:10]}",
            object_id=object_id,
            chunk_index=len(chunks),
            chunk_kind=chunk_kind,
            statement_indices=statement_indices,
            raw_sql=raw_sql,
            tables_read=tables_read,
            tables_written=tables_written,
            columns_written=columns_written,
            join_tables=join_tables,
            conditions=conditions,
            dependency_hints=dependency_hints,
            confidence=round(confidence, 3),
            contains_control_flow=contains_control_flow or buffer.saw_semantic_block,
            contains_join=bool(join_tables),
        )
    )
    buffer.clear()


def _leading_keyword(stmt_text: str) -> str:
    stripped = _strip_leading_comments(stmt_text)
    match = re.match(r"[A-Za-z]+", stripped)
    return match.group(0).upper() if match else ""


def _strip_leading_comments(text: str) -> str:
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos].isspace():
            pos += 1
        elif text[pos:pos + 2] == "--":
            nl = text.find("\n", pos)
            pos = n if nl == -1 else nl + 1
        elif text[pos:pos + 2] == "/*":
            end = text.find("*/", pos + 2)
            pos = n if end == -1 else end + 2
        else:
            break
    return text[pos:]


def _pop_matching_block(stack: list[str], stmt_text: str) -> None:
    if not stack:
        return
    stripped = _strip_leading_comments(stmt_text).upper()
    match = _CLOSER_RE.match(stripped)
    if match is None:
        stack.pop()
        return
    expected = match.group(1)
    if expected is None:
        stack.pop()
        return
    # Pop the most recent matching semantic block, or fall back to the top.
    expected = expected.upper()
    for i in range(len(stack) - 1, -1, -1):
        if stack[i] == expected:
            del stack[i:]
            return
    stack.pop()


def _sorted_unique(values) -> list[str]:
    return sorted({value for value in values if value})


def _dedupe_preserve_order(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
