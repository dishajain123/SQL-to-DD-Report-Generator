"""Core domain models shared across the pipeline.

These are the objects that flow between pipeline stages (parsing -> lineage ->
derivation -> report/CSV). Keeping them centralized avoids each module
inventing its own shape for the same concept.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Dialect(str, Enum):
    ORACLE = "oracle"
    MYSQL = "mysql"
    SQLSERVER = "sqlserver"


class ObjectType(str, Enum):
    PROCEDURE = "PROCEDURE"
    FUNCTION = "FUNCTION"
    TRIGGER = "TRIGGER"
    VIEW = "VIEW"
    UNKNOWN = "UNKNOWN"


class Intent(str, Enum):
    ANALYZE = "Analyze"
    EXPLAIN = "Explain"
    DERIVE = "Derive"
    GENERATE_DD = "Generate DD"
    GENERATE_EXCEL = "Generate Excel"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobPlan(BaseModel):
    """Output of the Intent Classifier (architecture step 3). Carried through
    the whole pipeline so later steps know whether DD Generation should run."""

    job_id: str
    intent: Intent
    company: str
    platform: str

    @property
    def requires_dd_generation(self) -> bool:
        return self.intent in (Intent.DERIVE, Intent.GENERATE_DD, Intent.GENERATE_EXCEL)


class SQLObject(BaseModel):
    """One split-out unit from an uploaded SQL file (architecture step 4)."""

    object_id: str
    name: str
    object_type: ObjectType
    dialect: Dialect
    raw_sql: str
    source_file: str


class StatementInfo(BaseModel):
    """Structural facts extracted from a single SQL statement inside an object."""

    statement_index: int
    statement_type: str  # SELECT, UPDATE, MERGE, INSERT, DELETE, CONTROL_FLOW
    raw_text: str
    tables_read: list[str] = Field(default_factory=list)
    tables_written: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    join_tables: list[str] = Field(default_factory=list)
    join_conditions: list[str] = Field(default_factory=list)
    # Columns actually assigned a value (SET clause target), keyed by the
    # table they were written to -- distinct from `columns`, which includes
    # every column referenced anywhere in the statement (WHERE/JOIN too).
    # This is what should drive DD row generation; using `columns` alone
    # produces false pairings (a column only seen in a WHERE clause getting
    # treated as if it were derived).
    set_columns_by_table: dict[str, list[str]] = Field(default_factory=dict)
    conditions: list[str] = Field(default_factory=list)
    parsed_ok: bool = True
    parse_error: Optional[str] = None


class VersionThreshold(BaseModel):
    """A detected date/period-based rule-versioning branch, e.g. `p_TIMEKEY > 26267`."""

    variable: str
    operator: str
    value: str
    raw_condition: str


class StructuralInfo(BaseModel):
    """Aggregated structural analysis for one SQLObject (architecture step 7)."""

    object_id: str
    statements: list[StatementInfo] = Field(default_factory=list)
    tables_read: list[str] = Field(default_factory=list)
    tables_written: list[str] = Field(default_factory=list)
    columns_written: list[str] = Field(default_factory=list)
    # The correct pairing for DD row generation: which specific columns were
    # actually set on which specific table (see StatementInfo.set_columns_by_table).
    columns_written_by_table: dict[str, list[str]] = Field(default_factory=dict)
    called_objects: list[str] = Field(default_factory=list)
    has_dynamic_sql: bool = False
    version_thresholds: list[VersionThreshold] = Field(default_factory=list)
    smart_chunks: list["SmartChunk"] = Field(default_factory=list)
    confidence: float = 1.0
    unsupported_constructs: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.confidence >= 0.5 and not self.has_dynamic_sql


class SmartChunk(BaseModel):
    """A dependency-aware logical chunk inside a SQLObject.

    Chunks preserve control-flow groupings such as IF/ELSE and CASE blocks
    while keeping sequential standalone statements separate.
    """

    chunk_id: str
    object_id: str
    chunk_index: int
    chunk_kind: str
    statement_indices: list[int] = Field(default_factory=list)
    raw_sql: str
    tables_read: list[str] = Field(default_factory=list)
    tables_written: list[str] = Field(default_factory=list)
    columns_written: list[str] = Field(default_factory=list)
    join_tables: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    dependency_hints: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    contains_control_flow: bool = False
    contains_join: bool = False


class LineageChain(BaseModel):
    """A group of objects linked by producer/consumer relationships
    (architecture step 9)."""

    chain_id: str
    object_ids: list[str]
    order: list[str]  # topologically sorted object_ids
    order_confidence: str = "high"  # "high" (acyclic, real topo sort) | "low" (fallback)


class CanonicalModel(BaseModel):
    """Single source of truth per lineage chain (architecture step 12)."""

    chain_id: str
    job_id: str
    object_ids: list[str]
    technical_summary: str
    business_summary: str
    glossary_terms: list["GlossaryTerm"] = Field(default_factory=list)
    derived_rules: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class DerivationOption(str, Enum):
    FORMULA_EXPRESSION = "Formula Expression"
    DECISION_TABLE = "Decision Table"


class ColumnType(str, Enum):
    TEMPORARY = "Temporary"
    PHYSICAL = "Physical"


class DDStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    PENDING_REVIEW = "PENDING_REVIEW"


class DDRow(BaseModel):
    """One row of the Derivation Dictionary output — matches the platform's
    Derivations export schema exactly (Entity Name, Column Name, ...)."""

    entity_name: str
    column_name: str
    column_type: ColumnType
    derivation_option: DerivationOption
    display_derivation_expression: str = ""
    effective_start_date: date
    status: DDStatus = DDStatus.PENDING_REVIEW
    data_type: str
    decision_table_json: Optional[str] = None
    conditional_json: Optional[str] = None
    business_meaning: str = ""

    # Traceability (not in the platform export, used internally / in the report)
    source_chain_id: str
    source_object_ids: list[str] = Field(default_factory=list)
    # One human-readable entry per source write site that fed this row's
    # expression -- e.g. "npa.sql stmt #30 (role=NULL_RESET)" -- so a
    # reviewer (or the report) can trace a generated condition back to the
    # exact statement(s) in the source SQL it came from, not just the
    # object as a whole. Populated in
    # app/derivation/dd_generator.py::_generate_for_column from the same
    # _AssignmentSite data already used to build the LLM's prompt context.
    source_statement_refs: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    validation_errors: list[str] = Field(default_factory=list)
    advisory_notes: list[str] = Field(default_factory=list)


class ReviewAction(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    EDIT = "EDIT"
    OVERRIDE = "OVERRIDE"


class ReviewDecision(BaseModel):
    dd_row_index: int
    action: ReviewAction
    edited_expression: Optional[str] = None
    reviewer: str = "unassigned"
    comment: str = ""


class GlossaryTerm(BaseModel):
    term: str
    definition: str
