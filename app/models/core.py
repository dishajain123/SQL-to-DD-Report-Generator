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
    normalization_notes: list[str] = Field(default_factory=list)


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
    # Everything that needs attention: real parse failures PLUS coverage-ledger
    # blockers (procedure branches, temp staging, MERGE, CATCH, anomalies).
    unsupported_constructs: list[str] = Field(default_factory=list)
    # Only statements sqlglot genuinely could not parse. Kept separate so a
    # well-formed UPDATE inside an IF branch is never reported as unparseable.
    parse_failures: list[str] = Field(default_factory=list)

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


class ReviewState(str, Enum):
    """Human/process review lifecycle — independent of platform Status."""

    GENERATED = "GENERATED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    APPROVED = "APPROVED"
    UNSUPPORTED = "UNSUPPORTED"


class ExecutionStep(BaseModel):
    """One source write to a DD column, in SQL execution order.

    The DD formula folds every step into one expression; the steps keep the
    sequence visible so an overwritten assignment (e.g. a value set in step 2
    and replaced in step 3) is still traceable.
    """

    step: int
    statement_ref: str = ""
    source_line: Optional[int] = None
    operation: str = "UPDATE"
    # "Main" or "Exception handler (CATCH)".
    scope: str = "Main"
    # Procedure-wide gate label (e.g. "Gate 1") this write sits under.
    workflow_gate: Optional[str] = None
    row_condition: str = ""
    # "JOIN Table ON a = b" for UPDATE/INSERT … FROM … JOIN sources: an INNER
    # JOIN also limits which rows the statement touches.
    join_conditions: list[str] = Field(default_factory=list)
    assigned_value: str = ""
    notes: list[str] = Field(default_factory=list)


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
    review_state: ReviewState = ReviewState.GENERATED
    data_type: str
    decision_table_json: Optional[str] = None
    conditional_json: Optional[str] = None
    business_meaning: str = ""

    # Traceability (not in the platform export, used internally / in the report)
    source_chain_id: str
    source_object_ids: list[str] = Field(default_factory=list)
    # One human-readable entry per source write site that fed this row's
    # expression -- e.g. "npa.sql stmt #30 (ordinal=1)" -- so a
    # reviewer (or the report) can trace a generated condition back to the
    # exact statement(s) in the source SQL it came from. Populated by the
    # v2 AST derivation pipeline from mutation sites.
    source_statement_refs: list[str] = Field(default_factory=list)
    # The actual raw SQL text behind each source_statement_refs entry, in
    # the same order. Lets downstream stages (alias resolution, dependency
    # extraction) work from the specific statement(s) a row's formula came
    # from instead of the whole object's SQL -- important because a short
    # alias like "A" is routinely reused for a different table in a
    # different statement elsewhere in the same object, which is
    # unresolvably ambiguous at the whole-object level but perfectly
    # resolvable within the one or two statements a given row actually
    # derives from.
    source_statement_sql: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    validation_errors: list[str] = Field(default_factory=list)
    advisory_notes: list[str] = Field(default_factory=list)
    # Position of this column's first write in the (comment-stripped) source
    # SQL. Reports sort rules by it so they read in execution order.
    execution_order: Optional[int] = None
    execution_steps: list[ExecutionStep] = Field(default_factory=list)
    # Procedure-wide IF gates this column's writes sit under, as
    # "Gate N := IF <original SQL condition>". Report context only — the
    # exported formula never references them.
    workflow_gates: list[str] = Field(default_factory=list)
    # Formula for the BEGIN CATCH path, kept apart from the main (TRY) formula.
    exception_handler_expression: str = ""


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