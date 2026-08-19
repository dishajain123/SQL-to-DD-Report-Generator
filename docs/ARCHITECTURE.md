# Architecture Reference

Quick module map. For the full pipeline diagram/rationale, see the
architecture SVGs produced earlier in this project's design phase — this
doc is a terse code-to-concept index, not a repeat of that discussion.

| Pipeline stage | Module | Notes |
|---|---|---|
| Input Guardrails | `app/guardrails/input_guardrails.py` | File type/size/security, company/platform required |
| Intent Classifier | `app/models/core.py::JobPlan` | `requires_dd_generation` gates DD Generation |
| Object Split & Classification | `app/parsing/object_splitter.py` | Handles both multi-proc-per-file and one-proc-per-file |
| Dialect Detection + Parser | `app/parsing/dialect.py`, `app/parsing/sql_parser.py` | sqlglot-based, Oracle + MySQL |
| Structural Analysis + Guardrails | `app/parsing/structural_analysis.py`, `app/guardrails/structural_guardrails.py` | Tables, SET-clause columns per table, TIMEKEY thresholds |
| Cross-Object Dependency & Lineage | `app/lineage/dependency_graph.py` | networkx; read-after-write, explicit calls, weak shared-write fallback |
| Context Building | `app/rag/chroma_store.py`, `app/rag/ingest.py` | Domain RAG + Platform RAG (4X grammar doc) |
| AI Understanding | `app/derivation/llm_client.py::technical_reasoning/business_reasoning` | Per lineage chain |
| Canonical Understanding Model | `app/derivation/canonical_model.py` | One per chain, not per object |
| DD Generation | `app/derivation/dd_generator.py` | Chain collapse, grammar targeting, retry-on-failure, TIMEKEY versioning |
| Grammar validation | `app/grammar/fourx_grammar.lark`, `app/grammar/validator.py` | Real formal grammar, not a heuristic |
| AI Output Guardrails | `app/guardrails/output_guardrails.py` | Grammar validity + evidence/hallucination check |
| Human Review | `app/review/review_store.py`, `app/review/streamlit_app.py` | SQLite-backed queue |
| Report Generator | `app/report/report_generator.py` | One combined doc, DD section only if generation ran |
| DD CSV Export | `app/report/dd_export.py` | Matches the platform's real Derivations schema exactly |
| Persistence / Audit | `app/utils/db.py` | Jobs, DD rows, review decisions, audit log |
| Orchestration | `app/orchestration/pipeline.py` | LangGraph `StateGraph`, conditional DD-generation routing |

## Data flow (types, not prose)

```
SQLObject (per split unit)
  -> StructuralInfo (per object: tables, columns_written_by_table, thresholds)
     -> LineageChain (groups + orders related objects)
        -> CanonicalModel (per chain: technical + business summary)
           -> DDRow[] (per chain: one or more rows per written column,
                        split by Effective Start Date when versioned)
              -> DD CSV + Combined Report
```

## Where the real engineering risk still lives

Everything above `DD Generation` in the table is deterministic and fully
tested. `DD Generation` itself depends on an LLM call whose *correctness*
(not just syntactic validity) can't be unit-tested the same way — that's
why `app/derivation/dd_generator.py` validates every output against the
real grammar and retries once, and why low-confidence/failed rows always
land in Human Review rather than being silently accepted. Build a golden
dataset (expected DD rows for a few real procs, starting with simple
single-CASE columns) before trusting this stage unattended — see the
"Known limitations" section of the README.
