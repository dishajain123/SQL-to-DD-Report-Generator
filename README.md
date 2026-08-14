# DD Automation

Turns Oracle/MySQL stored procedures into a Derivation Dictionary (DD): a
technical + business report, and (optionally) a DD Excel export matching a
banking platform's own Derivations schema. Built around a LangGraph
pipeline: parse SQL -> build cross-procedure lineage -> understand what the
logic does -> translate it into the platform's Formula Expression grammar
-> validate -> human review for anything uncertain -> report + Excel.

## What's real vs. what needs an API key

Parsing, lineage graph construction, grammar validation, guardrails, Excel
export, report generation, persistence, and the review UI are deterministic
code — no LLM involved, fully covered by tests, nothing to configure.

The **reasoning steps** (understanding what a chain of SQL does, and
translating it into the 4X Formula Expression grammar) call an external LLM
API for real. This requires `LLM_API_KEY` and a model/provider choice in
`.env`. The test suite mocks this client (see `tests/conftest.py`) so
`pytest` never makes a network call or needs a key. Prompt templates live in
separate YAML files under `app/derivation/prompts/` so they can be edited
without touching Python code.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set LLM_API_KEY / LLM_MODEL_NAME if you want to run the LLM-backed reasoning for real
```

Requires Python 3.10+.

## How To Run

Use this sequence for a clean local run:

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy [`.env.example`](/Users/dishajain/Downloads/DD_Automation/.env.example) to [`.env`](/Users/dishajain/Downloads/DD_Automation/.env) and set `LLM_API_KEY` and `LLM_MODEL_NAME`.
4. Run the tests to verify the repo is healthy.
5. Start the API or the Streamlit UI depending on what you want to use.
6. Optionally run the LLM smoke test to see the reasoning outputs step by step.

Example commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set:

```env
DEFAULT_COMPANY_NAME=Acme Bank
DEFAULT_PLATFORM_NAME=4X
DEFAULT_INTENT=Generate DD
DEFAULT_FUNCTION_REFERENCE_PATH=samples/platform_docs/4x_functions_operators.md
DEFAULT_ENTITY_NAME_MAP_JSON={}
LLM_PROVIDER=auto
LLM_API_KEY=your_real_key_here
LLM_MODEL_NAME=gpt-4.1
```

## Run Check

Before you start the app, it’s a good idea to verify the repo with:

```bash
pytest
```

You can also run layer-specific checks:

```bash
python3 scripts/run_layer1_tests.py
python3 scripts/run_layer2_tests.py
python3 scripts/run_layer3_tests.py
```

If you want to see the reasoning output step by step, use the smoke script:

```bash
python3 scripts/run_groq_smoke.py
```

If the script says the API cannot be reached, run it on a machine with internet access.

## Configuration

All configuration is environment variables (loaded from `.env` via
`python-dotenv`), read in `app/utils/config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `auto` | Chooses `openai` or `groq`. `auto` infers it from `LLM_MODEL_NAME` or `LLM_BASE_URL`. |
| `LLM_API_KEY` | (empty) | API key for the selected provider. |
| `LLM_MODEL_NAME` | provider-specific | Primary model used for reasoning/DD generation. |
| `LLM_BASE_URL` | provider-specific | Override for the provider's chat-completions endpoint. |
| `CHROMA_PERSIST_DIR` | `.chroma` | Where the RAG vector store persists to disk. |
| `SQLITE_DB_PATH` | `dd_automation.db` | Job history / DD rows / review decisions / audit log. |
| `OUTPUT_DIR` | `output` | Where reports and DD Excel exports are written. |
| `STRUCTURAL_CONFIDENCE_THRESHOLD` | `0.5` | Below this, Structural Guardrails fail an object. |
| `OUTPUT_GUARDRAIL_CONFIDENCE_THRESHOLD` | `0.7` | Below this, a DD row is flagged for review. |

If you want to override the model, set `LLM_MODEL_NAME` in `.env`. If you
want to switch providers, change `LLM_PROVIDER` and the matching key / base
URL. The app will pick up the new values on restart.

## LLM setup and output check

If you want to verify the LLM flow step by step, use the smoke script:

```bash
python3 scripts/run_groq_smoke.py
```

What it does:

1. Checks whether `LLM_API_KEY` is present in your environment.
2. Runs the technical reasoning step and prints the summary.
3. Runs the business reasoning step and prints the summary.
4. Runs DD formula generation and prints the final expression text.

By default it uses the sample SQL files in `samples/sql/` and the platform
reference in `samples/platform_docs/4x_functions_operators.md`. You can also
pass your own files:

```bash
python3 scripts/run_groq_smoke.py \
  --sql path/to/proc1.sql \
  --sql path/to/proc2.sql \
  --function-reference path/to/function_reference.md
```

## Running the tests

```bash
pytest
```

Layer-specific runners:

```bash
python3 scripts/run_layer1_tests.py
python3 scripts/run_layer2_tests.py
python3 scripts/run_layer3_tests.py
```

74 tests across `tests/unit`, `tests/integration`, `tests/e2e` — all run
offline against real sample procs, no API key needed.

## Running the sample end-to-end workflow

This runs the full pipeline against the real sample procs included in
`samples/sql/`, using a mocked LLM client (no API key needed) so you can see
the whole thing work immediately:

```bash
python3 -c "
from app.models.core import Intent, JobPlan
from app.orchestration.pipeline import build_pipeline
from tests.conftest import MockLLMClient
from app.utils import db

db.init_db()

with open('samples/sql/PRO_DPD_Calculation_StoredProcedure_2.sql') as f: dpd = f.read()
with open('samples/sql/PRO_MaxDPD_ReferencePeriod_Calculation_StoredProcedure.sql') as f: maxdpd = f.read()
with open('samples/sql/PRO_NPA_Date_Calculation_StoredProcedure_1.sql') as f: npa = f.read()
with open('samples/platform_docs/4x_functions_operators.md') as f: func_ref = f.read()

job_plan = JobPlan(job_id='sample-job-1', intent=Intent.GENERATE_DD, company='Acme Bank', platform='4X', include_dd_excel=True)
pipeline = build_pipeline(llm_client=MockLLMClient())
result = pipeline.invoke({
    'job_plan': job_plan,
    'uploaded_files': {'dpd.sql': dpd, 'maxdpd.sql': maxdpd, 'npa.sql': npa},
    'function_reference': func_ref,
    'entity_name_map': {'AccountCal_Stg': 'FCT_NPA_PRODUCT'},
})
print('DD rows generated:', len(result['dd_rows']))
print('Report:', result['report_path'])
print('Excel:', result.get('excel_path'))
"
```

This correctly reconstructs the real dependency chain across the three
sample procs (`DPD_Calculation` -> `MaxDPD_ReferencePeriod_Calculation` ->
`NPA_Date_Calculation`), generates DD rows, and writes a report + Excel file
to `output/sample-job-1/`.

To run it for real (actual LLM calls instead of the mock), set
`LLM_API_KEY` and `LLM_MODEL_NAME` in `.env` and use `LLMClient()` instead of
`MockLLMClient()`.

## Running the application

**API:**
```bash
uvicorn app.main:app --reload
```
Then `POST /api/jobs` with `{"company": "...", "platform": "...", "intent": "Generate DD", "files": {"proc.sql": "..."}}`.
`GET /health` for a liveness check, `GET /api/jobs/{job_id}/status` for job status.

**Intake + Human Review UI:**
```bash
.venv/bin/streamlit run app/review/streamlit_app.py
```
Provides a DD intake tab for submitting SQL jobs to the API and a review tab
that lists every DD row currently `PENDING_REVIEW` (grammar validation
failures, low-confidence rows, synthetic-date version rows) with Approve /
Reject / Edit / Override actions. The UI is preconfigured for the company,
platform, intent, function reference, and entity name map defined in
`.env`, so you do not need to type those in every time.

## Recommended Start Order

If you want the shortest reliable path from clone to working app:

1. `cp .env.example .env`
   2. Set `LLM_API_KEY` and `LLM_MODEL_NAME` in `.env`
3. `pip install -r requirements.txt`
4. `pytest`
5. `uvicorn app.main:app --reload`
6. In a second terminal, `.venv/bin/streamlit run app/review/streamlit_app.py`
7. Optional: `python3 scripts/run_groq_smoke.py`

## GitHub Ready Notes

- Keep `.env` local; only commit `.env.example`.
- If you need to change the company/platform the UI submits under, update
  `DEFAULT_COMPANY_NAME` and `DEFAULT_PLATFORM_NAME` in `.env`.
- If you want to change the default prompt context or entity mapping,
  update `DEFAULT_FUNCTION_REFERENCE_PATH` and
  `DEFAULT_ENTITY_NAME_MAP_JSON` in `.env`.
- Generated files in `output/`, SQLite DBs, Chroma caches, and `.venv/`
  are ignored by [`.gitignore`](/Users/dishajain/Downloads/DD_Automation/.gitignore).
- The repository is designed to run tests without an API key; only the LLM
  reasoning smoke test needs live network access and a real key.

## Project structure

```
app/
  models/core.py          Shared domain models (JobPlan, SQLObject, StructuralInfo,
                           CanonicalModel, DDRow, ...)
  parsing/                Object splitting, dialect detection, SQL statement parsing
                           (sqlglot-based), structural analysis
  lineage/                Cross-object dependency graph (networkx)
  grammar/                4X Platform Formula Expression grammar (Lark) + validator
  derivation/              LLM client, canonical model builder, DD generator, versioning
  guardrails/              Input / structural / output validation
  rag/                     Chroma-backed RAG for domain + platform knowledge
  review/                  Review queue + Streamlit UI
  report/                  Combined report + DD Excel export
  orchestration/pipeline.py  LangGraph wiring of the whole pipeline
  api/                     FastAPI routes/schemas
  utils/                   Config, logging, SQLite persistence

samples/
  sql/                     Real sample procs (Oracle) + a MySQL sample + a
                            synthetic multi-object file
  derivations/              A real sample Derivations export (target DD schema)
  platform_docs/            4X function/operator reference + banking domain glossary
                            (used for RAG ingestion)

tests/
  unit/                    Fast, no I/O beyond tmp files
  integration/             DD generation with mocked LLM, FastAPI request validation
  e2e/                     Full pipeline runs against real sample procs
```

## Known limitations (documented in code, not hidden)

- **TIMEKEY -> calendar date** requires the platform's own day-matrix table
  (`SysDayMatrix` in the sample procs). Without it, a deterministic
  synthetic date is used and the row is flagged `PENDING_REVIEW` rather than
  silently trusted. See `app/derivation/versioning.py`.
- **Lineage ordering** for objects that mutually read/write the same
  staging table (a real pattern in the sample procs) can't always be
  resolved from table names alone; the graph builder falls back to
  submission order and marks `order_confidence="low"`. See
  `app/lineage/dependency_graph.py`.
- **RAG embeddings** default to a dependency-free hashing embedding (no
  network calls, works everywhere, but not real semantic search). Chroma's
  default embedding function tries to download a model from the internet,
  which failed in this project's own build environment — pass a real
  embedding function into `ChromaStore` for production-quality retrieval.
  See `app/rag/chroma_store.py`.
- **DD Generation quality** depends on the LLM call at runtime like any
  real LLM-integrated system; grammar validity is guaranteed (every
  expression is parsed against the real 4X grammar before acceptance,
  with one retry on failure), but business-logic *correctness* for deeply
  cascaded multi-statement derivations should be checked against a golden
  dataset before trusting it unattended — start with simple single-CASE
  columns, not the deeply cascaded ones.

## Preserving existing DD data across re-runs

Re-running the pipeline (e.g. after one proc changes) does not wipe out a
previously-generated DD Excel. Pass `existing_dd_path` to `export_dd_rows`
(or set `existing_dd_excel_path` in the pipeline state) and rows are merged
by `(Entity Name, Column Name, Effective Start Date)`: new rows replace a
matching existing row, everything else is preserved untouched. See
`app/report/excel_export.py::merge_dd_rows`.
