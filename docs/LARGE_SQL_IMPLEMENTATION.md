# Large SQL processing: implementation and validation

The live pipeline now processes large procedures without letting repeated
self-dependent updates expand indefinitely. Completeness is a separate gate
from formula syntax: unresolved source semantics prevent automatic ACTIVE
status. This is a conservative review system, not proof of SQL equivalence.

## Findings from the supplied procedure

Input: `01_S00_PRO.InsertDataforAssetClassficationRBL.StoredProcedure.sql`.
It is UTF-16, 204,690 decoded characters and 4,452 lines.

The existing generation path is deterministic; neither narrative nor formula
generation calls an LLM. Increasing model tokens, parallel LLM requests or
truncating the input would not fix this bottleneck.

A profiled baseline was interrupted after more than five minutes without a
completed export. A separate serial profile recorded over one billion calls;
recursive AST pruning/signature comparison dominated its runtime. Prior-value
substitution creates shared subtrees: treating them as an ordinary tree can
expand exponentially. There were also repeated per-column assignment scans,
coverage comparisons and report parsing.

After the changes, a full unprofiled run on this machine completed in 30.667
seconds (an earlier intermediate implementation took 55.143 seconds). These
are local observations, not an SLA or a controlled speedup ratio: profiling
adds overhead, and machine load/cache state affect elapsed time.

That run produced:

- 318 candidate DD rows, with 275 nonempty candidate formulas.
- 2,033 recorded execution steps and 356 parsed write-ledger entries.
- 12 parser failures, retained in the output diagnostics.
- Four targets requiring an ordered workflow because flat expansion is too
  large: CUSTOMERCAL.SysNPA_Dt, CUSTOMERCAL.SysAssetClassAlt_Key,
  CUSTOMERCAL.DEGREASON and ACCOUNTCAL.NPA_Reason.
- Zero ACTIVE rows; source completeness is **not** verified for this procedure.
  The platform CSV/XLSX therefore contain headers only. All 318 candidates
  remain available in `dd_rows.json` and the QA report; they are withheld from
  deployment exports, not silently discarded.

The write count is a parser inventory, not a claim that every SQL operation
was successfully translated. Independent inventory disagreements remain
explicit blockers. Source failures include spaced comparison operators,
comment/statement-boundary handling, and statements containing EXEC or ALTER.

## Implemented approach

1. **Retain evidence.** Batch ingestion stores the original upload bytes.
   Each pipeline run saves submitted SQL text with SHA-256 hashes, all DD rows
   and execution steps in `dd_rows.json`, and source workflow companions.
2. **Prepare once per object.** `MutationSourceIndex` captures scanners and
   groups UPDATE assignments by target column before workers start. It keeps
   source positions, duplicate assignments and original statement indexes.
   It rejects reuse with different SQL. No size-based input truncation is used.
3. **Bound representation size.** Before recursive pruning, count the expanded
   AST size through its shared graph in time proportional to unique nodes.
   Limits are 12,000 container nodes, depth 160 and 200,000 string characters.
   Exceeding a limit yields an explicit validation error and no accepted flat
   formula. Source mutations and ordered execution steps remain available;
   this is a required workflow representation, not a shortened formula.
   Within those limits, pruning reuses shared subtree results.
4. **Remove repeated work.** Index coverage by entity, column and source
   statement. Preserve string-literal case and whitespace when comparing
   statements. Index advisory lookups by column. Reuse bounded caches for
   report SQL parsing, reference names and threshold comment scanning.
5. **Check completeness before acceptance.** Reconcile the independent write
   inventory with parsed writes and matching formula evidence. Structural
   failures, unresolved writes, validation errors and semantic advisories gate
   all rows of the affected object, since an omitted write can affect later
   columns. Retain formulas for review; set PENDING_REVIEW/NEEDS_REVIEW.
   Human approval remains an explicit existing override.
6. **Fix observed omissions.** Recognize CTE-prefixed writes starting with
   `;WITH`. The independent inventory no longer drops real tables whose names
   are one or two characters; resolved aliases are deduplicated by position.

`COMPLETED` remains the job lifecycle status meaning artifacts were produced.
It does not certify business correctness. Consult `completeness.json` and row
review status. The normal batch CLI preserves its historical exit behavior;
the benchmark/check command below returns a nonzero exit code for review.

## Reproduce

```bash
.venv/bin/python -m scripts.benchmark_large_sql \
  --input /path/to/procedure.sql \
  --output-dir output/large_sql_check
```

This does not execute the SQL or call an LLM. Each invocation creates a new job
folder and writes timing/result data to `benchmark.json`. Exit codes: 0 means
coverage checks passed, 1 means generation failed, 2 means review is required.

Inspect these artifacts together:

- `source/original.sql`: exact input bytes for batch runs.
- `source/manifest.json` and numbered SQL files: submitted text and hashes.
- `dd_rows.json`: all candidate rows, source fragments, execution steps and errors.
- `completeness.json`: write-level evidence and unresolved blockers.
- `source_workflow.json` / `.md`: ordered source review specification.
- `qa_coverage_report.md`, `report.md`, `dd_export.csv` and `dd_export.xlsx`.

The workflow companion is **not executable** on the destination platform.
JSON and source SQL retain evidence that presentation tables may summarize.

## Validation and remaining acceptance work

The full suite passed 443 tests after the core implementation. New regressions
cover exponential shared graphs, preservation of all 45 self-dependent writes
and steps, source-index equivalence, wrong-source rejection, literal-sensitive
coverage identity, missing-write gating and fully covered simple writes.
Batch tests additionally check original-byte preservation and complete JSON
row counts. Two already failing tests were updated to their current contracts:
dynamic LIKE raises an explicit unsupported error, and the isolated report
test mocks its database stage update. Existing CTE failures were fixed in code.

To certify this particular procedure's business output, remaining work is:

1. Resolve every listed parser/inventory discrepancy using the preserved SQL;
   distinguish invalid source syntax from parser limitations. Do not silently
   repair operators or discard commented/control-flow blocks.
2. Implement and validate platform workflows for set operations, procedure-wide
   gates, ordered mutations, temporary tables and exception handling.
3. Supply the actual schemas, lookup/reference data, called procedures and
   TIMEKEY calendar mapping. These are necessary to establish SQL semantics
   that cannot be inferred from a standalone procedure.
4. Execute the original SQL in an isolated SQL Server test database and compare
   destination behavior on golden fixtures: nulls, boundary dates, duplicate
   joins, branch precedence, resets and error paths. Check values, affected row
   sets and order-dependent outcomes, not only formula grammar.

Until then, the correct result is explicit review-required output. Neither
this patch nor an LLM can honestly guarantee all arbitrary SQL translations
are correct solely from syntactic validation.
