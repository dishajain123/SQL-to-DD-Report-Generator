# Conditional Json fix — changes so far and what remains

Date: 2026-09-21

## Problem

DD conditions shown on the frontend were wrong. Two separate symptoms were
reported and investigated:

1. Conditions appeared as `condition1 AND condition2 ...` whereas the platform
   syntax is `AND(condition1,condition2,...)` (same for `OR(...)`).
2. The **Conditional Json** column was wrong. The Display Derivation Expression
   column was reported as correct.

## Findings

Ground truth is `samples/derivations/sample_derivations.csv` (the platform's
own export). It must be read with `csv.DictReader(..., skipinitialspace=True)`;
the file is space-padded, and without that flag the quoted expressions are
mis-split into extra cells.

- The real export uses the call form: 6 rows use `AND(...)`, 1 uses `OR(...)`,
  none use infix.
- The generator emits infix (`a AND b`). Sources: `_render_sql_condition_to_4x`
  (`f"{left} AND {right}"` / `f"{left} OR {right}"`) plus about eight other
  `" AND ".join(...)` sites in `app/derivation/dd_generation_engine.py`.
- Nothing prevents it: `app/grammar/fourx_grammar.lark` accepts both `and_call`
  and `and_op`; the prompts (`dd_generation.yaml`, `retry_with_error.yaml`)
  never ask for `AND(...)`; and `samples/platform_docs/4x_functions_operators.md`
  lists infix as valid.
- **Root cause of the wrong Conditional Json.** `_condition_links_from_guard`
  split only the top-level ` AND ` and then treated each parenthesised group as
  one comparison, so `(a <= x AND b >= x AND ...) AND (c == d)` became a single
  link whose `value` held the rest of the condition text. `OR` was never split.
  `AND(a,b)` form was also unsupported (wrapper stripped, arguments not split).
  In the stored run `sample-job-1`, 15 of 49 rows with Conditional Json were
  affected — under the current *infix* output, not only under `AND(...)`.
- Link `name`/`type` did not follow the platform convention (81 real links
  checked): direct column `"E"."Col"` is `name == Col`, `type ENT`; the
  generator emitted `name: "E"`, `type: "REL"`.

## Changes made

All in `app/derivation/dd_generation_engine.py`:

- `_and_leaf_conditions` (new): parses a guard with the project's Lark grammar
  and flattens `AND(...)`, infix `AND`, parentheses and any nesting into atomic
  conditions. Returns `None` on an unparseable guard.
- `_condition_links_from_guard`: uses the above; falls back to the legacy text
  split only if parsing fails.
- OR groups become one intact `EXPR` link (a platform link list is an implicit
  AND and has no OR connector).
- `_reference_operand` / `_strip_outer_parens` (new): a link is only built for
  a real column reference (or `COALESCE(ref, literal)`); computed operands such
  as `(End - Start) >= 90` become an `EXPR` link instead of a wrong column.
- `_column_link_name`: platform `name`/`type` convention (ENT / TEMP `var` /
  REL foreign-key path).
- `_parse_condition_link`: routed through the helpers above; quoted literals
  are unquoted, references/expressions keep their text.
- Tests: 6 regression tests added to
  `tests/unit/test_process_metadata_and_dd_json.py`. Full suite: 443 passed
  (437 before).

Effect on the 72 stored rows of `sample-job-1`, re-derived: rows with a garbled
link went from 15 to 0. 40 links are still flagged by the audit heuristic but
are legitimate comparisons whose right-hand side is an expression, e.g.
`DPD_NoCredit >= COALESCE("...","DPD_IntService", 0)`, kept as text.

The Display Derivation Expression column was **not** changed.

## Remaining

1. **Expression column still infix.** The real export uses `AND(...)` /
   `OR(...)`; the generator emits infix (16 of 72 stored rows). Reported as
   "coming right", so left alone pending confirmation. If needed:
   - add a `canonicalize_logical_operators` pass at the end of
     `_finalize_platform_expression` (flatten chains into `AND(c1,c2,...)`);
   - make the validator reject infix `AND`/`OR` so the LLM retry path fires;
   - update `dd_generation.yaml`, `retry_with_error.yaml` and
     `4x_functions_operators.md` to require the call form;
   - update the ~26 test assertions across 7 files that expect infix.
2. **OR representation in Conditional Json.** The real Decision Table JSON
   encodes OR-of-AND groups as a lookup link (`IN` with an empty value). We
   emit a single `EXPR` link holding the OR text. Needs a real example to
   emulate faithfully.
3. **Expression-valued right-hand sides.** Comparisons like `A >= COALESCE(B,0)`
   or `A == B` keep the right side as text in `value`. No real example shows
   how the platform models column-to-column comparisons.
4. **Should Conditional Json exist for Formula rows at all?** In the real export
   Conditional Json is empty in all 31 rows (including formula rows with AND);
   only Decision Table Json is populated. The generator emits Conditional Json
   for every IF-based formula row.
5. **Existing jobs are not regenerated.** The DB and `output/` files still hold
   the old Conditional Json; re-run a job to get the corrected output.
6. **Job `job-497d33b623` is `FAILED` with no `error_message` and no rows.**
   Not investigated; the audit used the mock-LLM run `sample-job-1`.
7. **Minor:** `README.md` passes `include_dd_excel=True` to `JobPlan`, which has
   no such field (pydantic silently ignores it); `sample_derivations.csv` is
   padded and needs `skipinitialspace=True` to parse.
