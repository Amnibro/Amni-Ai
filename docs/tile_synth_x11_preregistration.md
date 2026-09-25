# X11 pre-registration: prompt-steered search (locked 2026-09-25 08:56, before any X11 search code)

This file fixes the splits, the systems, the metrics and the decision rule for X11 before any search code is changed or any X11 pool is looked at. The split files live in `data/code_eval/` (not in git). Their sha256 hashes are in `docs/x11_locked_manifest.json` and `docs/x11_dev_manifest.json`, written by `amni.compute.eval_lock.lock` (`scripts/x11_splits.py lock`).

## Why a new locked split

The X10 locked split (`lcb_easy_fn` + `apps_fn_lock`) was read once by X10 and is spent. Fresh function-level data with hidden tests is scarce:

| source | licence | call-based rows | usable | why |
|---|---|---|---|---|
| APPS train, interview | MIT | 558 | 53 | 488 are LeetCode rows with a single test case (the APPS copy has one test and broken quoting); 17 have no usable signature |
| APPS test, all difficulties | MIT | 38 | 24 | 13 reference solutions fail their own tests in the jail; 1 bad signature |
| BAAI/TACO | Apache-2.0 | 3,298 | 0 | 3,098 share a problem URL with APPS; the other 200 (geeksforgeeks) give tests as C-style text with only the 2 statement examples |
| newfacade/LeetCodeDataset, Easy | Apache-2.0 (dataset card) | 686 | 451 | 48 linked-list/tree; 13 no parsable example; 8 < 5 parsable tests; 8 oracle fails; 158 removed by decontam/dedup (mostly the LiveCodeBench easy problems X10 used) |
| LiveCodeBench medium | card says `cc` without a variant | – | not used | licence not verifiably MIT/Apache/CC-BY |
| CodeContests | CC-BY-4.0 | 0 | 0 | stdin/stdout only |

No X lane has seen APPS interview or APPS test rows, TACO, or LeetCodeDataset.

## Locked split (521 problems): never tuned on, evaluated once per system

| file | rows | construction |
|---|---|---|
| `apps_x11_lock.jsonl` | 70 | APPS train interview call-based (53) + APPS test call-based (24), minus 7 near-duplicates of earlier sets. Same builder as X10's APPS rows (`x10_splits.apps`): full statement as docstring, first 2 test cases (repr ≤ 300 chars) as `>>>` examples with outputs from the reference solution in the jail, all test cases hidden. |
| `lcd_easy_lock.jsonl` | 451 | Every usable LeetCodeDataset Easy problem (train + test files). The `class Solution` method becomes a plain function. Docstring = statement with the "Example" blocks removed and constraints kept. The statement's "Input:" examples (up to 3) become `>>>` examples; their outputs come from running the dataset's reference solution in the jail. Hidden tests = every `assert candidate(...) == literal` in the dataset's `check` (median about 100 per problem), converted to positional calls; the reference must pass all of them in the jail. |

- **Hidden test comparison:** as X10 (tuples = lists, 1e-5 relative float tolerance, bool is not int). APPS rows accept a one-element wrapped output; LeetCode rows do not.
- **Test size caps:** as X10 (256 KB per test, 4 MB per problem).
- **Contamination:** every row was checked with `amni.compute.decontam` (13-gram) against HumanEval, MBPP, `mbpp_fresh`, `mbpp_holdout`, `apps_fn_dev`, `apps_fn_lock` and `lcb_easy_fn`, and deduplicated by (function name, statement head) against the same sets and within the split. The default shared index (`decontam.index()`) now covers the new files. Four dev rows share a 13-gram with the locked split; the shared grams are the alphabet (`a b c d …`) and the "quick brown fox" pangram, so nothing was removed.
- **Guards:** `x5_mine.BENCH` (the X8 query-time corpus guard and the exact-body check) now includes both locked files. The X8 guard's `gdc` check uses the shared index, which includes them too. Any new tile mining must use the shared index.
- **Licence note:** LeetCodeDataset is Apache-2.0 per its dataset card; the statements are LeetCode's. APPS is MIT. The sets are used only for local evaluation and are not redistributed.
- **Eval-only (statement copyright).** Every LeetCode-derived row is evaluation-only: `lcd_easy_lock` (all 451 rows), the 23 `apps_x11_lock` rows with leetcode.com URLs, and `lcb_easy_fn` in dev. They never enter tile mining, training, or any tile, prior or ranker table. The manifests carry an `eval_only` note. `amni.compute.eval_lock.assert_trainable` enforces it: the X5 miner (`mine`, `tables`, `harvest`), the X11 miner and every X11 table trainer assert on their inputs.

## Dev split (2,432 problems): all tuning and training happens here

| file | rows | note |
|---|---|---|
| `mbpp_holdout.jsonl` | 464 | X10 dev |
| `apps_fn_dev.jsonl` | 1,333 | X10 dev |
| `lcb_easy_fn.jsonl` | 135 | X10 locked, spent; LeetCode style; **eval-only**: dev measurement, never training |
| `apps_fn_lock.jsonl` | 500 | X10 locked, spent |

- Hidden-test labels and reference solutions of dev rows may be used for training tables, except for the eval-only `lcb_easy_fn` rows. Any learned component is selected with cross-validation grouped by problem on dev only.
- Iteration may use a fixed seeded dev subsample; the final dev numbers are on the full dev split.

## Systems

- **X9 and X10:** the current code default (commit `36a92c9`: X10 ranker on, `TS_TOPN=16`) with the X10 pre-registered environment:
  ```
  TS_SK=on X0_EVJ=4 TS_XSTORE=data/code_eval/harvested_tiles.jsonl TS_PRIOR=data/pycode/x5_prior_ret.npz TS_PCFG=b=50,idf=0,ga=0,gv=0,gr=0,norm=0 TS_JOBS=4 TS_RANK=af=byte,at=-1.6002203560588433,anl=1
  ```
  One run; the X9 pick is `x9_pick`, the X10 pick is the submitted pick, on the same pools.
- **X11:** the frozen X11 code (prompt-steered search, execution-guided pruning, new tiles; the ranker may be retrained on dev pools) with the same environment and the same per-problem search budget (`TS_BUDGET` 10 s) and the same number of search workers (4).
- Both runs use the X11 runner: every search and every candidate execution runs inside a bwrap jail, with CPU ≤ 6 threads, total RSS < 6 GB and at least 12 GB of system RAM kept available. The X10 system run changes nothing in search or ranking; only the process isolation differs.

## Metrics

- **Primary:** pass@1 on the hidden tests of the locked split (521), X11 against X10. Different searches give different pools, so the comparison is paired by problem (same problem, two systems). An exact two-sided McNemar test on the discordant problems is reported alongside.
- **Secondary:**
  - X9 pass@1 (same run as X10).
  - **No verified candidate:** the number of problems whose pool has no program that passes every prompt example (`ts_found` false), for the X10 run and the X11 run, on locked and on dev. Target: X11 ≤ half of X10.
  - Precision when found, oracle (any candidate right), and per-file breakdown (APPS / LeetCode).
- **Continuity:** HumanEval with HE+ for X10 and X11 (seen by the X lanes; not used for any X11 decision except rule 2 below).

## Decision rule

Adopt X11 as the tile-synthesizer default only if both hold:
1. X11 − X10 ≥ +1 net pass@1 on the locked split;
2. HE+ on HumanEval is not lower than X10's rerun.

Otherwise the default stays X10, and the X11 components stay behind their switches (default off).

## Order of operations

1. Develop on dev only.
2. Freeze and commit the X11 code and every model table the defaults depend on. Record the commit and the code hash.
3. Run the X10 system and the X11 system on the locked split, once each, back to back, and on HumanEval.
4. Record every number through `eval_lock.record(kind="prod")`.

No change of any kind follows the locked runs.
