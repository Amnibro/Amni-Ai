# STORE-2 results: pixel-nonce text, facts as rank under prediction, nonce resolver (2026-10-06)

Pre-registration: `docs/adam_t/store2_prereg.md` (commit d87216b, before any eval item existed), with Amendment 1 (19:09:19, before test scoring). Eval set committed and sha256-locked in 128fff6. Code frozen in e4c1f96; test scored once in 55e57ee. Every number below cites a saved file and field under `data/store2/results/` unless another path is given.

## Bottom line
| gate | field (file) | pre-registered | result | |
|---|---|---|---|---|
| G1 | `total_bytes` (sizes_f2.json) | <= 300,000,000 | **340,640,415** | FAIL |
| G2 | all `parity_*_mismatch` (parity_f2.json); `roundtrip_bitexact` (build.json) | 0; true | **all 0; true** | PASS |
| G3 | `fact_bits_per_fact` (sizes_f2.json) | <= 7.0 | **9.811** | FAIL |
| G4 | `e2e_acc_instore` (test_scores.json) | >= 0.80 | **0.7895** | FAIL |
| G5 | `ent_at1` (test_scores.json) | >= 0.90 | **0.8889** | FAIL |
| G6 | `prop_acc` (test_scores.json) | >= 0.85 | **0.8655** | PASS |
| G7 | `abstain_prec` (test_scores.json), `abstain_n` 51 | >= 0.90 | **0.6667** | FAIL |
| G8 | `lat_median_ms` (test_scores.json) | <= 5 | **7.562** | FAIL |
| G9 | `lat_p99_ms` (test_scores.json) | <= 25 | **44.866** | FAIL |

- The drive is exact (0 mismatches over all 55,062,201 facts, 8,513,727 qualifiers, 11,900,718 entities and 15,753,155 name keys) and 340.6 MB, 7.36% of v3 (`ratio_vs_v3` 0.0736) and 83.05% of STORE-1's 410.2 MB (`ratio_vs_store1` 0.8305). It is 40.6 MB over the 300 MB gate.
- Facts cost 9.81 stored bits/fact (STORE-1: 11.21). Prediction removes 1.59 ideal bits/fact; 7.0 needs about 2.8 more.
- The resolver reaches 0.79 store-answer accuracy on the new test set (3.8x the v3 resolver's 0.2061, `v3_acc_instore`), with property accuracy 0.8655 (pass). Entity@1 (0.889) and abstain precision (0.667) fail.
- Latency fails on test under a loaded machine (load average about 15 from other sessions): unseen median 7.6 ms, p99 44.9 ms; warm 3.9 / 26.5 ms. The same protocol measured 4.2 / 20.3 ms on dev in the ablation runs (`a_full_dev.json`).

## Eval set (Part E; `data/store2/evalset/`)
- 570 questions written by an isolated agent with no repo, store or eval access (CC0). The first decontamination found 53 items sharing a 6-gram with the locked suites or the STORE-1 set, then 1, then 0 (`decontam.json`, 217 files incl. 3 STORE-1 eval files).
- Annotated mechanically (`annotate_log.json`): 18 Wikidata items dropped (v3 value differs or subject missing); 7 absent items dropped by the strict check (an author answer matched some fact of a candidate entity, or the property phrase mapped to a present PID; e.g. `absent check: prop:Q1650294:P580`). 7 commonsense items have no store key and count as "none".
- Split seed 20261017: dev 268 / test 275, locked (`lock.json`).

## Part T: text layer as pixel nonces
- Vocabulary (`data/store2/vocab/trainer.json`): 83,521 = 17^4 ids, one RGBA pixel each: 512 byte-fallback ids (mid-word and word-final) plus 83,009 BPE pieces (HF `tokenizers` 0.23.2 trainer, `end_of_word_suffix` `</w>`, WhitespaceSplit, `limit_alphabet` 1000, `min_frequency` 2) trained in 41.8 s on 29,185,954 lines of lowercased v3 labels plus v3 normalized names (CC0, corpus sha256 recorded). Merged piece ids equal 1975 + merge rank, so the piece table stores only parent pairs (0.36 MB, section `bpe`).
- Case channel per word: lower, Capitalised, UPPER, segment mask (capital after a non-letter), per-letter mask, raw bytes. Over all labels/titles/strings: 0 roundtrip failures; 1,375 raw labels (`tstat_full.json` `roundtrip_bad`, `raw_fields`).
- Storage forms measured on all text (`tstat_full.json`; MB at 4.25 bits per GF(17) digit):

| field | fixed 4 digits/piece | 0-order 17-ary Huffman | 0-order rANS-17 | context (order-1, t=8) incl. tables |
|---|---:|---:|---:|---:|
| labels | 106.41 | 92.26 | 88.03 | **70.14** |
| title suffixes | 19.24 | 15.77 | 15.03 | **12.30** |
| aliases (no copy model) | 34.07 | 29.64 | 28.29 | **25.41** |
| strings | 5.84 | 4.68 | 4.46 | **4.19** |

- Context coding saves 20% on labels over the 0-order rANS form and 34% over the fixed hot form. Built drive: labels + titles + case channel = 85.49 MB (`sizes_f2.json` `components.labels_titles`, STORE-1 121.50); aliases with a word-copy model from the label and earlier names = 21.51 MB (`aliases_names`, STORE-1 40.65).
- Compute (hot) form: 152,520 entities with >= 40 sitelinks stored one pixel per piece (1.71 MB, `hot_table`). Label decode: hot 1.16 us median vs rANS storage 108 us on the same entities with caches cleared (`hot_vs_storage.json`).
- The rANS coder is base 17 (M = 17^4, state in [17^6, 17^7)): its output stream is GF(17) digits packed 4 per Reffelt pixel; STORE-1's text stream was bytes. Within 1% of the ideal code length (Rust test `rans17_symbols_uniform_roundtrip`).

## Part F: facts as rank under prediction
Each fact codes an aux symbol (preferred flag + qualifier count, context: property, index, previous aux) and a slot: RULE (P17 via P131/P276/P159, P27 via P19/P20, P20 copies P19), MTF j (the j-th most recent fact of the property in the 32-entity block), CTX j (rank in a stored top-8 list for the subject's signature, P31, or P31 value), NEAR (value = subject index +-8), NEWSTR (first use of a string), CLS (the value's country equals the subject's: rank inside that country's list), or ESC (class symbol + rank in the property's class list). Ranks are coded as log2 bucket + uniform offset. All entropy coding is rANS-17 in 32-entity blocks; codebooks are rANS-coded too (12.56 MB vs STORE-1's 22.18 MB).
- Lossless with cycle checks: predictor references form a graph; a DFS over all 55M facts downgraded 84 facts that closed a cycle (`build.json` `cycles_broken`). Decoding is lazy: symbols decode without values, values resolve on demand.
- Bits per fact, measurement tool over all facts (ideal entropy of the coded symbols; `fstat_*.json`, block 64, rank-bucket payload):

| configuration | ideal bits/fact | change |
|---|---:|---:|
| no predictor (per-property rank buckets) | 10.515 (`fstat_abl_none.json`) | |
| all predictors | 8.926 (`fstat_k8_nocyc_b64.json`) | -1.589 |
| minus class lists (CLS) | 9.275 | +0.349 |
| minus context lists (CTX) | 9.231 | +0.305 |
| minus rules | 9.100 | +0.175 |
| minus MTF recency | 8.988 | +0.062 |
| minus aux context | 8.983 | +0.058 |
| minus NEAR | 8.978 | +0.052 |
| explicit per-entity country symbol instead of rules (block 32) | 9.145 (`fstat_k8_ectry.json`) | worse; costs 7.5 bits/entity |

- Per property (`fstat_k8_nocyc.json` vs `fstat_abl_none.json`, ideal bits/fact): P31 4.93 (6.92), P1082 13.90 (13.96), P569 15.63 (15.64), P106 7.97 (8.05), P131 12.78 (15.02), P19 13.45 (13.55), P161 14.84 (16.82), P570 15.92 (15.94), P54 11.17 (14.23), P69 11.15 (12.86). Dates and populations are almost unpredictable by these predictors; P31, P131, P54, P161 gain most.
- Built drive: 9.811 stored bits/fact (`sizes_f2.json` `fact_bits_per_fact`; the fact share of the core stream from the encoder's own bit accounting, `core_bits`, including flush and index); 11.991 with codebooks and context lists (`fact_bits_with_tables`). STORE-1: 11.21 / 14.43.
- TORUS-A2b as a ranker: it covers 151,898 P19, 80,558 P131, 67,173 P17 and 129,796 P171 facts (2.7-24% of each). Even if every covered fact cost 1 bit, it would save at most 0.88 MB (`torus_bound.json` `upper_bound_saving_bytes`), while its layer is 25.35 MB. Not used. TORUS-C was not used (CTO audit).
- Inverse/symmetric mirrors are not exploited directly (the decoder has no reverse index); NEAR and MTF catch adjacent series members.

## Part P: the drive (`sizes_f2.json`, `build.json`)
| component | STORE-2 bytes | STORE-1 bytes |
|---|---:|---:|
| fact stream | 67,524,176 | 77,159,982 (fact levels) |
| fact codebooks | 12,562,137 | 22,178,394 |
| fact context lists | 2,448,079 | - |
| signatures + counts | 28,231,977 | 37,745,523 (sig levels + codebook + counts) |
| header: QID gaps + sitelinks | 11,829,305 | 16,238,536 (QID map + sitelinks) |
| qualifiers | 14,114,931 | 13,000,587 |
| labels + titles | 85,485,451 | 121,497,239 |
| aliases / name lists | 21,514,761 | 40,648,298 |
| strings (+ ids) | 4,117,586 | 6,282,461 |
| model tables (all streams) + case classes | 5,868,078 | 9,372,807 (text model) |
| piece table | 356,186 | - |
| hot table | 1,706,310 | - |
| name table | 84,875,881 (F=2) | 66,030,453 (F=0) |
| **total** | **340,640,415** | **410,155,258** |

- Container `store2.gf17pg`: 8 logical 4096^2 RGBA pages, `roundtrip_bitexact` true. Name tables: F=2 84,875,881 B, F=3 94,266,879 B (`build.json` `name_tables`); keys are 64-bit hashes of piece-id sequences, 0 key collisions (`piece_key_dups`).
- Parity (`parity_f2.json`): `parity_groups/facts/quals/labels/names/strings/sitelinks/qid/ix/hot/ents_mismatch` all 0 over 55,062,201 facts, 8,513,727 qualifiers, 11,900,718 entities, 152,520 hot labels and 15,753,155 name keys; absent-name false-positive rate 4.12e-4.
- Reverse index: not built. 0 of the in-store dev Wikidata items needed an object-to-subject step (gold keys are forward by construction, as in STORE-1).
- Lookup speed (`bench_f2.json`, random entities, one core, block caches mostly cold): value list 123 us, label 55 us, groups 24 us per call on an uncached block; store open 146 ms.

## Part R: resolver
`amni/serve/store1.py` in STORE-2 mode (`AMNI_STORE2=1`): span keys and property cue keys are piece-nonce hashes computed by the drive (`s2_keys`); the property map is mined (`data/store2/props_mined.json`: 197 properties, 24,115 keys from labels, aliases, descriptions, WordNet synonyms/derivations/glosses, plus answer classes who/org/where/when/how/what mined from the store) plus the STORE-1 hand table as a supplement; absence needs an unlinked proper-noun run of >= 2 tokens that no multiword linked span covers (all but one token); the class check uses only the drive (P31 -> P279 labels), not the v3 store; the top-3 candidates are verified against their decoded names (catches F=2 fingerprint false positives); a hedged type-fitting fallback answers time/place questions whose cue matched nothing.

### Dev changes after the first dev result (all on dev only; file `<name>_dev.json`)
| time | run | change | e2e / ent@1 / prop / abstain_prec |
|---|---|---|---|
| 18:42:49 | d01_full | first dev result | 0.7289 / 0.8659 / 0.8232 / 0.6538 |
| 18:44:51 | d02 | partial proper-noun coverage no longer forces abstain; fallback limited to time/place; standalone WordNet synonyms/derivations; "what was the date" = when; latest point-in-time tiebreak; warm-up loads commonsense indexes | 0.7333 / 0.8598 / 0.8232 / 0.6792 |
| 18:53:03 | d03 | new drive (header stream); class check from the drive only | 0.7289 / 0.8720 / 0.8110 / 0.6545 |
| 18:57:48 | d04 | which/what typed by WordNet noun class; top-3 name verification | 0.7289 / 0.8598 / 0.8293 / 0.6607 |
| 18:58:16 | d05 | verification uses the linked key string | 0.7289 / 0.8598 / 0.8293 / 0.6607 |
| 18:59:30 | d06 | "org" answer class mined; verb unigrams; wh-conditioned stop-verbs | 0.7333 / 0.8659 / 0.8354 / 0.6727 |
| 19:01:32 | d07 | label-case check only when the question uses capitals | 0.7422 / 0.8902 / 0.8354 / 0.6852 |
| 19:02:07 | d08 | uncovered-token absence rule (reverted: abstain_prec fell) | 0.7511 / 0.8902 / 0.8354 / 0.6111 |
| 19:02:35 | d09 | run counts as covered by a multiword span covering all but one token | 0.7511 / 0.8902 / 0.8354 / 0.7200 |

### Dev ablations (`ablate_a.log`, `a_<name>_dev.json`; unseen latency on a loaded machine)
| config | e2e_acc_instore | ent_at1 | prop_acc | abstain_prec (n) | absent_false_answer |
|---|---|---|---|---|---|
| full (mined + hand + templates) | 0.7511 | 0.8902 | 0.8354 | 0.7200 (50) | 0.1628 |
| **no templates (selected)** | **0.7556** | **0.8963** | **0.8476** | **0.7347 (49)** | 0.1628 |
| mined only (no hand supplement) | 0.6400 | 0.8720 | 0.7744 | 0.4684 (79) | 0.1395 |
| hand only (STORE-1 table) | 0.7022 | 0.9085 | 0.7683 | 0.6230 (61) | 0.1163 |
| no WordNet expansion | 0.7467 | 0.9024 | 0.8293 | 0.6981 (53) | 0.1395 |
| no descriptions | 0.7511 | 0.8902 | 0.8476 | 0.7200 (50) | 0.1628 |
| STORE-1 absence rule | 0.7422 | 0.8902 | 0.8354 | 0.6667 (57) | 0.1163 |
| no fallback | 0.7422 | 0.8841 | 0.8232 | 0.6545 (55) | 0.1628 |
| fallback also for numbers | 0.7511 | 0.8902 | 0.8354 | 0.7111 (45) | 0.2558 |
| no latest-value tiebreak | 0.7511 | 0.8902 | 0.8354 | 0.7200 (50) | 0.1628 |
| no popularity prior | 0.7289 | 0.8415 | 0.8354 | 0.7200 (50) | 0.1628 |
| no type gate | 0.6978 | 0.9085 | 0.7988 | 0.6271 (59) | 0.1395 |
| no ConceptNet/WordNet route | 0.6800 | 0.8902 | 0.8354 | 0.5441 (68) | 0.1395 |
| no calculator route | 0.6667 | 0.8902 | 0.8354 | 0.5294 (68) | 0.1628 |
| F=3 | 0.7511 | 0.8902 | 0.8354 | 0.7200 (50) | 0.1628 |

- Selection (pre-registered rule; recorded in Amendment 1): no configuration reaches abstain_prec 0.90; within one dev item of the best e2e, no-templates has the highest abstain_prec, so templates are off. F=2 meets the dev latency gates (4.19 / 20.34 ms, `a_full_dev.json`) and ties F=3, so F=2.
- The mined map is not enough alone: mined-only and hand-only reach the same property accuracy (0.774 vs 0.768), but each misses phrasings the other covers, and the combination is best. The hand table therefore remains a supplement that still adds 0.11 dev e2e.

### Test, scored once (`test_scores.json` from `final_test_test.json`)
- `e2e_acc_instore` 0.7895 · `ent_at1` 0.8889 · `ent_at5` 0.9591 · `prop_acc` 0.8655 · `abstain_prec` 0.6667 (`abstain_n` 51) · `abstain_rec` 0.7234 · `absent_false_answer` 0.2766 · `v3_acc_instore` 0.2061 (`v3_acc_instore_wd` 0.2749).
- By category: wd_entity 0.7966, wd_value 0.7925, commonsense 0.7353, compute 0.8261; absent items abstained 0.7317.
- Pre-registered mined-only report on test (`mined_only_test_test.json`, not selected): 0.6228 / 0.8363 / 0.7778 / 0.4271.
- The dev-to-test gap is small (0.756 dev -> 0.790 test e2e; 0.848 -> 0.866 prop); the STORE-1 hand-rule collapse (0.83 -> 0.67) did not recur. Abstain precision is the main failure: 17 in-store items abstained on test, and 13 of the 47 not-in-store items were answered (11 of the 41 absent-category items).

### Latency (Part C; test_scores.json; one core, CPUQuota=100%, load average about 15 from other sessions)
| protocol | median ms | p99 ms |
|---|---:|---:|
| unseen (gated): caches cleared per question, page cache warm (Amendment 1) | 7.562 | 44.866 |
| warm (second pass) | 3.925 | 26.493 |
| cold (page cache dropped per question) | 67.786 | 285.984 |

The same unseen protocol gave 4.19 / 20.34 ms on dev (`a_full_dev.json`); test ran at higher machine load. Decoder work per question is about 20-40 block decodes (25-100 us each) plus Python resolver code.

## Part D: wiring
- `AMNI_STORE2=1` (default off): `amni/serve/adam_t_harness.py` `fill_s1` routes to the STORE-2 resolver and emits v4 material `mat('store2', ['<label>, <property>: <answer>.'], ...)` (items joined by one space); hedges `sure=False` answers; `_BG['store2']` uses the know settings. `AMNI_STORE1=1` still serves STORE-1. `gguf_runtime.py` untouched; the locked E2E suite was not run.
- `amni/compute/store1.py`: `Drive(..., pre='s2')` binds the `s2_*` C ABI (`keys`, `pieces`, `clear`, `cold` added) and `drive2()`; reads `AMNI_STORE2_DIR/LIB/FD` (default `data/store2/drive`, F=2).
- Tests (main-tree environment, `.venv-linux`): main-tree checkout (its own code) `test_store1.py` + harness suites 104 passed; this branch's code with the main tree's untracked `scripts/neutral_filter.py` visible: `test_store1.py`, `test_store2.py`, `test_adam_t_harness.py`, `test_adam_t_harness4.py` 109 passed; a pure worktree run fails 15 harness tests only on `ModuleNotFoundError: scripts.neutral_filter` (an untracked shared-tree file). Rust: 4 crate tests pass (3 STORE-1 + rANS-17 roundtrip).

## Where the paradigm was left
- The resolver is Python with regex tokenising, WordNet lemmas and float scores (as STORE-1). No matmul anywhere.
- The rANS coder's state is a binary 32-bit integer; its output is GF(17) digits in Reffelt pixels. Model frequency tables (147k) and decoded blocks live in RAM hash maps at run time, not in pixel pages.
- BPE pieces were trained with the HF `tokenizers` library (no model weights). Property answer classes and fact counts were mined offline from the v3 store.
- The commonsense route (`cognition.know` fallback) can still consult the v3 store; the class check no longer does.
- Pixel packing costs 4.25 bits per digit (4% over log2 17) on every component.

## What failed
- G1 size: 340.6 MB vs 300. The name table at F=2 (84.9 MB, required F in {2,3}) and labels (85.5 MB) dominate.
- G3 bits/fact: 9.81 vs 7.0. Dates, populations and cast lists are close to incompressible for these predictors; class coding conflicts with the country rules (cycles), so only one of them can carry a subject's country.
- G4, G5, G7 on test (0.7895, 0.8889, 0.6667): property paraphrases that neither WordNet nor aliases cover ("lose his life", "official tongue", "comes after"), inverse relations, multi-valued answers and over-eager answers on absent items.
- G8/G9 latency on test (7.6 / 44.9 ms) at machine load about 15; warm it is 3.9 / 26.5 ms.
- Absent verification still cannot catch absent items whose subject holds a closely related fact (Titanic's worldwide box office for an opening-weekend question).

## Next steps
1. Size: a signature dictionary for the frequent property sets (about -4 MB), order-2 label contexts and entity-class contexts for title suffixes (est. -8 to -12 MB), block size 64 for the core stream (-2 MB). Even so, 300 MB needs the name table below F=2 or a cheaper entity code in it.
2. Facts: inverse/mirror facts through a small reverse map for P155/P156/P40/P26; date offsets (P570 from P569); conditioning ESC buckets on the subject's signature.
3. Resolver: learn property paraphrases from the store itself (question templates generated from each property's top value classes), a direction model for inverse relations, and an abstain model on dev that combines cue weight, link margin and type fit.
4. Latency: cache decoded header/core blocks across questions in the serving process and move the resolver's scoring loop into the Rust drive.
5. Re-pre-register with a fresh eval set before any new test claim; this split is now spent.
