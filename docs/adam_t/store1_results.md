# STORE-1 results: the exact GF(17) store as the drive (2026-10-06)

Pre-registration: `docs/adam_t/store1_prereg.md` (commit 3db6e4d, before any eval item existed), with Amendment 1 (09:16:50, before test scoring). Every number below cites the file and field it comes from. Test was scored once (commit 070bc72).

## Bottom line
- **Part B passes.** The compact drive answers every v3 fact and name resolution identically: 0 mismatches over all 55,062,201 facts, 8,513,727 qualifiers, 11,900,718 entities and 15,753,155 name keys. It is **410,155,258 bytes, 8.86% of v3's 4,629,551,616**, all in Reffelt-packed GF(17) pages with a bit-exact roundtrip.
- **Part A fails on test.** The resolver beats the current v3 resolver by more than 5x, but misses all four gates. Dev was 0.83 store-answer accuracy; test is 0.67. The hand-written cue table over-fit the dev phrasing.
- **Part C fails on test.** Median 7.5 ms, p99 58 ms. The fingerprint choice (F=0) was made on a dev latency that was measured with warm caches on the same questions; see "What failed".
- **Part D done.** `AMNI_STORE1=1` (default off) feeds store answers to the harness as v4 material. The locked E2E suite was not run.

## Coverage audit (Part A1; separate agent, aggregates only; `data/store1/coverage_audit.md`)

| E2E know (n=102), primary class, strict | n | share |
|---|---:|---:|
| Wikidata entity fact | 33 | 32.4% |
| Wikidata value (date, number, string) | 14 | 13.7% |
| ConceptNet / WordNet | 38 | 37.3% |
| Wikipedia passages | 17 | 16.7% |
| computation | 0 | 0 |
| none | 0 | 0 |

| E2E explain (n=59), primary class, strict | n | share |
|---|---:|---:|
| ConceptNet / WordNet | 17 | 28.8% |
| passages | 40 | 67.8% |
| none | 2 | 3.4% |

- **Multi-label reach (know, strict):** Wikidata 47, concepts 78, passages 101, compute 4.
- **Subject in Wikidata, property missing:** 51 of 102 know items. Most are superlatives, counts and properties not kept.
- **Current v3 resolver (`W.ask`):** answers 23 know items, 21 of them correctly. It misses 26 of the 47 that Wikidata strictly holds.
- **What the drive must hold:** about half of "know" lives in Wikidata, the rest in concepts. Explain needs passages, which are outside this drive.

## Resolver (Part A)

`amni/serve/store1.py` over the drive reader `amni/compute/store1.py` (ctypes to `libadam_store1.so`). Everything is a lookup; there is no matmul.
- **Entity linking:** every span of 1 to 8 tokens is hashed (the v3 64-bit name hash) and looked up in the GF(17) name table in one batch.
- **Candidate score:** span length, capitalisation, name kind, sitelink prior, a type hint from the question noun, and a label case check. Penalties apply for proper-noun runs that are only partly covered and for proper nouns that cannot be linked at all; these are the absence evidence.
- **Property map:** a table keyed by hashed lemma n-grams, conditioned on the wh-word. It is built from 197 Wikidata property labels and 2,234 English aliases (`data/store1/props_en.json`, fetched once, CC0) plus a hand-written CC0 cue table, which reuses `wikifact.WHEN/WHO/NOUN`.
- **Join:** over (entity, property) pairs present in the entity's presence signature, with a type gate (`cognition.qclass/isa`, plus Wikidata P31/P106/P279 labels) and qualifier matching (years, character roles) for multi-valued properties.
- **Other routes, same output format:**
  - ConceptNet / WordNet / Wikidata keys for antonym, synonym, young, group, sound, count, made-of, used-for, colour, where, eat/drink, part-of and can-do;
  - a prime solver, an arithmetic phrase solver and `amni.math`.
- **Output:** a ranked list of (store, key, confidence). The resolver abstains ("not in store") only on absence evidence. Weak answers carry `sure=False`, and the harness hedges them.

### Eval set (`data/store1/evalset/`, CC0)
- **Authoring:** 540 questions written by an isolated agent with no repo or eval access. 25 items that overlapped a locked suite in 6-grams were replaced, and the final check is 0 dirty items (`decontam.json`).
- **Gold:** annotated mechanically from the v3 store. 14 Wikidata items were dropped as pre-registered (v3 value differs or the subject is missing; `annotate_log.json`).
- **Split:** dev 260 / test 266, seed 20261006, sha256-locked in `lock.json`.

### Dev (final frozen configuration, F=0; `data/store1/results/final_dev_dev.json`, `metrics`)

`e2e_acc_instore` 0.8289 · `ent_at1` 0.9085 · `ent_at5` 0.9608 · `prop_acc` 0.9085 · `abstain_prec` 0.7073 (`abstain_n` 41) · `abstain_rec` 0.9062 · v3 baseline `v3_acc_instore` 0.1228.

Ablations on dev (`data/store1/results/ablate.log`):

| config | e2e_acc_instore | ent_at1 | ent_at5 | prop_acc | abstain_prec (n) | absent_false_answer |
|---|---|---|---|---|---|---|
| full, F=3 | 0.8289 | 0.9085 | 0.9608 | 0.9085 | 0.7073 (41) | 0.0938 |
| full, F=2 | 0.8289 | 0.9085 | 0.9608 | 0.9085 | 0.7073 (41) | 0.0938 |
| full, F=1 | 0.8289 | 0.9085 | 0.9608 | 0.9085 | 0.7000 (40) | 0.1250 |
| **full, F=0 (verify by decode), chosen** | 0.8289 | 0.9085 | 0.9608 | 0.9085 | 0.7073 (41) | 0.0938 |
| no popularity prior | 0.8202 | 0.9020 | 0.9608 | 0.9150 | 0.6905 (42) | 0.0938 |
| no type gate | 0.8202 | 0.9085 | 0.9673 | 0.9020 | 0.6905 (42) | 0.0938 |
| no hand cue table (Wikidata aliases only) | 0.4825 | 0.7712 | 0.9216 | 0.5621 | 0.2308 (130) | 0.0625 |
| no Wikidata aliases (hand table only) | 0.8202 | 0.9020 | 0.9608 | 0.9020 | 0.6667 (45) | 0.0625 |
| no ConceptNet/WordNet route | 0.7281 | 0.9085 | 0.9608 | 0.9085 | 0.5000 (60) | 0.0625 |
| no calculator route | 0.7237 | 0.9085 | 0.9608 | 0.9085 | 0.4677 (62) | 0.0938 |
| no qualifier matching | 0.8070 | 0.9085 | 0.9608 | 0.9085 | 0.7073 (41) | 0.0938 |
| no subject-type hint | 0.8246 | 0.9020 | 0.9608 | 0.9020 | 0.7073 (41) | 0.0938 |
| no label case check | 0.8333 | 0.9150 | 0.9608 | 0.9085 | 0.7000 (40) | 0.1250 |

No configuration reached dev `abstain_prec` 0.90, so the selection constraint could not be met (Amendment 1). The hand cue table carries the accuracy; the Wikidata aliases add one or two items.

### Test, scored once (`data/store1/results/test_scores.json`)

| gate | field | pre-registered | test | result |
|---|---|---|---|---|
| GA1 | `e2e_acc_instore` | >= 0.80 | **0.6667** | FAIL |
| GA2 | `ent_at1` | >= 0.90 | **0.8797** | FAIL |
| GA3 | `prop_acc` | >= 0.85 | **0.7722** | FAIL |
| GA4 | `abstain_prec` | >= 0.90 | **0.3881** (`abstain_n` 67) | FAIL |

- **Also reported:** `ent_at5` 0.9684 · `abstain_rec` 0.7429 · `absent_false_answer` 0.2571.
- **By category (accuracy):** wd_entity 0.6696, wd_value 0.7826, commonsense 0.5208, compute 0.72.
- **v3 baseline on the same items:** `v3_acc_instore` 0.1212 (`v3_acc_instore_wd` 0.1772). The drive resolver is 5.5x the current resolver.

Why test is lower than dev (looked at after scoring; nothing was retuned):
- **Property paraphrases missing from the cue table.** Examples: "passed away", "gave birth to" (mother), "first queen", "was behind" (director), "owns", "took over from".
- **Over-eager absence evidence:** 41 in-store abstains. Proper-noun runs such as "Roman Polanski's Chinatown" read as unknown names.
- **Wrong value in multi-valued properties.**

The dev-to-test drop (0.83 to 0.67) is the same hand-rule generalisation gap the harness missions saw.

## Ultra-compact drive (Part B)

Code: `crates/adam-store1` (builder, reader, parity, bench, C ABI). It extends `crates/adam-torus` (generic digit rows, `Table::row`); nothing is copied. Data: `data/store1/drive/` (the build input `data/store1/build/nmenc.bin` comes from `scripts/store1_nmenc.py`).

Design:
- **Facts:**
  - 17-ary canonical Huffman codes per property, over the fact symbol (value or rank, unit, preferred flag, qualifier count).
  - Code digit j of every key is stored in a 1-digit GF(17) ribbon/BuRR level table j (A1 solver: w=64, bumping, spill layer).
  - Codebooks are canonical, lexicographic-delta coded in base-17 EV varints, with block offsets every 32 entries.
- **Presence:**
  - A per-entity signature: the ordered list of (property rank, single/multi/derived), Huffman-cascaded and keyed by entity.
  - A count table for multi-valued groups. There are no per-fact fingerprints, so absence is exact.
- **Qualifiers:** their own cascade, keyed (subject, property, index, j), with per-parent-property codebooks.
- **Strings:** string facts keep a 5-digit string id; the texts live in the text codec.
- **Derivable facts:** rules P17 <- P131 then P17, and P27 <- P19 then P17. A group is flagged derived only if the rule reproduces it exactly, and cycles are removed.
  - Coverage: 1,969,795 groups and 1,970,574 facts (`build.json`, `derived_groups` / `derived_facts`).
  - Net saving against a build without rules: 0.91 MB. Facts save 1,492,737 B; the signature levels cost 111,602 B more and the signature codebook about 0.47 MB more (`drive_meta/build.log` vs `drive_meta/build_noderive.log`). This is above the 0.5 MB bar, so the rules are kept.
  - Parity covers every derived group (rule + exceptions = exact).
- **Names:**
  - (64-bit name hash, chain index) maps to (entity, name kind) in a solved GF(17) table, with chains in v3 order (entity ascending).
  - F=0 stores no fingerprint: a candidate is verified by decoding its names. F>=1 adds fingerprint digits plus an END-marker overlay (17^-9 fingerprint) where a fingerprint false positive would extend a chain.
- **Text:**
  - The record per entity is title kind, label, title suffix or title, then the name list with label/title markers.
  - Codec: a static order-3 LUT context model (446,281 contexts) and rANS, in 16- and 32-entity blocks.
  - Order 2 and 4 are measured larger (`--only text`: 201.3 MB / 177.5 MB vs 175.9 MB).
  - The existing KS coder (ks2-clean1) gets 2.96 bits/byte on a label sample, about the same density as order 3 here. But its random-access unit is a 4 KB page that needs a model pass (about 8.6 ms per label), so it was not used (`results/ks_label_compare.json`).
- **Pages:**
  - Everything is GF(17) digits in Reffelt 17-bit pixel words (`adam_gf17::pack17`), as logical 4096x4096 RGBA pages, mmapped with `MADV_RANDOM`.
  - Byte streams are stored as one 16-bit word per pixel.
  - Container roundtrip and per-page check: `roundtrip_bitexact` true, 7 pages (`build.json`). The ribbon level files are proven by the full parity read-back.

### Bytes per component (F=0; `data/store1/results/sizes.json`, `components`)

| component | bytes |
|---|---:|
| fact levels | 77,159,982 |
| fact codebooks | 22,178,394 |
| qualifier levels / codebooks | 10,734,662 / 2,265,925 |
| presence signature levels / codebook | 17,730,404 / 17,373,573 |
| multi-value counts | 2,641,546 |
| string ids | 1,924,229 |
| sitelinks | 7,035,158 |
| QID map | 9,203,378 |
| text model | 9,372,807 |
| labels + titles | 121,497,239 |
| aliases (name lists) | 40,648,298 |
| strings | 4,358,232 |
| name table F=0 | 66,030,453 |
| **total** | **410,155,258** |
| v3 | 4,629,551,616 |
| **ratio (`ratio_vs_v3`)** | **0.0886** |

- **Other name-table budgets:** F=1 75,946,156, F=2 84,882,364 and F=3 94,256,334 bytes (`other_name_tables_bytes`). Totals would be 9.27% (F=2) and 9.45% (F=3).
- **Bits per fact** (`sizes.json`, `bounds.json`):

| measure | bits per fact |
|---|---:|
| fact levels | **11.21** |
| fact levels + codebooks | 14.43 |
| 0-order entropy of the stored fact symbol given the property | 10.30 |
| per-property codebook bound (the A1-style bound) | 15.02 |
| A1 store | 38.46 |
| all fact-side structures (facts, qualifiers, presence, counts, string ids) | 22.09 |

  The fact levels sit at 1.09x the 0-order entropy.
- **Reverse index** (object to subjects): not built.
  - 0 of 153 in-store dev Wikidata items need an object-to-subject step, against the pre-registered 3% bar. This is partly by construction, because gold keys were annotated forward.
  - Estimated compact size 108 MB (`results/reverse_estimate.json`); v3 spends 344 MB. With it, the total would be about 11%.

### Parity (gate GB1)
- **F=1, 2, 3** (`results/drive_meta/parity_f{1,2,3}.json`): every field 0, namely `parity_groups_mismatch`, `parity_facts_mismatch` (55,062,201 facts), `parity_quals_mismatch`, `parity_labels_mismatch`, `parity_names_mismatch` (name lists), `parity_sitelinks_mismatch`, `parity_qid_mismatch`, `parity_ix_mismatch` and `parity_ents_mismatch` (15,753,155 name groups).
- **Absent-name false-positive rate** (`absent_name_fpr`): F=1 6.9e-3, F=2 4.2e-4, F=3 2.8e-5.
- **F=0** (`results/parity_names_f0.json`): `parity_ents_mismatch` 0 over all 15,753,155 name keys with decode verification, and `hash_collisions_distinct_strings` 0. Facts and labels share the same tables, checked above.
- Python-side spot checks against v3 for names, labels, values and strings: `tests/test_store1.py`.

| gate | field | result |
|---|---|---|
| GB1 | all parity mismatch fields | 0: **PASS** |
| GB2 | `ratio_vs_v3` <= 0.15 | 0.0886: **PASS** |
| GB3 | `roundtrip_bitexact` | true: **PASS** |

### Lookup speed and RAM (`results/drive_meta/bench_f0.json`, `bench_f3.json`; one core; machine load average 5-8 from other jobs)

| measure | F=0 | F=3 |
|---|---:|---:|
| (entity, property) value list, warm | 7.1 us (`values_ns_warm`, about 141k/s/core) | 7.3 us |
| name lookup, warm | 1.0 us (`ents_ns_warm`, plus Python-side verification) | 3.3 us |
| label decode, warm | 15.8 us | 16.5 us |
| presence signature, warm | 0.78 us | 0.81 us |
| value list, cold (after DONTNEED) | 217 us | 207 us |
| name lookup, cold | 123 us | 253 us |
| store open | 160 ms | 80 ms |
| anonymous RAM (text-model tables) | about 85 MB | about 85 MB |

`rss` minus `RssFile` gives the anonymous RAM; the rest of the 1 GB RSS is mapped pages, mostly the v3 arrays the bench samples from.

A value list costs 7 us rather than A1's 0.5 us per key because one answer is a signature decode plus 1-3 cascade levels per fact plus a 32-entry codebook block.

## Speed (Part C)

| gate | field (`test_scores.json`) | pre-registered | test | result |
|---|---|---|---|---|
| GC1 | `lat_median_ms` | <= 5 | **7.53** | FAIL |
| GC2 | `lat_p99_ms` | <= 25 | **58.3** | FAIL |

- **How it was measured:** after a warm-up pass over dev, in a `CPUQuota=100%` scope.
- **Dev for comparison** (`final_dev_dev.json`): median 1.39 / p99 5.6 ms. That figure is optimistic, because dev was both the warm-up and the measured set, so every cache was hot.
- **Post-hoc diagnostic** (dev, warm-up on 15 unrelated questions, no scope): F=0 median 5.1 / p99 22.4 ms; F=3 median 3.1 / p99 15.0 ms. F=0's decode verification is the extra cost on unseen questions.

## Wiring (Part D)
- **Route:** `amni/serve/adam_t_harness.py` gains `fill_s1`, called first in `fill_know` when `AMNI_STORE1=1` (default off). It returns material `mat('store1', [line], …)`, where the line is `"<label>, <property>: <answer>."` or the calculator/concept line. Items are joined by one space, in the v4 format.
  - Answers with `sure=False` become harness hedges ("I think it's X, but I couldn't confirm it.").
  - "Not in store" falls through to the existing path.
- **Blend gate:** `_BG['store1']` uses the know settings.
- **Untouched:** `gguf_runtime.py`. The locked E2E suite was not run.
- **Tests:**
  - `tests/test_store1.py` passes 4 of 4.
  - The existing `test_adam_t_harness*.py` show the same 15 failures before and after the change. They fail identically with the backed-up original harness (`backups/store1/adam_t_harness.py.v4.bak`), so they are pre-existing in this worktree.
  - Rust: 3 crate tests plus the adam-torus ribbon test pass.

## Where the paradigm was left
- **Python on the resolver path:** regex tokenising, lemma lookups (WordNet morphy) and the scoring arithmetic use floats in Python. The stored tables and all lookups are GF(17) LUT reads with integer adds; there is no matmul anywhere.
- **Not GF(17)-native:**
  - The rANS text coder works in binary state; its byte stream is stored as 16-bit words per pixel.
  - The text-model frequency tables are decoded into a RAM hash map at open.
- **A RAM-resident Python dict:** the property cue table, built at load from `props_en.json` plus the hand table. It is not stored as a GF(17) page yet.
- **The type gate's `cognition.isa`** still falls back to the v3 store for P31 labels (reused as-is).

## What failed
- GA1-GA4 on test (0.667 / 0.880 / 0.772 / 0.388) and GC1-GC2 (7.5 / 58 ms).
- Dev abstain precision never reached 0.90 under any configuration.
- **The fingerprint decision (F=0) rested on a cache-hot dev latency.** Measured on unseen questions, F=3 would have met the median gate and F=0 would not.
- **The pre-registered "absent" gold has one known flaw:** at least one dev "untracked property" item (Avatar box office, P2142) is in fact in the store. Absent items with real subjects were not checked against all properties.

## Next steps
1. **Property paraphrase coverage instead of hand cues.**
   - Expand every cue verb and noun through WordNet synsets and derivations: die / pass away / perish; birth / bear / mother.
   - Mine (question n-gram, property) pairs from the Wikidata property examples or descriptions.
   - Measure on a fresh split, since this one is spent.
2. **Make absence evidence stricter:** require an unlinked run to be at least 2 proper tokens with no linked prefix, and drop the single-token unlinkable penalty.
3. **Switch the name table to F=2 or F=3** (+19-28 MB) so no names are decoded on the lookup path. Then measure latency on unseen questions only.
4. **Move the property cue table into the container** as a GF(17) solved table, and decode text-model contexts lazily.
5. **Re-pre-register** with a fresh eval set (the same authoring protocol) before any new test claim.
