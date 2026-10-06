# STORE-2 pre-registration: pixel-nonce text, facts as rank under prediction, nonce resolver

Written 2026-10-06, before any STORE-2 eval item exists and before any resolver change is run on eval data. Mission STORE-2 (CEO-approved design 2026-10-06). Lineage: `store1_prereg.md`, `store1_results.md` (STORE-1; its eval split is spent and is never used again for selection or test), `torus_a2b_results.md`, `torus_c_results.md` (including the CTO audit). Outputs go to `data/store2/` (peak cap 15 GB). Version v6.24.0.

## What was looked at before this file was written (disclosure)
- STORE-1 prereg, results and code, including the failure phrasings quoted in its results: "passed away", "gave birth to", "first queen", "was behind", "took over from", "owns", the "Roman Polanski's Chinatown" absence false positive and the mislabelled "Avatar box office" absent item. I have not opened the per-item rows of the STORE-1 dev/test files or of any locked suite, and I will not.
- CTO store measurements (no eval data): names text 18,715,359 names, 73.1M words, 2.71M distinct words; an 83,521-piece BPE gives 1.02 pieces/word, 152 MB at 4 fixed digits, 130 MB with a 0-order 17-ary Huffman code; 4,913 pieces lose (158 MB); 57,737 characters fall outside a 400-character alphabet. Per-property 0-order bits/fact: P31 6.74, P1082 13.25, P569 15.41, P106 7.89, P131 14.42, P19 13.49, P161 16.12. About 890k inverse/symmetric mirror facts (~2 MB).
- Store-only statistics measured while building (no eval data) are reported in the results with their files.

## Part T: text layer as pixel nonces
- Vocabulary: 83,521 = 17^4 pieces, so one piece id is one RGBA pixel (4 GF(17) digits, Reffelt `[1,17,289,4913]`). Ids 0-255 are byte-fallback pieces (raw UTF-8 bytes); the rest are BPE pieces trained (HF `tokenizers` BPE trainer, a library, no model) on licence-clean lowercased text only: Wikidata normalized names and lowercased labels/aliases (CC0). The trainer config (vocab size, pre-tokenizer, word-end marker, alphabet limit, min frequency, training corpus and its sha256) is recorded in `data/store2/pieces/trainer.json`. Pieces never cross a word boundary and word-final pieces carry an end-of-word marker, so a piece sequence is self-delimiting.
- Every label, title, alias, normalized name and string fact is a piece sequence. Labels keep case through a case channel (per word: lower, Capitalised, UPPER, or explicit per-character mask); any word whose case/space/Unicode roundtrip is not exact is stored as byte-fallback pieces. Lossless is proven by full parity.
- STORAGE form, measured three ways and the smallest exact one kept: (a) fixed 4 digits per piece; (b) 0-order 17-ary canonical Huffman digit codes packed 4 digits per pixel (several common pieces per pixel); (c) context coding: each piece predicted from the previous piece(s) and the field type (label, title suffix, alias, string), with escape to the field's 0-order model, entropy-coded by a base-17 rANS coder whose output stream is GF(17) digits (no binary state leaves the coder). Reported: bits per piece and MB for (a), (b), (c), and the model-table bytes for (c).
- COMPUTE form: a fixed one-pixel-per-piece hot table (piece id -> spelling table, plus the labels of the hottest entities as one pixel per piece) for direct texture-fetch decode. Its bytes count in the total; decode latency hot vs storage is reported.
- The same vocabulary is the nonce space of the resolver: questions are cut into the same pieces, and name keys are hashes of piece-id sequences.

## Part F: facts as rank under prediction
- Each fact value is coded as its position in a predicted candidate list (ranks 0..K-1), or an escape followed by the value's rank in the property's value codebook (bucketed by frequency, offset within bucket uniform). The preferred flag and qualifier count are a separate small symbol with its own context. Entropy coding: base-17 rANS in entity blocks (random access through a block offset index), or the STORE-1 17-ary Huffman ribbon cascade over the rank symbol; both are measured and the smaller exact one is kept.
- Predictors (integer/LUT only, no matmul), each measured as bits/fact saved when added, per property:
  1. rules: P17 from P131's P17; P27 from P19's country; P570 as a year offset from P569; inverse/symmetric mirrors (P155/P156, P26, P3373, P1365/P1366, P22/P25 vs P40) where the decoder can reach the mirror;
  2. subject context: per-property value frequency given the subject's presence signature or its P31 value (stored top-k lists per context, charged to the tables);
  3. block neighbours: the same property's value on the preceding entities of the decode block (batch-imported items share values);
  4. index proximity: an entity value close to the subject's own index (series, episodes, parts);
  5. the TORUS-A2b relational torus (`data/torus/a2b/layer2_test_final.gf17l2p`) as a ranker for P131/P17/P19 where its 615,156 entities cover the fact; it is kept only if its measured saving exceeds its own bytes. TORUS-C geometry is not used (CTO audit).
- Lossless: rank plus predictor must reconstruct every fact exactly (full parity below).

## Part P: everything else in the drive
- Presence (property signature, multi-value counts, rule flags), qualifiers, sitelinks, QID map and string ids are re-coded with the same coder where it measures smaller; otherwise the STORE-1 component is kept. Each component's bytes are reported against STORE-1.
- Name table: (hash of the name's piece-id sequence, chain index) -> (entity, kind) in a GF(17) solved table with F = 2 or 3 fingerprint digits plus the END overlay (STORE-1 design). F is chosen by the dev latency rule below. F=0 (decode verification) is not a candidate.
- Reverse index: built only if at least 3% of in-store dev Wikidata items need an object-to-subject step (STORE-1 rule).

## Part R: resolver on nonces
- Entity linking: every word span (1-8 words) of the question becomes a piece-id n-gram nonce key looked up in the name table; candidates ranked by span length, name kind, sitelink prior, a type hint, and case evidence (STORE-1 features).
- Property mapping (replaces the hand cue table as the main source), a table keyed by hashed lemma n-gram nonces conditioned on the wh-word:
  1. mined from Wikidata property labels, aliases and descriptions (`data/store1/props_en.json`, CC0; refetched for any added property);
  2. WordNet synonym and derivation expansion of the property words (die -> pass away/perish/decease; mother -> bear/give birth; owner -> own/possess);
  3. question-pattern templates generated mechanically from each property label (who is the X of; what is the X of; when was S <verb>ed; where was S <verb>ed; how many/much X);
  4. a small CC0 hand supplement (the STORE-1 table, trimmed); its ablation is reported, and the mined sources must carry the result: the "mined only" configuration is reported on test as well.
- Stricter absence: before abstaining, the resolver checks every property of the linked entity and of the top-k (k=5) candidates for a value of the asked answer type; it abstains only when no candidate holds such a property, or the subject span does not link at all and at least two proper tokens are unlinked. Single unlinked tokens no longer count as absence evidence.
- Non-Wikidata routes (ConceptNet/WordNet keys, prime/arithmetic solver, `amni.math`) are reused from STORE-1 with the same output format.
- Output: ranked (store, key, confidence); "not in store" is explicit.

## Part E: new eval set (CC0, written from scratch, then frozen)
- At least 500 natural questions written by a separate agent that sees no repo code, no store, and no eval data. Target mix: Wikidata entity-valued relations >= 200 (at least 30 relations, at least 3 different phrasings per relation, including "passed away", "gave birth to", "owns"-type phrasings, possessives and inverted word order), Wikidata dates/numbers >= 90, commonsense (ConceptNet/WordNet) >= 70, computation >= 40, truly absent >= 80 (fictional subjects, and real subjects asked for an attribute the author believes untracked, each with the true answer when the author knows it).
- Gold = (store, key, accepted answers), annotated mechanically from v3 as in STORE-1 (`store1_evalset.py` method). Wikidata items whose v3 value differs from the author's answer, or whose subject is missing, are dropped and logged, not relabelled.
- Absent verification (fixes the STORE-1 flaw): for every absent item, every proper-name span that resolves in v3 is collected; ALL facts of ALL candidate entities of those spans are rendered and matched against the author's accepted answers, and the author's property phrase is mapped to Wikidata PIDs by exact label/alias match and checked for presence on every candidate. Any hit drops the item from the set (logged).
- Decontamination: zero shared 6-grams (`amni.compute.decontam`, n=6) with every json/jsonl under `data/adam_t/eval/**` (locked chat/E2E suites, probe8, all reports, which hold the CTO's 30 hybrid failure questions in `report_h1-hybrid-qwen3-4b-q8-C8_0_e2e_tune_dry.json`) and with `data/store1/evalset/*.jsonl`; checked by script, dirty items rewritten by the author agent.
- Split: dev/test 50/50 stratified by category, seed 20261017, written once to `data/store2/evalset/{dev,test}.jsonl` with a sha256 lock (`lock.json`) before any resolver code runs on it. Test is scored exactly once, with frozen code, by `scripts/store2_eval.py --split test`, writing `data/store2/results/test_scores.json`; the script refuses a second test run.

## Metrics (field names fixed here; same definitions as STORE-1)
- `e2e_acc_instore`, `ent_at1`, `ent_at5`, `prop_acc`, `abstain_prec` (+ `abstain_n`, `abstain_rec`, `absent_false_answer`), per-category accuracy, and the v3 resolver baseline, exactly as defined in `store1_prereg.md`.
- Latency, question to material, CPU, one process, store mmapped, scope `CPUQuota=100%`:
  - `lat_median_ms`, `lat_p99_ms` (gated, "unseen" protocol): fresh process; store opened; a warm-up of 40 fixed script-generated questions that name no real entity (fictional names, arithmetic); then every eval question once, in file order, with all in-process caches (Python LRU caches, decoded-text caches) cleared before each question. The OS page cache is not dropped (cold-ish).
  - `lat_cold_median_ms`, `lat_cold_p99_ms` (reported): as above, plus `posix_fadvise(DONTNEED)` on every drive file and `madvise(DONTNEED)` on the maps before each question.
  - `lat_warm_median_ms`, `lat_warm_p99_ms` (reported): a second pass over the same questions.
- Sizes (`data/store2/results/sizes.json`): `total_bytes` = every file the reader opens (container, all level/table files, name table at the chosen F, hot table, piece table, models, codebooks). `fact_bits_per_fact` = 8 x (bytes of the per-fact fact stream: rank/escape symbols, escaped in-codebook offsets, preferred/qualifier-count symbols, and that stream's block offset index) / 55,062,201 (all v3 facts, rule-derived ones included, as in STORE-1). Value codebooks and predictor tables are `fact_tables_bytes`, reported with `fact_bits_with_tables`.
- Parity (`data/store2/results/parity.json`): `parity_groups_mismatch`, `parity_facts_mismatch` (all 55,062,201), `parity_quals_mismatch` (all 8,513,727), `parity_labels_mismatch` (label and title), `parity_names_mismatch` (name/alias lists), `parity_strings_mismatch`, `parity_sitelinks_mismatch`, `parity_qid_mismatch`, `parity_ix_mismatch` (all 11,900,718 entities), and `parity_ents_mismatch` over every distinct v3 normalized name string (content and order of the entity list equal to v3 `ents`); `roundtrip_bitexact` for the pixel container.

## Gates
| gate | field | file | pass |
|---|---|---|---|
| G1 | `total_bytes` | sizes.json | <= 300,000,000 |
| G2 | every `parity_*_mismatch`; `roundtrip_bitexact` | parity.json | all 0; true |
| G3 | `fact_bits_per_fact` | sizes.json | <= 7.0 (with-tables figure reported) |
| G4 | `e2e_acc_instore` | test_scores.json | >= 0.80 |
| G5 | `ent_at1` | test_scores.json | >= 0.90 |
| G6 | `prop_acc` | test_scores.json | >= 0.85 |
| G7 | `abstain_prec` | test_scores.json | >= 0.90 (needs `abstain_n` >= 10, else not evaluable) |
| G8 | `lat_median_ms` | test_scores.json | <= 5 |
| G9 | `lat_p99_ms` | test_scores.json | <= 25 |

## Dev protocol
- Ablations on dev only: full; mined only (no hand supplement); hand only; minus WordNet expansion; minus templates; minus descriptions; STORE-1 absence rule instead of the strict one; minus popularity prior; minus type gate; minus non-Wikidata routes; F=2 vs F=3.
- Selection: highest dev `e2e_acc_instore` subject to dev `abstain_prec` >= 0.90. If no configuration meets the constraint, the STORE-1 Amendment-1 rule applies (pre-registered here): among configurations within one dev item of the best `e2e_acc_instore`, the highest `abstain_prec`; then fewer bytes, then lower latency.
- F: F=2 if its dev "unseen" latency meets both latency gates and its dev `e2e_acc_instore` and `abstain_prec` are within one item of F=3; otherwise F=3.
- Every change after the first dev result is logged with a timestamp in the results. Any change to a gate, metric or protocol is an amendment written here with a timestamp and whether it helped.

## Wiring
- `AMNI_STORE2=1` (default off) makes the harness `fill_s1` route read the STORE-2 drive and resolver and emit v4 material (`Label: content` items joined by one space). `amni/serve/gguf_runtime.py` is untouched. The locked E2E suite is not run. `tests/test_store1.py` (and a new `tests/test_store2.py`) and the harness suites run in the main tree environment; pass counts are reported, with worktree-only failures named.

## Rules
CPU only; heavy commands under `systemd-run --user --scope -p MemoryMax=12G -p CPUQuota=300% nice -n 15`; Rust with its own target dir, deleted after; `data/store2/` <= 15 GB peak; deletions logged in `data/ks/logs/deletions.log`; v3 and `data/store1` are not deleted. Licence-clean data only. Failures are reported as failures, and any departure from the paradigm is declared.
