# STORE-1 pre-registration: the exact GF(17) store as the drive (resolver, coverage, ultra-compact store, speed)

Written 2026-10-06, before any eval item is written, before the coverage audit, and before any resolver is built or scored. Mission: STORE-1 (CEO direction 2026-10-06: the exact GF(17) store becomes ultra-compact and the drive for fast, accurate answers; the blender only phrases). Lineage: `torus_space_proposal.md`, `torus_a_results.md` (A1 exact store), `harness.md` (v3/v4 material routes, answer-type gate), the HYBRID-1 CTO audit (`hybrid1_results.md`). All outputs go to `data/store1/` (cap 15 GB).

## What was already looked at before this file was written (disclosure)
- The mission brief quotes four failure questions from the HYBRID-1 E2E run (Mona Lisa painter, largest planet, number of continents, and the retrieval-failure field showing one season question) and three commonsense examples ("opposite of hot", "what do cows drink", "smallest prime"). I read the report's aggregate `retrieval` block and one `retrieval_failures` example while locating the per-item fields; I did not read the per-item list.
- Store statistics measured on v3 only (no eval data): entity-valued facts 45,015,432 with 0-order entropy H(value | property) = 9.42 bits/fact, a 17-ary Huffman code 10.13 bits/fact, fixed per-property width 15.98 bits/fact; time 5,337,458 facts (Huffman 14.13 bits), quantity 3,992,822 (12.84 bits), strings 716,489; 503,869 preferred-rank facts; 6,447,294 facts with qualifiers (8,513,727 qualifiers); 42,692,129 (subject, property) groups of which 4,710,949 hold more than one value; 678,103 distinct (property set + multi flag) signatures over 10,142,453 subjects, H = 9.32 bits, 17-ary Huffman 2.47 digits per subject; labels 271.8 MB (order-3 static char model, 10% held-out: 2.61 bits/char; word 0-order 1.95 bits/char plus a 35 MB dictionary); 8,128,474 enwiki titles equal the label and 794,582 are the label plus a parenthetical. v3 on disk: 4,629,551,616 bytes.

## Part A: resolver and coverage

### A0 coverage audit (classification only, no tuning)
- Items: the 102 `know` and the 59 `explain` items of the locked E2E chat dev suite, read from `data/adam_t/eval/report_h1-hybrid-qwen3-4b-q8-C8_0_e2e_tune_dry.json` (`suites.chat.items`, `cat` in {know, explain}).
- Done by a separate agent; the resolver author (me) reads only the aggregate table, never the per-item rows, while the resolver is developed. Classes (first that applies, plus a multi-label column): `wd_entity` (Wikidata v3 entity fact: subject QID + property), `wd_value` (Wikidata v3 non-entity value: date, quantity, string), `concepts` (ConceptNet 5.7 / WordNet in `data/concepts`), `passages` (Wikipedia passages in `data/ks`), `compute` (math, primes, unit conversions; `amni/math`), `none`. Each row cites the key it found (QID+PID, ConceptNet edge, WordNet synset relation, passage title, or the computation). Output: `data/store1/coverage_audit.json` (+ `.md` table). Nothing is tuned on these items; they are never used as dev data.

### A1 resolver (lookup tables, no matmul)
- Entity linking: every token n-gram span (1 to 10 tokens) of the normalised question (`amni.compute.wikifact.nrm`) is hashed (same 64-bit hash as v3) and looked up in the GF(17) name table; candidates are ranked by a stored popularity prior (sitelinks, fact count from the presence code) plus answer-type compatibility and property presence.
- Property mapping: a lookup table keyed by hashed head-word n-grams (wh-word, verb lemma, nouns), built from (a) the English labels and all English aliases of the kept Wikidata properties (fetched once from the Wikidata API, CC0, provenance recorded) and (b) a small hand-written CC0 table of question-word patterns, reusing the existing in-repo `WHEN`/`WHO`/`NOUN` tables of `amni/compute/wikifact.py`.
- Non-Wikidata routes use the same output format: ConceptNet/WordNet relation keys (antonym, made-of, used-for, at-location, capable-of, has-property/colour, part-of, young-of, has-a counts) and calculator/solver keys (`amni/math`, a prime solver, unit conversion).
- Output: a ranked list of (store, key, confidence); the answer-type gate reuses `cognition.qclass` / `cognition.isa`. "Not in store" is an explicit output (abstain).

### A2 new eval set (CC0, written from scratch, then frozen)
- At least 400 natural questions written by a separate agent that sees no repo code and no eval data. Planned mix: Wikidata entity-valued relations ~200, Wikidata dates/numbers ~80, commonsense ~90, computation ~50, absent-from-every-store ~60 (for abstain precision).
- Gold = (store, key, accepted answers). Wikidata gold keys are annotated mechanically from the author's (subject name, intended property, answer): the subject entity is the v3 entity whose label/alias equals the subject name and whose property value matches an accepted answer (most sitelinks breaks ties); every property of that entity holding an accepted answer is in the gold property set. If v3 holds a different value from the author's answer, the item is dropped and logged (not relabelled). Commonsense gold = the ConceptNet edge or WordNet relation holding the answer; computation gold = the expression and its value; absent gold = `none`, verified by checking the intended key is missing from every store.
- Decontamination: zero shared 6-grams (`amni.compute.decontam`, n=6, its standard token and distinct-word rules) with every file under `data/adam_t/eval/**` (locked chat/E2E suites, probe8, all reports, which contain the 30 failure questions), checked by script; offending items are rewritten by the author agent.
- Split: dev/test 50/50, stratified by category, seed 20261006, written once to `data/store1/evalset/{dev,test}.jsonl` with a sha256 lock (`data/store1/evalset/lock.json`) before any resolver code runs on it. Test is scored exactly once, with the frozen resolver, by `scripts/store1_eval.py --split test` writing `data/store1/results/test_scores.json` and a lock entry.

### Metrics (field names fixed here)
- `ent_at1`, `ent_at5`: share of in-store Wikidata items (`wd_entity` + `wd_value`) whose gold subject entity is the resolver's top-1 / within the top-5 distinct subject candidates.
- `prop_acc`: share of in-store Wikidata items whose top-1 property is in the gold property set.
- `e2e_acc_instore`: share of in-store items (gold store in {wd, concepts, compute}) whose top answer from the store matches an accepted answer. Matching: `nrm` equality, or an accepted answer appearing as a whole-word sequence in the rendered answer; numbers equal within 0.5% (exact for integers below 1000); dates: same year, and same month/day when the gold has them.
- `abstain_prec`: among items where the resolver outputs "not in store", the share whose gold store is `none`. Reported with `abstain_n`; `abstain_rec` (share of `none` items abstained) and `absent_false_answer` are reported, not gated.
- Per-category breakdown and the v3 baseline (`amni.compute.wikifact.ask`, the current resolver) on the same items are reported, not gated.

### Gates (test split, scored once)
| gate | metric field | pass |
|---|---|---|
| GA1 | `e2e_acc_instore` | >= 0.80 |
| GA2 | `ent_at1` | >= 0.90 |
| GA3 | `prop_acc` | >= 0.85 |
| GA4 | `abstain_prec` | >= 0.90 (needs `abstain_n` >= 10, else reported as not evaluable) |

### Dev protocol
- Every resolver feature is developed and ablated on dev only: full; minus property presence; minus popularity prior; minus type gate; minus hand patterns (Wikidata aliases only); minus non-Wikidata routes; name-table fingerprint F in {0 (verify by decoding the candidate's names), 1, 2, 3}.
- Selection rule: highest dev `e2e_acc_instore` subject to dev `abstain_prec` >= 0.90; ties go to fewer bytes, then lower latency. The fingerprint budget is the smallest F whose dev `e2e_acc_instore` and `abstain_prec` are within one item of the best F and whose dev latency meets the Part C gate.
- Any change to a gate or metric after a dev result is an amendment with a timestamp and a note on whether it helped the result.

## Part B: ultra-compact drive

### Design to build (each choice measured, alternatives reported)
- Facts: A1 ribbon/BuRR retrieval (`crates/adam-torus`, extended, not copied) as 1-digit levels of a 17-ary canonical Huffman cascade: per-property codebooks (value rank within the property's value set, code length set by frequency); the fact symbol carries the value (+unit), the preferred flag and the qualifier count. Fixed per-property width is reported as the alternative.
- Presence: a per-subject property-presence signature (property set + multi-value flag, Huffman-coded, keyed by entity) plus a count table for multi-value groups, instead of per-key fingerprints; both costs are measured and the smaller exact option is kept.
- Non-entity values (dates, quantities with units, strings) in their own codebooks; qualifiers in their own cascade keyed (subject, property, index, j) with per-parent-property codebooks.
- Reverse index (object -> subjects): built only if at least 3% of dev Wikidata items need an object-to-subject step; otherwise excluded and its compact size estimated.
- Derivable facts: measured for P17 <- P131 then P17 (and P30 <- P17 then P30, P27 <- P19 then P17); a rule is stored as a group flag plus exceptions only if it saves >= 0.5 MB, and rule + exceptions is verified exact on every group it covers.
- Names: name -> entity as a GF(17) solved table keyed (64-bit name hash, i) with an ordered chain for ambiguous names (v3 order: entity ascending), fingerprint F chosen by the dev rule above; an END marker is stored where a fingerprint false positive would extend a chain. id -> label/title/aliases with a dedicated static order-k context-model codec (LUT model, range coded), blocked for random access; the existing KS coder is measured on a sample for comparison if it runs on CPU.
- Pages: every component in GF(17) digit pages (4096 x 4096 RGBA logical pages, Reffelt 17-bit pixel words via `adam_gf17::pack17`), mmapped and streamed, bit-exact roundtrip checked.

### Parity (exact, API level)
For all 11,900,718 entities and all 55,062,201 facts: `values(e, p)` lists (kind, value, unit, preferred flag, qualifier list, order), `label`, `title`, `name`, `names`, sitelinks, QID <-> index; for every name key of the v3 alias index (17.6M (hash, entity, kind) rows): `ents(x)` identical in content and order. Not in scope (deliberate, documented): the raw `names.tsv` build input (not used by the API), the `facts.gf17` duplicate, the reverse indexes unless the rule above builds them. Absent-name false-positive rate is reported.

### Gates
| gate | metric field (in `data/store1/parity.json` / `sizes.json`) | pass |
|---|---|---|
| GB1 | `parity_facts_mismatch`, `parity_names_mismatch`, `parity_labels_mismatch`, `parity_ents_mismatch` | all 0 (or each exclusion documented) |
| GB2 | `total_bytes` / `v3_bytes` | <= 0.15 (v3 = 4,629,551,616 bytes, so <= 694,432,742 bytes) |
| GB3 | `roundtrip_bitexact` | true |

Size justification: the measured information content (entity facts ~59 MB in 17-ary Huffman digits, dates/quantities ~17 MB, presence ~14 MB, names ~56 MB at 6 digits per (name, entity) pair, text ~100-130 MB at ~2.6 bits/char, codebooks ~25 MB) sums to roughly 300-350 MB, about 7% of v3; 15% leaves room for GF(17) packing overhead (4.25 bits per digit, 17 bits per 16-bit word) and slack. Reported, not gated: bits per fact vs the 0-order entropy bound and vs the codebook bound, lookups/s/core warm and cold, RAM.

## Part C: speed
- End-to-end question -> material latency on CPU (resolver + lookups + label decode, excluding the blender), over the test set, one process, store mmapped, after a warm-up pass over dev. Gate GC1: `lat_median_ms` <= 5 and GC2: `lat_p99_ms` <= 25 (fields in `data/store1/results/test_scores.json`). Cold-cache latency reported, not gated.

## Part D: wiring
- Harness material route `AMNI_STORE1=1` (default off) puts store answers into the v4 material format (`Label: content` items joined by one space). `amni/serve/gguf_runtime.py` untouched. The locked E2E suite is not run.

## Rules
CPU only; heavy commands under `systemd-run --user --scope -p MemoryMax=12G -p CPUQuota=300% nice -n 15`; Rust built with its own target dir, deleted after; deletions logged in `data/ks/logs/deletions.log`; v3 is not deleted. Failures are reported as failures.

## Amendments (each written before the test split was scored)

### Amendment 1 (2026-10-06T09:16:50-0400): configuration selection when no dev configuration meets the abstain constraint
- What changed: the dev selection rule ("highest dev `e2e_acc_instore` subject to dev `abstain_prec` >= 0.90") cannot be applied as written, because no dev configuration reaches `abstain_prec` 0.90 (best 0.7073; `data/store1/results/ablate.log`). Added tie rule: among configurations within one dev item of the best `e2e_acc_instore`, keep the one with the highest `abstain_prec`; remaining ties go to fewer bytes, then lower latency.
- Effect: it keeps the full configuration (dev `e2e_acc_instore` 0.8289, `abstain_prec` 0.7073) instead of the case-penalty ablation (`no_case`: 0.8333, 0.7000). It costs one dev item of `e2e_acc_instore` and gains 0.0073 `abstain_prec`; it does not help the headline metric. No gate threshold and no metric definition changed.
- Fingerprint budget, by the pre-registered rule: F=0 (verify by decoding the candidate's names) ties F=2 and F=3 on dev `e2e_acc_instore` and `abstain_prec` and meets the latency gate on dev (`fd0` row of `data/store1/results/ablate.log`), so F=0 is selected; F=1 loses one absent item.
