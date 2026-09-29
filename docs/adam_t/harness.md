# ADAM-T-HARNESS: verify-and-complete answer loop around ADAM-T

Status: CPU prototype for mission ADAM-T-HARNESS, 2026-09-29; revised as v2 by mission ADAM-T-HARNESS2 and as v3 by mission ADAM-T-COMMONSENSE (next sections, newest first). The sections from "Summary (v1)" on describe v1.

- **Code:** `amni/serve/adam_t_harness.py`, wired into `adam_t_client.answer()` behind `AMNI_ADAM_T_HARNESS=1` (**off by default**). `gguf_runtime.py` is untouched, so emergency-first is unchanged.
- **Tests:** `tests/test_adam_t_harness.py` (41 tests). The `test_adam_t_route`, `test_user_toroid`, `test_user_memory_*` and `test_decontam_adam_t` suites still pass.
- **Eval:** `scripts/adam_t_harness_eval.py`, own CC0 dev set in `data/adam_t/eval/harness/`. The locked E2E chat dev suite ran once per condition (dry/tune) through `scripts/adam_t_eval.py run --e2e`.
- **Limits honoured:**
  - CPU only; everything ran under `systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=200% nice -n 15`.
  - Model: a read-only snapshot of `exports/adam_t/t30m_sft2/ckpt` at **step 4607**. `core.bin`, `mem.bin`, `state.txt`, `steps.txt` and `log.jsonl` had stable size and mtime over a 5 s window, were unchanged after the copy, and were `cmp`-identical. Exported to `.gf17` in scratch (sha256 `47a3540d…`, argmax parity 97.6%).
  - Binary built with its own `crates/target` and copied to `adamt-har`. Model copies and the target dir were deleted afterwards.

## v3 (mission ADAM-T-COMMONSENSE, 2026-09-29)

v3 targets the "answered" gap that blocks milestone C2. It adds three lookup-first pieces to the harness/cognition path: an answer-type gate on best guesses, a commonsense resolver over WordNet, ConceptNet 5.7 and Wikidata v3, and an extractive explain route over the live Wikipedia index. `AMNI_ADAM_T_HARNESS` is still **off by default**, and each new piece has its own flag, all defaulting to on under the harness. `gguf_runtime.py` and `adam_t_client.py` are untouched, and `gguf_runtime.py` is still CRLF and byte-exact, so emergency-first is unchanged.

- **Model:** `exports/adam_t/t30m_sft2/model.gf17`, used read-only (step 7773, the same file as the v2 "final" E2E runs). Binary built with its own target dir in scratch and copied to `adamt-cs`. The target dir was deleted right after the copy.
- **Tests:** `tests/test_adam_t_harness.py` and `tests/test_cognition.py` gain 20 tests (explain detection, type gate, commonsense slot and hedge, paraphrase grounding, extractive explain, ConceptNet store, `isa`/`qclass`, `know`). 247 pass across `test_adam_t_harness`, `test_cognition`, `test_adam_t_route`, `test_adam_t_chat_eval` and `test_user_toroid`.

### What changed

- **Answer-type gate** (`_tgate`, `cognition.qclass`, `cognition.isa`). The expected class comes from the question noun: "which/what N ...", "the SUP N", "which N" at the end, "what is the X N of", or person for "who". A best-guess entity has to be a hyponym of that class before it is offered:
  - WordNet hypernym closure (instance hypernyms included);
  - otherwise ConceptNet `IsA`;
  - otherwise Wikidata P31/P279/P106/P171 values mapped back into WordNet.
  Unknown multi-word names pass only for person classes (name shape). Common-noun "what do X V" questions reject capitalised or non-WordNet guesses. When every candidate fails, the reply is an honest hedge ("I'm not sure which organ that is.") instead of "I think it's Like". The gate also applies to passage answers that the reader and ranker agree on. `AMNI_ADAM_T_HARNESS_TGATE=0` disables it.
- **ConceptNet 5.7 store** (`scripts/build_conceptnet.py`, `data/concepts/conceptnet/`):
  - Source: plain HTTPS download of `conceptnet-assertions-5.7.0.csv.gz` (497,963,447 B, sha256 `accd65fe…`). Filtered to English-English edges, 18 relations (IsA, UsedFor, CapableOf, PartOf, HasA, MadeOf, AtLocation, Causes, Desires, ReceivesAction, HasProperty, Antonym, CreatedBy, HasPrerequisite, CausesDesire, MotivatedByGoal, DefinedAs, LocatedNear), weight ≥ 1.0, and nodes of at most 6 words / 60 chars. Duplicates were merged.
  - Result: **382,613 edges, 261,089 nodes, 22.3 MB on disk**. The layout follows the wikifact store: hash-sorted node table plus text blob, CSR forward and reverse edge arrays, and an `edges.gf17` base-17 twin whose round trip is verified exact.
  - `manifest.json` records the licence (CC BY-SA 4.0), the ConceptNet attribution text, the source hash and the filter. `NOTICE` carries the attribution. The raw download was deleted after the build, and the deletion is logged in `data/ks/logs/deletions.log`.
  - Cognition reads the store through `cnid`/`cnv`/`cne`, plus a target-word index `_cnx`. `relate` and `common` gain ConceptNet paths (`cn1`, `cnshare`; source `cn`, not in the strict factual set).
- **Commonsense resolver** (`cognition.know`). Lookup only, and every answer carries provenance lines:
  - antonym, synonym, plural, young-of and group-of from WordNet lemmas, the exception list and glosses, and ConceptNet Antonym;
  - "how many N does X have" from number+noun patterns in WordNet glosses, ConceptNet HasA/PartOf/DefinedAs, or the subject's own Wikipedia article (possessive context only);
  - "what is X made from" from WordNet glosses, ConceptNet MadeOf/ReceivesAction and Wikidata P186, with an optional class filter;
  - "where do X live" from ConceptNet AtLocation, WordNet gloss places and Wikidata P2974/P183;
  - "what do X V" (object) from ConceptNet CapableOf/Desires/UsedFor, ReceivesAction "V-ed by X", WordNet glosses, and Wikidata P1034/P1056/P2283 (only when the Wikidata subject's label matches). A low-confidence fallback votes over the ConceptNet edges of hypernyms and siblings;
  - class and superlative questions ("which organ produces bile", "the tallest animal"): class hyponyms whose gloss has the rarest question word or the superlative directly before a class word (regional qualifiers rejected), ConceptNet reverse lookups on the question words, ConceptNet DefinedAs/IsA/HasProperty "SUP class" edges, and Wikipedia sentences "X is the SUP ... class" whose subject passes the class check.
  In the harness (`_cs`), antonym, count, made, object, where and define answers return right after a strong Wikidata miss. Class and superlative answers wait until the passage reader and ranker have had their chance to agree. Confidence below `AMNI_ADAM_T_HARNESS_CS_SURE` (0.6) gives "I think …". "Who" questions stay on the Wikidata/passage path. Generic "we/our" questions that the memory path cannot answer are tried here before "You haven't told me that yet." `AMNI_ADAM_T_HARNESS_CS=0` disables it.
- **Explain route** (`fill_expl`). How/why/what-causes questions that are not advice, not about the user, not a pronoun follow-up and not an ellipsis go to a new `explain` slot:
  - Topic articles come from the question's noun phrases, subject first. The title must match the phrase, and works, people and places are rejected (lead-sentence pattern and Wikidata P31 without P279).
  - Their passages are scored together with BM25 hits (`AMNI_ADAM_T_HARNESS_EXPLAIN_BM25`) on question-word coverage (idf, stem prefix), mechanism cues ("because", "due to", "caused by", "when X", "by …ing", "results in"…), title match, lead position and penalties (history/years, discourse openers, entity titles). A BM25 passage needs title overlap, coverage ≥ 0.5 and a strong cue.
  - The reply is 1–2 checked sentences (≤ 55 words, ≤ 380 bytes), marked `grounded` with `prov=["Wikipedia: <title> (passage <id>)"]`.
  - ADAM-T may rephrase it (`AMNI_ADAM_T_HARNESS_EXPLAIN_PARA`). The rephrase is used only if it adds no new content words or digits, keeps ≥ 60% of the extract's content words, has ≥ 60% of its trigrams in the passage and question, and ends a sentence. Otherwise the extract is returned as-is.
  - When nothing passes, the question falls back to the old model path. `AMNI_ADAM_T_HARNESS_EXPLAIN=0` disables the route.
- **Counters:** `hxe` (explain slots), `hcs` (commonsense slots) and `htg` (type-gate hedges) are added to the harness counters.

### Results

**Own dev set** (`data/adam_t/eval/commonsense/`, CC0, written from scratch for this mission):
- Contents: 157 explain questions (with expected mechanism terms) and 161 everyday-knowledge questions (with accepted answers).
- Overlap check: zero shared 6-grams with `chat/chat_src.jsonl`, the harness and cognition dev/test sets, and the toroid dev sets (22 draft questions were rewritten until the check passed).
- Split: per category into tune (158) and test (160), seed 20260929.
- Runs: through the live route (`gguf_runtime.chat` with AMNI_ADAM_T, harness and cognition on) by `scripts/adam_t_cs_eval.py`. "Before" is HEAD c828a2c on the same binary and model. The test half was scored once before and once after, with the code frozen.

| split | category | answered | coherent | fact | explain term hit | grounded |
|---|---|---|---|---|---|---|
| test (held out) | explain (79) | 0.291 → **0.772** | 0.861 → 1.000 | | 0.228 → 0.595 | 0.000 → 0.899 |
| test (held out) | know (81) | 0.333 → **0.568** | 0.975 → 0.901 | 0.222 → **0.494** | | |
| test (held out) | all (160) | 0.313 → **0.669** | 0.919 → 0.950 | | | |
| tune | explain (78) | 0.333 → 0.769 | 0.782 → 0.949 | | 0.218 → 0.513 | 0.000 → 0.795 |
| tune | know (80) | 0.288 → 0.763 | 0.988 → 0.963 | 0.150 → 0.688 | | |

The knowledge gain shrinks from tune (0.76) to test (0.57). The rules generalise, but not fully. Test declines rose from 0.025 to 0.148 because the type gate now says "I'm not sure …" where v2 offered a wrong-typed guess.

**Locked E2E chat dev suite** (410 items):
- Run: once, harness and cognition on, `--tune`, recorded as dry: `report_adam-t-t30m-sft2-cs-h1_7773_e2e_tune_dry.json`.
- Baseline: the v2 "final-h1" run on the same model file.

| metric | v2 final-h1 | v3 | target |
|---|---|---|---|
| answered | 0.551 | **0.637** (lo95 0.589) | ≥ 0.70 (C2: 0.80) |
| fact | 0.577 | **0.705** | ≥ 0.55 |
| coherent | 0.942 | **0.954** | ≥ 0.90 |
| good | 0.480 | 0.578 | |
| context use | 0.563 | 0.563 | |
| 911 lead | 1.000 | 1.000 | no regression |
| near-miss false positives | 0.037 | 0.037 | no regression |
| duplicate replies | 0.042 | 0.022 | |
| median / p90 latency | 688 / 2237 ms | 542 / 2109 ms | |

By category (answered): explain 0.254 → 0.610, know 0.637 → 0.784, chat 0.494 → 0.481, and advice, multi, nearmiss and emergency unchanged:
- Explain: 26 items gained and 5 lost. 46 of 59 explain items took the new route and 35 of those were answered. The other 13 fell back to the model (1 answered).
- Know: 17 gained and 2 lost.
- Chat: one "why" small-talk question now gets a Wikipedia sentence and lost its lexical overlap.

### Targets

- **Missed:** E2E answered ≥ 0.70 (0.637), and therefore C2's 0.80.
- **Met:** fact ≥ 0.55 (0.705), coherent ≥ 0.90 (0.954), and no regression in 911 lead or near-miss false positives.

The remaining gap is spread across categories:
- chat (0.48): lexical scorer, deliberately left alone;
- nearmiss (0.33) and advice (0.71): untouched by this mission;
- explain questions whose topic article is not reachable by title or has no cue sentence (13/59 fall back);
- held-out knowledge coverage.

### Limitations and next steps

- Topic selection is title-based. Wrong-sense titles still slip through ("Contract" for muscles contracting is kept out only by coverage scoring), and redirect-only topics ("sweat" → Perspiration) are missed.
- The explain extract is chosen by coverage and cue words, not by answering the "why". Some replies are on-topic facts rather than mechanisms (kw hit 0.60 on test).
- ADAM-T's paraphrase almost never passes the grounding check, so replies are effectively extractive.
- Commonsense rules were tuned on 80 questions; held-out precision is lower. Observed errors include hypernym answers ("the vertebrate"), Wikipedia superlatives with the wrong subject, and the count template with plural subjects ("A spiders has eight legs").
- ConceptNet crowd edges are noisy (for example "dogs CapableOf run across field"). Answers built on them alone stay at hedge confidence.

### Reproduce (v3)

```
python scripts/build_conceptnet.py --src <conceptnet-assertions-5.7.0.csv.gz or URL>
systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=200% nice -n 15 env AMNI_ADAM_T_HARNESS=1 AMNI_COGNITION=1 \
  python scripts/adam_t_cs_eval.py --split <tune|test> --label <label> --proto-cmd '<bin copy> serve <model.gf17>'
systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=200% nice -n 15 env AMNI_ADAM_T=<model.gf17> AMNI_ADAM_T_BIN=<bin copy> AMNI_ADAM_T_HARNESS=1 AMNI_COGNITION=1 \
  python scripts/adam_t_eval.py run --model proto --suites chat --e2e --tune --proto-cmd '<bin copy> serve <model.gf17>' --step 7773 --name adam-t-t30m-sft2-cs-h1
.venv-linux/bin/python -m pytest -q tests/test_adam_t_harness.py tests/test_cognition.py
```

## v2 (mission ADAM-T-HARNESS2, 2026-09-29)

v2 applies the CTO decisions: a best guess in every knowledge hedge, hedges that do not restate the question, the agreement check on by default, and the cognition layer wired in. The flag `AMNI_ADAM_T_HARNESS` is still **off by default**. `gguf_runtime.py` is untouched (CRLF, byte-exact), so emergency-first is unchanged.

- **Model:** the same snapshot as v1. `t30m_sft2/ckpt` was still paused at **step 4607**; the files were stable over a 5 s window, unchanged after the copy and `cmp`-identical. The re-export has the same sha256 (`47a3540d…`) as v1. Binary built in the worktree's own `crates/target`, copied to `adamt-h2`; model copies and the target dir were deleted afterwards.
- **Baselines reproduce exactly.** The off and v1 E2E conditions were re-run once each on this build and gave 410/410 identical replies and identical metrics to the v1 reports, so the v1 reports stand as the off/v1 rows below.
- **Tests:** `tests/test_adam_t_harness.py` (62 tests). `test_adam_t_route`, `test_cognition`, `test_user_toroid`, `test_user_memory_*` and `test_decontam_adam_t` still pass (186 total).

### What changed

1. **Best guess in every knowledge hedge.** When a knowledge or reading slot fails verification, the harness offers its top unverified candidate. The order is: a verified span the ranker disagreed with; an ADAM-T span that appears in the passage; the rknow ranker's top candidate; an uncorroborated weak Wikidata answer; last, an ADAM-T span not found in the passage. For open (`any`) questions, only short sentences found in the passage and the ranker's candidate are used.
   - **Dates and numbers stay coarse**, as in the v6 reader data (`_coarse`): decade for years from 1500 on ("in the 1940s"), century otherwise or with BC ("in the 1st century BC"), the month when there is no year; numbers by magnitude ("somewhere in the thousands", "somewhere between ten and a hundred", "fewer than ten") or a 10-point percent band.
   - **When there is no candidate at all,** the slot goes to the old context + ADAM-T path instead of saying "I'm not sure …".
   - **User-memory slots keep the honest "You haven't told me that yet."**
2. **Hedge wording.** Hedges no longer restate the question: "I think it's X, but I couldn't confirm it." A second or third hedge in the same reply rotates to "Probably X, though I couldn't verify that." and "Maybe X, but I'm not certain.", which keeps 3-gram repetition down. A single-slot memory miss is "You haven't told me that yet."; in a multi-part reply the sub-question is named so the user knows which part is missing.
3. **Agreement check on by default** (`AMNI_ADAM_T_HARNESS_AGREE`, default now `1`).
   - Agreement compares years for dates and values for numbers (`_same`), not strings, so "July 18, 1918" agrees with "18 July 1918".
   - A Wikidata answer is flagged as a conflict when all of these hold:
     - the question expects a typed answer (not an open `any` question);
     - the ADAM-T passage reading is verified, short and not a list;
     - the reading disagrees with Wikidata;
     - the rknow ranker's top candidate does not support the Wikidata answer.
   - The reply is then "I think Marie Curie was born in Freta Street, or maybe Warsaw."
4. **Cognition wired in** (`AMNI_COGNITION=1`). Before slot decomposition, `cog()` reads `cognition.mode()` and `cognition.parse()`.
   - **Engaged** for relate, common, analogy, abstract, deduce and connect operations, and for creative mode. **Factual mode stays on the strict slot path**: across both sets, 0 of 40 factual mode items engaged cognition.
   - **Context:** `cognition.context()` lines go into the prompt one per line. `adam_t_client.template()` and `AdamT.gen()` gained `sep` (default `' '`, so existing prompts are byte-identical); the cognition route passes `'\n'`.
   - **Coherence checks (every mode):**
     - no role leak;
     - no unfinished trailing sentence;
     - at most `AMNI_ADAM_T_HARNESS_COG_WORDS` (55) words;
     - 3-gram repetition ≤ 0.25;
     - not an echo of the question;
     - mostly alphanumeric.
   - **Fact checks (analytical only; creative relaxes them):**
     - at least 4 words;
     - no copied `X, rel: Y` line format;
     - mentions a queried term;
     - uses at least one checked-link term that is not in the question;
     - at most a third of its content words unsupported by the lines or the question;
     - no unsupported capitalised names;
     - the analogy answer is present;
     - yes/no agrees with `holds`.
   - **When the checks fail:**
     - analytical replies are composed from the checked lines (`compose`: "Tennis and badminton are both kinds of racket sport and court game.", "Japan. Paris is the capital of France. Tokyo is the capital of Japan.");
     - creative replies that fail coherence, and creative requests with no lines, defer to the old path.
   - **Decoding:** the JSON-lines protocol has no per-request sampling field, so mode-driven decoding was not wired.
5. **Routing fixes found on the tune half and in the v1 E2E hedges:**
   - "we/our" only counts as personal when it does not name a generic noun ("our solar system", "the planet we live on");
   - "between A and B" is not split;
   - "good way / best way / what helps" questions are advice and go to the model;
   - a knowledge slot that shares words with the user's own statements in a personal message hedges as unknown memory instead of guessing.
6. **Memory recall:**
   - typed attributes (job, age, name, birthday) bind without the predicate-verb match ("What does my sister do for a living?" ← "My sister is a nurse …");
   - a relaxed retry drops time words ("my meeting tomorrow");
   - when the toroid parser finds no slot but one of the user's own statements holds the entity, a question content word and the question's qualifiers, the harness quotes it: "You mentioned you play the cello in a community orchestra." (`_said`, counter `hsd`).
7. **Counters** (no user text): `hgs` guesses, `hsd` quoted statements; cognition uses `hg`, `hgl` lines, `hgv` model reply accepted, `hgc` composed, `hgk` creative, `hgms`.

### Results

**Own dev set, held-out test half** (125 items; the tune half is where rules were developed; for the test half only aggregates and the ids of the false-memory items were looked at):

| test half | off | v1 | **v2** (agree on) | v2, agree off |
|---|---|---|---|---|
| slot correct (answerable) | 0.139 | 0.387 | **0.490** | 0.490 |
| … memory / knowledge / calc | 0.16 / 0.10 / 0.18 | 0.37 / 0.43 / 0.27 | 0.49 / 0.52 / 0.27 | same |
| multi-part fully answered | 0.000 | 0.247 | 0.321 | 0.321 |
| false memory (never-stated parts) | 0.800 | 0.133 | **0.133** (2/15) | 0.133 |
| answered (≥ 1 non-hedged sentence) | 0.944 | 0.632 | 0.648 | 0.680 |
| overclaim | 0.921 | 0.467 | 0.460 | 0.451 |
| latency p50 / p90 ms | 772 / 2428 | 1179 / 3961 | 1346 / 4983 | 730 / 3763 |

Tune half, v2: slot correct 0.637 (v1 0.587), false memory 0.027 (v1 0.027), answered 0.712 (v1 0.688). The two test false memories are the same two items v1 got wrong, not new ones. With only 15 never-stated test slots, one slot is 6.7 pp.

**Locked E2E chat dev suite** (410 items, dry/tune, one run per condition):

| metric | off | v1 | **v2** |
|---|---|---|---|
| chat_ans | 0.588 | 0.488 | **0.559** |
| chat_good | 0.415 | 0.422 | **0.480** |
| chat_fact | 0.403 | 0.423 | **0.577** |
| chat_ctx | 0.451 | 0.437 | 0.535 |
| chat_coh | 0.902 | 0.822 | **0.927** |
| chat_dec | 0.022 | 0.161 | **0.017** |
| chat_dup / nmfp / em | 0.012 / 0.037 / 1.00 | 0.000 / 0.037 / 1.00 | 0.027 / 0.037 / 1.00 |
| milestones | C1 | none | **C1** |
| route mix | adam_t 396 | adam_t 246, harness 150 | adam_t 251, harness 145 |
| latency median / p90 ms (harness route) | – | 1115 / 2148 | 1446 / 2448 |

- **By category** (answered / good, off → v2):
  - know: 0.824 / 0.314 → 0.667 / 0.510;
  - multi: 0.521 / 0.366 → 0.577 / 0.465;
  - advice: 0.814 / 0.814 → 0.814 / 0.814.
  - chat, explain, nearmiss and emergency are unchanged.
  - No E2E item triggered cognition: the suite has no relation, analogy or abstraction requests.
- **Why answered stops at 0.559 (target 0.60 missed by 4.2 pp, 17 items).**
  - 26 items answered by off are not answered by v2. 22 of them are best-guess hedges whose candidate is not a gold keyword (17 entity guesses, 5 coarse numbers or dates), 3 are honest "You haven't told me" replies, and 1 is other.
  - Of the 25 lost items that have a gold answer, off's answer was factually right on only 4.
  - For `know` items, the locked scorer counts any reply of up to 4 words as on-topic ("Humid." for "What do bees make?"), but a longer reply only when it contains a keyword. So a wrong one-word guess counts as answered and a wrong hedged guess does not.
  - v2 gains 14 items that off missed.
  - Declines are now 1.7%, below off's 2.2%.

**Cognition dev set** (310 items; replies through the live route; `hit` = a gold keyword stem in the reply; answered and coherent use the locked chat scorer with the gold stems as keywords):

| dev | off | v1 | **v2** |
|---|---|---|---|
| hit (250 op items) | 0.096 | 0.088 | **0.840** |
| … relate / common / analogy / abstract | 0.11 / 0.14 / 0.04 / 0.08 | 0.09 / 0.12 / 0.06 / 0.08 | 0.79 / 0.84 / 0.86 / 0.92 |
| answered (all 310) | 0.145 | 0.119 | **0.845** |
| coherent | 0.787 | 0.529 | **0.977** |
| cognition engaged | 0 | 0 | 0.839 (factual 0/20, analytical 8/20, creative 5/20 mode items) |
| model reply accepted (else composed or deferred) | – | – | 0.119 |
| latency p50 / p90 ms | 2116 / 2406 | 1875 / 2646 | 477 / 1409 |

Held-out cognition test (145 items, run once, off vs v2): hit 0.129 → **0.765**, answered 0.166 → 0.690, coherent 0.717 → 0.972. By op, relate / common / analogy / abstract are 0.63 / 0.87 / 0.70 / 0.95.

- **ADAM-T is a weak verbaliser of these links.** Only 12% of its replies pass the fact checks. Typical failures are single nouns ("Albert.", "Belemnite."), copied line syntax, or new facts. Most accurate answers come from `compose`, which states only the checked links.
- **Creative mode** passes coherent model text through, but the model often answers creative requests with "Sure, I'd be happy to help". Creative requests with no cognition lines (most poem, story and slogan prompts, since `parse` finds no operation) defer to the old path.

### Cost of the agreement check (own dev test half, v2 with and without it)

- **Accuracy:**
  - test half: no change (slot correct 0.490, false memory 0.133 both ways), with 2 conflicts flagged;
  - tune half: one conflict, the Marie Curie → Freta Street case, now "…Freta Street, or maybe Warsaw" (slot correct 0.631 → 0.637).
- **Answered:** 0.680 → 0.648 on the test half. Four replies became hedges, from conflicts and from the second sentence-only read on passage answers.
- **Latency:** each strong Wikidata slot now retrieves passages and runs one ADAM-T read. On the test half, p50 goes 730 → 1346 ms and p90 3763 → 4983 ms. On the tune half, p50 goes 4 → 1259 ms.
- **Earlier rules that were dropped:**
  - The v1 conflict rule compared strings. It hedged correct dates ("July 18, 1918" vs "18 July 1918") and a list span ("Canberra, or maybe Adelaide, Brisbane, …").
  - A stricter rule that also required the ranker to back the passage flagged nothing on the dev set, Freta Street included.
- **E2E caveat:** the one E2E run used that stricter rule (1 conflict flagged in 145 harness replies). The shipped rule was settled on the own dev set afterwards; the E2E suite was not re-run.
- **Recommendation:** the check is on as the CTO asked. Its measured benefit is one caught Wikidata quirk, at about 0.6 s median latency and 3 pp of answered rate on the test half. It is worth revisiting when the Wikidata v3 KEEP list lands.

### Targets

| target | result | met |
|---|---|---|
| E2E answered ≥ 0.60 | 0.559 | **no** (−4.2 pp; see above) |
| harness dev test-half fact ≥ 0.42 | 0.490 | yes |
| harness dev test-half false memory ≤ 10% | 13.3% (2/15, same items as v1) | **no** (one slot) |
| C1 (answered ≥ 0.50, coherent ≥ 0.60) | 0.559 / 0.927 | yes |

### Reproduce (v2)

```
AMNI_COGNITION=1 python scripts/adam_t_harness_eval.py run --conds on,on-noagree --split all --bin <adamt-h2> --gf17 <model.gf17> --out data/adam_t/eval/harness/results_v2.jsonl
python scripts/adam_t_harness_eval.py report --out data/adam_t/eval/harness/results_v2.jsonl --summary data/adam_t/eval/harness/summary_v2.json
AMNI_COGNITION=1 python scripts/adam_t_harness_eval.py cog --cond v2 --cogset dev --bin <adamt-h2> --gf17 <model.gf17> --out data/adam_t/eval/cognition/harness_dev.jsonl
python scripts/adam_t_harness_eval.py cogreport --out data/adam_t/eval/cognition/harness_dev.jsonl --summary data/adam_t/eval/cognition/harness_dev_summary.json
env AMNI_ADAM_T_HARNESS=1 AMNI_COGNITION=1 AMNI_ADAM_T_HARNESS_AGREE=1 python scripts/adam_t_eval.py run --model proto --suites chat --e2e --tune --proto-cmd '<adamt-h2> serve <model.gf17>' --name adam-t-t30m-sft2-h2v2 --step 4607
```

Every command ran under `systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=200% nice -n 15`. The cognition `off` and `v1` rows used `--cond off` and a frozen copy of the v1 code.


## Summary (v1)

The CEO's loop, "A was asked, B was in memory, return B, then check A+B and complete with C", runs outside the model:

1. **Decompose** the message into slots.
2. **Fill** each slot from memory, then the calculator, then Wikidata, then passages read by ADAM-T. Provenance and confidence are kept.
3. **Verify** each fill: answer type, evidence span, binding, and agreement.
4. **Complete** with one targeted retry per failed slot. Anything still unverified is hedged.
5. **Compose** one reply from short templates.

Results on the new dev set (250 items, 170 multi-part):

- **Correct parts** roughly triple: 16% → 48% of all answerable slots, and 14% → 39% on the held-out test half.
- **Fully answered multi-part questions:** 0.6% → 36% (test: 0% → 25%).
- **False memory** falls from 69% to 6% (test: 80% → 13%).

The cost is a **lower answered rate**: 95% → 66%. The harness says "I'm not sure …" or "You haven't told me …" instead of guessing, and 97% of its hedges cover a real gap.

On the locked E2E chat dev suite (dry/tune, run once each):

- **Gains:** `chat_fact` rises 0.403 → 0.423 and `chat_good` 0.415 → 0.422. Median latency drops 564 → 417 ms.
- **Costs:** `chat_ans` falls 0.588 → 0.488 because declines rise 0.022 → 0.161, all from honest hedges on `know` and `multi` items. Coherence falls 0.902 → 0.822.
- **C1 lost:** the harness-on run no longer meets C1, because `chat_ans` ≥ 0.50 is missed by 1.2 pp.

**Recommendation:** keep the flag off in production until the hedge wording and the passage reader improve (section 7). Enable it where honesty about memory matters more than the answered rate.

## 1. Decompose (`split`, `coord`, slot resolution)

**Split.** The message is cut on sentence ends, on ", and / but / also / then + question word" (optionally after an adverb, as in "and roughly how many …"), and on ", and …" whose tail holds a wh-word ("…, and an octopus has how many hearts").
- Leading discourse words are stripped ("Also, sodium's chemical symbol?").
- A bare wh-word inherits the previous part's frame: "when was he born, and where?" becomes "where was he born".
- A leading entity fragment becomes the topic of later parts that have no entity: "In South Korea, what's the capital and …"; "Don Quixote, who wrote it …".

**Coordination (`coord`).** A trailing "and X" splits only when one of these general rules applies:
- X is a known memory entity: its label replaces the entity in the first part.
- X is a "my …" noun phrase: it replaces the owner's noun phrase.
- X is a single attribute word (name, age, job, …).
- X is a verb ("live and work").
- X is a capitalised entity. This covers lists ("capitals of Norway, Sweden and Denmark", singularised) and prepositional ellipsis ("and in Sweden"). It does not split when "A and X" is a Wikipedia title ("Pride and Prejudice"), per `rknow.istitle`.
- X starts with a possessive ("and its atomic number").
- X is a determiner noun phrase that replaces the subject ("A spider has how many legs, and an insect?").

**Question types** come from the toroid 5W parser (`user_toroid.ask`: W plus attribute) and from `ray_learn.qtype`, mapped to expected answer types:
- where → place;
- when / what year → time;
- how many / old / much → number;
- who / name → name;
- what colour → colour;
- job → an occupation-like (lowercase) value.

**Pronouns across slots.** Resolution is done by lookup, outside the model:
- **After a memory slot:**
  - he/she go to the verified value when that slot asked for a name ("my favourite painter" → "Frida Kahlo", then "when was she born" becomes a knowledge slot);
  - "it" goes to a thing value ("allergic to penicillin" → "who discovered penicillin");
  - "there" goes to a place value;
  - otherwise the pronoun goes to the memory entity ("where does she live" → "where does my sister live").
- **After a knowledge slot:** it/its/they/their go to the previous question's entity ("What's Japan's capital, and what currency does it use?" → Japan). he/she go to the previous answer (`adam_t_client._resolve`: "Who composed the Moonlight Sonata and where was he born?" → Beethoven).
- **Shared generic pronouns** are left alone (the "they" in "What money do they use in X, and in Y?").
- **Ellipsis follow-ups** in the history ("and New Zealand?") use `_resolve` on the conversation history.

**Routing.** A slot is personal if it has I/my/we, a name the user taught, or content words that overlap the user's earlier statements.
- **Personal slots** go to memory first.
- **Advice questions** ("how do I…", "should I…", anything addressed to "you") need a 0.9 memory confidence, or they are left to ADAM-T.
- **Non-factoid questions** (why / how-to / "what is X" definitions / symptoms, tips, …) with no Wikidata hit are marked `other`.
- **When every slot is `other`,** the harness returns `None` and the old ADAM-T path runs unchanged.
- **Single `other` slots inside a multi-part message** get the old context-plus-ADAM-T answer for that sub-question.

## 2. Fill: sources in precedence order, with provenance

| order | source | how | provenance |
|---|---|---|---|
| 1 | user memory | An ephemeral toroid store: the user's persisted `.torus` (when `AMNI_UMEM_TOROID=1`), else the legacy `user_memory` facts rendered as sentences, plus this session's user statements. Retrieval goes through `user_toroid.recall` (tier-1 slot plus tier-2 soft spans, with the null probability). | `mem` / slot or soft tier |
| 2 | calculator | `adam_t_client._math` (conf ≥ `AMNI_MATH_MIN_CONF`) | `calc` |
| 3 | Wikidata | `_wd(R, q, weak=True, forms)`. It tries the question plus canonical forms built from the entity and a relation noun ("Which city is Japan's capital" → "what is the capital of Japan"; money → currency). Strong means conf ≥ `THR` and lf ≥ 0.85, as before. | `wd` |
| 3b | weak Wikidata | A low-confidence Wikidata answer is kept only if an independently retrieved passage contains it (`_know(…, w=False)`, so retrieval is not biased toward the answer). This recovers "who wrote War and Peace" (conf 0.17) and rejects "who wrote Frankenstein → Alexandre Desplat" (the film score). | `wd+psg` |
| 4 | passages + ADAM-T | One passage per prompt ("single fact per line"): `User: <Title: passage> <question>\nAdam:`. Up to two passages; the second is the retry. | `psg` |

`_wd` gained an optional `weak` flag and a `forms` list. With the defaults it behaves exactly as before, and `_know` gained an optional precomputed `w`. Existing callers, including the SFT builder, are unchanged.

## 3. Verify

- **Type.** `fits(expected, value)` uses the toroid time scanner, number words, capitalisation (honorific-only values such as "Mr" fail), the colour lexicon and reason/means openers. Typed attributes must be the ones asked for, so an age cannot answer a what-question.
- **Evidence.**
  - A memory value must appear in its source utterance.
  - A passage answer span must appear in the passage.
  - Wikidata is its own evidence, bound by the label check in `_wd`.
- **Binding** (the RCA failure: picking a same-type distractor). For memory:
  - entity match, including the head noun;
  - possessor match: "my flight" does not accept "my brother's flight";
  - predicate class: "when does my flight land" does not accept a departure time;
  - the question's noun qualifiers must appear in the source ("what did I plant by the mailbox");
  - the value must not merely repeat the question.

  For passages:
  - the sentence holding the span must share a key token with the question;
  - the title or that sentence must contain the question's entity;
  - no unresolved anaphor may remain ("that island", "those").
- **Agreement.**
  - A passage answer is asserted only if ADAM-T's span agrees with the top candidate of the rknow extractive ranker. Otherwise it is offered as "I think it's X, but I couldn't confirm that."
  - A weak Wikidata answer needs passage corroboration.
  - Optional `AMNI_ADAM_T_HARNESS_AGREE=1` adds an ADAM-T second-prompt check (sentence-only context) and a Wikidata-vs-passage conflict flag ("I think …"). It is off in all runs reported here.
- **Confidence.** Memory confidence is the toroid probability renormalised over type-fitting candidates plus the null. `AMNI_ADAM_T_HARNESS_MEM` (0.5) asserts; `AMNI_ADAM_T_HARNESS_MEM_HEDGE` (0.3) gives "I think …, but I'm not certain."

## 4. Complete

Every failed slot gets one targeted retry:
- **memory:** a relaxed lookup without the predicate or attribute, still bound by the original question's predicate and qualifiers;
- **knowledge:** the next passage, or the alternative resolution (answer entity vs question entity), or the unresolved text.

Still unverified, the slot is hedged:
- **memory:** "You haven't told me where your brother lives yet." This uses an indirect-question rewrite (`ind`) that swaps I/my for you/your and inflects the verb.
- **knowledge:** "I'm not sure how many legs a spider has."

## 5. Compose

Verified slots are joined in the order asked:
- **memory:** `user_toroid.say` templates, or the user's own clause with I/my swapped for you/your when the template would be ungrammatical;
- **Wikidata:** property templates ("X was written by Y.", "The capital of X is Y.");
- **passages:** ADAM-T's own sentence that contains the verified span;
- **calculator:** the calculator line as is.

The model never has to choose among facts.

## 6. Wiring and privacy

- **Wiring.** `answer()` calls `_har()` first when the flag is on.
  - It returns `None` for statements, forget commands, non-questions and all-`other` messages.
  - Harness exceptions (except `ProtoError`) are counted as `har_err` and fall back to the old path.
- **Logging.** The route is logged on the existing debug line as `route=harness`, with counters only: `hs` slots, `hm`/`hk`/`hc`/`ho` kinds, `hv` verified, `hh` hedged, `hr` retries, `hx` conflicts, `hms` milliseconds. A test checks that no user text reaches the log.
- **Memory is read-only.** The ephemeral store is built per request and never saved. The persisted store is copied, not mutated.
- **E2E runner changes.** `adam_t_eval.py --e2e` keeps `AMNI_ADAM_T_HARNESS` across its env reset, records it in the adapter info, maps the `harness` route, and hashes `adam_t_harness.py` and `user_toroid.py` as live-route code.

## 7. Evaluation

### 7.1 Own dev set (`data/adam_t/eval/harness/`)

**Contents.**
- 250 CC0 items, written for this mission by a separate agent that saw no repo code or eval data.
- Kinds: `multi_mem` 50, `multi_know` 45, `mix` 50 (memory + knowledge ± calc, 12 with a never-stated part), `followup` 40, `nomem` 40, `single` 25.
- 425 slots: 186 memory, 171 knowledge, 16 calculator, 52 never-stated.

**Decontamination.** There is zero 6-gram overlap with `chat_src.jsonl` and `toroid_dev.jsonl`. 42 items were reworded to reach that (checked by `build`).

**Split.** Odd ids are the tune half, where rules were developed. Even ids are the test half; only aggregates were looked at.

**Scoring (lexical, per slot).**
- **Answerable slot:** correct if one reply sentence contains an accepted answer and no forbidden value. A value that is the answer to another slot is allowed.
- **Never-stated slot:** false memory if the reply contains a forbidden value or no abstention phrase.

**Metric definitions.**
- `answered`: at least one non-hedged sentence.
- `hedge_warranted`: the share of hedged replies where some part really was wrong or unknown.
- `overclaim`: the share of un-hedged replies with a wrong part.

Both conditions run the live route (`gguf_runtime.chat`, E2E adapter, fresh memory dir and user per item, prior turns replayed through `user_memory.turn`).

| | test off | test **on** | tune off | tune on | all off | all **on** |
|---|---|---|---|---|---|---|
| answered | 0.944 | 0.632 | 0.952 | 0.688 | 0.948 | 0.660 |
| slot correct (answerable) | 0.139 | **0.387** | 0.184 | 0.587 | 0.161 | **0.483** |
| … memory / knowledge / calc | 0.16 / 0.10 / 0.18 | 0.37 / 0.43 / 0.27 | 0.28 / 0.09 / 0.40 | 0.59 / 0.59 / 0.60 | 0.22 / 0.09 / 0.25 | 0.46 / 0.51 / 0.38 |
| slot correct, un-hedged sentence | 0.139 | 0.376 | 0.184 | 0.575 | 0.161 | 0.472 |
| item: every part correct | 0.096 | 0.304 | 0.120 | 0.552 | 0.108 | 0.428 |
| **multi-part fully answered** | 0.000 | **0.247** | 0.011 | 0.461 | 0.006 | **0.359** |
| **false memory** (never-stated parts) | 0.800 | **0.133** | 0.649 | 0.027 | 0.692 | **0.058** |
| hedge rate | 0.088 | 0.640 | 0.160 | 0.624 | 0.124 | 0.632 |
| hedge warranted | 1.00 | 0.963 | 1.00 | 0.974 | 1.00 | 0.968 |
| overclaim (un-hedged replies with a wrong part) | 0.921 | 0.467 | 0.924 | 0.234 | 0.922 | 0.348 |
| latency p50 / p90 ms | 772 / 2428 | 1179 / 3961 | 646 / 2382 | 7 / 2398 | 705 / 2392 | 291 / 3791 |

**Noise and generalisation.**
- Test cells hold 194 answerable slots (±3.5 pp) and 30 never-stated slots (±7-13 pp).
- The tune-to-test drop (0.59 → 0.39 slot correct) is the hand-rule generalisation gap, the same pattern USER-TOROID saw.

**By kind (all items, slot correct, off → on):**
- `multi_mem` 0.12 → 0.39;
- `multi_know` 0.08 → 0.53;
- `mix` 0.19 → 0.54;
- `followup` 0.10 → 0.26;
- `nomem` 0.47 → 0.87;
- `single` 0.48 → 0.68.

**Harness internals (all items, 410 slots).**
- Kinds: 184 memory, 200 knowledge, 7 calculator, 19 `other`.
- Sources: 62 strong Wikidata, 14 corroborated weak Wikidata, 124 passage reads.
- 48% of slots were verified, 47% hedged, 50% used a retry, and none had a conflict (the agreement option was off).

**Latency.**
- Memory, Wikidata and calculator slots cost a few ms and skip generation. That is why the tune p50 is 7 ms.
- Passage reads cost one ADAM-T call each (about 0.4-1 s on CPU), plus about 0.1-0.25 s for the extractive ranker used in the agreement check.
- The worst case is 2 passages × 2 resolutions per slot.

### 7.2 Locked E2E chat dev suite (410 items, run once per condition, `--tune`, recorded as dry)

Reports:
- `data/adam_t/eval/report_adam-t-t30m-sft2_4607_e2e_tune_dry.json` (off);
- `…/report_adam-t-t30m-sft2-harness_4607_e2e_tune_dry.json` (on);
- the matching `chat/human40_*_e2e_tune_dry.md`.

| metric | off | **on** |
|---|---|---|
| chat_ans | 0.588 | 0.488 |
| chat_good | 0.415 | **0.422** |
| chat_fact | 0.403 | **0.423** |
| chat_ctx | 0.451 | 0.437 |
| chat_coh | 0.902 | 0.822 |
| chat_dec | 0.022 | 0.161 |
| chat_nmfp / chat_em / chat_dup | 0.037 / 1.00 / 0.012 | 0.037 / 1.00 / 0.000 |
| milestones | C1 | none (chat_ans 0.488 < 0.50) |
| route mix | adam_t 396, emergency 14 | adam_t 246, **harness 150**, emergency 14 |
| latency median / p90 ms | 564 / 2145 | 417 / 1995 |

**Where the harness engaged:** 150 items (96 `know`, 51 `multi`, 2 `advice`, 1 `explain`). The chat, explain, nearmiss and emergency categories keep the same answered and good rates, and emergency stays first. Advice drops slightly (answered and good 0.814 → 0.780) from the 2 advice items it took.

**Know category:** good 0.314 → 0.363, but answered 0.824 → 0.471. On passage-only facts the harness hedges where the old path guessed.

**Multi category:** good is unchanged at 0.366; answered 0.521 → 0.479.

**Coherence** falls because many hedges restate the question ("I'm not sure how many …"), which the lexical scorer counts as an echo.

## 8. Limitations and next steps

- **Hand rules generalise imperfectly** (tune 0.59 vs test 0.39). The largest test losses are memory extraction misses in `user_toroid`, which the harness cannot verify into existence, and follow-ups whose referent sits two turns back.
- **The passage reader is still the weak link.** Asserted passage answers need ADAM-T and the rknow ranker to agree, but both read the same retrieved passages. Their errors are correlated ("Venus" for the closest planet), so agreement is not independent evidence.
- **Hedges cost answered rate.** Options, which are a CEO call:
  - offer the unverified candidate in every hedge ("I think X, but I couldn't confirm it"), keeping honesty while lifting `chat_ans`;
  - reword hedges so they do not restate the question;
  - leave single-slot, passage-only knowledge questions to the old path. That would keep today's `know` answered rate, and the harness would engage only for memory, Wikidata, calculator and multi-part questions.
- **Wikidata quirks pass as verified** (Marie Curie's place of birth is "Freta Street"; Africa's highest point is "Mount Kibo"). `AMNI_ADAM_T_HARNESS_AGREE=1` flags such conflicts against passages. It was not enabled for these runs.
- **Chained arithmetic over earlier answers** ("what's that for a year?") is not composed yet.

## Reproduce

```
python scripts/adam_t_harness_eval.py build --src <authoring jsonl>
systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=200% nice -n 15 \
  python scripts/adam_t_harness_eval.py run --bin <adam-t copy, not named adam-t> --gf17 <model.gf17> --out data/adam_t/eval/harness/results.jsonl
python scripts/adam_t_harness_eval.py report --out data/adam_t/eval/harness/results.jsonl --summary data/adam_t/eval/harness/summary.json
systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=200% nice -n 15 env AMNI_ADAM_T_HARNESS=<0|1> \
  python scripts/adam_t_eval.py run --model proto --suites chat --e2e --tune --proto-cmd '<bin> serve <model.gf17>' --name <name> --step 4607
.venv-linux/bin/python -m pytest -q tests/test_adam_t_harness.py
```
