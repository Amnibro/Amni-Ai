# USER-TOROID: a 5W user-context memory for ADAM-T

Status: design plus CPU prototype for mission USER-TOROID, 2026-09-28. The CEO made two additions during the mission, relayed by the CTO:

- **Tier 2**, soft retrieval over raw user utterances (section 6);
- **"Borrow what worked"**: both tiers run on the `MemoryBus` spine, and confident recalls are answered directly (section 7).

- **Code:**
  - `amni/serve/user_toroid.py`, wired into `amni/serve/adam_t_client.py` behind `AMNI_UMEM_TOROID=1`, which is **off by default**;
  - `amni/serve/memory_bus.py`: one line. Re-learning a value lifts its old suppression.
- **Tests:** `tests/test_user_toroid.py` (32 tests). The `test_user_memory_*` and `test_adam_t_route` suites still pass.
- **Eval:** `scripts/adam_t_toroid_eval.py`. Data and results are in `data/adam_t/eval/toroid/`.
- **Limits honoured:**
  - CPU only; no GPU was used.
  - The `adam-t-sft2` run was not touched. Its step-3806 checkpoint was copied read-only (size and mtime identical before and after, `cmp` identical), then exported to `.gf17` in scratch.
  - It was served by a copy named `adamt-tor`, under `systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=200% nice -n 15`.

## Summary

- **Retrieval.** Structured retrieval gets the right slot for 66% of answerable held-out questions. The current path gets the fact to ADAM-T in 14% of next-session cases, and in same-session cases buries it among 0.6 same-type distractors on average.
- **Answers.** Answering confident recalls directly from memory (the MemoryBus tier-0 path) lifts end-to-end answer accuracy on the held-out test half:
  - same session: 20% → 49%;
  - next session: 19% → 50%.
- **Hint line.** A memory line placed next to the question lifts accuracy only to 33% and 28%. The 30M model barely uses the new format zero-shot. Even a perfect oracle line gets only 40-49%.
- **Tuning source.** A tuning-set source for the format is needed. It is proposed in section 11.
- **False recall.** Direct answers carry a real cost. On no-fact questions, direct answering claims a memory 40% of the time (8 of 20 items), against 25% (same session) and 20% (next session) for the current path. Most of that comes from confident but wrong extraction.
  - The "unknown" rendering is the best guard found: 10% false recall in the same session and 0% in the next session.
  - Combining the two as `directU` keeps 51% accuracy and brings false recall back to today's level (25% and 20%).

| (held-out test half, 100 dialogues) | current | hint line (both tiers) | hint + "unknown" line | direct answers from memory | direct + "unknown" (`directU`) | oracle line (perfect retrieval) |
|---|---|---|---|---|---|---|
| same session: accuracy | 0.20 | 0.33 | 0.38 | 0.49 | **0.51** | 0.49 |
| next session: accuracy | 0.19 | 0.28 | 0.31 | 0.50 | **0.51** | 0.40 |
| same session: false recall on no-fact items | 0.25 | 0.25 | **0.10** | 0.40 | 0.25 | 0.10 |
| next session: false recall on no-fact items | 0.20 | 0.20 | **0.00** | 0.40 | 0.20 | 0.05 |

**The retrieval side works:**

- **Extraction.** The 5W extractor finds 72% of gold facts. The old parser finds 15%, and 0% of why/how facts.
- **Retrieval.** hit@1 on the test half is 69% with both tiers, 64% with slots alone and 66% with soft retrieval alone. The tune half, where the rules were developed, scores 85-90%.
- **Distractors.** The memory line carries 0.05-0.21 same-type distractors, against 0.6-0.7 in the history.
- **Cost.** Retrieval costs under 0.5 ms at p95.

**The biggest weakness is generalisation of hand rules to unseen phrasing:** 88% tune against 69% test. The confidence signal does not separate those errors well either: direct answers are 92% precise on tune but 65% on test.

## 1. Why this exists

The ADAM-T RCA (`docs/adam_t/rca_grounding.md`, ADAM-T-RCA branch) found:

- **One fact copies well.** With one relevant fact adjacent to the question, sft2 copies it 67-83% of the time.
- **Distractors break it.** With one to four same-type distractors, it falls to chance, and it often copies the first one listed. There is no key-to-value lookup inside the model.

The user-memory path gives it exactly the hard case:

- **Same session.** The fact sits in the history among same-type facts, 0.6 distractor values per question on this dev set.
- **Next session.** The fact reaches the model only if the old parser caught it, which happened in 14% of dialogues (section 9).

So the toroid does the key-to-value lookup outside the model, with hashes and digit compares, then does one of two things:

- **Confident:** answers directly;
- **Otherwise:** hands ADAM-T one short line for the exact slot, immediately before the question.

## 2. Data model (tier 1: slots)

A fact is a slot: `(entity, predicate, W, attribute) -> value`, plus confidence and time.

| field | meaning | examples |
|---|---|---|
| entity | who or what the fact is about. `me` is the user. Others are the noun phrase the user used, lowercased, without determiners or throwaway adjectives. | `me`, `sister`, `dentist appointment`, `favorite color`, `sister's husband` |
| predicate | verb lemma plus verb class | `live` (move, relocate, reside), `keep` (park, store, hide, leave), `go` (fly, travel, commute, head), `drive`, `work`, `be` |
| W | what, where, when, why, how | |
| attribute | a sub-key inside W, 17 values (one GF(17) digit) | '' name color model job breed kind favorite age home work location origin destination likes allergy birthday |
| value | the span as the user wrote it | `dark green`, `the top drawer of my desk`, `because the rent was cheaper` |
| confidence | sixteenths | 0.9 copula/"called", 0.8 marker span, 0.6 ellipsis/alias, 0.5 generic object |
| time | statement clock, cyclic in Z(17^3) (section 3) | |

**Supersession.** `(entity, W, attribute)` identifies a slot, and the latest version wins.
- For `me`, and for what-slots with a real verb, the verb lemma is part of the key. "I moved to Boston" supersedes "I live in Denver", but "why I moved" and "why I sold the boat" stay separate.
- A correction without a verb ("Actually it's red.", "I'm in Sheffield now") inherits the verb and attribute of the slot it corrects.
- Rows written in one statement with the same value are value-linked aliases. "I parked in lot B" writes `me` and `car`, and a correction to one updates both.
- Every write carries `supersedes=<old value>` through the MemoryBus (section 7).

**Entity resolution** is pure lookup:
- "the appointment" resolves to the most recent known entity with that head noun ("dentist appointment"), folding singular and plural.
- "it", "that" or a bare value inside a correction resolve by type. The value's W and attribute pick the most recent non-self entity with that slot, so "it's 9:40 am" goes to the flight and "It's Muffin" to the cat.
- "he", "she" and "they" go to the last person or pet entity.
- Names map through stored name slots: "Priya" becomes `sister`.
- A verb's object ("I quit coffee because ...") and a motion destination ("I go to the gym every Tuesday") also receive the statement's W spans. Real questions address such facts by the object ("Why did I give up coffee?").

## 3. Addressing: GF(17) torus coordinates and pages

Every slot is a point on a discrete torus. Each axis is a run of base-17 digits, each cyclic mod 17:

| axis | digits | torus | source |
|---|---|---|---|
| entity | 8 (two RGBA pixels) | Z17^8 | 32-bit hash of the entity label (`user_memory.bh`) |
| predicate class | 4 | Z17^4 | 16-bit hash of the verb class |
| W, attribute | 2 (u16 = W·17 + attr) | Z5 × Z17 | |
| time | 3 | Z17^3 = Z4913 | statement clock mod 4913 |
| confidence, source | 2 | Z17 × Z17 | |
| verb lemma | 4 | Z17^4 | 16-bit hash; separates "park" from "keep" |

**Pages.** One slot is one 48-pixel row. Each pixel holds the 4 GF(17) Reffelt digits of a u16, one per RGBA channel, via the same `encode_fp16_to_rgba4` as `user_memory`.

- **Pixels 0-7:** the coordinates.
- **Pixels 8-47:** the label, verb and value text.

Tier 2 utterances use 32-pixel rows:
- **Header row:** row count, clock, key count, byte length, and up to 14 entity/noun key hashes, 2 pixels each.
- **Payload rows:** raw deflate with a fixed chat preset dictionary.

`<salted user key>.torus` holds a header with CRC32, then both pages as raw digits. The decoder rejects any CRC mismatch or any digit ≥ 17. The file is mode 0600, written atomically, and sits beside the `user_memory` `.ptex` file.

**Retrieval is a sub-torus selection.** A question fixes some axes and asks for another:

- "What color is my car?" fixes entity = h(car), W = what, attribute = color.
- "Why did I move?" fixes entity = h(me), W = why, predicate = h(live).
- The lookup is a vectorised digit-column mask `isin(entity, question entities) & (W == w)`. Attribute and predicate agreement score the survivors.
- There is no matmul and no embedding.
- **Latency:**
  - per-dialogue store: 0.11 ms p50 for slots, 0.35 ms for both tiers;
  - all 200 dialogues pooled as one heavy user (594 slots, 277 utterances): 12 ms.

**Time on the torus.** Age is `(head - t) mod 4913` and the youngest version wins. This is serial-number arithmetic on Z4913, correct while the live span stays under 2456 statements.
- `Store.rebase()` runs on save. When the oldest row nears half the circle, it renumbers rows by age rank, which keeps the order and frees the circle.
- A test crosses the wrap point and forces a rebase.

**Lorentz / rapidity.** Considered and not forced in (`docs/adam_t/lorentz_prereg.md`):
- Memory only needs order and relative age, and the cyclic clock gives both exactly.
- The one natural fit is tier 2's recency term. Linear in rank in log space is e^(−ρ·age), a boost with fixed rapidity ρ (0.35 per step after tuning).
- A content-gated rapidity is the next step if recency ever needs to depend on the fact type.

## 4. Extraction from user turns (no ML)

`extract()` is a lookup and pattern parser layered on `user_memory`. It reuses:

- the MEM-PARSE forget detector `_fg`, with all vetoes;
- the question and request filters `_isq` and `_REQ`;
- the kin/pet class bits (`CLS & 32`), the colour lexicon (`GX`) and the function-word set (`FUN`);
- the word hash.

Per clause (split on sentence ends, ", and my ...", ", I ...", ", <NP> is ..."):

1. **Subject.**
   - "my"/"our" plus a noun phrase, including possessive chains ("my dog's name" → dog + name) and "my sister Priya" (a name slot).
   - "I"/"we" → `me`.
   - Pronouns, known names, "the X", or a bare "X" noun phrase followed by a verb.
2. **Auxiliaries and verb.** Copula, "going to", modals, particles ("signed up", "locked up"). Coordinated verbs ("lives in Austin and works at a bakery") re-key the later spans.
3. **W spans in one left-to-right pass, mostly word-table lookups.**
   - **why:** because / since / so (that) / in order to / "to" + base verb (not after a catenative) / "for" + purpose noun.
   - **when:** a token time-phrase scanner over days, months, clock times, years, ordinals, units and glue ("the second Friday of June", "every other Tuesday", "at the end of the month").
   - **how:** by + means, on foot, over the phone/email, online, via, with/using + tool. After a cooking verb, "in a cast iron pan" is how.
   - **where:** a preposition plus a proper name, a place noun, or a determiner noun phrase, with "of ..." and one locative extension ("the blue flowerpot by the back door", "lot B"). The attribute comes from the verb class: live → home, work → work, from → origin, keep/park/be → location, motion + to → destination.
   - **what:** the copula or object complement, typed as colour, age, job (people), breed (pets), name or kind. Also "called/named", "allergic to", "works as", "drive a", and "have/bought/adopted a X named Y", which creates entity X.
4. **Corrections.**
   - Leading discourse is stripped: "actually", "wait", "scratch that", "make that", "turns out the vet says".
   - "not X" contrasts and fillers ("these days", "after all") are dropped.
   - A bare value or "it's X" is typed and attached to the slot it corrects.

**Dev set result (all 200 dialogues, 487 gold facts):**

| extractor | precision | recall | what R | where R | when R | why R | how R |
|---|---|---|---|---|---|---|---|
| user_toroid | 0.46 | **0.72** | 0.69 | 0.81 | 0.72 | 0.67 | 0.64 |
| user_memory (old, prod mode) | 0.65 | 0.15 | 0.37 | 0.24 | 0.04 | 0.00 | 0.00 |

- **Why precision is low.** The toroid stores facts the gold data does not list: the user's own "what" objects ("me – drink – green tea"), alias rows, and weak entities.
- **Why that is mostly harmless.** A question selects one slot, so these rows rarely surface. When a false positive does surface, it shows up as a wrong direct answer or line, and that is measured below.

## 5. Retrieval by question type

`ask()` parses the question into `(entities, W, attribute, predicate)`:

- It strips "remind me", "do you remember", "did I say/tell you" and "again", and splits "time's my" into "time is my".
- It maps wh-forms to W:
  - "what time/day/date" → when;
  - "what's the point/reason", "how come" and "what for" → why;
  - "how old" → what/age;
  - "what colour/kind of/model/breed" → attribute plus candidate entity;
  - "who" → name;
  - "what does X do" → job.
- The where attribute comes from the verb.
- Entities are "my/our" noun phrases, then "the X", then known labels found anywhere in the question (unigram/bigram lookup), then `me`. The first is primary.

`lookup()` scores the candidate slots:

- **entity:** primary 3.0, other 2.0;
- **attribute:** same +2, compatible (kind/breed/model) +1, conflicting −2.5, question attribute against a row without one −0.5;
- **predicate:** class +1.5, lemma +0.5;
- confidence, and a tiny age term.

For `me`, a row with a different content verb is not a candidate, so "Why am I flying?" never answers "why I walk to work". The exception is generic motion in how-questions. A slot needs a score of at least 2.0.

## 6. Tier 2: soft retrieval over raw utterances (CEO addition 1)

**Store.** Every user statement, not questions or requests, is stored whole and compressed in the utterance page.
- It is keyed by up to 14 hashes: the resolved entities first, then "my X" phrases, `me`, and content nouns.

**Retrieve and score.**
1. The question's entity hashes select candidate utterances with one vectorised `isin` over the header key digits.
2. The candidates are decompressed and re-parsed, with the entity context they had at store time, into typed candidate spans. Those are the extractor's spans plus every raw W span and every colour word.
3. Each span gets a linear score:
   - entity: primary 2.0, any question entity 1.0;
   - W type fit: 2.5;
   - attribute: +1 on a match, −1 on a conflict;
   - predicate class: +1 on a match, −1.5 for a mismatched self verb;
   - recency: −0.35 per newer candidate utterance;
   - superseded by a newer same (entity, W, attribute): −3;
   - agreement with the tier 1 slot: +1.5;
   - clause subject: +0.5.
4. **Softmax** at temperature 0.75, with a **null candidate** at 4.5. The null candidate is the probability that memory does not hold the answer. Identical spans pool their probability.

**Weights.** Null, supersession, recency and temperature were tuned on the tune half only:
- the first three by accuracy, with ties broken toward the more conservative (higher) null;
- the temperature by NLL.
- The other weights are hand-set.

**Combination.** A tier 1 slot, when present, supplies the line or answer. Its confidence is the soft probability mass that agrees with it. Tier 2 fills in when there is no slot.
- An earlier version let a confident tier 2 override the slot. It lost on corrections, because tier 2's per-utterance re-parse cannot see store-time inheritance ("I'm in Sheffield now").
- On test, tier 2 fills 9 of 100 questions that had no slot, 5 of them correctly. It breaks none that the slot tier got right.

## 7. MemoryBus spine and the answer policy (CEO addition 2: "borrow what worked")

The toroid is not a parallel memory. **Writes and reads go through `amni.serve.memory_bus.MemoryBus`.** Each user gets their own bus instance, with the per-user `.torus` store as its homes:

| MemoryBus home | per-user binding | notes |
|---|---|---|
| `answer_lut` (tier 0, `store/lookup/commit_recent/rollback_recent`) | `SlotLUT`: tier 1 slots | the exact key `slot::<ent>␟<verb>␟<W>␟<attr>␟<conf>` for verify; free questions go through `recall()` and return only when the slot confidence ≥ `AMNI_RECALL_DIRECT_GATE` (0.90) |
| `sem_lut` (tier 2, `lookup_soft(q, k=1, return_diag=True)`) | `SoftLUT`: tier 2 top span | returns `(span, None, p_top, p_2nd)`. The bus's own gate (0.90) and margin (0.04) apply. |
| `kb`, `adam` (global PTEX `teach`), `learning_atlas` | **None** | user facts never reach global, federable or shared stores |
| ledger | `os.devnull` | the anti-pattern set lives per bus instance. No plaintext user text is appended to `data/corrections.jsonl`. |

**Write.** `ingest()` calls `record_learning(key, value, kind='user_fact', provenance='user:toroid', exactness='exact', supersedes=<old value or None>)` for every fact.
- The bus stores the fact, re-reads it through `recall(key)`, and commits it only if the read returns the intended value. This is verify-after-write.
- Otherwise the row is rolled back. A test forces a failed verify and checks the rollback.
- Corrections supersede. The bus puts the old value in the anti-pattern set, so it can never be served as a tier-0 answer. Re-learning a value lifts its suppression; that is the one-line `memory_bus.py` change.

**Read (`respond()`).**
1. **Direct.** `bus.recall(question)` returns a value at tier 0 (confident slot) or tier 2 (confident soft span, margin-checked). The reply comes from a short template, with no ADAM-T call: "Your car is red.", "Your brother lives in Tucson.", "Because the rent was cheaper.", "By train.", "It's next Tuesday at 3 pm.", "Your dog's name is Biscuit." "my/I/our" in the value become "your/you".
2. **Hint.** Otherwise, if the toroid line's confidence is at least `AMNI_UMEM_TOROID_HINT` (0.5), the line goes next to the question and ADAM-T phrases the answer.
3. **Low confidence**, one of:
   - the top-2 "(unsure)" line, if p_top ≥ 0.25;
   - with `AMNI_UMEM_TOROID_UNK=1`, the "unknown" line for a known entity without that slot;
   - with `AMNI_UMEM_TOROID_ABSTAIN=1`, a direct "You haven't told me that yet." when the null probability is at least the gate;
   - otherwise the model answers alone.

**PII and federation boundary.**
- **Federation.** The federable export (`federation_store._select_lessons`) reads only the global `adam.sem_lut`. A user-fact bus has `adam=None`, so it never teaches that LUT.
- **Metadata.** `record_learning` marks `kind='user_fact'` as `federable: False`, because it is not one of `fact`/`skill_lesson`.
- **Separation.** The global agent `memory_bus` never receives user facts.

**Serve integration** (`adam_t_client.context()`, with `AMNI_UMEM_TOROID=1`):
- **Statements** write both stores (legacy `user_memory` and the toroid).
- **Direct answers** set `c['reply']`, so `answer()` returns without generating. `AMNI_UMEM_TOROID_DIRECT=0` turns direct answers into hint lines.
- **Hint and unsure lines** go last in `rc`, which the template puts immediately before the question: `User: Memory: your car, color: red. What color is my car?\nAdam:`.
  - With a line present, the legacy preamble is dropped (`AMNI_UMEM_TOROID_PRE=1` keeps it), and the calculator and knowledge retrieval are skipped.
- **No toroid output** leaves everything exactly as before.
- **Errors** are counted in `c['n']['tor_err']` and never break the answer path.

## 8. Privacy and forget

- **Per user and local.**
  - The file name is `user_memory`'s salted SHA-256 key. The file is mode 0600 and written atomically.
  - The per-user bus writes nowhere else, and its ledger is `/dev/null`.
  - The module logs nothing. The client's debug log only gains a `tor:<n>` count.
- **Forget uses `user_memory._fg`**, so every MEM-PARSE veto holds ("I forgot my keys", "Delete the file on my desktop", "Clear my schedule", "Never mind" all do nothing, and a test checks them). "Erase my dentist appointment" is also vetoed. That is the existing MEM-PARSE behaviour, since it may be a calendar command.
  - **Entity forget:** "Forget what I told you about my car" removes all of that entity's slots.
  - **Slot forget:** "Delete where my brother lives" removes that W only. For `me` it also narrows by verb: "Forget where I parked" removes only the park slots.
  - **Aliases:** value-linked aliases from the same statement go too.
  - **Utterances:** every tier 2 utterance keyed by the entity is removed. For a slot forget, only those containing a span of that W. This deliberately over-deletes utterances that mention two people.
  - "Forget everything" deletes the file.
- **Stated plainly:** tier 2 keeps raw user sentences. They are deflated and digit-encoded, which is not encryption. That is the same protection as today's `.ptex` user memory, with more text. At-rest encryption (Amni-Crypt) should come before this flag defaults on.

## 9. Offline evaluation

**Dev set:** `data/adam_t/eval/toroid/toroid_dev.jsonl`, 200 dialogues, CC0. It was written fresh for this mission by a separate agent that saw neither the parser nor the chat eval set.

- **Mix:** 40 per W. Per W: 13 plain (chit-chat and other-entity distractor turns), 9 corrections (3 with an elided entity), 5 forgets, 9 multi-entity (2-3 same-type entities), 4 no-fact.
- **Gold:** per-turn facts, the question address, accepted answers and forbidden values.
- **Decontamination:** 0 six-gram overlap with `data/adam_t/eval/chat/chat_src.jsonl`, checked by `build`. Eight generic chit-chat lines were reworded to reach 0.
- **Split:** odd ids are the **tune** half, which rules and weights were developed on. Even ids are the **test** half. Only test aggregates were ever looked at, never test items.

**Retrieval hit@1.** For answerable questions, the right value comes first and no forbidden value appears. For unanswerable ones, nothing is returned.

| split | tier | hit@1 | answerable | abstain ok | correction (supersession) | what | where | when | why | how | p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test | slot | 0.64 | 0.60 | 0.80 | 0.60 | 0.65 | 0.60 | 0.60 | 0.65 | 0.70 | 0.11 / 0.14 |
| test | soft | 0.66 | 0.63 | 0.80 | 0.60 | 0.60 | 0.70 | 0.65 | 0.65 | 0.70 | 0.30 / 0.41 |
| test | both | **0.69** | **0.66** | 0.80 | 0.64 | 0.65 | 0.75 | 0.60 | 0.70 | 0.75 | 0.35 / 0.46 |
| tune | slot | 0.90 | 0.88 | 0.96 | 0.90 | 1.00 | 0.95 | 0.90 | 0.80 | 0.85 | 0.10 / 0.15 |
| tune | soft | 0.85 | 0.84 | 0.88 | 0.75 | 0.85 | 0.90 | 0.90 | 0.80 | 0.80 | 0.27 / 0.41 |
| tune | both | 0.88 | 0.89 | 0.84 | 0.90 | 0.90 | 0.95 | 0.90 | 0.80 | 0.85 | 0.33 / 0.46 |

Test by kind, both tiers: plain 0.73, correction 0.64, forget 0.70, multi-entity 0.60, no-fact 0.90.

**What the current injection shows ADAM-T** (`cover`: the real `user_memory.turn` + `adam_t_client._mem` on the same dialogues):

| setting | answer present in context | same-type forbidden values in context | toroid line has the answer | forbidden values in the line |
|---|---|---|---|---|
| same session, test | 0.95 (history) | 0.59 | 0.60 | 0.21 |
| next session, test | **0.14** (preamble; empty 71% of the time) | 0.06 | 0.60 | 0.21 |
| same session, tune | 0.97 | 0.67 | 0.87 | 0.05 |
| next session, tune | 0.15 | 0.08 | 0.87 | 0.05 |

**Soft probability calibration.** Confidence is the top-span probability when a line is emitted, otherwise the null probability. Correct means hit@1.

| split | tier | ECE | Brier | mean p | accuracy |
|---|---|---|---|---|---|
| tune | soft | 0.10 | 0.14 | 0.80 | 0.85 |
| test | soft | 0.21 | 0.26 | 0.78 | 0.66 |
| test | both | 0.26 | 0.28 | 0.84 | 0.69 |

The test reliability curve is roughly monotone, but the ≥ 0.9 bin is right only 74% of the time. The probability ranks candidates, but it is over-confident off the tune distribution.

**Direct-answer precision against the gate** (retrieval only, no model):

| gate | tune: rate / precision / direct answers on no-fact items | test: rate / precision / direct answers on no-fact items |
|---|---|---|
| 0.80 | 0.66 / 0.92 / 0.00 | 0.61 / 0.64 / 0.20 |
| 0.90 (bus default) | 0.59 / 0.92 / 0.00 | 0.55 / 0.66 / 0.20 |
| 0.95 | 0.49 / 0.94 / 0.00 | 0.50 / 0.66 / 0.20 |
| 0.98 | 0.41 / 0.93 / 0.00 | 0.36 / 0.72 / 0.15 |

Raising the gate barely helps on test. The errors are extraction mistakes that tier 1 and tier 2 share, so their agreement is not independent evidence.

**Cost:**
- ingest 0.22 ms per turn;
- 192 bytes per slot and about 256 bytes per utterance on disk;
- all 200 dialogues pooled: 185 KB for 17 KB of raw text. The digit encoding expands the payload 4× by design.

## 10. End-to-end answer accuracy with ADAM-T (CPU)

**Setup.**
- **Model and serving:** sft2 at step 3806, exported to `.gf17`, served on the CPU LUT path with the serve's greedy defaults.
- **Prompts:** built with the real `template()` (budget 2928 bytes, reply cap 160 bytes).
- **Two settings:**
  - **same session:** the dialogue is in the history;
  - **next session:** the facts exist only in stored memory, and the history is empty. The current path then relies on the `user_memory` preamble.
- **Scoring.** An answerable item is correct when an accepted answer appears and no forbidden value does. An unanswerable item is correct when the reply abstains and names no forbidden value.
- **False recall.** On the no-fact items (20 in test, 45 overall), the reply claims a memory without abstaining. That covers a forbidden value, "you told me ...", "your X is/lives ...", a bare "Because ..."/"By ..." answer, or any direct answer.
- **Noise:** n = 100 per test cell (±9-10 pp), and 20 for false recall (±20 pp).

**Conditions:**
- `cur`: today's path (history plus the `user_memory` preamble).
- `slot0`/`soft0`/`both0`: the memory line, zero-shot. The legacy preamble is dropped whenever a line exists.
- `slotF`/`bothF`: the same, plus three in-prompt demos.
- `bothU`: `both0` plus the "unknown" line.
- `turnB`: the retrieved source sentence, re-injected as the last user turn. This is the format sft2 was tuned on.
- `gold0`/`goldF`: an oracle line built from gold annotations.
- **The MemoryBus answer policy of section 7:**
  - `direct`: direct at ≥ 0.90, hint at ≥ 0.5, unsure at ≥ 0.25, otherwise the model alone;
  - `directA`: `direct` plus a direct abstention;
  - `directU`: `direct`, but the "unknown" line replaces the unsure line.

| condition | test same: acc | test same: false recall | test next: acc | test next: false recall | all same: acc | all next: acc |
|---|---|---|---|---|---|---|
| cur | 0.20 | 0.25 | 0.19 | 0.20 | 0.17 | 0.18 |
| slot0 | 0.30 | 0.25 | 0.25 | 0.20 | 0.32 | 0.30 |
| soft0 | 0.30 | 0.25 | 0.26 | 0.30 | 0.33 | 0.33 |
| both0 | 0.33 | 0.25 | 0.28 | 0.20 | 0.34 | 0.32 |
| slotF | 0.35 | 0.25 | 0.23 | 0.20 | 0.36 | 0.26 |
| bothF | 0.38 | 0.25 | 0.22 | 0.20 | 0.37 | 0.26 |
| bothU | 0.38 | **0.10** | 0.31 | **0.00** | 0.40 | 0.34 |
| turnB | 0.19 | 0.25 | 0.29 | 0.30 | 0.18 | 0.31 |
| gold0 (oracle line) | 0.49 | 0.10 | 0.40 | 0.05 | 0.50 | 0.46 |
| goldF | 0.50 | 0.15 | 0.39 | 0.10 | 0.48 | 0.41 |
| **direct** | 0.49 | 0.40 | 0.50 | 0.40 | 0.54 | 0.57 |
| directA | **0.52** | 0.35 | **0.53** | 0.30 | 0.59 | 0.62 |
| **directU** | 0.51 | 0.25 | 0.51 | 0.20 | 0.59 | 0.60 |

The tune half follows the same order at higher levels. For example, `direct` scores 0.59 (same session) and 0.65 (next session), and `directU` 0.67 and 0.69.

**Routes for `direct` (test half; counts, then accuracy):**

| route | count | same session | next session |
|---|---|---|---|
| direct | 55 | 65% | 65% |
| hint | 12 | 42% | 33% |
| unsure | 13 | 38% | 31% |
| model alone | 20 | 15% | 30% |

**Findings:**

1. **The line helps, but the model barely reads the new format.**
   - Test accuracy rises from 0.20 to 0.33 in the same session and from 0.19 to 0.28 in the next session.
   - Even the oracle line reaches only 0.49 and 0.40. The ceiling is the model, not retrieval.
   - Typical failures:
     - how/why questions read as advice requests ("You can take a few hours to get to work.", "You can cook your rice by using a simple ... method");
     - replies about "memory" itself ("You can use the following memory: 1. ...");
     - the RCA's confabulation template ("You told me he's a bus.").
   - why/how are weakest even with the oracle (0.30/0.25 in the test same session).
2. **In-prompt demos are unstable.**
   - They add 5 pp in the same session but cost 2-6 pp in the next session.
   - They double the prompt (about 600 bytes against 300) and the generation time (1.5 s against 0.8 s p50).
3. **The "unknown" line is the cheapest false-recall guard.** False recall falls from 0.25 to 0.10 (same session) and from 0.20 to 0.00 (next session), with no loss of accuracy.
4. **Re-injecting the source sentence (`turnB`)** helps only in the next session (0.29). In the same session (0.19) the sentence is already in the history.
5. **Direct answers from memory are the largest lever.**
   - Test accuracy is 0.49-0.53, against 0.20 for today's path.
   - They cost false recall on no-fact items (0.40 for `direct`), because a confident wrong extraction is answered as fact.
   - `directU` keeps most of the gain (0.51) and brings false recall back to today's level (0.25 and 0.20).
6. **Calibration of confidence against final correctness:**
   - `direct` policy: ECE 0.33 on test, 0.12-0.13 on tune;
   - hint conditions: ECE 0.45-0.65, because the model often fails to use a confident line.
7. **Latency (CPU LUT, 2 threads):**
   - `respond()` takes 0.5 ms p50;
   - a direct answer skips generation entirely. The hint path costs 0.8 s (same session) or 0.35 s (next session) p50. The p50 generation time in the direct rows reads 0 because most questions skip the model;
   - the hint line adds about 35 prompt bytes.

**Code-version note.** The ten hint and oracle conditions ran before the final rule and weight edits. The `direct*` rows and all offline numbers use the final code. Replaying the final code on the same dialogues reproduces the recorded hint lines for 196/200 (`both0`), 200/200 (`slot0`), 186/200 (`soft0`) and 196/200 (`bothU`).

## 11. Tuning-set source: needed. Proposal `tmem`

**It is needed:**
- A perfect memory line gets only 40-49% zero-shot.
- The model treats "Memory:" as a topic, and demos are unstable.
- Direct answers sidestep this for about 55% of questions. But the hint and unsure paths, 25% of questions, and every paraphrase that misses the gate still need ADAM-T to read the line.

**Proposed SFT source `tmem`:** CC0, templated, with no eval text.

- **Prompt:** the exact serve rendering. Optional history of 0-6 turns, including same-type distractor facts, then `User: Memory: <label>, <prop>: <value>. <question>\nAdam:`.
- **Reply:** a short natural answer that copies the value verbatim, drawn from the section 7 templates plus two or three paraphrases each ("Your car is red.", "It's red.").
- **Coverage:**
  - 5 W × 17 attributes;
  - self, kin, pets, objects and events;
  - 30% multi-token and invented values;
  - about 40 question paraphrase patterns per W ("remind me", "did I say", "what's the point of", "how do I usually ...").
- **Contrastive twins** (RCA fix 2). Every question appears in five forms:
  1. with the right line → the answer;
  2. with an `unknown` line → "You haven't told me that yet.";
  3. with no line and no relevant history → the same abstention. This targets sft2's 57% confabulation rate;
  4. with an `X or Y (unsure)` line → "I think it's X, but you also mentioned Y.";
  5. with the line plus history holding a superseded value → the line's value (the line beats the history).
- **Binding practice** (RCA fix 1 data): 20% of prompts carry two memory lines for different entities. The answer comes from the line whose label matches the question.
- **Size and weight:**
  - about 8k examples at weight 3 in the next mix (umem is 4);
  - 6-gram decontamination against `chat_src.jsonl` and `toroid_dev.jsonl`;
  - the builder sits beside the existing SFT builder.
- **Acceptance.** Re-run `adam_t_toroid_eval.py model` on the tuned model. Targets:
  - `gold0` ≥ 0.85;
  - `both0` within 5 pp of retrieval hit@1;
  - `bothU` false recall ≤ 0.05;
  - `directU` ≥ 0.70 on test.

## 12. Limitations and open decisions

- **Brittle rules.** Hand rules are brittle on unseen phrasing: 88% tune against 69% test. More patterns help linearly. A learned span tagger on top of the lookups would help more, but it breaks "no ML" and is a CEO decision.
- **Confident errors.** Direct answers make wrong extractions confident. On test they are 65% precise, and they claim a memory on 20% of no-fact items. **Recommendation:** ship `directU` (direct plus "unknown") only after the tuning source lands, and add a user-visible "not right? tell me" path that feeds `supersedes`.
- **Entity identity is the label.** "my friend" and "my best friend" are different entities, and two unnamed friends collide.
- **Tier 2 is stateless per utterance.** That is why the combination is slot-first.
- **Raw text at rest is not encrypted** (section 8).
- **Existing `.ptex` facts are not imported.** A one-shot migration through `user_memory.render()` plus `ingest()` is not built.

## Reproduce

```
python scripts/adam_t_toroid_eval.py build                     # authoring files -> toroid_dev.jsonl + 6-gram check (toroid_dev.jsonl is canonical)
python scripts/adam_t_toroid_eval.py offline --tune --out data/adam_t/eval/toroid/offline.json
python scripts/adam_t_toroid_eval.py cover --fw data/adam_t/eval/toroid/offline.json --out data/adam_t/eval/toroid/cover.json
systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=200% nice -n 15 \
  python scripts/adam_t_toroid_eval.py model --gf17 <model.gf17> --bin <adam-t copy, not named adam-t> \
  --fw data/adam_t/eval/toroid/offline.json --out data/adam_t/eval/toroid/model_results.jsonl \
  [--conds cur,slot0,slotF,soft0,both0,bothF,bothU,gold0,goldF,turnB,direct,directA,directU]
python scripts/adam_t_toroid_eval.py report --out data/adam_t/eval/toroid/model_results.jsonl --summary data/adam_t/eval/toroid/model_summary.json
.venv/bin/python -m pytest -q tests/test_user_toroid.py
```
