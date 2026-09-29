# ADAM-T-HARNESS: verify-and-complete answer loop around ADAM-T

Status: CPU prototype for mission ADAM-T-HARNESS, 2026-09-29.

- **Code:** `amni/serve/adam_t_harness.py`, wired into `adam_t_client.answer()` behind `AMNI_ADAM_T_HARNESS=1` (**off by default**). `gguf_runtime.py` is untouched, so emergency-first is unchanged.
- **Tests:** `tests/test_adam_t_harness.py` (41 tests). The `test_adam_t_route`, `test_user_toroid`, `test_user_memory_*` and `test_decontam_adam_t` suites still pass.
- **Eval:** `scripts/adam_t_harness_eval.py`, own CC0 dev set in `data/adam_t/eval/harness/`. The locked E2E chat dev suite ran once per condition (dry/tune) through `scripts/adam_t_eval.py run --e2e`.
- **Limits honoured:**
  - CPU only; everything ran under `systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=200% nice -n 15`.
  - Model: a read-only snapshot of `exports/adam_t/t30m_sft2/ckpt` at **step 4607**. `core.bin`, `mem.bin`, `state.txt`, `steps.txt` and `log.jsonl` had stable size and mtime over a 5 s window, were unchanged after the copy, and were `cmp`-identical. Exported to `.gf17` in scratch (sha256 `47a3540d…`, argmax parity 97.6%).
  - Binary built with its own `crates/target` and copied to `adamt-har`. Model copies and the target dir were deleted afterwards.

## Summary

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
