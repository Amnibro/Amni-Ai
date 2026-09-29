# RCA: why ADAM-T does not ground its answers in context

Status: controlled-probe RCA for mission ADAM-T-RCA, 2026-09-28. CPU only. No GPU was used, and the `adam-t-sft2` run was not touched: its checkpoint was copied read-only, with size and mtime checked before and after the copy.

## Summary

The CTO's hypothesis is **partly right**. ADAM-T does copy from context, but it cannot pick the *right* item. The cause is not "it won't copy". It is "it has no working key-to-value lookup". Five findings:

1. **Single-fact copying works.** With one fact in the prompt, the SFT models copy it:
   - sft2: user memory 53-83%, passages 43-77%, Wikidata 40-100%, calculator 90-100%;
   - sft3: the same, except user memory drops to 23-67% because of the templates in finding 3.
   - This holds at 900 bytes of distance, for rare and 9-token invented names, and for counterfactuals: swapped real capitals such as "Italy, capital: Madrid." are copied in 30/30 items by both SFT models.
   - Context beats world priors when it is the only candidate.
2. **Binding fails.** Add one to four same-type distractor facts and every model falls to about chance among the in-context candidates.
   - The model copies *a* name of the right type. For passages and Wikidata lines it usually takes the **first** one listed, not the one that belongs to the asked entity.
   - Every model does this, base included. On the synthetic induction test (a repeated random word list) the base model predicts only 6-10% of the second copy. SFT models reach 14-36%. A working induction circuit gets more than 90%.
   - This matches the E2E "Mongolia -> Hohhot" and "13 gauge" failures. Live context always carries several same-type entities: two passages plus a Wikidata line.
3. **Template prior on memory questions.** When SFT v3 is asked about a fact the user stated one turn earlier, it answers "I don't know ... You haven't told me yet" 47% of the time. It does this although it assigns the answer 100% multiple-choice accuracy, and once given the reply frame it puts a median p=0.76 on the answer's first token (0.97 on the example below).
   - Paraphrased questions push this to 70%.
   - The round-2 SFT (`sft2`, v5 mix) cuts the template rate to 12-13%. But it now confabulates when no fact exists ("You told me he's called Tomato.": 57% of no-fact memory questions).
4. **SFT did not damage copying.** In identical plain-text framing, SFT v3 beats the base model on every copy metric:
   - forced-frame multiple-choice accuracy 86% vs 75%;
   - teacher-forced top-1 70% vs 24%;
   - induction 23-36% vs 6-10%.
5. **No serving bug.**
   - Tokenisation is byte- and token-identical between the SFT builder and the served prompt (1470/1470 probes; 2400/2400 SFT examples re-tokenised).
   - LUT vs burn: first reply token agrees on 50/50 probe prompts and serve-decoded replies are identical on 48/50.
   - One train/serve template drift exists: Wikidata lines gained a property label on 2026-09-28, after v3 and v5 were built. The drift is benign: the new format scores higher.

Ranked root causes and fixes are at the end. The top lever is a key-matching copy path (a pointer head) plus associative-recall training data. SFT mix rebalancing is second.

## What was run

| item | path / value |
|---|---|
| probe set | `data/adam_t/probe/probes.jsonl` (1470 items, 49 cells of 30, own CC0 synthetic text, invented entities, no eval-set content), `meta.json` |
| script | `scripts/adam_t_probe.py` (`build`, `run`, `report`, `induct`, `fid`, `tok`, `audit`) |
| chat template | `amni.serve.adam_t_client.template(q, history, 2928, (), rc)`, the exact serve assembler (history turns, then `User: <rc items> <question>\nAdam:`) |
| base framing | context lines + `\nQuestion: <q>\nAnswer:` (the base model never saw `User:/Adam:`) |
| models | `base` = `exports/adam_t/t30m_bpe4096/model.gf17` (step 35000, 573M tokens); `sft3` = SFT v3 step 4638 exported from `exports/adam_t/t30m_sft/ckpt` (sha256 `02caacbe...`, byte-identical to the model in the E2E report); `sft2` = round-2 SFT (v5 mix) step 2993, a stable snapshot of `exports/adam_t/t30m_sft2/ckpt`; `sft3_plain` = sft3 on the base framing |
| serving | LUT path (`adamt-rca serve <model.gf17>`), serve decoding defaults (greedy, rep_penalty 1.15, no_repeat_ngram 3, stop_repeat); base uses plain greedy |
| new serve ops | `crates/adam-t/src/main.rs`: `enc` (tokenise), `nxt` (next-token log-probs, optional raw-token tail, batched, both paths), `probe` (one KV prefill: first-reply-token distribution, the model's own serve-decoded reply, then per-token log-prob and rank of each candidate continuation after a forced frame; `lut::Kv` is now `Clone`); `gen` accepts a prefilled state |

### Cells (30 items each)

- Fact type:
  - `umem`: a user-memory turn `User: <fact>\nAdam: Got it, thanks for telling me.`, the serve's fixed ack, placed after 2 filler turns;
  - `passage`: a sentence in a `<Title>: <sentences>` passage;
  - `wikidata`: a `<Label>, <property>: <value>.` line, the current `_wl` format;
  - `calc`: a `Calculator: a x b = c.` line.
- Factors, varied one at a time from the base cell (1-token common answer, fact adjacent to the question, no distractors, exact phrasing):
  - distance: umem 0/1/2/4/6 filler turns, 45-613 bytes; passage and Wikidata about 50/200/450/900 bytes;
  - distractors: 1/2/4 same-type facts about other people or entities, in random order with the true fact;
  - answer class: `c1` 1 BPE token common, `c2` 2-3 tokens common, `r2` 2-3 tokens invented, `c4` 4+ tokens common (about 7), `r4` 4+ tokens invented (about 9.4); calc uses result size;
  - phrasing: exact vs paraphrased question;
  - `nofact`: the fact is removed (control);
  - Wikidata only:
    - `real_true` / `real_cf`: a real country with its true or a swapped capital, testing prior vs context;
    - `fmt old` / `fmt_cf old`: the pre-2026-09-28 line format `<Label>: <value>.` that SFT v3/v5 were trained on.
- Scores:
  - `exact`: the answer appears verbatim and no distractor value is in the reply;
  - `span`: case/punctuation-insensitive containment;
  - `tpl`: a canned or fallback reply ("haven't told me", "don't know", "thanks for telling", "not sure", ...) that does not contain the answer;
  - `MC`: forced-frame 4-way choice. Log-prob of `<frame> <candidate>` for the answer vs 3 alternatives: the distractor values first, then same-class pool values. Chance is 25%. With k distractors the effective chance among in-context candidates is 1/(k+1).
  - `TF`: teacher-forced greedy. Every answer token is top-1 after `<frame>`.
  - `p(open copy)` / `p(open tpl)`: probability of the first reply token being the answer's or the frame's first token, vs the sum over canned openings (" I", " You", " Oh", " Got", " Thanks", " Good", " Nice", " Okay", " Sure", " My", " If", " Sorry", " Hmm", " Noted", " Great").
- Noise: n=30 per cell gives a 95% interval of about ±15-18 pp per cell. Conclusions rely on patterns across cells and models, not on single cells.

## Results

Cells show `exact/span/tpl/MC/TF` in %.

### User memory (`umem`)

| cell | dist B | ans tok | base | sft3 | sft2 | sft3_plain |
|---|---|---|---|---|---|---|
| base c1 | 45 | 1.0 | 0/0/13/83/60 | 53/53/47/100/77 | 83/83/13/97/87 | 30/30/37/90/67 |
| dist 1 turn | 146 | 1.0 | 3/3/13/93/50 | 33/33/67/100/80 | 73/73/13/100/67 | 27/30/40/90/53 |
| dist 2 | 239 | 1.0 | 7/7/10/90/43 | 33/37/60/97/77 | 67/67/17/93/77 | 43/43/30/93/70 |
| dist 4 | 422 | 1.0 | 0/0/0/97/57 | 57/60/30/100/87 | 73/73/7/97/73 | 23/33/13/93/73 |
| dist 6 | 613 | 1.0 | 7/7/3/90/53 | 53/57/27/97/77 | 67/67/3/97/63 | 30/33/17/93/60 |
| distr 1 | 84 | 1.0 | 0/0/43/47/30 | 47/47/3/47/43 | 43/43/3/47/40 | 27/30/20/43/30 |
| distr 2 | 116 | 1.0 | 0/0/67/40/23 | 33/33/0/30/27 | 47/47/0/50/43 | 20/20/7/33/23 |
| distr 4 | 184 | 1.0 | 0/0/90/30/17 | 30/30/0/37/30 | 33/33/0/47/37 | 20/20/13/47/27 |
| cls c2 | 45 | 2.7 | 0/0/33/100/67 | 67/67/33/100/87 | 73/73/23/100/87 | 40/40/33/100/70 |
| cls r2 | 45 | 3.0 | 7/7/20/100/53 | 47/47/50/100/73 | 77/77/13/100/83 | 30/30/43/100/67 |
| cls c4 | 45 | 7.0 | 10/10/33/100/77 | 53/53/40/100/80 | 73/73/20/100/93 | 37/37/40/100/60 |
| cls r4 | 45 | 9.4 | 0/0/20/100/53 | 53/53/40/100/70 | 53/53/23/100/63 | 27/27/47/100/67 |
| paraphrase | 45 | 1.0 | 0/0/33/97/80 | 23/30/70/100/77 | 70/73/20/100/97 | 33/33/33/97/70 |
| no fact | - | 1.0 | 0/0/0/13/0 | 0/3/80/23/0 | 0/0/43/27/0 | 0/0/10/20/0 |

### Passage sentence

| cell | dist B | ans tok | base | sft3 | sft2 | sft3_plain |
|---|---|---|---|---|---|---|
| base c1 | 79 | 1.0 | 0/0/0/87/13 | 60/60/0/97/70 | 53/53/0/93/77 | 57/57/0/97/80 |
| dist 200 | 225 | 1.0 | 3/3/0/97/33 | 60/60/0/100/87 | 63/63/0/100/97 | 60/60/0/100/97 |
| dist 450 | 486 | 1.0 | 0/0/0/83/43 | 57/57/0/97/80 | 70/70/3/100/100 | 47/47/0/97/83 |
| dist 900 | 938 | 1.0 | 0/0/0/87/43 | 53/53/0/100/70 | 50/50/0/97/93 | 47/47/0/97/87 |
| distr 1 | 103 | 1.0 | 0/0/0/53/17 | 37/37/0/67/50 | 50/50/0/70/53 | 47/47/0/73/70 |
| distr 2 | 126 | 1.0 | 0/0/0/53/20 | 23/23/0/53/43 | 30/30/3/67/43 | 33/33/0/60/57 |
| distr 4 | 176 | 1.0 | 0/0/0/43/7 | 10/10/0/33/20 | 20/20/0/50/33 | 17/17/0/37/30 |
| cls c2 | 80 | 2.8 | 17/17/0/100/47 | 57/60/0/100/77 | 63/63/0/100/90 | 73/73/0/100/80 |
| cls r2 | 80 | 3.0 | 10/10/0/100/47 | 67/67/0/100/63 | 77/77/0/100/83 | 77/77/0/100/87 |
| cls c4 | 81 | 6.8 | 0/0/0/100/57 | 60/60/0/100/73 | 77/77/0/100/90 | 57/57/0/100/97 |
| cls r4 | 81 | 9.4 | 3/3/0/100/37 | 43/43/0/100/60 | 60/60/0/100/77 | 50/50/0/100/80 |
| paraphrase | 81 | 1.0 | 0/0/0/83/33 | 43/43/0/100/60 | 43/43/0/100/77 | 47/47/0/100/87 |
| no fact | - | 1.0 | 0/0/0/27/0 | 0/0/0/33/0 | 0/0/13/30/0 | 0/0/0/30/0 |

### Wikidata line

| cell | dist B | ans tok | base | sft3 | sft2 | sft3_plain |
|---|---|---|---|---|---|---|
| base c1 | 1 | 1.0 | 13/13/0/53/0 | 77/77/0/87/63 | 67/67/3/90/53 | 73/73/0/83/60 |
| dist 200 | 231 | 1.0 | 3/3/0/63/0 | 80/80/0/100/83 | 57/57/7/93/73 | 63/63/0/97/73 |
| dist 450 | 489 | 1.0 | 3/3/0/33/0 | 57/57/0/100/70 | 53/53/3/100/77 | 57/57/0/90/67 |
| dist 900 | 936 | 1.0 | 3/3/0/33/0 | 80/80/0/100/73 | 53/53/0/100/77 | 43/43/0/97/70 |
| distr 1 | 16 | 1.0 | 3/3/0/23/0 | 27/27/10/50/40 | 23/23/0/37/20 | 37/37/0/43/37 |
| distr 2 | 29 | 1.0 | 7/7/0/27/0 | 20/20/3/33/23 | 7/7/0/30/13 | 20/20/0/33/23 |
| distr 4 | 46 | 1.0 | 3/3/0/23/0 | 10/10/0/27/13 | 7/7/0/13/0 | 10/10/0/27/10 |
| cls c2 | 1 | 2.8 | 17/17/0/97/0 | 97/97/0/100/97 | 93/93/0/100/93 | 93/93/0/100/93 |
| cls r2 | 1 | 2.9 | 17/17/0/100/0 | 93/93/0/100/90 | 83/83/0/100/67 | 83/83/0/100/80 |
| cls c4 | 1 | 6.6 | 27/27/0/100/0 | 93/93/0/100/93 | 87/87/0/100/63 | 97/97/0/100/90 |
| cls r4 | 1 | 9.4 | 23/23/0/100/0 | 73/73/0/100/73 | 77/77/0/100/57 | 77/77/0/100/73 |
| paraphrase | 1 | 1.0 | 27/27/0/47/0 | 73/73/0/80/47 | 40/40/0/83/33 | 60/60/0/83/47 |
| no fact | - | 1.0 | 0/0/0/30/0 | 0/0/20/23/0 | 0/0/37/27/0 | 0/0/0/27/0 |
| real, true capital | 1 | 3.8 | 3/3/0/97/0 | 100/100/0/100/100 | 100/100/0/100/100 | 100/100/0/100/100 |
| real, swapped capital | 1 | 3.4 | 3/3/0/97/0 | 100/100/0/100/100 | 100/100/0/100/100 | 100/100/0/100/100 |
| old format `<L>: <v>.` | 1 | 1.0 | 10/13/0/50/0 | 53/53/0/90/63 | 47/47/7/87/30 | 57/57/0/90/67 |
| old format, swapped | 1 | 3.7 | 7/7/0/97/0 | 100/100/0/100/100 | 90/90/0/100/80 | 100/100/0/100/100 |

### Calculator line

| cell | ans tok | base | sft3 | sft2 | sft3_plain |
|---|---|---|---|---|---|
| base (<1000) | 2.0 | 7/7/0/63/0 | 100/100/0/100/100 | 100/100/0/100/100 | 100/100/0/100/100 |
| 4-6 digits | 4.5 | 0/0/0/90/0 | 97/97/0/100/97 | 97/97/0/100/100 | 100/100/0/100/100 |
| 7+ digits | 8.2 | 0/0/0/73/0 | 83/83/0/100/100 | 90/90/0/100/90 | 93/93/0/100/90 |
| paraphrase | 2.0 | 0/0/0/33/0 | 100/100/0/100/100 | 97/97/0/100/100 | 100/100/0/100/100 |
| no fact | 2.0 | 0/0/0/20/0 | 0/0/0/17/0 | 0/0/0/27/0 | 0/0/0/23/0 |

### Per model, all fact-present items (n=1350)

| model | exact | span | tpl | MC | TF top-1 | p(ans tok given frame) | p(open copy) | p(open tpl) | tpl opening preferred |
|---|---|---|---|---|---|---|---|---|---|
| base | 5 | 5 | 8 | 75 | 24 | 0.11 | 0.07 | 0.23 | 88 |
| sft3 | 59 | 60 | 11 | 87 | 70 | 0.58 | 0.50 | 0.30 | 35 |
| sft2 | 63 | 64 | 4 | 87 | 70 | 0.54 | 0.45 | 0.27 | 35 |
| sft3_plain | 55 | 55 | 8 | 86 | 70 | 0.56 | 0.45 | 0.18 | 33 |

The base model's free generation in QA framing is incoherent at 30M ("What are my dog's name?"). So its `exact` numbers mean little, and base-vs-SFT comparisons use MC, TF and log-probs.

### Knows the answer but answers with a template (fact-present items)

| model | type | template reply | template while MC picks the answer | p(open copy) | p(open tpl) | tpl opening preferred | no fact: template | no fact: other reply |
|---|---|---|---|---|---|---|---|---|
| sft3 | umem | 36 | 35 | 0.41 | 0.48 | 64 | 80 | 20 |
| sft3 | passage | 0 | 0 | 0.40 | 0.28 | 34 | 0 | 100 |
| sft3 | wikidata | 1 | 0 | 0.60 | 0.23 | 21 | 20 | 80 |
| sft3 | calc | 0 | 0 | 0.64 | 0.00 | 0 | 0 | 100 |
| sft2 | umem | 12 | 12 | 0.47 | 0.43 | 51 | 43 | 57 |
| sft2 | passage | 1 | 0 | 0.37 | 0.23 | 38 | 13 | 87 |
| sft2 | wikidata | 1 | 1 | 0.52 | 0.22 | 26 | 37 | 63 |
| sft2 | calc | 0 | 0 | 0.31 | 0.03 | 8 | 0 | 100 |

Example (`umem.base.0.c1.03`, sft3): "My teacher's name is Mar." is followed by "What is my teacher's name?", and the reply is "I don't know your teacher's name yet. Want to tell me?". Given the frame "Your teacher's name is", the model puts p=0.97 on " Mar". Across the sft3 umem items that got a template while MC picked the answer, the median p(answer token given frame) is 0.76 (mean 0.59).

No-fact replies from sft2 include "You told me he's called Tomato.", "You told me he's called "Server"." and "If I remember right, it's chess.".

### Distractors: which in-context value gets copied (distractor cells, 90 items per type)

| model | type | exact, fact listed last | exact, fact not last | distractor copied | MC | copied value's position (first / middle / last / none) |
|---|---|---|---|---|---|---|
| sft3 | umem | 46 (n=37) | 30 (n=53) | 53 | 38 | 26 / 18 / 37 / 9 |
| sft3 | passage | 18 (n=28) | 26 (n=62) | 33 | 51 | 33 / 10 / 8 / 39 |
| sft3 | wikidata | 3 (n=35) | 29 (n=55) | 39 | 37 | 41 / 7 / 4 / 38 |
| sft2 | umem | 38 | 43 | 46 | 48 | 33 / 16 / 29 / 12 |
| sft2 | passage | 32 | 34 | 18 | 62 | 27 / 9 / 10 / 44 |
| sft2 | wikidata | 9 | 15 | 29 | 27 | 20 / 7 / 8 / 53 |
| sft3_plain | wikidata | 0 | 36 | 43 | 34 | 55 / 2 / 2 / 31 |
| base | all | 0 | 0-7 | 0-6 | 24-50 | mostly none |

- The effective chance level among the in-context values is 50% at k=1, 33% at k=2 and 20% at k=4. MC sits at that level for every model. There is no key matching at all.
- For passages and Wikidata lines the model copies the **first-listed** same-type value. When the true fact is the last Wikidata line, sft3 gets it 3% of the time.

### Answer length (base and class cells, excluding calc)

| model | ans tokens | n | exact | TF all tokens top-1 | TF per token | p(first token given frame) |
|---|---|---|---|---|---|---|
| sft3 | 1 | 90 | 63 | 70 | 70 | 0.53 |
| sft3 | 2 | 25 | 56 | 80 | 90 | 0.76 |
| sft3 | 3 | 155 | 74 | 81 | 94 | 0.69 |
| sft3 | 5+ | 177 | 62 | 75 | 97 | 0.72 |
| sft2 | 1 | 90 | 68 | 72 | 72 | 0.52 |
| sft2 | 3 | 155 | 80 | 85 | 95 | 0.68 |
| sft2 | 5+ | 177 | 71 | 73 | 96 | 0.69 |
| base | 1 | 90 | 4 | 24 | 24 | 0.12 |
| base | 5+ | 177 | 10 | 37 | 90 | 0.20 |

Once the first answer token is right, later tokens of multi-token names copy at 90-97% per token, even for invented 9-token names. The first token is the bottleneck.

In the class cells MC is 100% for multi-token answers because the alternatives are not in context. Only the distractor cells test discrimination.

### Induction test (`induct`): L random single-token words, then " .", then the list again; top-1 accuracy on the second copy (30 lists per length)

| model | L=10 | L=50 | L=200 |
|---|---|---|---|
| base | 6 | 10 | 10 |
| sft3 | 23 | 32 | 36 |
| sft2 | 14 | 25 | 34 |

A transformer with a working induction head predicts the second copy of a random sequence almost perfectly. ADAM-T has no general induction circuit. Pretraining data offers little signal for one: 6.3% of tokens in `adam_t/stream` hold windows are predictable by an exact earlier 3-gram in the same 1024-token window. The SFT sets are at 11.5% (v3) and 12.2% (v5).

### Serving fidelity and tokenisation

- **Tokenisation** (`tok`, `data/adam_t/probe/tokenization.json`):
  - For all 1470 probe prompts, the builder's Python BPE (`build_adam_t_sft.Bpe`, loaded from the builder source) equals the served Rust BPE, both on the serve prompt and on the full training-format example `prompt + " <reply>\n\n"`.
  - The serve prompt is always an exact token prefix of that example, so `Adam:` is a token boundary and replies start with " ".
  - 200 random examples per source (umem, kctx, wikidata, calc, mem, squad) of SFT v3 and v5 re-tokenise to exactly the stored ids: 1200/1200 each. All start with `User: ` and end with `\n\n`.
- **LUT vs burn**:
  - The export parity on hold windows is argmax agreement 97.75% for sft3 (mean |dlogit| 0.037, max 0.61) and 97.85% for sft2.
  - On 50 random fact-present probe prompts (`fid`, sft3 LUT vs CPU ndarray burn, `data/adam_t/probe/fidelity.json`):
    - first reply token top-1 agreement is **50/50**;
    - the largest |dlogp| over the burn top-5 averages 0.071 (max 0.18);
    - serve-decoded replies (40 bytes) are identical in **48/50**, and the first word matches in 50/50.
  - The two divergences come mid-sentence after a correct opening, and both replies are wrong either way ("Halwistead is a rock band..." vs "Halwistead was the lead singer of Halew..."). They cannot explain the grounding gap.
  - Note: burn's CPU path sleeps on the GPU-governor `THROTTLE_FILE`, so `fid` was run with `THROTTLE_FILE` pointing at a missing file.
- **Template drift**:
  - `amni/serve/adam_t_client._wl` began emitting the property label ("Sweden, capital: Stockholm.") in 628fc27 (2026-09-28 10:19).
  - SFT v3 (built 09-26) and the v3/v4 rows inside v5 (which sft2 trains on) contain 0 property lines; 98% of wikidata-source prompts use `<Label>: <value>.`. The E2E report (09:40) was run with the old format ("Mongolia: Ulaanbaatar.").
  - The probes show the drift is **benign**. The new, never-trained format beats the trained one: sft3 77 vs 53, sft2 67 vs 47, sft3_plain 73 vs 57. The property word is an extra key the model can match.
  - The v6 reader source (built 17:51) already uses the new format.

### SFT data audit (`audit`, 3000 sampled examples per source; `data/adam_t/probe/sft_audit.json`)

| source (v5) | train ex. | sft2 weight | reply content in prompt | unknown-template reply | hedge reply |
|---|---|---|---|---|---|
| umem | 2927 | 4 | 65% | 27% | 0% |
| mem | 15267 | 0.75 | 22% | 17% | 13% |
| wikidata | 6494 | 3 | 78% | 0% | 8% |
| kctx | 4373 | 3 | 74% | 0% | 20% |
| squad / adversarial_qa / tydiqa | 24558 | 1 | 99% | 0% | 0% |
| hotpot_rc | 29080 | 1 | 93% | 0% | 0% |
| hotpot_cb / mintaka (closed book) | 38602 | 1 | 5-6% | 0% | 16-28% |

- About 22-26% of the Wikidata and knowledge-context training targets are answered from outside the given context, e.g. "The Final Destination was directed by David R. Ellis." after passages about a different film. These teach that the context is optional.
- The v3 training stream contains 947 "Oh, cool! Thanks for telling me." replies (mem-source acknowledgements). That line is the E2E's canned 43/410; v4/v5 dropped it.

## Verdicts

| hypothesis | verdict | evidence |
|---|---|---|
| (a) weak copy / induction at 30M ternary after 573M tokens | **Confirmed, refined.** Copy of a *single* salient value works. *Key-based* lookup (induction, binding) is essentially absent. | Induction 6-10% base, 14-36% SFT. MC with distractors at chance for all 4 model/framing runs. Primacy heuristic for passages and Wikidata. No distance decay to 900 B, so this is not a length problem. Pretraining is 6% induction-predictable. |
| (b) multi-token rare-name copying fails more | **Refuted as a main cause.** | Later tokens copy at 90-97%. Classes c2/r2/c4 are at or above c1. Only r4 (about 9.4 tokens) is 0-20 pp lower in some cells (umem sft2 53 vs 83, passage sft3 43 vs 60), inside the noise elsewhere. |
| (c) template priors from SFT imbalance dominate when copy confidence is low | **Confirmed for user-memory questions**, not for passages, Wikidata or calc. | sft3 answers umem with a template on 36% of fact-present items, 35 of those 36 points while MC picks the answer; the opening prefers the template in 64%; paraphrase gives 70%. The template is not triggered by low copy confidence: p(answer given frame) has a median of 0.76 in those items. sft2 cuts the template rate to 12% but confabulates on 57% of no-fact questions. In the SFT data, umem is 27% and mem 17% unknown-template replies, and 22-26% of wikidata/kctx targets lie outside the context. |
| (d) serving-fidelity bug (LUT/burn, prompt/template) | **Refuted as a cause of the quality gap.** | Token-identical prompts. Probe prompts: first reply token 50/50, replies 48/50 identical, both divergences mid-sentence in already-wrong replies. Hold-window parity 97.8% argmax. The one template drift (Wikidata property label) helps rather than hurts. |
| (e) SFT damaged the base model's copying | **Refuted.** | Same plain framing: sft3 MC 86 vs 75, TF 70 vs 24, p(ans given frame) 0.56 vs 0.11, induction 23-36 vs 6-10. The base binds no better. |

## Ranked root causes and proposed low-level fixes

### 1. No key-to-value lookup: binding / associative recall is at chance

- **What happens.** Real context always has several same-type entities (two passages plus a Wikidata line, several remembered facts). The model copies a plausible name, usually the first listed, not the one attached to the asked key. This is the dominant E2E failure.
- **Fixes, in order of expected impact:**
  - **Pointer / copy head (lookup-friendly output mixture).** Add a gated copy distribution. Take one designated attention head in the last block, with its query aimed at the question's key tokens. Its attention weights over context positions are scattered onto those positions' token ids: `p = g * p_vocab + (1 - g) * scatter_add(attn_w, ctx_ids)`, where `g` is a sigmoid of a ternary projection of the final state.
    - Serving stays table-lookup plus index/scatter: no new matmul, and it fits `lut::Lm::step` (the head's weights already exist in the KV cache).
    - Train it from base or SFT with the existing loss; the copy target comes for free wherever the gold token occurs in context.
    - Expected: distractor cells move from 10-47% toward the single-fact level (55-80%). Knowledge grounding in E2E (27/76) should rise substantially, since most failures there are wrong-candidate copies.
  - **Associative-recall and induction pretraining data.** Mix 2-5% synthetic key-value lists, repeated random spans and "attribute of X" lookups into the stream (currently 6% induction-predictable). Or continue pretraining past 573M tokens with such a curriculum. Test with `induct`: the target is more than 80% at L=50.
    - Also test whether the hashed n-gram memory (`mem`, ngrams 2/3/4/6) crowds out attention-based copying. `crates/adam-t/mem_ab.sh` gives the A/B harness; compare `induct` with and without mem. This is untested here.
  - **Hard-negative reading SFT.** Every reading and memory example should carry 1-4 same-type distractors with the answer's position uniformly distributed, so neither primacy nor recency works. v6 "reader" adds passages but not position-balanced same-type distractors.

### 2. Template prior on memory questions (SFT mix)

- **What happens.** The model knows the answer (MC 100%, median p 0.76 given the frame) but opens with "I don't know / You haven't told me". sft2's reweighting (umem x4, v4 acks) traded this for confabulation.
- **Fix: contrastive twin examples.** Every unknown-template umem/mem example gets a twin with the same question and the fact present, and the reverse. Add paraphrased questions (the "What color is my car?" style), since paraphrase raises the template rate to 70%. Cap the unknown-template share per question type at the real no-fact rate. Add no-fact twins for "You told me ..." so confabulation is penalised.
- **Expected:** umem template-while-known drops from 35% (sft3) or 12% (sft2) to under 5%, and no-fact confabulation (sft2 57%) drops below 15%.

### 3. SFT label noise that teaches "context is optional"

- **What happens.** 22-26% of wikidata/kctx targets are not in the context, and 8-20% are hedges built around a prior guess.
- **Fix.** For context-bearing sources, keep only examples whose answer span is in the context, or turn them into explicit "not in the passage" hedges that do not name a prior answer. Up-weight span-extraction sources (squad/tydiqa/adversarial_qa, 99% in-context) relative to wikidata/kctx.
- **Expected:** a moderate gain in grounding with distractors. It compounds with fix 1.

### 4. Serving and template: no bug; housekeeping only

- The LUT path is at its known parity level.
- For consistency, rebuild the v3/v4 Wikidata rows with the current `_wl`, or let v6 reader rows carry the new format. The property label measurably helps.
- Low priority.

### 5. Multi-token rare names

Not a primary cause. It needs no work beyond fix 1, because the first-token choice is the bottleneck.

## Reproduce

```
python scripts/adam_t_probe.py build
python scripts/adam_t_probe.py run --model sft3=<model.gf17> [--plain] --threads 1
python scripts/adam_t_probe.py report --models base,sft3,sft2,sft3_plain
python scripts/adam_t_probe.py induct --models base=<gf17>,sft3=<gf17>
python scripts/adam_t_probe.py fid --gf17 <model.gf17> --run <run dir with ckpt/ and log.jsonl>
python scripts/adam_t_probe.py tok --gf17 <model.gf17>
python scripts/adam_t_probe.py audit --sets sft,sft_v5
```

- The binary comes from `ADAMT_BIN`, default `scratchpad/rca/bin/adamt-rca`: a copy of `adam-t`, so `sft.sh`'s `pgrep -x adam-t` guard is not tripped. It needs the `enc`/`nxt`/`probe` ops from this change.
- Run everything under `systemd-run --user --scope -p MemoryMax=8G -p CPUQuota=300% nice -n 15`.
- One model over the full set takes about 45 min on one thread.
- Outputs: `data/adam_t/probe/results/<model>.jsonl`, `summary.json`, `induction.json`, `fidelity.json`, `tokenization.json`, `sft_audit.json`.
