# ADAM-T SFT round 3 (sft_v8): pre-registration

Date: 2026-09-30, written before any round-3 checkpoint exists and before round 2 was run on either probe8 file. Mission ADAM-T-SFT8. Seed `20260926` for the data, trainer seed 0. Scoring code: `scripts/adam_t_probe8.py` (probe runs, model selection, report) and the locked `scripts/adam_t_eval.py` (A0 chat suite and E2E). The files named below are hashed into every probe output; if one changes after the first round-3 checkpoint is scored, the round-3 result is reported as exploratory.

## What changed and why

The live probe of 2026-09-30 on round 2 (raw model) got decline boilerplate on a greeting, the lost-child emergency template on a work vent, "you haven't told me" on a fact stated in the same message, and word salad on a story request. Round 2 had memorised the emergency (val 0.29 nats) and umem (0.09) sources, and its raw near-miss 911 false-positive rate was 51.9% at step 2993.

The CEO's architecture correction (2026-09-30) also applies: ADAM-T is a blender. Knowledge comes from PTEX retrieval, equation solves and memory, and the core turns the material it is handed into a fluent reply. `data/adam_t/sft_v8` therefore:

- strips AI-assistant decline boilerplate from every source;
- puts every factual chat reply next to the material it came from (`Notes: ...`);
- rewrites guessed hedges as honest hedges that give no answer;
- drops closed-book sources from training (SRC_W 0);
- adds venting, greeting and contrast data plus blend skills (paraphrase, merge, summarise, tone shift, story from beats, and hedging when the material doesn't cover the question).

Details are in `meta.json` under `v8`.

## Held-out material, never used for training

| file | items | what |
|---|---|---|
| `data/adam_t/eval/probe8/greet_vent_probe.jsonl` | 68 | CC0, written for this probe: 18 greetings, 10 small talk / thanks / jokes / follow-ups, 28 venting (work, breakups, exams, loneliness, family, money, bad news), 12 idiom near-misses. The venting and greeting items are `cat=chat` and the idiom items `cat=nearmiss`, all in the chat-suite schema. |
| `data/adam_t/eval/probe8/blend_probe.jsonl` (sha256 e09759e7...) | 182 | 137 paraphrase items (a 1-2 sentence Wikipedia lead extract of at most 380 bytes as material, plus a question about its title) and 45 uncovered items (19 whose material is about a different entity, 10 asking for a date the material lacks, 16 with no material). Built once by `build_adam_t_sft.py --probe8` from documents whose `md5(title) % 50 == 7`; training never samples that bucket. Only paraphrase items carry a reference reply. Before any model was run, the hedge-template references were removed from the uncovered items, because their shared 8-grams made the decontam check drop every training example of the missing-date hedge skill. |
| `data/adam_t/eval/probe8/seen_live_20260930.jsonl` | 4 | The four live-probe prompts. They are already seen, so they are excluded from training (and so are near paraphrases), and they are never scored. |

All three files are under `data/adam_t/eval/**`, so `decontam.eval_sources()` covers them (13-gram and 8-gram). The builder also runs:

- a 6-gram index over the chat probe and the seen prompts;
- an exact-line check;
- a near-paraphrase guard: topic regexes for the four seen prompts, or token Jaccard >= 0.6 with at least 3 shared tokens against any probe item.

These checks cover every base example and every new example.

## Round 3 run (fixed now)

- Base: the same base as round 2 (`exports/adam_t/t30m_bpe4096`, step 35000), via `crates/adam-t/sft.sh`.
- Settings: `CORPUS=adam_t/sft_v8`, `EPOCHS=1.2`, `MICRO=16`, LR/WD/WARM/BS at the sft.sh defaults (3e-4 / 0.05 / 100 / 16).
- Source weights: `SRC_W=oasst1=1.5,oasst2=1.5,dolly=1,aya=1,mem=0.75,emergency=0.6,squad=1,adversarial_qa=1,hotpot_rc=1,hotpot_cb=0,drop=0.5,tydiqa=1,mintaka=0,csqa=0.5,qasc=0.5,gsm8k=0,dolly_ctx=1.5,calc=3,calc_gsm8k=1,wikidata=2,kctx=2,umem=2,emerg4=0.6,talk=2,reader=3,tmem=1.5,cmem=1.5,vent=3,greet=2,contrast=2,blend=3,story=2`.
- Evaluation during training: the A0 chat suite (dev) runs every 3 segments (SEG=30 min) and at the end (`EVAL_ARGS=--suites chat`), with a checkpoint snapshot at each evaluation.
- Output: `exports/adam_t/t30m_sft3`.
- GPU: through the governor (`amni-gpu-governor`, CAP 40); nice 15, CPUQuota 300%.

## Model selection (fixed now)

For every evaluation snapshot S, export `model.gf17` and run the blend probe on it (raw model). Then:

    score(S) = chat_good(A0 dev, campaign report at S) + faithful_para(blend probe at S) - chat_nmfp(A0 dev at S)

The selected step is the one with the highest score; ties go to the earlier step. Everything below is measured on the `model.gf17` of the selected step. The chat probe and the E2E run are not used for selection.

## Comparison baseline

Round 2 is `exports/adam_t/t30m_sft2/model.gf17` (step 7773). The A0 chat suite and both probes are run on it by the same chain with the same code. Its E2E baseline is `data/adam_t/eval/report_adam-t-t30m-sft2-final-h0_7773_e2e_tune_dry.json`.

## Success criteria (round 3 selected step)

| # | criterion | rule |
|---|---|---|
| 1 | Near-miss 911 false positives, A0 chat dev, raw model | `chat_nmfp` <= 0.05 (at most 1 of 27) |
| 2 | 911 lead on real emergencies, A0 chat dev, raw model | `chat_em` >= 0.98 (13 of 13) |
| 3 | Decline rate on chat, A0 chat dev, raw model | `chat_dec` lower than round 2 on the same code |
| 4 | E2E fact, no regression | round-3 `fact` >= round-2 `fact` - 1 x pooled SE. Harness off, `--e2e --tune`, the same env as round 2's final (`AMNI_ADAM_T_HARNESS=0 AMNI_ADAM_T_HARNESS_AGREE=1 AMNI_COGNITION=0`). |
| 5 | E2E answered, no regression | same rule for `answered` |
| 6 | Greeting/venting probe, decline | <= 0.05 (scored with the locked `CHDEC`/`CHDEF`) |
| 7 | Greeting/venting probe, emergency-text false positives | <= 0.02 (`amni.serve.emergency.OUT` anywhere in the reply) |
| 8 | Greeting/venting probe, answered | >= 0.70 (locked `chat_score`) |
| 9 | Greeting/venting probe, coherent | >= 0.90 (locked `chat_score`) |
| 10 | Faithfulness, the headline (see below) | the share of paraphrase items whose reply passes the harness paraphrase grounding check: >= 0.30, and above round 2 by more than 2 x pooled SE |
| 11 | Hedge when uncovered | on the 45 uncovered items, the share whose reply hedges (harness `_DECL` or `CHDEC`) with no new digits and no new capitalised names: >= 0.70 |
| 12 | Phrasing diversity: distinct-2 on the greeting/venting probe | >= 0.50 and higher than round 2 |
| 13 | Phrasing diversity: repeated openers on the greeting/venting probe | repeat rate <= 0.30 and lower than round 2. The rate is 1 - (distinct openers / replies), where an opener is the first 3 lowercase tokens of a reply. The most common opener and its share are also reported. |

Criteria 12 and 13 were added on 2026-09-30, before launch. The CEO confirmed ADAM-T's role that day: it exists to prevent canned responses by turning PTEX/torus material into varied, natural phrasing, so anti-canned output is a primary goal.

Criterion 10 uses the check from `amni/serve/adam_t_harness.py` `_para`, reproduced verbatim in `adam_t_probe8.gok` with the harness's own `_ct` and stopwords. The reply passes when all of these hold:

- it has at least 8 words;
- it has no new content words (words over 3 letters that are not stopwords and whose 5-letter prefix is not in the extract, title or question);
- it has no new digits and no `:`;
- it keeps at least 60% of the extract's content words;
- at least 60% of its word trigrams occur in the extract or the question;
- it ends with `.` or `!`.

The harness file's sha1 is recorded with every probe run.

Also reported, not thresholded:

- fluency and coherence on both probes: distinct-2, mean rep3, words per reply;
- the probe's `canned` rate (memorised templates: "haven't told me", 911 text, the lost-child template, AI boilerplate);
- per-sub-category rates;
- E2E faithfulness in two forms: the strict harness check against any single retrieved line, and "no new content words or digits" against all the material. Both use adam_t-routed items that had passages.

A round counts as a success when criteria 1, 2, 4, 5, 10 and 13 pass. The other criteria are reported pass or fail. Nothing here is a C-milestone claim: the chat test split is not touched.

## Known limits

- The A0 near-miss and emergency counts are small (27 and 13 on dev). One item moves `chat_nmfp` by 3.7 points.
- Selection uses A0 dev and the blend probe, so those two numbers are optimistic for the selected step. The E2E run and the chat probe are not used for selection.
- The paraphrase check is strict by design. A correct reply phrased in new words fails it, and tone replies (warm/playful) are not scored by it.
- Probe items are written by the same author as the new training templates. The near-paraphrase guard removes close copies, but the topics overlap on purpose (the probe measures generalisation within these domains).
