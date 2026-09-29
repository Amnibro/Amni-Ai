# ADAM-T-BIND: pointer/copy head and recall/induction pretraining data, pre-registered A/B

Status: built and CPU-tested on 2026-09-28. **The GPU A/B has not been run.** It runs when the CTO schedules GPU time, with `crates/adam-t/bind_ab.sh` (section 6). No GPU was used for this change; the `adam-t-sft2` run was not touched.

- **Why:** `rca_grounding.md` found that ADAM-T copies a single fact but has no key-to-value lookup. With 1-4 same-type distractors, every model is at chance among the in-context candidates, and the base model predicts only 6-10% of a repeated random list (a working induction head gets more than 90%). Only 6.3% of pretraining tokens are in-window repeats.
- **What this adds**, both behind env flags. With both unset, training, export and serving are unchanged (section 5).
  - `COPY=1`: a gated pointer/copy head (section 1).
  - `RECALL_MIX=<r>`: synthetic associative-recall and induction windows mixed into pretraining (section 2).
- **Code:**
  - `crates/adam-t/src/lib.rs`: `Cp`, `cmix`, the `copy`/`rmix` flags in `Cfg`, `tb`/`xy`, and the copy-head optimiser and checkpoint in `train`.
  - `crates/adam-t/src/lut.rs`: `cmix`, `Lm::cp`, `Kv::kc`, and the `w{4L+1}`/`cpy` tables.
  - `crates/adam-t/src/recall.rs`: the generator.
  - `crates/adam-t/src/main.rs`: `adam-t recall`, and the `room` field of the `probe` op.
  - `crates/adam-t/src/bind_tests.rs`: the tests.
  - `scripts/adam_t_probe.py`: the `binding` subcommand, `run --no-gen --room`, and `induct --lens`.
  - `crates/adam-t/bind_ab.sh`: the A/B launcher.
  - `crates/adam-t/t30m_v2.sh`: passes `COPY` and `RECALL_MIX` through.

## 1. The copy head (`COPY=1`)

**Forward.** Let h_i be the final residual (the input of the vocab head) at position i, and x the input tokens.

1. **Projection.** One ternary linear map `d -> 2*dc + 1` (dc = d/H, 64 at both the 6.5M and 30M shapes) on h_i gives a copy query q_i, a copy key k_i and a gate logit z_i. It uses the same RMSNorm, 255-level activation grid and STE trits as every other linear map.
2. **Attention.** Copy attention is causal and strict: a_ij = softmax_j(q_i·k_j/sqrt(dc)) over j < i.
3. **Copy distribution.** Each key's successor gets the key's attention mass: p_copy,i(w) = Σ_{j<i} a_ij·[x_{j+1} = w].
   - This is induction by construction: the model matches the current context against an earlier one and copies what came next.
   - For "A is B ... what is A?" the query at the frame's last token finds the key position that ends "A ... is", and the copied token is B's first token. Later tokens of a multi-token value repeat the same step one position further on.
4. **Gate and output.** g_i = sigmoid(z_i + gb). The gate bias gb is one learned scalar, initialised to 2.0, so the head starts at 88% vocabulary. The output is the log of the mixture:

   log p_i = log(g_i·softmax(logits_i) + (1 − g_i)·p_copy,i)

   It is computed stably as a two-term log-sum-exp from log_sigmoid(±z), log_softmax(logits) and log(max(p_copy, 1e-30)). Position 0 has nothing to copy from, so its gate is forced to 1 (z + 10^4).
5. **Consumers.** A COPY model's "logits" are these normalised log-probabilities. Every consumer works on them unchanged: cross-entropy, bpb, `lp`, `nxt`, `probe`, argmax, and the rep-penalty and temperature decoding.

**Design choices** (fixed now, not tuned):
- **Keys and queries are content-only.** They get no LPE boost, so matching does not depend on distance. Recency is still available through the residual.
- **The scatter reads tokens, not weights.** The copy distribution is a scatter-add over context token ids, and the only new weights are one ternary map and one scalar.
- **Separate module.** The head is its own module (`ckpt/copy.bin`, optimiser `ckpt/ocp.bin`, AdamW with the core's settings), so existing `core.bin` checkpoints load unchanged.
- **Continuation and SFT.** `INIT` continuation and SFT inherit the head from their base unless `COPY` is set. A base without the head gets a fresh one when `COPY=1`.

**Serving (LUT path, lookup-only):**
- **Weights.** The copy map is exported as trit table `w{4L+1}` and runs through the same 243-entry sum tables as every linear map. Its γ is appended to `g`, and table `cpy` holds dc and gb exactly. There is no new float matmul on weights.
- **Per token.** The copy attention is float dot products of activations, like normal attention. The copy distribution is a scatter-add of those weights onto `Kv.x[j+1]`, and the mixture is elementwise.
- **KV cache.** `Kv.kc` caches the copy keys, so a decode step adds one key.
- **Short cut.** The mixture skips exp/ln when |a − b| > 17, where the f32 result is bit-identical to the full formula.
- **Compatibility.** It works with LPE (the LPE boost is untouched) and with lazy and grown PTEX memory (`adam-t grow` copies the new tables).

**Cost:**
- **LUT decode**, 6.5M BPE shape (d 256, L 8, H 4, T 256, V 4096, LPE, random ternary weights, 3 threads; `copy_lut_bench`, three runs): 3.33-3.41 ms/token without the head, 3.37-3.44 with it, **+1.0-1.9%**. Most of the added cost is the log-softmax over 4,096 entries.
  - At the tiny smoke shape (d 64, L 2), the same work is ×1.14, because the core is almost free there. As with `mem_ab`, that does not predict the 6.5M ratio.
- **Trainer:** one [b·t, 129] ternary map, one [b, t, t] single-head score matrix, and a [b·t, V] scatter plus about ten elementwise passes over [b·t, V] per micro-batch. At MICRO 4, 6.5M BPE, that is about 64 MB per pass. Estimate: +3-10% GPU time per step.
- **Params:** d·(2dc+1) + 1 = 33,025 at 6.5M, 66,049 at 30M.

## 2. Recall/induction pretraining data (`RECALL_MIX=r`)

**Mixing:**
- **Which windows.** Window j of step `it` is replaced by a synthetic window when hash(seed, it, j) < r. Every other window is the same natural-text window as without the flag: the same memmap offset and the same batch order.
- **Stream.** The windows come through the existing reader and CORPUS path. The 25 GB stream is not rebuilt, and nothing is written to disk.
- **Generation.** Windows are generated on the fly and deterministically from (seed, it, j), so resume is exact. Episodes are separated by `\n\n`, encoded with the model's own BPE (or bytes), and cut to CTX+1 tokens.
- **Loss.** The loss is the ordinary LM loss on every token.
- **Where it applies.** Pretraining only. SFT ignores the flag.

**Episode types** (own CC0 templates and word lists in `recall.rs`; entity names are invented from syllables; everything is natural-language-ish):

| type | share of episodes | form |
|---|---|---|
| binding ("A is B ... what is A?") | 35% | 2-7 entities, so 1-6 same-type distractors, sharing one attribute such as "favorite fruit", "locker code" or "travel bag". In 30% of episodes each entity has two attributes, so the other attribute's facts are extra distractors. 5 fact templates. The facts are shuffled, so the answer position is uniform. Then 1-3 questions in 5 question templates ("Question: What is X's A? Answer: V.", "What is the A of X? It is V.", cloze, ...), each about a uniformly chosen fact |
| key:value list, then a query | 25% | 2-12 `key<sep>value` items (4 separators, 4 joiners, 6 headers). Keys are invented words, nouns, "item N" or adjective-noun pairs. Values are numbers, colours, invented words or multi-word values. Then 1-3 queries in 4 templates |
| property lines | 15% | 2-7 `<Label>, <property>: <value>.` lines, the serve's Wikidata line format, with properties the RCA probe never uses (mascot, founding year, population, official colour, area code, motto, main export, tallest tree, river, landmark). Then 1-2 questions |
| repeated random spans | 25% | 4-48 random words (12%) or 4-40 random alphabetic BPE tokens (13%), then 1-2 verbatim repeats after one of 6 separators. The RCA `induct` format (lowercase single-token words, `" ."`, repeat) is never produced, so the probe measures transfer |

**Measured** (`recall_generator_format_mix_and_decontam`, BPE-4096, 4,000 episodes; and `adam-t recall` on 20,000 episodes):

| check | result |
|---|---|
| episodes by kind (bind / kv / lines / words / tokens) | 1431 / 941 / 613 / 493 / 522 |
| every lookup answer bound by an in-context fact and repeated after it | 5,587 / 5,587 |
| answer position, mean relative (0 = first fact, 1 = last) | 0.502; first 21.2%, last 21.7% |
| same-type distractors | 1-6, all present |
| multi-token keys / values (BPE-4096) | 99.7% / 87.7% |
| realised mix at r = 0.04 (48,000 draws) | 0.0401 |
| RECALL_MIX unset | batch sampler identical to the old one |
| RECALL_MIX set | only the hit windows change |
| 20,000 episodes | 6.59 MB, 2.71M tokens (135 tokens and 2.43 bytes/token per episode) |

**Decontamination:**

| check | result |
|---|---|
| word 13-grams vs the 102 registered eval sources, `decontam.dirty`, cached index `idx13_6e7f096b…` | 0 of 20,000 episodes |
| 8-grams and 13-grams vs the ADAM-T eval sets (`data/adam_t/eval/**`: NQ MC, RC dev, D3 dev, chat, ARC-E, PIQA) plus the RCA probe set | 0 |
| unit test: 8-grams vs 10 sources, against 172,556 generated 8-grams | 0 |

`bind_ab.sh` repeats both checks on a fresh 20,000-episode sample and refuses to train on any hit.

## 3. Tests (CPU NdArray; `cargo test --release -p adam-t`)

All 23 tests pass: 20 lib tests (17 earlier, 3 new) and 3 main tests. Two benches are ignored.

1. **`copy_mix_fwd_bwd_matches_reference_and_lut`.** `cmix` under `Autodiff<NdArray>` is compared against an independent f64 reference that forms the mixture explicitly, with central-difference gradients over the logits, the copy/key/gate map and gb. The LUT `lut::cmix` is compared against the same reference.

   | check | max error |
   |---|---|
   | output | 4.9e-7 |
   | gradients (largest gradient 7.4) | 1.5e-7 (logits), 2.4e-7 (map), 6.7e-8 (gb) |
   | LUT vs reference | 3.4e-7 |
   | row normalisation | 9.2e-8 |

   Position 0 equals the vocabulary distribution, and the vocab-only output differs by 2.8, so the test is sensitive.
2. **`copy_head_learns_toy_kv_and_lut_is_exact`.** A toy key-value task: 6 random `Kv` letter pairs, then `?K v` queries. The model is a 1-layer ternary core, d 64, LPE, lazy PTEX memory 2^6 rows, 1,000 steps. One layer cannot build an induction circuit.

   | arm | answer accuracy (384 held-out answers) |
   |---|---|
   | without COPY | 6.2-7.6% |
   | COPY=1 | **98.4%** |

   Also checked on the COPY run:
   - Export: roundtrip bytes and trits exact.
   - Trainer vs LUT: argmax 100%, max |Δ| 1.3e-5 on the export windows and 1.4e-5 on a held-out window.
   - KV-cached copy decoding: bit-identical to the full window.
   - Pause and resume: bit-identical.
   - `INIT` continuation: inherits the head and reproduces the LUT output bit for bit.
3. **`recall_generator_format_mix_and_decontam`** (section 2).
4. **`copy_lut_bench`** (ignored, section 1).

**End-to-end CPU smoke** (tiny shape d 64, L 2, CTX 128, V 4096 on `adam_t/stream`, 30 steps, all 4 arms × 2 seeds, NdArray):
- **Training and export.** All exports were roundtrip-exact.
- **Serving.** `serve <run dir>` (burn) and `serve model.gf17` (LUT) agree to 5e-7 on `nxt`.
- **Probes.** `adam_t_probe.py binding` and `induct` ran on every model.
- **Checks and verdict.** `lutbench`, the launcher's decontam block and its result block (on fabricated harness reports) ran. They gave the expected verdict: no arm adopted at that shape, because the ×1.14 tiny-shape decode ratio fails the speed guard.

## 4. Metrics used by the A/B

- **Binding** (primary). `scripts/adam_t_probe.py binding` on `model.gf17` (LUT), in the base-model plain framing, over the RCA probe set's 270 distractor items: {user memory, passage, Wikidata} × {1, 2, 4 distractors} × 30.
  - **Score:** forced-frame 4-way multiple choice. The frame and each candidate are scored by log-probability: the answer against 3 alternatives, distractor values first. This is the RCA's `MC`.
  - **Room.** `--room 16` keeps the last 240 prompt tokens at ctx 256. The frame is 14 tokens at most and the answer 1. Six of the 270 prompts lose at most 12 leading filler tokens, and no fact and no question is ever cut (checked with the builder's BPE). The summary counts `truncated` and `facts_cut`.
  - **Chance.** In-context chance is 1/min(k+1, 4): 0.361 on average. `lift` is reported as well.
- **Induction** (primary). `adam_t_probe.py induct --lens 10,50,100`: 30 random lists per length, top-1 on the second copy, averaged over the three lengths. L=200 does not fit ctx 256.
- **Guards:**
  - harness `bpb_clean2` and `bpb_stream` (`scripts/adam_t_eval.py run --suites bpb` through `serve <run dir>`, 8,192 seeded windows);
  - LUT decode speed, `adam-t lutbench <model.gf17> 128 128`: KV decode ms/token p50, 3 threads, GF(17) LPE path, 5 interleaved repetitions over the 8 models, median per model.
- **Recorded, not used in the decision:**
  - the trainer's val bpb and `gpu_ms_per_step`;
  - peak total-card VRAM;
  - the export parity;
  - the probe details: TF top-1, p(answer | frame), per type and per k, and induct per length.

## 5. Defaults unchanged

- **Same model file.** The pre-change binary (2c0b826) and this one trained the same tiny BPE, LPE, lazy-memory config for 40 steps with `COPY` and `RECALL_MIX` unset. They gave a **byte-identical `model.gf17`** (sha256 prefix f44b1378…), identical val nats (1.9515, 1.9025) and identical export parity.
- **Checkpoint files differ on every run.** `.bin` checkpoints differ between *any* two runs, including the same binary twice, because burn stores random parameter ids.
- **What changed without the flags.** The only change is two new keys in the first `log.jsonl` line: `"copy": false` and `"recall_mix": 0`. They come after `mem_opt`, so the existing `t30m_v2.sh` resume check still matches.
- **Old suite.** All 17 earlier lib tests and 3 main tests pass unchanged.

## 6. Pre-registered 6.5M A/B (not yet run)

**Question.** At the phase-1 size, with LPE and lazy memory, do the copy head and/or recall data give ADAM-T working key-to-value binding and induction, without costing bpb or LUT speed?

**Arms.** All arms are identical except for two flags. Run order per seed is both, base, copy, recall, so a GPU-side failure of the copy path shows up in the first run.

| arm | `COPY` | `RECALL_MIX` |
|---|---|---|
| base | 0 | 0 |
| recall | 0 | 0.04 |
| copy | 1 | 0 |
| both | 1 | 0.04 |

**Shared config** (fixed now). It is `smoke_bpe_tern_mem` (design section 1) plus the adopted LPE and the lazy optimiser, on the v2 base's corpus:
- **Shape:** d 256, L 8, H 4, ctx 256, FF 1024, BPE-4096 (`data/clean2/bpe4096.json`), TERN 1.
- **Memory:** MEM 1, NGRAMS 2,3,4,6, MEM_LOG2 15, `MEM_OPT=lazy`.
- **Position:** `LPE=1`, φmax 4, S 64 (adopted, `lorentz_ab/ab_result.json`).
- **Data:** `CORPUS=adam_t/stream`.
- **Schedule:** BS 64, MICRO 4, 3,000 steps (49.2M tokens), LR = LR_MEM = 2e-3, 200 warm-up steps, WD 0.05.
- **Checkpoints and eval:** checkpoint every 250 steps, trainer val every 500 (20 × 128 windows), SEG 20, PARITY_WINDOWS 8, no harness between segments.
- **Fixed recall and copy settings:** RECALL_MIX 0.04; copy head dc = d/H = 64, gate bias init 2.0.

**Seeds.** SEED 0 and 1 per arm. SEED sets the init and the sampler, so at one seed all arms see the same natural windows in the same order. The recall arms replace the same 4% of them.

**Metrics.** Section 4, all on the step-3,000 `model.gf17` or run dir.

**Decision rule** (fixed now), for each arm X in {recall, copy, both} against base:
- **Definitions:**
  - gain(m) = mean over seeds of X − mean over seeds of base;
  - spread(m) = max(|X_s0 − X_s1|, |base_s0 − base_s1|).
- **Primary win** on binding MC or induction: gain > max(spread, **0.05**). 5 points is the smallest effect worth a model-format or data change; the RCA expects binding to move from chance (0.36) toward the single-fact level (0.55-0.80).
- **Primary loss:** gain < −spread.
- **bpb guard:** for `bpb_clean2` and for `bpb_stream`, gain ≤ spread. X may not lose more than the seed spread.
- **Speed guard:** (X_s0 + X_s1)/(base_s0 + base_s1) of median decode ms p50 is at most **1.10**.
- **Adopt X** if at least one primary wins, neither primary loses, both bpb guards hold, and the speed guard holds.

**Consequence for the 30M v2 base** (`v2_pick` in `ab_result.json`):
- If no arm is adopted, v2 launches unchanged: `COPY=0 RECALL_MIX=0`.
- Otherwise, let best be the adopted arm with the highest binding MC. v2 takes the simplest adopted arm (in the order recall, copy, both) whose binding MC is within max(its spread, best's spread) of best's.
- The launch uses that arm's flags (`v2_env`) with `t30m_v2.sh`.

A tie within the spread is a loss for the arm. No rerun with another mix ratio, gate init or head size without a new pre-registration.

**What the script does** (`crates/adam-t/bind_ab.sh`, the same pattern as `mem_ab.sh`):
- **Refuses to start** without `THROTTLE_FILE`, while any other `adam-t-*` user unit is active, or while any `adam-t` process runs.
- **Freezes the binary:** copies the binary, `campaign.sh`, `smoke.sh` and `bind_ab.sh` into `exports/adam_t/bind_ab/bin`, with the sha256 and the commit. The CPU copy used for probes, recall dumps and `lutbench` is named `adamt-probe`, so no CPU job is ever named `adam-t`.
- **Checks the generator.** It dumps 20,000 recall episodes and runs the decontam checks of section 2. It stops on any hit (`recall_decontam.json`).
- **Trains, scores and times.** It trains the 8 runs (with a 5 s VRAM log) and runs the harness bpb suite (GPU). Then it runs the binding and induct probes (CPU, 4G scope) and `lutbench` (CPU).
- **Writes the result.** `exports/adam_t/bind_ab/ab_result.json` holds every number above, `adopted`, `v2_pick` and `v2_env`.
- **Can be rerun and paused.** Finished runs, reports and probes are skipped, and `touch <run>/STOP` pauses it. `DRY=1` prints the resolved config.

**Expected cost** (governor at 40%). The Lorentz arms took 57, 61, 55 and 60 min each: byte-level, the same core, dense memory, MICRO 1, 392 GPU-ms/step, including the export.

| factor | per-step effect | basis |
|---|---|---|
| BPE-4096 head and 4,096-way CE | ×1.00-1.10 | `smoke_bpe_tern_mem` cost about the same GPU minutes as the byte smoke (26.8 vs 26), design section 1 |
| lazy memory at 2^15 | ×1.00-1.15 | memscale section 7 |
| MICRO 4 | ×1.00-1.10 | more, shorter bursts |
| copy head (copy and both arms) | ×1.03-1.10 | section 1 |
| RECALL_MIX | ×1.00 | CPU-generated, about 2.6 windows per step |

| item | estimate |
|---|---|
| base and recall arms | 57-85 min each (about 66 most likely) |
| copy and both arms | 60-93 min each (about 70 most likely) |
| **training, 8 runs** | **about 8.5 h wall (7.8-11.9 h)** |
| harness bpb evals | 8 × about 2.5 min (Lorentz: 2.3-2.5 min each) ≈ 20 min of GPU |
| CPU phase (no GPU) | 8 × (binding about 2 min + induct about 1 min) + `lutbench` about 1 min + decontam gate about 1 min ≈ 30 min |
| **total** | **about 9.3 h wall (8.5-13 h)**, of which about 8.8 h (8-12.3 h) needs the GPU |

- **VRAM:** about the Lorentz arms' use, plus 3 × 128 MiB of lazy table and moments, plus the copy arms' [4096, 4096] mixture tensors (about 0.5 GB). Well under 10 GB.
- **Disk:** about 0.7 GB per run (checkpoint with lazy moments, plus `model.gf17`), about 6 GB in total.
- **Host RAM:** trainer `MEM_MAX` 12G, harness scope 6G, probe and `lutbench` scopes 4G.

**The command** (from the repo root, with a checkout that contains this change, after `cd crates && cargo build --release -p adam-t`):

```
THROTTLE_FILE=~/.local/share/amni-ascend/throttle.txt bash crates/adam-t/bind_ab.sh
```

`AB_DIR` (default `exports/adam_t/bind_ab`), `SEG` (default 20), `SEEDS`, `SPEED_REPS` (default 5), `MICRO` (default 4, exact accumulation) and `BIN_SRC` are the only knobs. Once the result exists, the v2 base takes its flags:

```
THROTTLE_FILE=~/.local/share/amni-ascend/throttle.txt <v2_env from ab_result.json> bash crates/adam-t/t30m_v2.sh
```

`t30m_v2.sh` refuses to resume a run dir started with other `COPY`/`RECALL_MIX` values. Give `RECALL_MIX` as the log prints it, for example `0.04`.

**Abort rules** (these pause a run; they are not results):
- A non-finite `train_nats` or `val_nats`.
- After the first segment, `gpu_ms_per_step` above 1.5 × 392 ms: `touch <run>/STOP` and report.

**Untested on GPU.** The copy path uses burn's `scatter` (add) and `max_pair` under Fusion/wgpu, and was only run on NdArray. A failure shows up in the first run, the `both` arm, as a trainer exit, and the script stops there.
