# ADAM-T-LORENTZ: Lorentz position encoding (LPE), pre-registered A/B

Status: built and CPU-tested on 2026-09-28. **The GPU A/B has not been run.** It runs when the CTO schedules GPU time, with `crates/adam-t/lorentz_ab.sh` (section 5).

- **Code:** `crates/adam-t/src/lib.rs` (`Lz`, `boost`, `G17`/`L17`, the `LPE*` flags in `Cfg`) and `crates/adam-t/src/lut.rs` (the `lpe` table, `Lm::qkv`).
- **Flags:** `LPE=1` turns it on. `LPE_PHI` (default 4) and `LPE_S` (default 64) set its shape. `LPE_LUT=ref` makes the LUT path use the direct exponent table instead of the GF(17) discrete-log path. With `LPE` unset, the trainer, export and LUT path behave exactly as before, and the full old test suite passes unchanged.

## 1. The idea and what it becomes in this model

A 1+1D boost with rapidity φ acts on light-cone coordinates u = x+t, v = x−t as u → e^φ·u, v → e^−φ·v. It preserves u·v, and boosts compose by adding rapidities.

**LPE maps this onto attention:**
- Each head's dims are paired (2m, 2m+1) = (u, v), like RoPE pairs.
- Position p gets rapidity φ_m(p) for pair m. The query is boosted by +φ(i), and the key by the same boost applied to its light-cone dual k̃ = (k_v, k_u).
- The score is the Minkowski form in light-cone coordinates, 2η(a, b) = a_u·b_v + a_v·b_u:

  2η(Λ(φ_i)q, Λ(φ_j)k̃) = e^{φ_i − φ_j}·q_u·k_u + e^{−(φ_i − φ_j)}·q_v·k_v

  It depends only on the rapidity difference, so relative position enters q·k exactly as with RoPE, with cosh/sinh-type growth instead of rotation.

**Why it is cheap:**
- In light-cone coordinates a boost is **diagonal**. It is one elementwise scale per dim, not a 2×2 mix like RoPE's rotation.
- So LPE is a fixed per-position factor table multiplied into the fused `qkv` output before attention:
  - u dims of q × e^{φ}, v dims of q × e^{−φ};
  - u dims of k × e^{−φ}, v dims of k × e^{φ};
  - v (values) untouched.
- The flash-attention kernels, the fused ternary linear and the WMMA GEMMs are unchanged. The Euclidean dot of the scaled q and k *is* the Minkowski form above.
- The swap that makes the form indefinite costs nothing either. It is a fixed permutation of the learned k projection's columns, which training absorbs.

**Learned absolute positions are dropped** when `LPE=1`:
- `pos` stays in the checkpoint, so the format is unchanged, but it is not added and not trained. The export writes it as zeros.
- The LPE arm therefore has t·d fewer trained parameters (65,536 at the phase-1 shape). `core_params` reports this.

## 2. Handling the exponential growth (design choice)

A boost is unbounded. Over a 256–1024 window, RoPE-like frequencies would give factors like e^{100}, which overflow f32 and saturate the softmax. The choice is **bounded, centred, quantised rapidity**:

1. **Bounded.** The fastest pair's total rapidity across the whole window is φmax = `LPE_PHI` = 4, so relative factors lie in [e^{−4}, e^{4}] = [0.018, 54.6].
2. **Centred.** φ_m(p) = δ·(s_m·p − c_m), with c_m = ⌊s_m(T−1)/2⌋.
   - The centring cancels in every q·k product, so the scores are exactly relative.
   - It halves the absolute factors to at most e^{±2} (7.4× and 0.135×). That keeps q and k well-conditioned for the ternary grid and for f32.
3. **Quantised.** Rapidities are integer multiples of one quantum δ = φmax / (S·(T−1)).
   - Pair m advances by an integer s_m quanta per position.
   - s_m falls geometrically from S = `LPE_S` = 64 (the fastest pair) to 1 across the dh/2 pairs, like RoPE's frequency ladder.
   - The slowest pairs are nearly boost-free (0.0625 total rapidity over the window), so they act as content channels, as RoPE's slow pairs do.
   - Integer exponents make the GF(17) form below exact.

| shape | δ | fastest pair, e-folding length | exponent range | table |
|---|---|---|---|---|
| phase 1: dh 64, T 256 | 2.45e-4 | 64 positions | ±8160 | 16,336 f32 = 1,021 rows × 16 |
| 30M: dh 64, T 1024 | 6.11e-5 | 256 positions | ±32736 | 65,488 f32 = 4,093 rows × 16 |

**What LPE can express**, per pair and per relative offset Δ ≥ 0 (causal): e^{ωΔ}·A + e^{−ωΔ}·B, where A = q_u·k_u and B = q_v·k_v are content-dependent. This is a learned, content-gated, multi-slope exponential recency or anti-recency bias, like ALiBi with a slope per pair chosen by content. Unlike RoPE, it has no oscillating component. The A/B tests whether that is enough to replace learned positions.

**Rejected alternatives:**
- **Normalising each pair by cosh(ωΔ).** It bounds the scores for any φ, but it is not separable into a q factor and a k factor, so it would need a custom attention kernel.
- **Saturating the rapidity with tanh(ωp).** It breaks exact relativity.
- **Wrapping the rapidity with period P.** It is not relative across the wrap.

## 3. GF(17) discrete-log form on the lookup-only path

In GF(17), 3 is a primitive root, so the multiplicative group is cyclic of order 16, and boosting u → g^k·u, v → g^{−k}·v is addition of k mod 16 in log space.

**The obstruction.** The float attention scores need real positive factors. The only homomorphism from Z/16 into the positive reals is the trivial one, so no real boost is periodic mod 16. (The characters of Z/16 into the complex numbers are 16th roots of unity, and those give rotations, which is RoPE.)

**The resolution.** The exact integer exponent n = s_m·p − c_m is carried as two digits:
- a **GF(17) discrete-log digit** lo = n mod 16, stored as the field element z = g^lo;
- an **octave** hi = ⌊n/16⌋.

**How the digits work:**
- **Composing boosts** multiplies z by g^{k mod 16} in GF(17), which adds the logs mod 16. A carry into hi happens when log z + (k mod 16) ≥ 16, and hi also gets ⌊k/16⌋.
- **The inverse boost** (the v-coordinate's e^{−φ}) is the field inverse z^{−1}, with the octave mirrored.
- **The factor** is one texel: row hi, column log z, of the 16-column table e^{δ·(16·hi + lo)}.

So a chain of boosts collapses to one exponent sum plus one table lookup. The table is not stored in the file. The `.gf17` export stores only the rule (`lpe` table: φmax as f64 bits and S), and the reader regenerates the texels, so the file stores coordinates, not numbers.

**Where each piece lives:**
- `Lz::gz` composes a boost.
- `Lz::inv` inverts.
- `Lz::gf` looks up.
- `Lz::uv` gives a position's (u, v) factors by either path.
- `Lm::qkv` applies them after the ternary qkv map, before K is cached. A cached key keeps its own position's boost, so KV-cache decoding stays exact.

## 4. Tests (CPU, NdArray; `cargo test --release -p adam-t`)

All 16 tests pass: the 13 old tests and 3 new ones. The micro-bench is a 4th, ignored test.

1. **`gf17_boost_table_exact_and_additive`:**
   - 3 generates all 16 nonzero elements, and log/exp are exact inverses.
   - g^a·g^b = g^{(a+b) mod 16} for all 256 pairs.
   - For all 256 (u, v) pairs, a 40-boost chain equals one boost by the exponent sum and keeps u·v mod 17.
   - For **every** exponent in the table, the GF(17) digit path, the direct table and `(δn).exp() as f32` are **bit-identical**, and the (hi, log z) digits reconstruct n.
   - The incremental chain state(p+1) = boost(state(p), s_m) equals the direct state at every position of all 32 pairs.
   - Random 8-boost chains (exponents ±1000) compose additively.
   - Factor products match e^{δ·s·(i−j)} to 1e-6, and the centred rapidities stay within ±φmax/2.
2. **`lpe_fwd_bwd_matches_minkowski_reference`:**
   - The trainer path (`boost` then `fa::fa` with its custom backward, under `Autodiff<NdArray>`) is compared against an independent f64 reference. The reference boosts q and the dual k̃ explicitly with exp(φ), scores with the light-cone Minkowski form using uncentred absolute rapidities, applies a causal softmax, and takes central-difference gradients.
   - Max |Δ out| = 6.3e-7 and max |Δ grad| = 2.1e-6, against grads of up to 5.6.
   - Without LPE the output differs by 1.26, so the test is sensitive.
3. **`lpe_tiny_model_trains_and_lut_is_bit_exact`:**
   - A tiny ternary + PTEX-memory model trains 60 steps with LPE on synthetic text. Hold nats go 2.65 → 1.55 → 1.27, with every loss finite.
   - Export roundtrip is bit-exact.
   - LUT against trainer: max |Δlogit| 9.5e-7, argmax agreement 100%, bpb 1.84043 on both.
   - The reloaded config keeps the LPE flags, and the file's regenerated table equals the trainer's.
   - LUT logits through the GF(17) path are **bit-identical** to the direct-exponent path.
   - KV-cache decoding matches the full-window pass.

**LUT-path CPU speed** (`lpe_lut_bench`, ignored test; 6.5M phase-1 shape: d 256, L 8, H 4, T 256, random ternary weights, 3 rayon threads, two runs):

| arm | full 256-window forward | KV decode step | boost cost per position per layer |
|---|---|---|---|
| learned positions | 285–291 ms (880–898 tok/s) | 3.26–3.29 ms | – |
| LPE, GF(17) path | 285–287 ms (893–899 tok/s) | 3.32–3.33 ms | 0.24 µs |
| LPE, direct table | 281–284 ms (901–910 tok/s) | 3.28–3.31 ms | 0.20 µs |

**What the speed table shows:**
- The boost itself costs about 1.9 µs per token over 8 layers, about **0.06%** of a 3.3 ms decode step.
- Between arms, the differences run from −2.4% to +1.7%.
- The GF(17) arm's KV step was 1.4–1.7% slower in both runs. That is 25× the boost's own measured cost, and the arm order was fixed (baseline first), so it is most likely order and thermal drift. The pre-registered speed check therefore interleaves the arms and measures an A/A spread.
- LPE removes the d-wide position add and adds d multiplies per layer, so its expected speed is the baseline's.

## 5. Pre-registered GPU A/B (not yet run)

**Question.** Does LPE beat learned absolute positions at the phase-1 6.5M shape?

**Arms.** Both arms are identical except for the position scheme:

| | baseline `pos` | `lpe` |
|---|---|---|
| position | learned absolute `pos` table (current) | LPE, φmax 4, S 64 (fixed now, not tuned) |
| everything else | the phase-1 config from `design.md` section 7 (`smoke_tern_mem`) | same |

**The shared config:**
- **Shape:** d 256, L 8, H 4 (head dim 64), ctx 256, FF 1024, bytes (v 256).
- **Model:** TERN 1, MEM 1 with NGRAMS 2,3,4,5,6,8 and MEM_LOG2 15.
- **Schedule:** BS 64, MICRO 1, STEPS 3000, LR 2e-3, WARM 200, WD 0.05.
- **Data:** CORPUS clean2.
- **Kernels:** the current defaults (flash attention, fused ternary linear, exact WMMA).

**Seeds.** SEED 0 and 1 for each arm. SEED sets both the init and the data sampler, so the two arms see the same bytes in the same order at the same seed. The run order is pos s0, lpe s0, pos s1, lpe s1.

**Metrics**, all on the final step-3000 checkpoint:
- **Primary:** harness `bpb_clean2`. That is `scripts/adam_t_eval.py run --suites bpb` through `adam-t serve <run dir>`, 8,192 seeded 256-byte windows of `clean2/hold.bin`.
- **Secondary:** harness `bpb_stream`, the same windows protocol on `data/adam_t/stream/hold.bin`.
- **Also reported, not used in the decision:**
  - the trainer's in-run val bpb (20 × 128 windows, seed 1234);
  - the export parity (`cpu_lut_bpb`, argmax agreement).
- **Speed:** `adam-t lut <model.gf17> 16` on CPU, 3 threads, GF(17) path. Five interleaved repetitions over the 4 models, median bytes/s per model.

**Decision rule.** LPE is adopted only if **all** of the following hold:
1. **Clean2 win:** mean_pos − mean_lpe on bpb_clean2 exceeds the seed spread. The spread is max(|pos_s0 − pos_s1|, |lpe_s0 − lpe_s1|).
2. **Stream guard:** LPE does not lose on bpb_stream by more than that metric's own seed spread.
3. **Speed:** LUT inference is not slower. The ratio (lpe_s0 + lpe_s1)/(pos_s0 + pos_s1) of median bytes/s must be ≥ 1 − ε, where ε is the A/A spread |pos_s0 − pos_s1|/pos_s0 in the same session, capped at 2%.

Anything else keeps the learned positions. A tie within the spread is a loss for LPE. There is no re-run with other φmax or S unless a new pre-registration is written first.

**The command.** Run it from the repo root. The checkout must contain the ADAM-T-LORENTZ commit, so build first with `cd crates && cargo build --release -p adam-t`:

```
THROTTLE_FILE=~/.local/share/amni-ascend/throttle.txt bash crates/adam-t/lorentz_ab.sh
```

`BIN_SRC=<path to adam-t>` picks a different build. `AB_DIR` (default `exports/adam_t/lorentz_ab`), `SEG` (default 20) and `SPEED_REPS` (default 5) are the other knobs. Nothing else is tunable.

**What the script does:**
- **Refuses to start** without the throttle file, while `adam-t-30m` or `adam-t-sft` is active, or while any `adam-t` process runs.
- **Freezes the binary:** it copies the binary, `campaign.sh` and `smoke.sh` into `exports/adam_t/lorentz_ab/bin`, and records the binary's sha256 and the git commit.
- **Runs each arm** through `campaign.sh` / `smoke.sh` with the usual governor rules: `SEG` = 20-minute segments with checkpoint and resume, `MEM_MAX` 12G, CPU 300%, nice 15, and a 1 Hz GPU busy log.
- **Scores each run** with the harness. The reports go to `data/adam_t/eval/report_lorentz-ab-<arm>-s<seed>_3000_dry.json`.
- **Runs the CPU speed check**, then writes `exports/adam_t/lorentz_ab/ab_result.json` with every number above and `adopt_lpe`.
- **Can be rerun.** Finished runs and reports are skipped, and `touch <run>/STOP` pauses it.

**Expected cost.** Phase 1 took about 26 GPU minutes per arm at full duty, before the custom kernels. So expect 4 runs × about 25–45 min under the governor, plus 4 bpb evals of a few minutes each: roughly 2–3.5 h of shared GPU.

## 6. HMEM (hyperbolic memory keys): design note, not built

**The proposal.** PTEX memory keys would get a depth/rapidity coordinate, so a hierarchy move is one exponent addition.

**Why it is not built now:**
1. **PTEX addressing is already additive in its one hierarchy coordinate.** Row = k·2^R + top_R(hash), so moving between n-gram orders is an add of 2^R. A GF(17) rapidity digit would relabel this, not remove work.
2. **Code cost.** It would change the trainer's hashing, the `.gf17` table layout and the LUT reader together, and it would need its own A/B.
3. **It does not address the binding limit.** The measurement in section 7 shows that limit is hash collisions in 2^15-row tables, which HMEM leaves unchanged.

**If it is pursued**, the form worth testing is a *relational* key: a skip-gram table keyed by (content of token a, content of token b, offset Δ), with Δ as the rapidity coordinate. A lookup at a different offset is then one exponent add on the key, and it brings information the contiguous n-gram tables do not have. It needs its own pre-registration after the LPE A/B.

## 7. Could Lorentz double-store data? (CTO question, analysis plus one CPU measurement)

**The algebra:**
- A GF(17) light-cone pair u = g^a, v = g^b splits into:
  - an **invariant** a+b mod 16. This is log(u·v), the "interval", unchanged by any boost: *what*.
  - a **rapidity** a−b mod 16, which a boost by k shifts by 2k: *where*.
- The map (a, b) → (a+b, a−b) mod 16 has the 2-element kernel {(0,0), (8,8)}, and the two outputs always have the same parity. So 256 pairs land on 128 images, and 1 bit is lost.
- Raw capacity therefore cannot double. By pigeonhole, no re-keying stores more distinct values than there are cells.

**The real win is dual addressing.** One stored entry can be reached by an invariant key (content, the same under every shift) and by a relational rapidity key. One entry then stands in for all of its shifted copies.

**Measured on `adam_t/stream`** (`train.bpe4096.u16`; 8 evenly spaced chunks; the 30M memory: orders 2, 3, 4, 6, 2^15 rows each, ctx 1024; numpy, scratch script):

| sample | order | occurrences | distinct n-grams (invariant key) | distinct (n-gram, position) (position key) | position key / invariant key | distinct n-grams per 2^15-row table |
|---|---|---|---|---|---|---|
| 1M tokens | 2 | 1.0M | 230K | 964K | 4.2× | 7 |
| 16M tokens | 2 | 16.0M | 1.12M | 13.1M | **11.7×** | 34 |
| 16M tokens | 3 | 16.0M | 5.80M | 15.6M | 2.7× | 177 |
| 16M tokens | 4 | 16.0M | 10.6M | 15.9M | 1.5× | 323 |
| 16M tokens | 6 | 16.0M | 14.5M | 16.0M | 1.1× | 443 |

At 16M tokens, 45%, 70%, 84% and 95% of the distinct n-grams (orders 2, 3, 4, 6) occur only once.

**What the measurement says:**
- **PTEX memory is already keyed on the invariant.** The row is a hash of the n-gram's content alone, with no position in the key, so every position-shifted copy of an n-gram already shares one row.
- **Against a position-keyed table, that invariance saves 11.7× (bigrams) down to 1.1× (6-grams)** on a 16M-token sample, and the saving grows with the corpus. That saving is already banked.
- **Against the current tables, invariant keying would dedupe almost nothing more.** The only position-dependent keys are the n-grams truncated at the window start (p < n−1). They are 0.10–0.49% of lookups (n−1 of every 1,024 positions), so collapsing them into lower-order rows would save at most about 0.5% of lookups and a negligible number of rows.
- **The binding constraint is collisions, not duplicates.** Even 16M tokens hold 34–443 distinct n-grams per row for every order, and at least 32,740 of the 32,768 rows in every table are touched after only 1M tokens. The capacity levers are:
  - more rows (the planned 2^18–2^20 rows per order on SSD pages);
  - smaller rows (the GF(17)-grid rows);
  - not storing singletons. 45–95% of distinct n-grams are seen once; a count-min or first-seen gate would keep them out of the tables.

**Recommendation.** No "double storage" from Lorentz keys in PTEX. Dual addressing only pays off with a relational key that brings new information, like the skip-gram offset table in section 6. It does not pay off as a re-keying of the existing n-gram tables.
