# Adam Rust crates

Cargo workspace for Adam's Rust port (phase R1: inference engine + answer memory; phase R2: trainer).
`amni_ai_core/` (the AmniAI host binary) is excluded from the workspace and still builds on its own.

| crate | what it is |
|---|---|
| `adam-ray` | `RayField` engine: v16 packed `.npz`, v18 and v19 raw `_sid.npy` loaders, wgpu compute kernel (`src/ray.wgsl`), CPU reference path, `prime` / `logits_for` / `advance` / `run` / `score` / `bpb` / `stream` / `beam`; `src/v19.rs` is the v19c working-memory stepper shared with `adam-train` |
| `adam-qa` | port of `amni/compute/ray_qa.py`: `words`, `stem`, index loader for `ray_qa_index.npz`, `search`, `overlap`, `hard`, `pick`, `guide` |
| `adam-ray-py` | PyO3 module `adam_ray` exposing a drop-in `RayField`, `QA`, `pick`, `guide`, `words`, `stem`, `hard` |
| `adam-train` | v16/v18 ray-hash-field trainer: wgpu compute passes (`src/wgsl/*.wgsl`) plus a CPU reference (`src/cpu.rs`), CLI `adam-train` with the Taichi trainers' env and checkpoint format |

## Build

Check free memory first (`free -g`). wgpu release builds want about 4 to 6 GB; keep jobs at `-j 8`.

```bash
cd crates
cargo build --release -j 8
PYO3_PYTHON=../.venv-linux/bin/python ./build_py.sh
```

`build_py.sh` builds `adam-ray-py` and copies the extension to the repo root as `adam_ray.cpython-3xx-*.so`, so `import adam_ray` works wherever the repo root is on `sys.path` (the server, tests and scripts all add it).

## Switch the live server

`amni/compute/ray_field.py::engine()` returns the Rust engine when the environment has

```bash
AMNI_RAY_IMPL=rust
```

It passes the same pack, synsets, words, wordcounts and arch that the Taichi `RayField` would use. `AMNI_RAY_ARCH=cpu` selects the Rust CPU path, and any other value (default `vulkan`) selects wgpu. If no GPU adapter is available, it falls back to CPU. `ADAM_RAY_ADAPTER=<name substring>` pins a specific adapter. To roll back, unset `AMNI_RAY_IMPL`.

## Tests

The parity references come from the Taichi `RayField` on CPU (`AMNI_RAY_ARCH=cpu`).

```bash
AMNI_HOME=/home/anmire/ai/Amni-Ai python crates/adam-ray-py/tests/ray_ref.py v16 /tmp/ref/v16.npz
AMNI_HOME=/home/anmire/ai/Amni-Ai python crates/adam-ray-py/tests/ray_ref.py v18 /tmp/ref/v18.npz
AMNI_HOME=/home/anmire/ai/Amni-Ai python crates/adam-ray-py/tests/ray_ref.py qa /tmp/ref/qa.tsv
cd crates
cargo test -j 8 --release -p adam-ray --test numpy_compat
AMNI_HOME=/home/anmire/ai/Amni-Ai ADAM_RAY_REF=/tmp/ref cargo test -j 8 -p adam-ray --test parity -- --nocapture --test-threads=1
AMNI_HOME=/home/anmire/ai/Amni-Ai ADAM_RAY_REF=/tmp/ref cargo test -j 8 -p adam-qa --test parity -- --nocapture
AMNI_HOME=/home/anmire/ai/Amni-Ai python crates/adam-ray-py/tests/ray_ref.py tables_v16 /tmp/ref/tab
AMNI_HOME=/home/anmire/ai/Amni-Ai ADAM_RAY_TABLES=/tmp/ref/tab cargo test -j 8 -p adam-ray --test tables -- --nocapture
cd .. && AMNI_HOME=/home/anmire/ai/Amni-Ai python crates/adam-ray-py/tests/test_parity.py
```

- `ADAM_RAY_ARCHS=cpu,vulkan` picks the backends for the Rust parity test.
- `ADAM_RAY_TEST_ARCH` does the same for the Python test.
- Tests skip if the packs or references are missing.
- The table dumps are about 600 MB, so delete them after the run.

Run the benchmark with `python crates/adam-ray-py/tests/bench.py v16 v18`, which starts one engine per subprocess, or with `cargo run --release -p adam-ray --example bench -- <cpu|vulkan> <v16|v18>`.

## Layout on the GPU

| table | texture | notes |
|---|---|---|
| slot cells (L levels × T slots) | `Rg32Uint` 2D array, 4096 × T/4096 × L | `.x` = sid0, sid1, sid2 bytes plus a base-5 trit code `u` (0..124, the pack's own `_unpack5` code); `.y` = two f16 levels (`lv1 = f16(0.5·scale)`, `lv2 = f16(scale)`); weight = sign(t)·lv[\|t\|]. v19 packs decode lv1 as 0.5·lv2 instead, the trainer's `qw` rule, which stays exact when the scale is an f16 subnormal (about 0.03% of v18 cells differ between the two rules); v16/v18 keep the stored lv1, the Taichi engine's rule. Every cell of both packs decodes value-exact. |
| dense bigram rows (256 + 65536) × 256 | `R16Float`, 4096 wide (16 rows per texel row) | the dense page is f16-exact in both formats |
| synset / hypernym / wordcount (2^20) | `Rgba32Uint` 1024 × 1024 | one fetch gives all three |
| bias | `R32Float` 256 × 1 | |
| topic cos/sin for `pw1 % 360` | `Rg32Float` 360 × 1 | host libm values, so the carried topic state matches CPU |

Storage buffers hold only what the kernel writes: the per-beam driver state (double-buffered, `NS = 40` u32 per beam), the per-step parent|byte stream, and the logits output. One workgroup of 256 threads handles one beam. Thread 0 advances the state and plans the cells. Every thread gathers its own class in the Taichi order. The hop top-3 is a tie-exact rank count in workgroup memory. There is no matmul anywhere.

## v19 inference (R1b)

`RayField` detects a v19 raw pack from its level count: L=41 is COPY=13 (question tori 21..23, copy 24..36, hops 37..40), L=42/43 is COPY=13 plus 1/2 memory-result levels (v19d, see below), L=44..46 is COPY=16 plus 0..2 memory-result levels (v19e echo and entity candidates, see below), L=37 is COPY=9, L=32 is COPY=4. The Python getters `copy` and `v19` report it; nothing else changes for callers.

- The copy band (levels 24..24+COPY) is kept as raw f16 slot-0 weights and never goes through the cell ternarizer.
- `v19::W19` is one beam's v19 state: the 40-u32 core state, the parser scalars (question words and their 4-byte prefixes, working-memory cursors, fact-question clock, pronoun subject), the current-word buffer, the 128x4 set-associative working memory and the 64-entry entity ring (about 31 KB). `W19::step` is the per-byte update, `W19::gates` the copy-candidate emit. The trainer's `Drv19` runs the same two calls per position.
- `v19::Host` holds 2 x 256 `W19` (double-buffered like the core state) plus the lexicons. Each step copies the parent beam's state into the child slot and steps it, so beams fork with their memory.
- CPU path: bias, dense, planned taps, then the copy gates (`CG` x the sum of the gate cells' slot-0 weights, added to the candidate byte), then the four hops. This is the trainer's forward order.
- GPU path: the CPU steps every beam and uploads, per want-step and beam, the core state, 16 (byte, gate) pairs (unused ones off) and the 2 memory-result cell entries (`EX = 74` u32, `u32::MAX` = off). The kernel skips its own state update, plans and gathers from the uploaded state, adds the gates, the memory-result cells and runs the hops. Steps without logits are not dispatched.
- v19 packs use the trainer's topic constant (`1 - 0.99f32`). v16/v18 keep the Taichi engine's `(1 - 0.99f64) as f32`, which drifts from the trainer's by 1e-9 and can move a topic cell on long inputs.

```bash
cd crates && cargo test --release -j 8 -p adam-train --test v19_fwd --no-run
AMNI_HOME=/home/anmire/ai/Amni-Ai ADAM_V19_PACK=<rayhash_..._sid.npy> [ADAM_V19_TEXTS=<\0-separated file>] \
  [ADAM_V19_REF=<taichi npz>] [ADAM_V19_GREEDY=90] [ADAM_V19_LEX=clean1] ADAM_RAY_ARCHS=cpu,vulkan target/release/deps/v19_fwd-* --nocapture
AMNI_HOME=/home/anmire/ai/Amni-Ai python scripts/v19_rust_eval.py exports/gf17_continuum/rayhash_v19c2_chat3_rust_s20k v19c2_rust
```

- `v19_fwd` compares `RayField::logits_for` (each arch) with the `adam-train` CPU forward of the same checkpoint, and with a Taichi reference when given. The Taichi reference is `Amni-Ai-v19/tests/ray_field_ref.py` with `REF_SCRIPT=ray_hash_field_ti_v19.py LOG2_ENTRIES=16 COPY=13 TI_FAST_MATH=0 TI_ADVANCED_OPTIMIZATION=0`. It compiles in about 1.8 GB at LOG2_ENTRIES=16. Never compile it at LOG2_ENTRIES=20, which needs more than 29 GB.
- `ADAM_V19_GREEDY=n` also greedy-decodes n bytes per text through `prime`/`advance` and through the trainer's full recompute, and counts identical outputs.
- `v19_rust_eval.py` runs chat3 `hold_eval` and suite D (`recall.jsonl` split into known and unknown, plus `twohop.jsonl`) and writes `data/concepts/eval/report_<tag>.json` under `AMNI_HOME`. `LEX` picks the lexicon under `data/` (default `clean1`; `big2` only for the retired packs in `exports/legacy_grok`). `C3_N` (default 0, meaning all 1000 dialogues) limits chat3, `EVAL_N` limits suite D, and `AMNI_RAY_ARCH` picks the backend (default cpu).
  - Decoding: `DEC=greedy` (default) is the raw argmax loop, which stops at byte 10 or 30. `DEC=stream` is `stream(topk=1)` and `DEC=beam` is `beam(BEAM_W, 4)`, both with `LOCK`, `SS` (`stop_sentence`), `REP` (stream only), `VETO` and `MAXB` (default 90 bytes).
  - Every reply gets two scores. The lenient one (`hit`) counts a reply as correct if it contains the value. The strict one (`strict`) needs all of the following:
    - the first sentence, up to the first `.`, `!` or `?` followed by whitespace or the end, contains the value as whole words (hyphens count as word characters);
    - no conflicting value appears anywhere, except after `not`, `instead of`, `rather than` or `changed/switched/moved (it) from`;
    - no content word appears 3 or more times (function words such as "the", "you" and "in" are exempt);
    - the reply is at most 200 bytes.
  - For an unknown item, the first sentence must contain a decline (`DECLINE` in `v19_eval_run.py`) and none of the candidate values.
  - Conflicting values: collect every answer string in chat3 and suite D that is at least 3 characters long. Keep those that occur case-sensitively in the dialogue's earlier `User:` lines, other than the answer and its sub- or superstrings. Outside `correction` items, a value that shares a `User:` line with the answer (the two-hop bridge city, for example) is exempt.

## Copy-lock decoding

`stream(..., lock, stop_sentence, rep)` and `beam(..., lock, stop_sentence)` take the Rust `Dec` options (Python keywords `lock`, `stop_sentence` and `rep`). All of them default to off, and with them off the outputs are byte-identical to the previous engine.

- `lock` works on v19 packs with `COPY >= 13`. At each step, a copy candidate (direct, hop, chain, and on COPY=16 packs echo or entity) is eligible when all of the following hold:
  - its gate byte is valid and its gate logit is positive;
  - the current word has at least one byte, and the character before the word is not an apostrophe or a hyphen;
  - its next value byte is the model's own argmax;
  - its value has not been banned.
  - The eligible candidate with the highest gate logit is locked, and its remaining value bytes are forced, across spaces and hyphens, until the value is complete. The chain candidate (12) can also start at a word boundary right after a completed chain word.
- After a locked value completes:
  - the next byte cannot be a word character, so "Detroit" cannot become "Detroited";
  - the value is banned: its gate contribution is subtracted wherever it pushes again, and any byte that would extend a word to the value's first 3 bytes, or to all of a shorter value, is masked;
  - when `stop_sentence` is also on and no hop candidate is still pending, a word start whose argmax re-opens a banned value ends the reply ("In Austin Austin" becomes "In Austin.").
- `stop_sentence` stops at the first `.`, `!` or `?` followed by a space or newline once a full word has been emitted, and never inside a locked value.
- `rep` is subtracted from the logit of every byte that would repeat a 6-byte sequence already in the reply.
- Everything is deterministic: a forced byte skips the sampler.
- The server's memory route (`amni/serve/gguf_runtime.py`) defaults to greedy, `lock`, `stop_sentence`, `rep=1` and `veto` off. The knobs are `AMNI_RAY_MEM_LOCK`, `AMNI_RAY_MEM_STOP_SENTENCE`, `AMNI_RAY_MEM_REP`, `AMNI_RAY_MEM_TOPK`, `AMNI_RAY_MEM_TEMP`, `AMNI_RAY_MEM_VETO` and `AMNI_RAY_MEM_BYTES`. The Taichi engine ignores the lock options.

## v19d memory-result levels

`MR=1|2` (trainer env, needs `VER=19 COPY=13`) appends `MR` ordinary 3-slot torus levels after the hops, so a v19c pack is a strict prefix of a v19d one (`PAGES` warm-starts levels 0..40 and the new levels start at the `sid0` pattern with zero weights).

| levels | L=43 (`MR=2`) |
|---|---|
| 0..1 | dense byte1 / byte2 |
| 2..19, 20 | byte and word tori, topic torus |
| 21..23 | question tori |
| 24..36 | copy gates (13 candidates, 4 cells each) |
| 37..40 | belief hops (`hb() = 37`) |
| 41 | memory result A (`mb() = hb() + hops`) |
| 42 | memory result B, answer template (`lv() = mb() + mr`) |

The signal comes from `W19` at every Adam-reply position (`qon == 1`), using the candidates picked at `Adam: `:

- `mstate()` = `(min(#direct candidates, 2) + 3·[two-hop candidate] + 6·[question relation parsed] + 12·who) · 4 + qtype`, where who is 0 no question entity, 1 the user, 2 another person or pet, and qtype is 0 word, 1 name, 2 place, 3 number. So `mstate / 4 % 3` is empty / one / several. The `who` split is what separates "memory is empty for a question about my uncle" (decline) from a general question (normal answer): generic chat2 turns are all empty, but only 10% of them name another person.
- `k = mstate · 16 + rel`, where rel is the top candidate's relation, or the parsed question relation when memory is empty.
- `Q::ap` counts bytes since `Adam: ` (0 at the byte that predicts the first answer byte); `ab` buckets it as 0, 1, 2, 3, 4..7, 8..15, 16..31, 32+.
- Level A key: `sl(hm(k / 64, min(ap, 23), 139), byte0)`: coarse state (empty / one / several, hop, parsed, who) × position × last byte.
- Level B key: `sl(hm(k, ab, 141), hm(byte0 | byte1 << 8 | byte2 << 16, byte3, 143))`: full state and candidate relation × position bucket × last 4 bytes, so the reply template ("You haven't told me…" versus "You told me…") is conditioned on the memory.
- Both taps have weight `MG = 4` (like the copy gates' `CG`), are added after the copy gates and before the hops, and train like any other cell (3 slots, replacement, ternary per cell).
- Anchor: in training batches a window that does not start at a dialogue separator (byte 30) has an incomplete memory, so `W19::q.anc` is false until the first byte 30 and the memory-result taps stay off. `W19::begin` (inference, `Engine::logits`) sets it true.

`crates/adam-train/tests/v19_probe.rs::probe_mr` checks the gating (answer only, anchor) and the empty / one states; `tests/v19_mstate.rs` (`ADAM_V19_MSTATE=<\0-separated kind\x01text>`) tabulates the state over an eval set.

## v19e echo and entity candidates

`COPY=16` (trainer env, `VER=19`) adds three copy candidates to the 13 of v19c/v19d, with their own 4-cell gate levels right after the chain gate, so the hops and memory-result levels move up by three.

| levels | L=46 (`COPY=16 MR=2`) |
|---|---|
| 24..36 | copy gates 0..12 (direct 0..7, hop 8..11, chain 12) |
| 37..38 | gates 13, 14: echo, the newest and second-newest facts committed in the current user turn (ring stamp above `qclk`) |
| 39 | gate 15: entity, the surface form of the current turn's possessed entity phrase ("brother", "sister-in-law", "lab partner") |
| 40..43 | belief hops |
| 44..45 | memory results A and B |

- Direct candidates exclude the current turn by design, so after a statement ("My brother works as a pilot.") only the echo candidates carry the new value, and the entity candidate carries "brother" for "Oh, your brother is a pilot?". After a question the entity candidate lets "Your ___" copy the asked-about person.
- The entity surface is captured by the parser: after `my`/`your`, every word while the possessive stays open is appended (with its `-`, `'` or space separator), and the buffer is committed whenever the word becomes the entity. It is reset at every `User: `.
- On COPY=16 packs, value offsets also advance across `-` inside a value ("sister-in-law", "cross-stitch").
- Gate cells on COPY=16 packs (all 16 candidates): cell 0 is `hm(kind, rank, ctx)` with `ctx = 1` inside a word and `2 + (pw1 & 4095)·2 + [last byte is a space]` at a word start, so the first-byte gate is conditioned on the previous word ("is a", "in", "Your"); cell 2 adds 4 to the question type when the user turn had no `?`, so gates learn separate statement-ack and answer behavior. Kinds: 0 direct, 1 hop, 2 chain, 3 echo, 4 entity. COPY=13 packs keep the v19d keys bit-exact.
- A pronoun in the turn ("He drives a Rivian.", "What does he drive?") reuses the surface of the last committed entity when the pronoun resolves to it, so the entity candidate still says "brother".
- Memory-result levels on COPY=16 packs:
  - `mstate()` adds 144 when the user turn had no `?`, so statement acknowledgements and answers get separate memory-result keys (question keys are unchanged from v19d).
  - Level A uses position slot 24 instead of `min(ap, 23)` at a value end (the current word completes a live direct, hop, echo or entity value), so "value just ended" is one dense key per state and last byte.
  - Level B, once a value has ended in the reply (`Q::vp` = bytes since the value end + 1), keys on `hm(k, 8 + min(d, 15))` and only the bytes typed after the value (`d` = bytes since the end), so the template tail (", if I remember right.", "! Nice to meet you.") does not depend on the value's own bytes.
- Copy-lock on COPY=16 packs can also lock at a word start (`n == 0`, value offset 0) when the candidate's gate is what makes its first byte the argmax (the argmax minus the gate is below the runner-up), so "Anthony" is not split into an article "A" plus a later copy. On every v19 pack, the stop-at-banned-value rule no longer fires right after an apostrophe ("Your sister's ...").
- The GPU engine uploads 16 (byte, gate) pairs plus the 2 memory-result entries per beam (`EX = NS + 34`).
- `PAGES=<v19d pack>` with `COPY=16` warm-starts levels 0..36 and moves hops and memory results from 37..42 to 40..45; levels 37..39 start fresh.
- `tests/v19_probe.rs::probe_echo` checks the echo, entity, hyphen and key layout; `examples/v19_dump.rs <pack> <lexicon dir> <text file> [bytes]` prints the ring, the question state, every live gate and a per-stage logit breakdown (dense, tori, question tori, gates, memory results, hops) at the end of the text.

## Trainer (R2): `adam-train`

`adam-train` replaces `scripts/ray_hash_field_ti_v16.py`, `ray_hash_field_ti_v18.py` and the v19c working-memory trainer (`ray_hash_field_ti_v19.py`, parser-fix revision). It reads the same environment, samples the same batches (numpy PCG64), and writes the same `exports/gf17_continuum/rayhash_<TAG>{.json,_dp.npy,_sid.npy,_sw.npy,_bias.npy}`. Those files load in `adam_ray`, in the Taichi `RayField`, and in the Taichi trainers (`PAGES=`).

```bash
cd crates && cargo build --release -j 8 -p adam-train
AMNI_HOME=/home/anmire/ai/Amni-Ai THROTTLE_FILE=/path/throttle.txt ARCH=vulkan \
CORPUS=chat2 LEVELS=21 HOPS=4 QT=1 TERN=1 TERN_SCOPE=cell QLEV=5 LOG2_ENTRIES=20 CTX=256 BS=128 \
STEPS=20000 LR=3e-3 LR_END=3e-4 SYNSETS=$AMNI_HOME/data/big2/synsets.txt TAG=v18_chat2_rust \
target/release/adam-train
```

- `QT=1` (the default) is v18 with question tori. `QT=0` is v16.
- `VER=19` is v19c working memory. `COPY=13` gives the relational memory: a 64-entry entity ring, 13 copy candidates and 4-cell gates. `COPY=4` and `COPY=9` give the byte-copy bands.
  - v19 also brings the byte-30 dialogue reset, the dialogue-start batch sampler, the copy band that skips the ternary quantizer, and the `PAGES_V18` and `PAGES_V19B` warm starts.
- `ARCH=cpu` runs the CPU reference. Any other value runs wgpu and falls back to the CPU if no adapter is found.
- Also supported: `SGRID/SLO/SHI`, `MULTI` (salted per corpus), `PAGES` (full warm start), `PAGES_V16` (v16 into v18), `SEED`, `RTH/RW/WD`, `TOPIC_ALPHA`, `GEN_T/GEN_K`.
- Not supported: `GATE` (it must be 0), the `SLOTS` values other than 3, and the inference-only `DUMP`/`GEN_PROMPT`/`WORDS`/`COUNTS`.
- `THROTTLE_FILE` is read at step 1 and every 50 steps. The trainer sleeps that many seconds per step.
- Extras:
  - `EXPORTS=<dir>` redirects the outputs.
  - `STEP_LOG=<file>` writes every step's loss and every eval with full precision.
  - `DUMP_F32=<dir>` writes the f32 tables and optimizer moments.
  - `PROF=1` prints per-pass GPU timestamps.

### Passes (one small WGSL shader each; no matmul)

| pass | grid | work |
|---|---|---|
| `state` | thread per sequence | the shared byte state machine (`adam-ray/src/state.wgsl::upd`), which writes 40 u32 of state per position |
| `plans` | thread per position | hashed torus cells, topic taps, question tori (`plan`), written as `[pos][level][4]` (entry, weight) |
| v19 driver | CPU, one thread per sequence (`adam-ray/src/v19.rs`, wrapped by `src/v19.rs::Drv19`) | word/parse events, the memory update and the candidate/gate emit, run sequentially per sequence; the `[pos][level][4]` taps and 13 copy target bytes are uploaded, and the GPU `state`/`plans` passes are skipped |
| `fwd` | workgroup per position | bias, dense byte1/byte2, all taps, the v19 copy gates (`4·Σ w·sw[slot 0]` added to the target byte), chained belief hops (tree-merge top-3), then CE, dlogits and loss |
| `hist`, `args`, `umap` | per tap / 1 / per key | unique (level, entry) keys, indirect dispatch sizes, key to slot map |
| `slots`, `dacc` | per tap / per position | gradient sums as exact 64-bit fixed-point atomics (order-free, so runs are bit-reproducible), plus the replacement candidate `atomicMax` |
| `dense`, `slot` | per dense key / per slot key | Adam on the dense rows and on slot weights; clip ±4, weight decay and the NaN guard; slot-id replacement (the first miss in Taichi order is the highest tap index) |
| `part`, `fin`, `lvl` | | bias gradient and step, and the per-level ternary thresholds (`TERN_SCOPE=level`) |

The tables are textures:

- `dp` is R32Float 4096×8192.
- `sid` is an R32Uint array with one layer per level (three id bytes).
- `sw` is an R32Float array with three layers per level.
- `bias` is R32Float.
- The lexicon and topic LUTs come from `adam-ray`.

Every table is sampled in `fwd` and bound as a `read_write` storage texture only in the pass that updates it. Adam moments, taps and accumulators are storage buffers.

### Parity

```bash
cd /home/anmire/ai/Amni-Ai
TI_FAST_MATH=0 TI_CPU_MAX_NUM_THREADS=1 ARCH=cpu <trainer env> .venv-linux/bin/python \
  <worktree>/crates/adam-train/tests/taichi_ref.py /tmp/ref
cd <worktree>/crates
ADAM_TRAIN_REF=/tmp/ref <same trainer env> cargo test --release -p adam-train --test parity -- --nocapture
python3 adam-train/tests/compare.py <rust dump dir> /tmp/ref
python crates/adam-train/tests/fwd_parity.py <rust _sid.npy>
```

- `REF_SCRIPT=ray_hash_field_ti_v16` selects the v16 reference.
- `fwd_parity.py` loads the checkpoint into both `adam_ray` and the Taichi `RayField` and compares logits.
- The CPU path matches Taichi CPU exactly (every loss and every table value) when Taichi runs with `TI_FAST_MATH=0` and one thread.
  - Taichi's default fast-math rewrites `a*x + (1-a)*y` as `a*(x-y) + y`.
  - Taichi's default parallel scatter makes its own segment order nondeterministic.
- The GPU path is deterministic run to run, but not bit-equal to the CPU because of:
  - GPU `exp` and `log`;
  - driver FMA contraction;
  - fixed-point gradient sums;
  - argmin ties in slot replacement, which amplify ulp differences.

### v19 parity

```bash
cd /home/anmire/ai/Amni-Ai
TI_FAST_MATH=0 TI_ADVANCED_OPTIMIZATION=0 AMNI_HOME=<home holding scripts/ray_hash_field_ti_v19.py> \
  .venv-linux/bin/python <worktree>/crates/adam-train/tests/v19_drv_ref.py /tmp/v19drv.npz
cd <worktree>/crates
ADAM_V19_DRV=/tmp/v19drv.npz ADAM_V19_DUMP=/tmp/v19rust VER=19 COPY=13 LEVELS=21 HOPS=4 QT=1 LOG2_ENTRIES=16 CTX=8192 BS=8 \
  CORPUS=chat3 SYNSETS=$AMNI_HOME/data/big2/synsets.txt cargo test --release -p adam-train --test v19_drv -- --nocapture
python3 adam-train/tests/v19_recall.py /tmp/v19rust
cargo test --release -p adam-train --test v19_probe -- --nocapture
```

- `v19_drv` compares the Rust driver with the Taichi `drivers` kernel over all 1,000 `data/chat3/hold_eval.jsonl` dialogues:
  - the IDX, W and CPB plan (CRC per batch);
  - the entity ring, the candidates and the question state.
- `v19_recall.py` prints the candidate recall@8 table from either dump.
- `v19_probe` ports the `probe` and `probe13` asserts of `tests/test_v19_memory.py`.
- Taichi's v19c CPU compile fits in about 2 GB only with `TI_ADVANCED_OPTIMIZATION=0`. That setting does not change results: the v18 parity stays bit-exact with it.
