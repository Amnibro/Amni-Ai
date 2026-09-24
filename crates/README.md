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
| slot cells (L levels × T slots) | `Rg32Uint` 2D array, 4096 × T/4096 × L | `.x` = sid0, sid1, sid2 bytes plus a base-5 trit code `u` (0..124, the pack's own `_unpack5` code); `.y` = two f16 levels (`lv1 = f16(0.5·scale)`, `lv2 = f16(scale)`); weight = sign(t)·lv[\|t\|]. Every cell of both packs decodes value-exact. |
| dense bigram rows (256 + 65536) × 256 | `R16Float`, 4096 wide (16 rows per texel row) | the dense page is f16-exact in both formats |
| synset / hypernym / wordcount (2^20) | `Rgba32Uint` 1024 × 1024 | one fetch gives all three |
| bias | `R32Float` 256 × 1 | |
| topic cos/sin for `pw1 % 360` | `Rg32Float` 360 × 1 | host libm values, so the carried topic state matches CPU |

Storage buffers hold only what the kernel writes: the per-beam driver state (double-buffered, `NS = 40` u32 per beam), the per-step parent|byte stream, and the logits output. One workgroup of 256 threads handles one beam. Thread 0 advances the state and plans the cells. Every thread gathers its own class in the Taichi order. The hop top-3 is a tie-exact rank count in workgroup memory. There is no matmul anywhere.

## v19 inference (R1b)

`RayField` detects a v19 raw pack from its level count: L=41 is COPY=13 (question tori 21..23, copy 24..36, hops 37..40), L=37 is COPY=9, L=32 is COPY=4. The Python getters `copy` and `v19` report it; nothing else changes for callers.

- The copy band (levels 24..24+COPY) is kept as raw f16 slot-0 weights and never goes through the cell ternarizer.
- `v19::W19` is one beam's v19 state: the 40-u32 core state, the parser scalars (question words and their 4-byte prefixes, working-memory cursors, fact-question clock, pronoun subject), the current-word buffer, the 128x4 set-associative working memory and the 64-entry entity ring (about 31 KB). `W19::step` is the per-byte update, `W19::gates` the copy-candidate emit. The trainer's `Drv19` runs the same two calls per position.
- `v19::Host` holds 2 x 256 `W19` (double-buffered like the core state) plus the lexicons. Each step copies the parent beam's state into the child slot and steps it, so beams fork with their memory.
- CPU path: bias, dense, planned taps, then the copy gates (`CG` x the sum of the gate cells' slot-0 weights, added to the candidate byte), then the four hops. This is the trainer's forward order.
- GPU path: the CPU steps every beam and uploads, per want-step and beam, the core state and 13 (byte, gate) pairs (`EX = 66` u32). The kernel skips its own state update, plans and gathers from the uploaded state, adds the gates and runs the hops. Steps without logits are not dispatched.
- v19 packs use the trainer's topic constant (`1 - 0.99f32`). v16/v18 keep the Taichi engine's `(1 - 0.99f64) as f32`, which drifts from the trainer's by 1e-9 and can move a topic cell on long inputs.

```bash
cd crates && cargo test --release -j 8 -p adam-train --test v19_fwd --no-run
AMNI_HOME=/home/anmire/ai/Amni-Ai ADAM_V19_PACK=<rayhash_..._sid.npy> [ADAM_V19_TEXTS=<\0-separated file>] \
  [ADAM_V19_REF=<taichi npz>] [ADAM_V19_GREEDY=90] ADAM_RAY_ARCHS=cpu,vulkan target/release/deps/v19_fwd-* --nocapture
AMNI_HOME=/home/anmire/ai/Amni-Ai python scripts/v19_rust_eval.py exports/gf17_continuum/rayhash_v19c2_chat3_rust_s20k v19c2_rust
```

- `v19_fwd` compares `RayField::logits_for` (each arch) with the `adam-train` CPU forward of the same checkpoint, and with a Taichi reference when given. The Taichi reference is `Amni-Ai-v19/tests/ray_field_ref.py` with `REF_SCRIPT=ray_hash_field_ti_v19.py LOG2_ENTRIES=16 COPY=13 TI_FAST_MATH=0 TI_ADVANCED_OPTIMIZATION=0`. It compiles in about 1.8 GB at LOG2_ENTRIES=16. Never compile it at LOG2_ENTRIES=20, which needs more than 29 GB.
- `ADAM_V19_GREEDY=n` also greedy-decodes n bytes per text through `prime`/`advance` and through the trainer's full recompute, and counts identical outputs.
- `v19_rust_eval.py` is `v19b_eval.py` with `adam_ray` greedy decoding: the same 934-byte prompt window, 90 bytes, and a stop at byte 10 or 30. It writes `data/concepts/eval/report_<tag>.json` under `AMNI_HOME`. `C3_N` (default 0, meaning all 1000 dialogues) limits chat3, `EVAL_N` limits the other suites, and `AMNI_RAY_ARCH` picks the backend (default cpu).

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
