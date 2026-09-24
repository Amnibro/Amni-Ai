# Adam Rust crates

Cargo workspace for Adam's Rust port (phase R1: inference engine + answer memory; phase R2: trainer).
`amni_ai_core/` (the AmniAI host binary) is excluded from the workspace and still builds on its own.

| crate | what it is |
|---|---|
| `adam-ray` | `RayField` engine: v16 packed `.npz` and v18 raw `_sid.npy` loaders, wgpu compute kernel (`src/ray.wgsl`), CPU reference path, `prime` / `logits_for` / `advance` / `run` / `score` / `bpb` / `stream` / `beam` |
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

## Trainer (R2): `adam-train`

`adam-train` replaces `scripts/ray_hash_field_ti_v16.py` and `ray_hash_field_ti_v18.py`. It reads the same environment, samples the same batches (numpy PCG64), and writes the same `exports/gf17_continuum/rayhash_<TAG>{.json,_dp.npy,_sid.npy,_sw.npy,_bias.npy}`. Those files load in `adam_ray`, in the Taichi `RayField`, and in the Taichi trainers (`PAGES=`).

```bash
cd crates && cargo build --release -j 8 -p adam-train
AMNI_HOME=/home/anmire/ai/Amni-Ai THROTTLE_FILE=/path/throttle.txt ARCH=vulkan \
CORPUS=chat2 LEVELS=21 HOPS=4 QT=1 TERN=1 TERN_SCOPE=cell QLEV=5 LOG2_ENTRIES=20 CTX=256 BS=128 \
STEPS=20000 LR=3e-3 LR_END=3e-4 SYNSETS=$AMNI_HOME/data/big2/synsets.txt TAG=v18_chat2_rust \
target/release/adam-train
```

- `QT=1` (the default) is v18 with question tori. `QT=0` is v16.
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
| `fwd` | workgroup per position | bias, dense byte1/byte2, all taps, chained belief hops (tree-merge top-3), then CE, dlogits and loss |
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
