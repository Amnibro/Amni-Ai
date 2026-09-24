# Adam Rust crates

Cargo workspace for Adam's Rust port (phase R1: inference engine + answer memory).
`amni_ai_core/` (the AmniAI host binary) is excluded from the workspace and still builds on its own.

| crate | what it is |
|---|---|
| `adam-ray` | `RayField` engine: v16 packed `.npz` and v18 raw `_sid.npy` loaders, wgpu compute kernel (`src/ray.wgsl`), CPU reference path, `prime` / `logits_for` / `advance` / `run` / `score` / `bpb` / `stream` / `beam` |
| `adam-qa` | port of `amni/compute/ray_qa.py`: `words`, `stem`, index loader for `ray_qa_index.npz`, `search`, `overlap`, `hard`, `pick`, `guide` |
| `adam-ray-py` | PyO3 module `adam_ray` exposing a drop-in `RayField`, `QA`, `pick`, `guide`, `words`, `stem`, `hard` |

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
