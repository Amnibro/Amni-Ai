# adam-t

Adam-T is Adam's homegrown ternary transformer core, with PTEX hashed n-gram memory.

- **Training:** burn on wgpu/Vulkan, with a straight-through estimator (STE) for the ternary weights.
- **Export:** a GF(3)-in-GF(17) `.gf17` file.
- **Inference:** lookup-only CPU LUT inference.

The design is in `docs/adam_t/design.md`, and the evaluation rules are in `docs/adam_t/eval_suite.md`.

## Build

```
cd crates && cargo build --release -p adam-t
```

The binary is written to `crates/target/release/adam-t`.

## Scoring interface: `adam-t serve` (proto 1)

```
adam-t serve exports/adam_t/<TAG>/model.gf17
adam-t serve exports/adam_t/<TAG>
```

The model can be given two ways:

- **A `.gf17` file:** the lookup-only CPU path. It needs no GPU. Generation uses a per-layer K/V cache, so each new symbol costs one step and the window is not rerun.
- **A training run directory:** the burn checkpoint path.
  - It needs `log.jsonl`, whose first line holds the config, plus `ckpt/`.
  - It runs on the GPU by default. Set `BACKEND=ndarray` to run on the CPU. GPU runs follow `THROTTLE_FILE`.
  - Use it for fast batched `lp`. Only one GPU job runs at a time, so pause training first.

The protocol is JSON lines on stdin and stdout, one reply per request, and all text is base64:

| request | reply |
|---|---|
| `{"op":"hello"}` | `{"model":"adam-t-<TAG>","step":N,"fmt":"plain","ctx":<window bytes, approx>,"ctx_symbols":T,"vocab":V,"proto":1}` |
| `{"op":"lp","items":[[ctx_b64,cont_b64],...]}` | `{"lp":[ln p(cont\|ctx),...]}` |
| `{"op":"gen","prompt":b64,"max":N,"stop":[b64,...]}` | `{"text":b64}` |
| `{"op":"bye"}` | `{"bye":true}`, then the server exits |
| anything that fails | `{"error":"..."}` |

`lp` details:
- Values are natural-log probabilities summed over the whole continuation.
- The context is truncated from the left.
- A continuation longer than the window is scored in strided windows, with at least T/2 symbols of left context each.
- `LP_BATCH` (default 16) sets how many windows go into one forward.

`gen` details:
- Decoding is greedy.
- `max` is counted in bytes.
- The text is cut at the first stop sequence, and the stop itself is not returned.

Byte models use symbols equal to bytes. BPE models (`v` > 256) carry their merge table inside the `.gf17` file, and they tokenize context and continuation separately.

The eval harness uses it like this: `python scripts/adam_t_eval.py run --model proto --proto-cmd "crates/target/release/adam-t serve exports/adam_t/<TAG>/model.gf17"`.

## Tokenizer (homegrown BPE)

- `crates/adam-t/bpe.py <V>` trains the merges on `data/clean2/train.bin` only and writes `data/clean2/bpe<V>.json`. The merges come out of HF `tokenizers` used as code.
- `adam-t tokenize <bpe.json> <in.bin> <out.u16>` is a lossless Rust encoder that runs in parallel, at about 160 MB/s on 8 threads. The trainer reads `data/<CORPUS>/{train,hold}.bpe<V>.u16` when `BPE=<bpe.json>` is set.
- `adam-t bpe-enc <bpe.json>` prints the ids for stdin.

## Training

`crates/adam-t/smoke.sh` is the runner that follows the GPU rules. Before it starts, it refuses to run without `THROTTLE_FILE`, or while another `adam-train` or `adam-t` process is running. It then:

- logs 1 Hz `gpu_busy_percent` to `$RUN_DIR/gpu_busy.log`;
- runs the job in a `systemd-run` scope with `MEM_MAX` (default 12G) and `CPU_Q` (default 800%).

```
THROTTLE_FILE=... RUN_DIR=exports/adam_t/<TAG> TAG=<TAG> TERN=1 MEM=1 STEPS=3000 crates/adam-t/smoke.sh crates/target/release/adam-t train
```

### Environment variables

| Group | Variables |
|---|---|
| Shape | `D`, `L`, `H`, `CTX`, `FF` |
| Model type | `TERN` (0 or 1); `MEM` (0 or 1) with `NGRAMS` and `MEM_LOG2` |
| Batch and length | `BS`, `STEPS` |
| Optimiser | `LR`, `LR_MEM`, `WD`, `WARM` |
| Data and output | `SEED`, `CORPUS` (default `clean2`), `OUT` (default `exports/adam_t/$TAG`) |
| Eval and checkpoints | `EVAL_EVERY`, `CKPT_EVERY`, `EVAL_BATCHES` |
| Limits | `MAX_MIN`: wall-clock limit per process; the run checkpoints, pauses, and resumes when rerun |
| Throttle | `THROTTLE_FILE`, `THROTTLE_UNIT_MS` |
| Vocabulary | `BPE` (path to `bpe<V>.json`; empty means bytes) |

The throttle file is read every step. The run sleeps `throttle × max(1, gpu_ms/THROTTLE_UNIT_MS)` per step.

At the end, a ternary run exports `model.gf17`. It then checks that the roundtrip is bit-exact and that the LUT output matches the trainer's.

### Other subcommands

- `adam-t bench`: training-step throughput.
- `adam-t mm`: GEMM microbenchmark comparing WGSL and burn.
- `adam-t lut <model.gf17> [windows]`: CPU LUT bits per byte on hold windows.
- `adam-t fa`: checks and benchmarks the custom GPU ops. `FA_MODE=check` compares flash attention's output and gradients with burn's attention; `FA_MODE=flash|ref` times attention forward+backward (`FA_B`, `FA_T`, `FA_H`, `FA_IT`); `FA_MODE=lin` checks the fused ternary linear against the autodiff STE reference; `FA_MODE=xmm` times the exact bf16 cooperative-matrix GEMMs.

### Custom GPU ops (`src/fa.rs`)

- **Flash attention:** CubeCL kernels (tiled online-softmax forward that stores only the log-sum-exp, and a recompute backward split into a dQ kernel and a dK/dV kernel). They read the fused `qkv` matmul output directly and write `o` in `[B*T, D]` layout, so no head transposes. Registered as a custom op for `CubeBackend`, `Fusion` and `Autodiff`; other backends use the reference formula. Needs head dim 64 and `T % 64 == 0`, otherwise it falls back. `ATTN=ref` restores burn's attention.
- **Fused ternary linear:** one autodiff op per linear map that saves only `x` and `W` and recomputes RMSNorm, the activation grid and the trits in backward, with the same STE gradients. `LIN=ref` restores the op-by-op autodiff path.
- **Exact cooperative-matrix GEMMs:** the ternary forward `xi · t` has integer operands (|xi| <= 128, t in {-1,0,1}), so it runs in bf16 on the matrix cores with f32 accumulation and stays bit-exact. Backward operands that are real floats are split by one kernel into three bf16 terms whose sum is the f32 value, so the products are exact too. `XMM=0` falls back to f32 GEMMs.
- **Pausing a run:** `touch $RUN_DIR/STOP` makes the trainer checkpoint and exit, and `campaign.sh` stops without an eval. `crates/adam-t/t30m.sh` is the 30M launcher.
