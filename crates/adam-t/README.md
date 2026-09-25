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

## Scoring interface (for eval harnesses)

The model can be loaded two ways:

- `--ckpt <run dir>`: a training run directory such as `exports/adam_t/<TAG>`. It needs `log.jsonl`, whose first line holds the config, plus `ckpt/core.bin`, and `ckpt/mem.bin` for memory models.
  - It runs on the GPU by default. Set `BACKEND=ndarray` to run on the CPU instead.
  - GPU runs follow the governor: set `THROTTLE_FILE`.
- `--gf17 <model.gf17>`: the exported lookup-only model, run on the CPU.
  - It needs no GPU and no checkpoint.
  - It matches `--ckpt` to about 1e-3 nats per byte.

Both commands read the whole of stdin as JSON lines and write one JSON line to stdout per input line, in the same order.

### logprob

```
adam-t logprob --ckpt exports/adam_t/smoke_tern_mem < items.jsonl
{"context": "The capital of France is", "continuation": " Paris"}
-> {"context": ..., "continuation": ..., "logprob": -10.97, "n_bytes": 6, "greedy": false, "truncated": false}
```

- `logprob` is the sum of natural-log probabilities over the continuation's UTF-8 bytes. The model is byte-level, so divide by `n_bytes` for a per-byte score, or by `n_bytes·ln2` for bits per byte.
- `greedy` is true when every continuation byte is the argmax.
- `truncated` is true when context plus continuation exceeded the model's context window (`ctx`, 256 bytes for the smoke models). Only the last `ctx+1` bytes are scored as the window.
- An empty context is replaced with `"\n"`.
- `LP_BATCH` (default 16) sets how many items go into one GPU forward.

### gen

```
adam-t gen --gf17 exports/adam_t/smoke_tern_mem/model.gf17 < prompts.jsonl
{"prompt": "Q: Who wrote Hamlet?\nA:", "max_new": 32, "stop": "\n"}
-> {"prompt": ..., "completion": " ...", "new_bytes": 12}
```

- Decoding is greedy, one byte at a time.
- `max_new` defaults to 64.
- `stop` defaults to `"\n"`, and is stripped from the completion. Use `""` for no stop.
- A plain-text input line is taken as `{"prompt": line}`.

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

The throttle file is read every step. The run sleeps `throttle × max(1, gpu_ms/THROTTLE_UNIT_MS)` per step.

At the end, a ternary run exports `model.gf17`. It then checks that the roundtrip is bit-exact and that the LUT output matches the trainer's.

### Other subcommands

- `adam-t bench`: training-step throughput.
- `adam-t mm`: GEMM microbenchmark comparing WGSL and burn.
- `adam-t lut <model.gf17> [windows]`: CPU LUT bits per byte on hold windows.
