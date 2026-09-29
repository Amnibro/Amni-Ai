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
| `{"op":"hello"}` | `{"model":"adam-t-<TAG>","step":N,"fmt":"plain","ctx":<window bytes, approx>,"ctx_symbols":T,"vocab":V,"proto":1,"gen":{decoding defaults}}` |
| `{"op":"lp","items":[[ctx_b64,cont_b64],...]}` | `{"lp":[ln p(cont\|ctx),...]}` |
| `{"op":"gen","prompt":b64,"max":N,"stop":[b64,...]}`, plus the optional decoding fields below | `{"text":b64}` |
| `{"op":"bye"}` | `{"bye":true}`, then the server exits |
| anything that fails | `{"error":"..."}` |

`lp` details:
- Values are natural-log probabilities summed over the whole continuation.
- The context is truncated from the left.
- A continuation longer than the window is scored in strided windows, with at least T/2 symbols of left context each.
- `LP_BATCH` (default 16) sets how many windows go into one forward.

`gen` details:
- Decoding is greedy unless `temperature` is set. The optional request fields below override the server defaults:

| field | effect | `plain` default | `chat` default |
|---|---|---|---|
| `no_repeat_ngram` | bans any token that would complete an n-gram already in the reply (0 = off) | 0 | 3 |
| `rep_penalty` | divides the positive logits (and multiplies the negative ones) of every token already in the reply, once per distinct token (1 = off) | 1.0 | 1.15 |
| `temperature`, `top_k` | above 0, samples from the `top_k` best logits (0 = all) at that temperature | 0, 0 | 0, 0 |
| `seed` | seeds the sampler so that a request is reproducible; unseeded sampling uses seed 0 | 0 | 0 |
| `stop_repeat` | ends the reply when a clause of 3+ words (split at `. ! ? ; , :` and newlines) repeats an earlier one, and cuts before the repeat | false | true |

- The controls act on the logits, on both the `.gf17` LUT path and the burn path. On the LUT path they add no float matmul.
- If every token is banned, the step falls back to the raw argmax.
- Setting `GEN_DEFAULTS='{"no_repeat_ngram":0,...}'` in the server's environment overrides the defaults for every request. `hello` reports the defaults in effect, so harness reports record them.
- `max` is counted in bytes.
- The text is cut at the first stop sequence, and the stop itself is not returned.

Byte models use symbols equal to bytes. BPE models (`v` > 256) carry their merge table inside the `.gf17` file, and they tokenize context and continuation separately.

The eval harness uses it like this: `python scripts/adam_t_eval.py run --model proto --proto-cmd "crates/target/release/adam-t serve exports/adam_t/<TAG>/model.gf17"`.

Tuned models (runs made by `adam-t sft`, or an exported `meta.json` with `"fmt": "chat"`) answer `hello` with `"fmt": "chat"`, so the harness uses its `User: …\nAdam:` templates. Their `gen` also always stops at `\n\n` and `\nUser:`, on top of the client's stops, so a reply ends at the end of the Adam turn.

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
| Memory scale-up | `INIT=<run dir>`: continued pretraining from that run's checkpoint (shape from its `log.jsonl`, fresh optimisers, step 0); its PTEX tables are grown exactly to `MEM_LOG2`, which must not be smaller. `MEM_OPT` (`dense` default, or `lazy`: row-lazy Adam that updates and stores moments only for the table, touched rows only) |
| Binding | `COPY` (0 or 1; 1 adds the gated pointer/copy head: a ternary `d -> 2*d/H + 1` map on the final residual gives a content-only query/key and a gate logit, attention over earlier positions puts mass on the token that followed each key, and the output is `log(g*softmax(logits) + (1-g)*copy)`; exported as table `w{4L+1}` plus `cpy`, served lookup-only), `RECALL_MIX` (fraction of pretraining windows replaced by synthetic associative-recall / induction windows from `recall.rs`, default 0; the other windows are the same as without it). `adam-t recall <out.txt> [episodes]` dumps generator episodes and prints their stats |
| Position | `LPE` (0 or 1; 1 replaces learned positions with Lorentz boosts), `LPE_PHI` (max rapidity across the window, default 4), `LPE_S` (quanta per position for the fastest pair, default 64); `LPE_LUT=ref` makes the LUT path use the direct exponent table instead of GF(17) discrete logs |

The throttle file is read every step. The run sleeps `throttle × max(1, gpu_ms/THROTTLE_UNIT_MS)` per step.

At the end, a ternary run exports `model.gf17`. It then checks that the roundtrip is bit-exact and that the LUT output matches the trainer's.

### Supervised tuning (`adam-t sft`)

`adam-t sft` tunes a finished base run on the chat/QA set in `data/adam_t/sft` (see its `meta.json`). `crates/adam-t/sft.sh` is the launcher:

```
THROTTLE_FILE=~/.local/share/amni-ascend/throttle.txt bash crates/adam-t/sft.sh
```

- **Start:** `BASE=<run dir>` names the base run. On the first launch, `sft.sh` snapshots the base checkpoint (`ckpt/core.bin`, `ckpt/mem.bin`, `state.txt`, and the first `log.jsonl` line) into `$RUN_DIR/base`, and the trainer takes the shape (`d`, `L`, `H`, `ctx`, `ff`, `v`, BPE, PTEX memory including `MEM_LOG2`, the LPE flags) from it. It also takes `MEM_OPT` from the base unless `MEM_OPT` is set. `sft.sh` sets `EVAL_MEM` to 8G for bases at 2^18 rows or more, and `DRY=1` prints the inherited shape and the resolved config. The first step loads the core and the PTEX memory tables from the base. The optimisers start fresh.
- **Safety checks:** `sft.sh` refuses to start without the throttle file, while an `adam-t` process runs, or while the base unit (`BASE_UNIT`, default `adam-t-30m`) is active. It also refuses to snapshot a base that has no `model.gf17` unless `ALLOW_PARTIAL_BASE=1` is set.
- **Packing:** whole examples are packed greedily into windows of CTX+1 tokens and never split. The remainder is padded, with mask 0. Examples longer than CTX+1 are skipped and counted.
- **Loss:** cross-entropy is taken only where the mask is 1 (Adam reply tokens, including the closing `\n\n`). It is normalised by the number of masked tokens in the whole step, so micro-batches add up exactly.
- **Schedule:** the defaults are `LR` 3e-4, `WARM` 100, then cosine to 0.1×, with `EPOCHS` 2. Set `EPOCHS=0 STEPS=N` for a fixed step count. `ckpt/steps.txt` records the resolved step count for `campaign.sh`.
- **Sampling:** each epoch is a fresh seeded shuffle. `SRC_W=name=w,...` weights sources by the idx `src` column: each example appears floor(w) times plus once more with probability frac(w), and `w=0` drops the source. Names come from `meta.json`, and indices also work.
- **Eval:** masked-loss eval runs on the whole hold split every `EVAL_EVERY` steps (default 250), or on `EVAL_BATCHES` × batch windows if that is above 0. It logs `val_nats` (per masked byte), `val_nats_per_tok` and `val_by_src`.
- **Same as pretraining:** the checkpoint format, `THROTTLE_FILE` every step, `STOP` and `MAX_MIN` pauses, and the final `model.gf17` export. `adam-t serve <sft run dir>`, `adam-t export` and the harness therefore load the tuned model unchanged.

### Other subcommands

- `adam-t bench`: training-step throughput.
- `adam-t mm`: GEMM microbenchmark comparing WGSL and burn.
- `adam-t lut <model.gf17> [windows]`: CPU LUT bits per byte on hold windows.
- `adam-t grow <model.gf17> <MEM_LOG2> <out.gf17>`: grows an exported model's PTEX tables exactly. Row `k·2^b + j` copies old row `k·2^a + (j >> (b − a))`, because `hrow` keeps the top bits of the hash. The output's logits are bit-identical to the input's.
- `adam-t lutbench <model.gf17> [prompt tokens] [decode steps]`: CPU LUT serve cost, meaning open time, prefill, per-token KV decode ms (p50, p90, first 16), RSS and page-cache counters, and a hash of every logit bit. `LUT_MB` (default 1024) sets the page-cache budget here and in `serve`.
- `adam-t fa`: checks and benchmarks the custom GPU ops. `FA_MODE=check` compares flash attention's output and gradients with burn's attention; `FA_MODE=flash|ref` times attention forward+backward (`FA_B`, `FA_T`, `FA_H`, `FA_IT`); `FA_MODE=lin` checks the fused ternary linear against the autodiff STE reference; `FA_MODE=xmm` times the exact bf16 cooperative-matrix GEMMs.

### Custom GPU ops (`src/fa.rs`)

- **Flash attention:** CubeCL kernels (tiled online-softmax forward that stores only the log-sum-exp, and a recompute backward split into a dQ kernel and a dK/dV kernel). They read the fused `qkv` matmul output directly and write `o` in `[B*T, D]` layout, so no head transposes. Registered as a custom op for `CubeBackend`, `Fusion` and `Autodiff`; other backends use the reference formula. Needs head dim 64 and `T % 64 == 0`, otherwise it falls back. `ATTN=ref` restores burn's attention.
- **Fused ternary linear:** one autodiff op per linear map that saves only `x` and `W` and recomputes RMSNorm, the activation grid and the trits in backward, with the same STE gradients. `LIN=ref` restores the op-by-op autodiff path.
- **Exact cooperative-matrix GEMMs:** the ternary forward `xi · t` has integer operands (|xi| <= 128, t in {-1,0,1}), so it runs in bf16 on the matrix cores with f32 accumulation and stays bit-exact. Backward operands that are real floats are split by one kernel into three bf16 terms whose sum is the f32 value, so the products are exact too. `XMM=0` falls back to f32 GEMMs.
- **Lorentz A/B:** `crates/adam-t/lorentz_ab.sh` runs the pre-registered position-encoding A/B (`docs/adam_t/lorentz_prereg.md`). It needs `THROTTLE_FILE`. `cargo test --release -p adam-t --lib lpe_lut_bench -- --ignored --nocapture` times the LUT path with and without LPE.
- **Pausing a run:** `touch $RUN_DIR/STOP` makes the trainer checkpoint and exit, and `campaign.sh` stops without an eval. `crates/adam-t/t30m.sh` is the 30M launcher.
- **Memory scale-up:** `crates/adam-t/t30m_mem.sh` continues the 35k 30M checkpoint with the PTEX memory grown to 2^18 rows per order (`MEM_OPT=lazy`), or keeps 2^15 with `CONTROL=1`. It snapshots the base into `<run>/init` and never writes to the base. The design, the VRAM/RAM/disk estimate, the CPU serve cost and the pre-registered comparison are in `docs/adam_t/memscale_prereg.md`.
- **Binding A/B:** `crates/adam-t/bind_ab.sh` runs the pre-registered 6.5M BPE-4096 A/B of `COPY` and `RECALL_MIX` (`docs/adam_t/bind_prereg.md`), scored with `scripts/adam_t_probe.py binding` and `induct`. `cargo test --release -p adam-t --lib copy_lut_bench -- --ignored --nocapture` times the LUT decode with and without the copy head.
- **Next runs:** `crates/adam-t/mem_ab.sh` is the pre-registered 6.5M from-scratch A/B (2^15 against 2^18, LPE and lazy, 2 seeds). `crates/adam-t/t30m_v2.sh` is the 30M v2 base from scratch (the v1 recipe plus `LPE=1`, `MEM_OPT=lazy`, `MEM_LOG2` default 18). Both refuse to start while another `adam-t-*` unit or `adam-t` process runs, and `DRY=1` prints the resolved config. See `docs/adam_t/memscale_prereg.md` sections 10 and 11.
- **Large tables:** the export streams the memory table page by page (`lut::save`), and its roundtrip check decodes page by page (`Store::page_words`), so export RAM does not grow with `MEM_LOG2`.
