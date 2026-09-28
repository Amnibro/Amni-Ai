# ADAM-T-MEMSCALE: growing the PTEX memory on the 30M checkpoint, pre-registered comparison

Status: built and CPU-tested on 2026-09-28. **Neither GPU continuation has run.** Both wait for the CTO to schedule GPU time (section 6). Sections 10 and 11 add a pre-registered 6.5M from-scratch A/B (`mem_ab.sh`) and the 30M v2 base launcher (`t30m_v2.sh`, LPE + lazy 2^18). Neither has run.

- **Code:** `crates/adam-t/src/lib.rs` (`Mem::grow`, `mrow`, `mleaf`, `Lazy`, the `INIT` and `MEM_OPT` flags in `Cfg`, the init path in `train`), `crates/adam-t/src/lut.rs` (`save`, `wr`, `check`, `grow`, streamed `finish`), `crates/adam-gf17/src/lib.rs` (`Writer::put_at`, `Store::page_words`), `crates/adam-t/src/main.rs` (`adam-t grow`, `adam-t lutbench`, `LUT_MB`).
- **Launcher:** `crates/adam-t/t30m_mem.sh` (`CONTROL=1` for the control arm).
- **Flags:** `INIT=<run dir>` starts a continued-pretraining run from another run's checkpoint and grows its memory to `MEM_LOG2`. `MEM_OPT=lazy` swaps the memory's dense AdamW for a row-lazy Adam. With both unset, the trainer, export and LUT path behave as before, and the old test suite passes unchanged.
- **Base:** `exports/adam_t/t30m_bpe4096`, step 35,000, MEM_LOG2 15, NGRAMS 2,3,4,6. It is only read: the launcher copies `ckpt/{core,mem}.bin`, `state.txt` and the first `log.jsonl` line into `<run>/init`.

## 1. Why

On 16M stream tokens, each 2^15-row table holds 34 (bigrams) to 443 (6-grams) distinct n-grams per row (`lorentz_prereg.md` section 7). The binding limit is collisions, not duplicates. Memory is lookup-only, so more rows cost RAM, VRAM and disk, but almost no FLOPs. The CEO wants the gaps at 30M closed before the core scales.

## 2. Exact growth: the proof

`hrow` (`lib.rs`) addresses order k at position p as

  row = k·2^R | (hsh(ngram, n) >> (64 − R))

`hsh` does not depend on R. So the index inside a table is the **top R bits** of one fixed 64-bit hash, and for b > a:

  top_b(h) >> (b − a) = (h >> (64 − b)) >> (b − a) = h >> (64 − a) = top_a(h)

Therefore the old row of new row `k·2^b + j` is `k·2^a + (j >> (b − a))`. Every old row is copied into 2^(b−a) consecutive new rows (`Mem::grow`: reshape [K·2^a, 1, d], `repeat_dim`, reshape [K·2^b, d]).

**The suggested seeding `old[new_idx & (2^a − 1)]` is not exact for this hash.** It would be exact only for a hash that keeps the low bits. With top-bit addressing, a low-bit mask picks an unrelated row. On 360,000 lookups (5 orders, both byte and u16 keying, every window position including truncated n-grams), `new >> (b − a)` recovered the old row 360,000 times out of 360,000, and the mask did so 5,677 times, which is the 1/64 chance rate for a = 6. The test also builds the mask-seeded table and shows that its logits differ (max |Δlogit| 0.43 and 0.49).

The embedding gather returns the same K rows as before, and they are summed in the same order, so **the grown model's logits are bit-identical** to the old model's. Training then lets sibling rows diverge.

## 3. What was built

**Continued pretraining (`INIT`).**
- `INIT=<dir>` reads the shape (d, L, H, ctx, FF, V, BPE, n-gram orders, TERN, MEM, LPE) from `<dir>/log.jsonl`. `MEM_LOG2` stays the requested one and must be at least the checkpoint's; shrinking is refused because it cannot be exact.
- At step 0 it loads `core` and `mem` from `<dir>/ckpt`, grows the tables, and starts fresh optimisers and a fresh schedule at step 0. Resuming after the first checkpoint is the usual resume, at the new size.
- `BASE` (SFT) and `INIT` are exclusive. The SFT path also goes through `grow`, which is the identity when the sizes match.

**Row-lazy Adam for the memory (`MEM_OPT=lazy`).**
- The dense path keeps a dense gradient, a gradient accumulator and two dense moments for the whole table, and updates every row each step. That is at least 5 table copies in VRAM (section 4).
- The lazy path takes the memory sum as an autodiff leaf (`mleaf`), reads its gradient after backward, scatter-adds it into the rows touched in this step (all micro-batches), and applies AdamW (β1 0.9, β2 0.95, ε 1e-5, no weight decay, global-step bias correction) to those rows only. The writes use `scatter_nd` Assign, which is fully parallel over rows × d.
- It keeps the table plus two moments: 3 copies. Untouched rows keep their moments frozen. That is TF-style LazyAdam, not bit-identical to dense AdamW beyond step 1.
- The moments are checkpointed as `ckpt/lm.bin` and `ckpt/lv.bin`.
- **Checked:**
  - On the first step, lazy equals burn's AdamW **bit for bit** on the touched rows (max |Δ| = 0), and leaves every untouched row unchanged.
  - On a second step, rows it does not touch stay unchanged, while dense AdamW keeps moving them by momentum.
  - A lazy run paused at step 1 and resumed matches the uninterrupted run bit for bit.

**Export and LUT serve for large tables.**
- `lut::save` streams the memory table to the `.gf17` writer one 4096×4096 page (8,388,608 f32) at a time, using `Writer::put_at`. The export roundtrip check (`lut::check`) compares each page decoded by `Store::page_words` with the source slice. Host RAM for export no longer grows with the table.
- `adam-t grow <model.gf17> <MEM_LOG2> <out.gf17>` grows an exported model directly. It is byte-identical to exporting the grown burn model, and its LUT logits and KV-cached decode are bit-identical to the original's.
- `LUT_MB` sets the serve page-cache budget (default 1024, as before).
- `adam-t lutbench <model.gf17> [prompt] [steps]` measures open time, prefill, per-token decode time, RSS and cache counters, and prints a hash of every logit bit.

## 4. MEM_LOG2 choice: 18 (RAM / VRAM / disk estimate)

K = 4 orders, d = 512, f32. One table copy is 4·2^R·512·4 B = 2^(R+13) B:

| | 2^15 | 2^18 | 2^20 |
|---|---|---|---|
| rows per order / total (4 orders) | 32,768 / 131,072 | 262,144 / 1,048,576 | 1,048,576 / 4,194,304 |
| table (1 copy) | 256 MiB | 2 GiB | 8 GiB |
| distinct 6-grams per row at 16M tokens | 443 | 55 | 14 |
| **VRAM**, dense AdamW (≈ 6 copies measured: table, 2 moments, dense grad, accumulator, update temporaries) | 1.5 GiB | ≈ 12 GiB | ≈ 48 GiB |
| **VRAM**, lazy Adam (3 copies measured, plus 1 if a scatter has to copy) | 0.75 GiB | 6–8 GiB | 24–32 GiB |
| trainer total, lazy (non-memory ≈ 4.8 GB + desktop 1.2 GB) | ≈ 6.8 GB | **≈ 12–14 GB of 16** | ≈ 30–38 GB |
| trainer total, dense | ≈ 7.5 GB (measured) | ≈ 18 GB: spills to GTT | ≈ 54 GB |
| checkpoint on disk (core 145 MB + oc 289 MB + table + 2 moments) | 1.2 GB | 6.8 GB | 26 GB |
| `model.gf17` (measured, section 5) | 250 MB | 1.89 GB | 7.50 GB |
| host RAM, trainer checkpoint write (≈ 2 copies transient) | 0.5 GiB | ≈ 4 GiB | ≈ 16 GiB |

- **The copy counts are measured on the CPU** (`adam-t bench`, `BACKEND=ndarray`, d 128, 4 orders, peak RSS at 2^12 against 2^17, where the 2^17 table is 256 MiB):
  - dense: 80 → 1,567 MB, which is **6.0 table copies**, and 75 → 902 ms/step, because the dense pass grows with the table;
  - lazy: 58 → 800 MB, which is **3.0 copies**, and 64 → 67 ms/step, flat.
  - The GPU fusion allocator can only add to these counts, because its memory pools hold on to freed buffers.
- **The non-memory figure:** the 30M run's measured peak is 7.5 GB, including 1.2 GB of desktop and about 1.5 GB of dense memory state (6 copies of 256 MiB). That leaves about 4.8 GB for everything else (design section 7c).
- **Why the dense trainer is out even at 2^18:** past 16 GB the driver spills into GTT. The phase-1 row in design section 7c shows the cost: 16.9 GB VRAM plus 16.7 GB GTT ran at 4.3 s/step, against 1.9 s. GTT is system RAM, which this machine does not have to spare.

**Decision: MEM_LOG2 = 18 with `MEM_OPT=lazy`.**
- 2^20 does not fit the 16 GB card with any optimiser that keeps per-element moments. It would need the table resident in host RAM or on SSD pages, with row-wise second moments. That is the planned scale-out (design section 2), not this run.
- 2^18 cuts the 6-gram load from 443 to about 55 n-grams per row and leaves 2–4 GB of VRAM headroom.
- The run directory is named after the size it uses, `exports/adam_t/t30m_mem18`, not `t30m_mem20`; `RUN_DIR` overrides it.

## 5. CPU serve cost (LUT, lookup-only, 3 threads)

Measured with the **real 35k model** (`exports/adam_t/t30m_bpe4096/model.gf17`, only read) and exact grown copies from `adam-t grow` (`scratchpad/memscale`, deleted afterwards).
- **Command:** `adam-t lutbench <file> <prompt> <steps>`, 3 rayon threads, a 300% CPU cap, nice 15, a prompt from `adam_t/stream` hold, then teacher-forced KV decode steps.
- **Load:** the machine was shared with other jobs, so the timings carry about ±20% noise between repeats.

| | 2^15 (base) | 2^18 | 2^20 |
|---|---|---|---|
| `.gf17` file | 250 MB | 1.89 GB | 7.50 GB |
| `adam-t grow` wall / peak RSS | – | 13.8 s / 532 MB | 54.6 s / 531 MB |
| open | 0.03 s | 0.03 s | 0.03 s |
| **logit bits** (hash over every prefill and decode logit) | `63458be5…` (256+256), `7a2137d9…` (512+512) | identical | identical |
| decode ms/token p50, 256 + 256, two runs | 12.6 / 16.0 | 13.2 / 17.8 | 13.5 / 14.6 |
| decode p50, 512 + 512 | 20.5 | 21.0 | 21.9 |
| decode p50, 960 + 64 | 20.3 | 20.6 | 25.7 |
| prefill, 256 / 960 tokens | 1.05–1.29 s / 4.57 s | 1.13–1.62 s / 4.87 s | 1.18–1.30 s / 5.08 s |
| peak RSS, `LUT_MB=1024`, 512 + 512 / 960 + 64 | 578 / 634 MB | 1,160 / 1,223 MB | 1,278 / 1,288 MB |
| peak RSS, `LUT_MB=256`, 512 + 512 | 371 MB | 408 MB | 474 MB |
| decode p50, `LUT_MB=256`, 512 + 512 | 18.9 | 22.5 | 19.9 |
| tile misses / evictions, `LUT_MB=1024`, 512 + 512 | 7,984 / 0 | 25,439 / 9,055 | 29,396 / 13,012 |

**What the table says:**
- **Growth is bit-exact at full scale.** Every logit of every run is the same at all three sizes.
- **Decode cost is about flat.** Each token reads K = 4 rows, which means at most 32 tile decodes whatever the table size. Pooled over six 256 + 256 runs, the p50 medians are 14.0 / 15.6 / 15.0 ms (×1.11 at 2^18, ×1.07 at 2^20). Only the long 960-token context at 2^20 showed +27%.
- **RSS follows the page-cache budget, not the file size:**
  - At 2^15, the whole decoded memory (8 pages × 1,024 tiles × 64 KiB = 512 MiB) fits the default 1 GiB cache and never evicts.
  - At 2^18 and 2^20, the decoded memory would be 4 GiB and 16 GiB. The cache fills to `LUT_MB` and evicts.
  - Serve RSS is therefore about 250 MB + `LUT_MB` + at most 80 MB of parsed plane tables (at 2^20). With `LUT_MB=256`, 2^20 serves in 474 MB.
- **Two serve fixes made RSS bounded.** Both are in this change, and neither changes a result bit:
  1. **Tile-span release.** `Store` released decoded tile spans with `MADV_DONTNEED` on 4 KiB pages. The kernel's 64 KiB fault-around kept the neighbouring file pages mapped, so RssFile grew with every miss: 845 MB after 512 tokens at 2^18. The release is now 64 KiB-aligned, and RssFile stays at 7–26 MB.
  2. **Prefill chunks.** `Lm::rows` asked the store for all prefill positions at once, so every touched tile was decoded before the budget applied. That meant 1.4 GB (2^18) and 1.6 GB (2^20) peak at `LUT_MB=256` for a 512-token prompt. It now asks 32 positions at a time: 408 MB and 474 MB.

## 6. Pre-registered comparison

**Arms.** Both start from a copy of the 35k checkpoint and differ only in `MEM_LOG2`:

| | `mem18` (treatment) | `mem15` (control) |
|---|---|---|
| launcher | `t30m_mem.sh` | `CONTROL=1 t30m_mem.sh` |
| run dir | `exports/adam_t/t30m_mem18` | `exports/adam_t/t30m_mem15` |
| PTEX memory | grown 2^15 → 2^18 rows per order (exact) | 2^15, as in the base |

**Shared config** (fixed now):
- **Shape:** the base shape, read from the base run: d 512, L 10, H 8, FF 2048, ctx 1024, BPE-4096, TERN 1, NGRAMS 2,3,4,6.
- **Data:** `CORPUS=adam_t/stream`, the same stream the base trained on. `SEED=1`, so the batch sequence is new: with seed 0 and the step counter restarted, the continuation would replay the base's first 9,000 batches. Both arms see the same batches in the same order.
- **Schedule:**
  - BS 16 × 1024 (MICRO 2), 9,000 steps, which is 147M tokens, about 26% of the base's 573M.
  - Fresh optimisers, 200 warm-up steps to LR 4e-4, then cosine to 4e-5. 4e-4 is the base schedule's LR at step ≈ 27,600, and the base ended at 2e-4.
  - `LR_MEM` = `LR`, WD 0.05 on the core, no weight decay on the memory.
  - `MEM_OPT=lazy` in **both** arms, so the optimiser is not a difference between them.
- **Checkpoints and eval:** a checkpoint every 250 steps, trainer val every 1,000 steps (10 × 128 windows), SEG 30 min, EVAL_SEGS 16 as in the 30M unit, so the harness runs at the final step. PARITY_WINDOWS 2.

**Metrics.** All use the step-9,000 checkpoint and the same harness suites as the base's step-35,000 report (`data/adam_t/eval/report_adam-t-t30m-bpe4096_35000.json`): qa, rc, mem, bpb, zs.
- **Primary:** harness `bpb_stream`, which is in-distribution for the memory.
- **Secondary:** harness `bpb_clean2`.
- **MC and zero-shot:** qa, rc and mem MC `acc_norm` (NQ, RC, D3), and ARC-E and PIQA `acc_norm`.
- **Recorded, not used in the decision:** the trainer's val bpb curve; the export parity; LUT decode ms/token and RSS at `LUT_MB=1024`, from `adam-t lutbench` on each `model.gf17`.

**Base numbers at step 35,000:**

| bpb_stream | bpb_clean2 | NQ MC | RC MC | D3 MC | ARC-E | PIQA |
|---|---|---|---|---|---|---|
| 1.4769 | 1.6192 | 0.2875 | 0.41 | 0.4294 | 0.386 | 0.566 |

**Effects.**
- mem_effect = ctrl − mem18. This is what the bigger memory buys at equal steps and data.
- step_effect = base − ctrl. This is what 9,000 more steps buy at 2^15.

**Decision rules**, fixed now:
1. **Memory helps:** mem_effect on `bpb_stream` ≥ **0.010**, and mem18 is not worse than ctrl on `bpb_clean2` by more than 0.005.
2. **Memory hurts:** mem_effect on `bpb_stream` ≤ −0.010.
3. **No detectable memory effect:** anything in between. A difference inside ±0.010 bpb is a tie. The comparison has one seed per arm and no measured seed spread at this size, so 0.010 is the fixed margin. It is twice the 0.005 bpb promotion gate in design section 4.
4. **MC and zero-shot suites:** a difference counts only beyond 2 × pooled SE, pooled SE = sqrt(SE_a² + SE_b²), as in `eval_prereg.md`. They are reported but cannot overturn rules 1–3.
5. **Steps alone:** step_effect ≥ 0.010 on `bpb_stream` means the extra steps helped by themselves, whatever the memory result.
6. **Serve guard for adopting 2^18:** rule 1 holds, and `lutbench` decode p50 at 2^18 is at most 1.5× the 2^15 figure at the default `LUT_MB=1024` (section 5). If only a larger `LUT_MB` passes the guard, that value becomes the new serve default, with its RSS stated.

Nothing else is tuned. A rerun with another size, LR or optimiser needs a new pre-registration first.

**Abort rules** (these pause a run; they are not results):
- After the first segment, `gpu_ms_per_step` in `log.jsonl` above 1.5 × the base's 1,250 ms, or peak VRAM above 15 GB (`mem_info_vram_used`): `touch <run>/STOP` and report.
- A non-finite `train_nats` or `val_nats`.

## 7. GPU cost (for scheduling)

- **Base:** 1,250 GPU-ms per step, 3.0 s/step wall under the 40% governor (design section 7c).
- **Lazy optimiser:**
  - It drops the dense AdamW pass over 67M memory parameters, the dense 256 MiB gradient per micro-step and the accumulator add.
  - It adds three `select`s and three fully parallel `scatter_nd`s over the touched rows (at most 131,072 × 512), plus one sequential-per-column `select_add` over 131,072 gathered gradients. That is the same kernel shape burn's embedding backward already runs in the base.
  - Estimate: 0–15% more GPU time per step, the same in both arms because the work scales with lookups, not table size.
  - The abort rule above bounds it.

| | steps | GPU busy (1.25–1.45 s/step) | wall under the governor (3.0–3.4 s/step) | harness eval (GPU, final step) | checkpoint writes |
|---|---|---|---|---|---|
| mem18 | 9,000 | 3.1–3.6 h | 7.5–8.5 h | ≈ 46 min | 36 × 6.8 GB |
| mem15 control | 9,000 | 3.1–3.6 h | 7.5–8.5 h | ≈ 46 min | 36 × 1.2 GB |
| **total** | 18,000 | **6.3–7.3 h** | **15–17 h** | **≈ 1.5 h** | |

- **One GPU job at a time.** The launcher refuses to start while `adam-t-30m`, `adam-t-sft` or `adam-t-lorentz` is active, or while any `adam-t` process runs. Run the arms one after the other, in either order.
- **Disk:**
  - mem18 needs about 9.1 GB: the init snapshot 0.4 GB, the checkpoint 6.8 GB, and `model.gf17` 1.9 GB.
  - The control needs about 1.9 GB.
- **Host RAM:**
  - Trainer: MEM_MAX 12G.
  - Harness: `EVAL_MEM` 8G for mem18, because `serve <run dir>` loads the 2 GiB table through the host; 4G for the control.

## 8. Launch commands (from the repo root, after `cd crates && cargo build --release -p adam-t`)

```
THROTTLE_FILE=~/.local/share/amni-ascend/throttle.txt bash crates/adam-t/t30m_mem.sh
THROTTLE_FILE=~/.local/share/amni-ascend/throttle.txt CONTROL=1 bash crates/adam-t/t30m_mem.sh
```

The same commands resume a paused run. `touch <run>/STOP` pauses at a checkpoint.

## 9. Tests (CPU, NdArray; `cargo test --release -p adam-t -p adam-gf17`)

- `mem_scale_tests::hash_is_prefix_consistent_and_grow_is_bit_exact`
  - Checks the top-bit identity on 360,000 lookups.
  - Checks that grown rows copy the right old rows.
  - Checks that grown logits are bit-identical for byte and u16 (BPE) keying.
  - Checks that low-bit seeding is not.
- `mem_scale_tests::grown_gf17_equals_grown_export_and_lut_is_bit_exact`
  - Uses 2^6 → 2^16 rows per order, so the memory table spans 2 pages.
  - Checks the streamed multi-page export roundtrip.
  - Checks that `adam-t grow` output is byte-identical to the grown export.
  - Checks that LUT logits and KV decode are bit-identical to the original's, and that shrinking is refused.
- `mem_scale_tests::lazy_step_matches_adamw_on_touched_rows_and_leaves_the_rest`: section 3.
- `mem_scale_tests::init_grows_memory_continues_training_and_resumes`
  - A tiny base is trained, then continued with `INIT` at 2^6 → 2^9 and LR 0, lazy and dense. The checkpointed model's logits are bit-identical to the base's, and the exported `model.gf17` gives bit-identical LUT logits, with the export roundtrip exact.
  - A lazy continuation trains: val 1.63 → 0.78 nats, with the children of 170 of 192 old rows diverged. Paused and resumed, it equals the uninterrupted run.
  - A dense continuation trains.
  - Shrinking, and `BASE` with `INIT`, are refused.
- `adam-gf17` `paged_put_matches_write`: page-streamed writes in any page order give the same file as the one-shot writer, and `page_words` decodes each page exactly.
- **Full scale** (section 5): `adam-t grow` of the real 35k model to 2^18 and 2^20 gives identical logit bits through 960 prefill tokens and 512 KV decode steps.
- **All suites pass:** 17 lib tests (13 earlier, 4 new) with the LPE bench still ignored, the 3 main-binary tests, and 6 `adam-gf17` tests (5 earlier, 1 new).

## 10. 6.5M from-scratch A/B: MEM_LOG2 15 against 18 with LPE (pre-registered, not yet run)

Section 6 grows a finished 30M checkpoint. This A/B asks the from-scratch question at the phase-1 size, with two seeds, before the 30M v2 base (section 11) commits a day and a half of GPU time to 2^18.

**Question.** Trained from scratch with LPE, does a 2^18-row PTEX memory beat 2^15 at 6.5M, without making LUT decode too slow?

**Arms.** Identical except for `MEM_LOG2`:

| | `mem15` (control) | `mem18` (treatment) |
|---|---|---|
| PTEX memory | 6 × 2^15 rows × 256 (192 MiB) | 6 × 2^18 rows × 256 (1.5 GiB) |
| run dirs | `exports/adam_t/mem_ab/mem15_s{0,1}` | `exports/adam_t/mem_ab/mem18_s{0,1}` |

**Shared config** (fixed now): the `lpe` arm of the Lorentz A/B (`lorentz_prereg.md` section 5), plus the lazy memory optimiser.
- **Shape:** d 256, L 8, H 4, ctx 256, FF 1024, bytes (v 256), TERN 1, MEM 1, NGRAMS 2,3,4,5,6,8.
- **Position:** `LPE=1`, φmax 4, S 64. The Lorentz A/B adopted it (`exports/adam_t/lorentz_ab/ab_result.json`: −0.028 bpb on clean2, seed spread 0.008, LUT speed ×1.003).
- **Optimiser:** `MEM_OPT=lazy` in **both** arms, as in section 6, so the optimiser is not a difference. The Lorentz arms used dense AdamW, so their numbers are a reference only, not a third arm.
- **Schedule:** BS 64, MICRO 1, 3,000 steps, LR = LR_MEM = 2e-3, 200 warm-up steps, WD 0.05 on the core.
- **Data and seeds:** `CORPUS=clean2`. SEED 0 and 1 per arm; SEED sets the init and the sampler, so both arms see the same bytes at the same seed. Run order: mem15 s0, mem18 s0, mem15 s1, mem18 s1.
- **Checkpoints and eval:** a checkpoint every 250 steps, trainer val every 500 (20 × 128 windows), SEG 20 min, PARITY_WINDOWS 8. The harness does not run between segments (`EVAL=0`).

**Metrics**, all on the step-3,000 checkpoint:
- **Primary:** harness `bpb_clean2` (`scripts/adam_t_eval.py run --suites bpb` through `adam-t serve <run dir>`, 8,192 seeded 256-byte windows). Reports go to `data/adam_t/eval/report_mem-ab-m<15|18>-s<seed>_3000_dry.json`.
- **Secondary:** harness `bpb_stream`, same protocol on `adam_t/stream/hold.bin`.
- **Speed:** `adam-t lutbench <model.gf17> 128 128` on CPU (3 threads, GF(17) LPE path, default `LUT_MB=1024`): KV decode ms/token p50. Five interleaved repetitions over the 4 models; the median per model.
- **Recorded, not used in the decision:** the trainer's val bpb, `gpu_ms_per_step` (the lazy step's cost at 2^15 against 2^18), peak total-card VRAM (sampled every 5 s from `mem_info_vram_used` into `<run>/vram.log`), and the export parity.

**Decision rule** (fixed now). mem18 is adopted only if **all** of these hold:
1. **Clean2 win:** mean_mem15 − mean_mem18 on `bpb_clean2` exceeds the seed spread, max(|mem15_s0 − mem15_s1|, |mem18_s0 − mem18_s1|).
2. **Stream guard:** mem18 does not lose on `bpb_stream` by more than that metric's own seed spread.
3. **Serve guard:** (mem18_s0 + mem18_s1)/(mem15_s0 + mem15_s1) of median decode p50 is at most **1.2**.

A tie within the spread is a loss for mem18. **Consequence for section 11:** if `adopt_mem18` is false, the 30M v2 base launches with `MEM_LOG2=15`. No rerun with other sizes, LRs or cache budgets without a new pre-registration.

**What the script does** (`crates/adam-t/mem_ab.sh`, same pattern as `lorentz_ab.sh`):
- **Refuses to start** without `THROTTLE_FILE`, while any `adam-t-*` user unit other than its own is active, or while any `adam-t` process runs. The self-exception lets it run as unit `adam-t-memab`.
- **Freezes the binary:** copies it with `campaign.sh`, `smoke.sh` and `mem_ab.sh` into `exports/adam_t/mem_ab/bin` and records the sha256 and the git commit.
- **Runs, scores and times** the four runs, then writes `exports/adam_t/mem_ab/ab_result.json` with every number above and `adopt_mem18`.
- **Can be rerun.** Finished runs and reports are skipped; `touch <run>/STOP` pauses it. `DRY=1` prints the resolved config and exits without touching anything.

**GPU cost** (governor at 40%). The Lorentz arms took 57, 61, 55 and 60 min each (390 GPU-ms/step at ~40% card busy), including the export.
- **Lazy optimiser:** 0–15% per step, and it scales with lookups (6 × 16,384 per step), not with the table.
- **mem18 extras:** 4.6 GB checkpoints (table + 2 moments) about 15 times per run, and a 1.4 GB export: about 2–5 min per run.
- **Estimate:** 57–70 min per mem15 arm and 60–75 min per mem18 arm, so **4.0–4.8 h** of training. Add 4 harness bpb evals (≈ 2.5–4 min each) and the CPU speed check (a few minutes, no GPU): **≈ 4.2–5.1 h wall** in total.
- **VRAM, mem18:** 3–4 copies of 1.5 GiB (4.5–6 GiB) plus the small phase-1 core and desktop, so about 7–9 GB of 16.
- **Disk:** about 6 GB per mem18 run (4.6 GB checkpoint + 1.4 GB `model.gf17`) and 0.8 GB per mem15 run, about 14 GB in total.
- **Host RAM:** trainer `MEM_MAX` 12G; harness scope 8G for mem18, 6G for mem15; `lutbench` scope 4G.

**Command** (repo root, after `cd crates && cargo build --release -p adam-t`):

```
THROTTLE_FILE=~/.local/share/amni-ascend/throttle.txt bash crates/adam-t/mem_ab.sh
```

**CPU smoke (2026-09-28).** The resolved config (from `DRY=1`) ran on a tiny shape (d 32, L 2, ctx 64, 30 steps) on the NdArray backend, with both arms and both seeds, lazy at the real 2^15 and 2^18. All four exports were roundtrip-exact, and `lutbench` ran on each. The result block, run on those logs, speed records and fabricated harness reports, gave the expected verdicts. At this tiny shape the 2^18 decode was ×1.25 slower (0.22 → 0.28 ms/token). A 2-layer d 32 core leaves the memory lookup as most of the cost, so this does not predict the 6.5M ratio. It does show that the serve guard can bind.

## 11. 30M v2 base from scratch (`crates/adam-t/t30m_v2.sh`, not yet run)

**Recipe:** the 30M v1 recipe (`t30m.sh`, design section 7c), with three changes: `LPE=1` (φmax 4, S 64), `MEM_LOG2` from the environment (default 18, allowed 15–18), and `MEM_OPT=lazy`.
- **Unchanged:** d 512, L 10, H 8, FF 2048, ctx 1024, BPE-4096, TERN 1, NGRAMS 2,3,4,6, BS 16 × 1024 (MICRO 2), 35,000 steps (573M tokens), LR = LR_MEM = 2e-3, 1,000 warm-up steps, cosine, WD 0.05, SEED 0, `CORPUS=adam_t/stream`.
- **Eval and checkpoints:** val every 1,000 steps (10 batches), a checkpoint every 250, SEG 30, `EVAL_SEGS=16`, PARITY_WINDOWS 2. `EVAL_MEM` is 8G at 2^18, because `serve` loads the 2 GiB table through the host, and 4G below that.
- **Run dir:** `exports/adam_t/t30m_v2`.
- **Guards:** the same as `mem_ab.sh`. It also refuses to resume a run dir whose `log.jsonl` was started with another `MEM_LOG2`, or without LPE or lazy. It freezes the binary and scripts into `<run>/bin`, with the sha256 and the commit. `DRY=1` prints the resolved config.

**VRAM estimate (lazy 2^18, LPE, d 512, ctx 1024, BS 16 / MICRO 2).**
- **Non-memory:** about 4.8 GB (section 4, from the measured 7.5 GB v1 peak).
- **Memory:** the table plus 2 moments, 3 × 2 GiB, plus 1 copy if a scatter copies: 6.4–8.6 GB.
- **LPE:** removes the position add and multiplies one fixed factor into the fused `qkv` output. The factor table is 1024 × 512 f32 (2 MB). At most one more saved `qkv`-sized activation per layer: 8 × 1024 × 1536 × 4 B = 50 MB × 10 layers = **≤ 0.5 GB**.
- **Desktop:** 1.2 GB.
- **Total ≈ 12.4–15.1 GB, most likely ≈ 13–14 GB, of the 16 GB card (15.9 GiB).**
- **Abort rule** (as in section 6): peak VRAM above 15 GB after the first segment, or `gpu_ms_per_step` above 1.5 × 1,261 ms → `touch <run>/STOP`. The fallback is `MICRO=4`, which is exact gradient accumulation with the same batch and halves the activations, at about 5–10% speed. It is not tuning, so it needs no new pre-registration.
- `MEM_LOG2=15` would need about 7.5 GB, as v1 did.

**Wall-time estimate (governor at 40%).**
- **Training:** v1 took about 27 h for 35,000 steps (1,261 GPU-ms/step, 2.98 s/step wall). Lazy costs 0–15% and LPE nothing measurable, so **27–31 h**.
- **Checkpoints at 2^18:** 6.8 GB each, about 140 scheduled plus about 60 at segment ends. At 3–6 s each, 10–20 min. A segment restart also reloads the 6.8 GB checkpoint, which adds another 5–10 min.
- **Harness evals:** about 4 (every 16 segments, plus the final one) at 21–46 min each: **1.4–3 h**.
- **Total ≈ 29–35 h wall (1.2–1.5 days)**, of which about 12–14 GPU-busy hours.
- **Disk:** 6.8 GB checkpoint + 1.9 GB `model.gf17`. The checkpoints write about 1.4 TB to the SSD over the run (v1: about 0.25 TB).

**SFT from a v2 base.** `adam-t sft` takes the shape from `BASE`:
- **Inherited:** d, L, H, ctx, FF, V, BPE, n-gram orders and the LPE flags. `MEM_LOG2` too: it is the base's, whatever the environment says.
- **MEM_OPT was not inherited before this change.** SFT fell back to dense AdamW, which at 2^18 needs about 6 table copies, ≈ 18 GB in total, and spills into GTT (section 4).
- **Now:** `train` takes `MEM_OPT` from the base when `MEM_OPT` is unset. An explicit `MEM_OPT` still wins. `sft_mode_end_to_end_cpu` checks this: a lazy LPE 2^8 base, tuned with a config that says dense / no LPE / 2^6, gives a lazy LPE 2^8 SFT run with `lm.bin`/`lv.bin` moments, and at LR 0 it keeps the base weights. Without the fix the test fails.
- **`sft.sh`:** sets `EVAL_MEM` to 8G when the base's `mem_log2` is 18 or more. It had been 4G, which is too small to serve a 2 GiB table. It also gained `DRY=1`, which prints the inherited shape and the resolved config.

**Commands** (repo root, after the build; `MEM_LOG2=15` if section 10 rejects 2^18):

```
THROTTLE_FILE=~/.local/share/amni-ascend/throttle.txt bash crates/adam-t/t30m_v2.sh
THROTTLE_FILE=~/.local/share/amni-ascend/throttle.txt BASE_RUN=/home/anmire/ai/Amni-Ai/exports/adam_t/t30m_v2 RUN_DIR=/home/anmire/ai/Amni-Ai/exports/adam_t/t30m_v2_sft BASE_UNIT=adam-t-30m-v2 bash crates/adam-t/sft.sh
```

**CPU smoke (2026-09-28).** The launcher's resolved config (from `DRY=1`) ran on a tiny shape (d 32, L 2, ctx 128, 40 steps) on the NdArray backend, keeping `MEM_LOG2=18`, lazy, LPE, BPE-4096 and `adam_t/stream`.
- **Base:** paused by `MAX_MIN` at step 21 and resumed to 40, with the lazy moments restored from `lm.bin`/`lv.bin`. The export was roundtrip-exact, and LUT and trainer agreed (max |Δlogit| 2.4e-7, argmax 100%).
- **SFT:** from that base, run with the config `sft.sh` resolves, on `adam_t/sft`. It logged `mem_log2` 18, `lpe` true and `mem_opt` lazy, and exported exactly.
