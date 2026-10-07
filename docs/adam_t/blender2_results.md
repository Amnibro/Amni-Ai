# BLENDER-2 results: current-generation blender, VRAM and quality measured together

Mission BLENDER-2, 2026-10-06 to 2026-10-07. Branch `worktree-agent-a8440e194319837f7` (based on `feat/ray-field-serve` b62845b), commits prefixed `BLENDER-2:`. Plan fixed before scoring: `docs/adam_t/blender2_eval.md` (with timestamped amendments). Models, licences, revisions: `docs/adam_t/blender2_models.md`. Every run file cited below is in `data/adam_t/eval/blender2_runs/` (copied from `~/ai/hybrid1/runs2/`).

## 0. Bottom line

1. **Recommended: Granite 4.2 3B (Apache-2.0), GPTQ-calibrated int4 (group 128), layers streamed from pinned RAM, embedding and lm_head on the CPU, int8 KV, 256-token prefill chunks.** Measured in one process (`dev_granite42_q4g_M.json` / `probe_granite42_q4g_M.json`): **0.82 GB whole-process VRAM** (`blender2.meas.peak_proc_vram_GB` 0.819 = `blender2.proc_vram_peak_all_GB` 0.819), 8.5 tokens/s decode, 881 tokens/s prefill, perplexity 7.745 (int8: 7.483); probe8 strict faithful **63.5%**, no-new-content 73.0%, hedge when uncovered 97.8%, blend coherent 91.8%, chat answered 80.9%, chat decline 2.9%, emergency-text false positives 0%. Host RAM: 1.67 GB pinned layers plus 1.03 GB of bf16 tables. Selected on the dev set by the amended rule (section 9); probe8 was not used to choose.
2. **The intermittent garbage is explained and fixed.** Root cause: PyTorch's expandable-segments allocator (set by the HYBRID-1 wrapper) corrupts memory reused after `empty_cache()` on torch 2.14.1+rocm7.2/gfx1200; it hit whichever configuration loaded second or later in a process. Reproduced in 5 of 5 pre-fix runs and in a model-free probe; the engine now forces it off; 5 consecutive post-fix runs are clean and fully deterministic (section 2). A second, latent bug (stream ring wrap race for odd layer counts) was found and fixed.
3. **Streamed equals resident, bit for bit, for every reported model** (Granite 4.2, MiniCPM5, Qwen3.5-2B, Qwen3.5-4B, and HYBRID-1's Qwen3-4B), at both bit-widths: token-identical on every prompt, logit max-abs-diff 0.0, 67 to 144 prompts each including multi-chunk prompts (section 5).
4. **Sub-1 GB with quality, measured:** Granite 4.2 int4 M 0.82 GB (above), Granite 4.2 int8 M 0.90 GB (probe8 faithful 61.3%), Qwen3.5-2B int4 M 0.82 GB (19.0%), Qwen3.5-2B int8 M 0.89 GB (8.0%), MiniCPM5 int8 M 0.70 GB (43.1%, but declines 26% of greetings), MiniCPM5 int4 M 0.65 GB (51.8%, declines 22%). None needed int4 KV.
5. **Calibrated int4 works.** GPTQ int4 costs 2 to 4.5% perplexity against int8 on every model (calibration-free int4 cost 19% on Granite 4.2), and Granite 4.2 int4 matches its int8 on probe8 strict faithful (63.5 to 65.0% vs 61.3 to 62.0%), at about half the host RAM and faster decode (13.9 vs 10.5 tokens/s in S mode).
6. **The Qwen3.5 Gated DeltaNet port works inside the time box** (text-only bake, hybrid cache, chunked prefill state carry-over, heterogeneous stream ring). It shrinks KV+state at 2,560 tokens (Qwen3.5-2B 0.036 GB vs Granite 0.108 GB, Qwen3-4B 0.192 GB) but not whole-process VRAM yet, and Qwen3.5's small models are weaker blenders here: probe8 strict faithful 7 to 19% (2B) and 52 to 58% (4B), dev no-new-content 36 to 65% vs Granite's 93.5%. The CEO's expectation that Granite 4.2 beats Qwen3.5 holds on the grounding metrics (strict faithful and no-new-content, dev and probe8); Qwen3.5 is level on hedging and declines and slightly ahead on probe8 chat answered (4B int8 S 85.3% vs 79.4%).
7. **Against HYBRID-1's Qwen3-4B (now excluded as too old):** its streamed configuration is bit-identical to resident and was scored on the streamed numerics (86.9% probe8 strict faithful), but its VRAM on the safe allocator is 1.63 GB, not 1.54 GB. Granite 4.2 int4 M is half that VRAM at 23 points less strict-gate pass on probe8.

## 1. Candidates

Verified on the model cards and the HF API (`blender2_models.md`): Granite 4.2 3B (Apache-2.0, dense, CEO priority 1), MiniCPM5-2B (Apache-2.0, dense Llama, priority 2), Qwen3.5-2B and Qwen3.5-4B (Apache-2.0, Gated DeltaNet hybrid with a vision tower; text weights only, priority 3). No dense Qwen3.8 of 4B or less exists on the Qwen org (only 27B dense and large MoE). No other dense, instruction-tuned, general-chat Apache/MIT model of 4B or less released after 2026-06 was found (the newer small releases are Qwen3.5 fine-tunes for agents or vision, a base-only Granite preview, MoE, or non-permissive licences; list and reasons in the models doc).

## 2. Determinism: the intermittent garbage, its root cause and the fix

### Reproduction (before the fix)

`scripts/hybrid1_equiv.py` exactly as merged by HYBRID-1 (frozen copy of commit b62845b), the HYBRID-1 config sequence (base, embed_cpu, head_cpu, stream, chunk256, stream_chunk256_cpu_tab, kv8, kv4, tab_q8, tab_q4), the HYBRID-1 environment (`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, as set by the HYBRID-1 `py` wrapper), the Qwen3-4B-Instruct-2507 int8 bake, 5 fresh processes. Files: `data/adam_t/eval/blender2_runs/determinism/repro_q3orig_es_{1..5}.json`.

| run | configurations that produced garbage (token match with base < 0.3; kv4 excluded because its numerics are broken by design) |
|---|---|
| 1 | stream_chunk256_cpu_tab |
| 2 | head_cpu, tab_q4 |
| 3 | chunk256, stream_chunk256_cpu_tab, kv8, tab_q8, tab_q4 (the HYBRID-1 pattern) |
| 4 | chunk256, tab_q8 |
| 5 | head_cpu, stream_chunk256_cpu_tab, tab_q8, tab_q4 |

5 of 5 runs failed. The first one or two loads in a process (base, embed_cpu) were always clean; garbage appeared only after the process had loaded, freed and `empty_cache()`d earlier models, and it hit whichever configuration happened to load then, not one code path. Outputs were either `!!!!` (token 0: NaN logits) or near-garbled text.

### Isolation

1. Same five-configuration load sequence, three repetitions in one process (`stress1.json`): with `expandable_segments:True`, 6 of 15 loads produced garbage (none in the first repetition; including plain `base` in the second); without it (`stress1_noes.json`), 0 of 15.
2. A model-free probe (`tests/alloc_reload_probe.py`): allocate 40 to 89 int8 tensors of 64 KB to 32 MB, fill each with a synchronous host-to-device copy, allocate and free a few fp32 temporaries, read everything back, free all, `torch.cuda.empty_cache()`, repeat. With `expandable_segments:True`: corrupted tensors per repetition `[0, 0, 0, 37, 56, 66, 60, 72]` (and `[0,0,32,36,43,55,0,64]`, `[0,0,0,36,57,53,0,64]` on reruns); without it: all zeros, also with a side-stream pinned copy.

**Root cause:** PyTorch's expandable-segments caching allocator on this stack (torch 2.14.1+rocm7.2, HIP 7.2.53211, gfx1200) returns memory whose contents do not match what was just copied into it once segments have been unmapped by `empty_cache()` and remapped. Weights loaded into such memory are silently corrupted. It is not the stream ring, not chunked prefill and not KV slot reuse; those were only the configurations that happened to load after the first `empty_cache()` in the HYBRID-1 sequence. The first load in a process was clean in every process observed here (5 sequence runs and 2 stress runs), which is consistent with the HYBRID-1 probe8 and measurement rows (one load per process) never showing the symptom; those rows' VRAM figures were, however, taken with this allocator setting.

**Fix:** `blender_atex.noexp()` sets `expandable_segments:False` at runtime (`torch._C._accelerator_setAllocatorSettings`) at every engine construction, so the environment variable can no longer turn it on; the HYBRID-1 wrapper's variable is removed from the BLENDER-2 runner. Regression test: `tests/test_blender_atex.py::test_reloads_with_expandable_segments_env_stay_exact` runs the probe with the variable set and requires 0 corrupt tensors; it fails with the fix disabled (`NOFIX=1`: `[0, 0, 0, 36, 44, 54, 60, 64]`) and passes with it.

### After the fix: 5 consecutive clean runs

Same config sequence, same bake, variable still set in the environment, new engine, 11 prompts (the 3 HYBRID-1 prompts plus 8 multi-turn prompts of 335 to 648 tokens so that 256-token chunked prefill really chunks; HYBRID-1's prompts were all shorter than one chunk, so its `chunk256` equivalence never exercised chunking), 48 greedy tokens. Files: `determinism/post_q3fix_es_{1..5}.json`.

- Every configuration produced exactly one token sequence per prompt across the 5 runs (`toks_sha1` identical in all 5 files): fully deterministic.
- No degenerate output except kv4 (broken by its numerics in every run, as HYBRID-1 reported).
- Against base: embed_cpu, stream 11/11 identical with logit max-abs-diff 0.0; head_cpu 11/11 identical (max-abs-diff 0.125, CPU vs GPU bf16 matmul); chunk256 and stream_chunk256_cpu_tab 10/11 (one 558-token prompt diverges at step 20; chunked prefill changes numerics by up to 1.06 logits on long prompts, so it is NOT numerics-neutral as HYBRID-1 stated); kv8 7/11; tab_q8 8/11; tab_q4 8/11; kv4 0/11.

### Second defect found on the way: odd layer counts in the stream ring

With an odd number of layers the ring prefetched layer 0 into slot 0 while the last layer (also slot 0) was still running, so the last layer could compute with layer 0's weights. None of the HYBRID-1 models has an odd layer count (36, 40, 42), so no reported number was affected. Fixed (no wrap prefetch for odd counts); `test_stream_ring_bit_exact_vs_resident[3,5]` fail on the old ring and pass on the new one.

## 3. Environment and tests

- Rebuilt `~/ai/hybrid1/venv` from `docs/adam_t/hybrid1_env.txt`: Python 3.12.14, torch 2.14.1+rocm7.2 (HIP 7.2.53211), triton-rocm 3.8.0, transformers 5.18.0, accelerate 1.15.0, safetensors 0.8.0, numpy 2.5.3, tokenizers 0.23.2, pytest 9.1.1; device 0 = gfx1200 (16 GB). Full freeze: `docs/adam_t/blender2_env.txt`.
- Runtime: `HIP_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; no `PYTORCH_HIP_ALLOC_CONF` (the engine forces expandable segments off anyway, section 2).
- `tests/test_blender_atex.py` and `tests/test_qgroup_gemv.py` in that venv: **all pass** (30 passed in the final run on 2026-10-07; the 20 HYBRID-1 tests also passed on the untouched code before any edit). New tests: stream ring bit-exact vs resident for 3, 4 and 5 layers; Gated DeltaNet hybrid (5 and 8 layers): stream bit-exact, chunked prefill carries the recurrent and conv state (max logit diff < 0.05 vs one-shot prefill), int8 KV on the full-attention layers only; KV capacity growth bit-exact; GPTQ beats round-to-nearest on output error; the allocator probe with `expandable_segments:True` in the environment shows 0 corrupted tensors.

## 4. Method (what one table row is)

One configuration = one process (`scripts/hybrid1_run.py --meas --ppl --set dev,probe`): load the bake with the frozen options; measure (64-token warm-up, `empty_cache`, then a 2,048-token prefill of fixed random ids and a 512-token greedy decode, governor sleep off); perplexity on 1,535 tokens of the dev text through the engine's own prefill (same chunking, KV and tables); score the dev set; score probe8 (governor obeyed). A sampler thread reads this process's `drm-memory-vram` from `/proc/self/fdinfo` every 20 ms for the whole run and records the peak per phase. Both output files of a row (`dev_<row>.json`, `probe_<row>.json`) carry the identical `blender2` block; the report checks this (`same process` column).

- `S` (streamed): `{"stream":true,"head":"q8","embed":"q8","kv":8,"chunk":256}`; `M` (minimal VRAM): `{"stream":true,"head":"cpu","embed":"cpu","kv":8,"chunk":256}`; both with the frozen `split4` prompt.
- int8 = group 64, calibration-free (MSE clip + least squares); int4 = group 128, GPTQ (128 x 512 calibration tokens from the ADAM-T SFT v8 train split, seed 11, damp 0.01 escalated x10 where the Hessian is not positive definite, sha256 of the calibration text `c7f0fa6b0e80...` in each manifest).
- Measure-phase peak = `blender2.meas.peak_proc_vram_GB` (pre-registered selection metric). Whole-run peak = `blender2.proc_vram_peak_all_GB` (includes load and the warm-up; reported next to it and used as the conservative headline).

## 5. Streamed-vs-resident equivalence (every reported model)

`scripts/hybrid1_equiv.py --set dev --long 8`: the 3 HYBRID-1 prompts, the 103 dev blend prompts, the 30 dev greeting/venting prompts and 8 multi-turn prompts of 335 to 664 tokens (several 256-token prefill chunks), 48 greedy tokens each. Reference = the same numerics with every layer resident (`R_S`, `R_M`); compared = the deployed streamed configuration (`S`, `M`). Logit max-abs-diff is over every compared step up to and including the first differing step. (The two Granite int4 files were run before the long prompts were added: 136 prompts.)

| file | config | reference | prompts token-identical to reference | logit max-abs-diff on identical prompts | max-abs-diff incl. divergence step | divergences (prompt index @ first differing step) | degenerate repeats |
|---|---|---|---|---|---|---|---|
| equiv_granite42_q8_A2.json | R_S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_granite42_q8_A2.json | S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_granite42_q8_B2.json | R_M | R_M | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_granite42_q8_B2.json | M | R_M | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_granite42_q4g_A.json | R_S | R_S | 136/136 | 0.0 | 0.0 |  | 0 |
| equiv_granite42_q4g_A.json | S | R_S | 136/136 | 0.0 | 0.0 |  | 0 |
| equiv_granite42_q4g_B.json | R_M | R_M | 136/136 | 0.0 | 0.0 |  | 0 |
| equiv_granite42_q4g_B.json | M | R_M | 136/136 | 0.0 | 0.0 |  | 0 |
| equiv_minicpm5_q8_A.json | R_S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_minicpm5_q8_A.json | S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_minicpm5_q8_B.json | R_M | R_M | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_minicpm5_q8_B.json | M | R_M | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_minicpm5_q4g_A.json | R_S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_minicpm5_q4g_A.json | S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_minicpm5_q4g_B.json | R_M | R_M | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_minicpm5_q4g_B.json | M | R_M | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_2b_q8_A.json | R_S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_2b_q8_A.json | S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_2b_q8_B.json | R_M | R_M | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_2b_q8_B.json | M | R_M | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_2b_q4g_A.json | R_S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_2b_q4g_A.json | S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_2b_q4g_B.json | R_M | R_M | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_2b_q4g_B.json | M | R_M | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_4b_q8_A.json | R_S | R_S | 67/67 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_4b_q8_A.json | S | R_S | 67/67 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_4b_q8_B.json | R_M | R_M | 67/67 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_4b_q8_B.json | M | R_M | 67/67 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_4b_q4g_A.json | R_S | R_S | 67/67 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_4b_q4g_A.json | S | R_S | 67/67 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_4b_q4g_B.json | R_M | R_M | 67/67 | 0.0 | 0.0 |  | 0 |
| equiv_qwen35_4b_q4g_B.json | M | R_M | 67/67 | 0.0 | 0.0 |  | 0 |
| equiv_qwen3_q8_A.json | R_S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |
| equiv_qwen3_q8_A.json | S | R_S | 144/144 | 0.0 | 0.0 |  | 0 |

**Every streamed configuration is token-identical and logit bit-identical (max-abs-diff 0.0) to its resident twin on every prompt, for every model and both bit-widths, including the hybrid DeltaNet Qwen3.5 stack.** The levers themselves are not numerics-neutral: Granite int8 with bf16 tables, bf16 KV and no chunking (`base` in `equiv_granite42_q8_A.json`) matches the S numerics on 116 of 136 prompts (max-abs-diff 5.7), which is why every quality number below is measured on the deployed numerics and not borrowed from another configuration. Engine v3 is token-identical to v1 on all 136 dev prompts for Granite int8 R_S/S/R_M/M (`equiv_granite42_q8_A2.json`/`_B2.json` vs `_A.json`/`_B.json`, field `toks_sha1`), and the v1, v2 and v3 Granite int8 S rows produced the same 250 probe8 replies.

The HYBRID-1 recommended configuration (Qwen3-4B-Instruct-2507 int8, C8 = this S) was also checked before its bake was deleted: S vs R_S token-identical on 144/144 prompts, max-abs-diff 0.0 (`equiv_qwen3_q8_A.json`). Its HYBRID-1 probe8 row (`data/adam_t/eval/hybrid1_runs/probe_qwen3_4b_q8g64_C8.json`, `step` = the C8 options with `stream: true`) was therefore scored on the streamed numerics after all (the audit note that it was resident does not match the saved `step` field), in a single-load process that the allocator defect cannot reach. Its VRAM, re-measured on engine v3 without expandable segments (`measonly_qwen3_q8_S.json`, `blender2.meas.peak_proc_vram_GB`), is **1.63 GB**, not 1.54 GB (the HYBRID-1 figure was taken with the corrupting allocator setting). Those two numbers still come from different processes and engine versions, so they are not paired in the Pareto below.

## 6. Bakes and calibration

| bake | bits / group | calibration | bake GB (`bake_bytes`) | layer cos mean | layer cos min | weight rel-err (weighted) | calibration-output rel-err (mean) | code roundtrip exact |
|---|---|---|---|---|---|---|---|---|
| granite42_q8g64 | 8 / 64 | none (MSE clip + LSQ) | 4.370 | 0.99999 | 0.99998 | 0.52% | n/a | true |
| granite42_q4g128_gptq | 4 / 128 | GPTQ | 2.699 | 0.99254 | 0.97460 | 11.67% | 4.92% | true |
| minicpm5_q8g64 | 8 / 64 | none | 3.176 | 0.99999 | 0.99996 | 0.52% | n/a | true |
| minicpm5_q4g128_gptq | 4 / 128 | GPTQ | 2.123 | 0.99217 | 0.87305 | 11.63% | 4.72% | true |
| qwen35_2b_q8g64 (text only) | 8 / 64 | none | 2.477 | 1.00043 (fp32 CPU reduction artefact; fp64 cosine was added after this bake) | 0.99996 | 0.53% | n/a | true |
| qwen35_2b_q4g128_gptq (text only) | 4 / 128 | GPTQ | 1.747 | 0.99201 | 0.98582 | 11.82% | 4.51% | true |
| qwen35_4b_q8g64 (text only) | 8 / 64 | none | 5.065 | 0.99999 | 0.99998 | 0.52% | n/a | true |
| qwen35_4b_q4g128_gptq (text only) | 4 / 128 | GPTQ | 3.169 | 0.99277 | 0.98509 | 11.36% | 4.42% | true |

Values from each bake's `bake_manifest.json` (`bake_bytes`, `fidelity.layers.*`), recorded before the losing bakes were deleted (deletions logged in `data/ks/logs/deletions.log`). GPTQ int4 has a larger weight error than calibration-free int4 (9.7% in HYBRID-1) because it trades weight error for output error; its output error on the calibration activations is 4.4 to 4.9%. Bakes sizes include the bf16 embedding and lm_head (quantised to int8 at load in the S configuration, kept on the CPU in bf16 in the M configuration).

Calibrated int4 (GPTQ, same affine bit-plane format, so the kernel is unchanged) recovers most of what calibration-free int4 lost. Perplexity through the engine (resident, bf16 tables, 1,535 dev tokens): Granite 4.2 int8 7.483, RTN int4 8.906 (+19%), GPTQ int4 7.703 (+2.9%) (`ppl_granite42_resident_bf16tab.json`); MiniCPM5 int8 9.612, GPTQ int4 9.901 (+3.0%; HYBRID-1 RTN int4 10.69) (`ppl_minicpm5_resident_bf16tab.json`); Qwen3.5-2B int8 8.204, GPTQ int4 8.577 (+4.5%) (`ppl_qwen35_2b_resident_bf16tab.json`); Qwen3.5-4B int8 6.137, GPTQ int4 6.271 (+2.2%) (`ppl_qwen35_4b_resident_bf16tab.json`). Fidelity fields: `bake_manifest.json` `fidelity.layers` of each bake (`rel_err_weighted` is the weight error, `out_rel_err_mean` the calibration-output error GPTQ minimises).

## 7. The Qwen3.5 Gated DeltaNet port

Time box: about one working day. Outcome: **done inside the box**; Qwen3.5-2B and Qwen3.5-4B both bake, stream and score through the ATEX engine (details below). What it took:

1. **Text-only bake from the multimodal checkpoint** (`scripts/bake_hybrid1.py`): keys `model.language_model.*` are mapped to `model.*`; the 24-block vision tower (`model.visual.*`, 331 M parameters in the 2B) and the multi-token-prediction head (`mtp.*`, 61 M) are dropped; the bake's `config.json` is the `text_config` with `architectures: ["Qwen3_5ForCausalLM"]` (`model_type qwen3_5_text`); 1-D fp32 tensors (`A_log`, `dt_bias`) keep fp32. All 2-D projections, including the DeltaNet `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` and `out_proj` and the gated full-attention `q_proj` (doubled width for the output gate), go through the same ATEX bit-plane kernel; the depthwise `conv1d` (3-D) stays bf16.
2. **Cache:** a hybrid cache (`DynamicCache(config)` gives a fixed-size `LinearAttentionLayer` per DeltaNet layer: conv state plus a recurrent state of heads x 128 x 128 in fp32) with the int8 KV layer swapped in for the full-attention layers only. The engine's KV/state accounting now counts conv and recurrent states as well as K and V.
3. **Chunked prefill across chunks:** transformers 5.18 already continues the recurrent state and the conv state when a chunk arrives with a non-empty cache (`initial_state=recurrent_state`, `update_conv_state`); verified on a 5- and 8-layer random hybrid model (chunked vs one-shot prefill, max logit diff < 0.05) and on the real model by the 144-prompt equivalence with 8 multi-chunk prompts.
4. **Stream ring for unequal layers:** the HYBRID-1 ring sized its two slots from layer 0 and assumed every layer has the same tensors; DeltaNet and attention layers differ. The ring now moves one flat, 256-byte-aligned block per layer (one host-to-device copy per layer instead of about 28), sized to the largest layer, with per-layer views precomputed. This also fixed the odd-layer-count wrap race.
5. **Kernels:** no DeltaNet kernel was written. The DeltaNet recurrence runs on transformers' torch fallbacks (`torch_chunk_gated_delta_rule` for prefill, `torch_recurrent_gated_delta_rule` for decode, `F.conv1d` for the short convolution), with `HF_HUB_OFFLINE=1` so no hub kernel is fetched. These are small per-layer torch ops (matmul and elementwise), outside the ATEX/TMU path; see section 11.

**Port result (measured).** Both Qwen3.5 models run through the ATEX engine with streamed, bit-identical layers (section 5). KV plus DeltaNet state at 2,560 tokens (`meas.kv_cache_GB_at_ctx` on the fixed accounting, amendment 4):

| model | layers (DeltaNet / full attention) | KV + state at 2,560 tokens, int8 KV | file |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 (dense, HYBRID-1) | 0 / 36 | 0.192 GB | `measonly_qwen3_q8_S.json` |
| Granite 4.2 3B (dense) | 0 / 40 | 0.108 GB | `probe_granite42_q8_S.json` |
| MiniCPM5-2B (dense, 2 KV heads) | 0 / 42 | 0.056 GB | `probe_minicpm5_q8_S.json` |
| Qwen3.5-2B | 18 / 6 | 0.036 GB | `measonly_qwen35_2b_q8_S.json` |
| Qwen3.5-4B | 24 / 8 | 0.094 GB (50 MB of it is the fixed fp32 recurrent state) | `probe_qwen35_4b_q4g_S.json` |

The hybrid layout shrinks the cache as expected, but how much depends on the context: per token, int8 KV costs 16 KB for Qwen3.5-4B (8 attention layers) and 6 KB for Qwen3.5-2B, against 40 KB for Granite 4.2 and 72 KB for Qwen3-4B, while the DeltaNet layers add a fixed recurrent state (fp32: 19 MB for the 2B, 50 MB for the 4B). At 2,560 tokens that makes the 2B 3 to 5x smaller than the dense models and the 4B about equal to Granite; at 8k tokens the 4B would need about 0.18 GB against Granite's 0.33 GB. It does not yet shrink whole-process VRAM: Qwen3.5-2B int8 M allocates 0.38 GB through torch but peaks at 0.89 GB whole-process, against 0.61 and 0.90 for Granite int8 M, so its non-allocator overhead is about 0.2 GB larger (most likely the rocBLAS/rocSOLVER code objects and workspaces behind the fallback's `torch.linalg.solve_triangular`; not isolated further), and the 248k-entry vocabulary makes the tied table large (0.64 GB at int8 for the 4B in S mode). Prefill is 80 to 130 tokens/s on the fallback. A Triton chunked delta-rule kernel would remove both costs; it was not written inside the time box.

## 8. VRAM x quality, one process per row

Columns cite the field they come from; both files of a row carry the identical `blender2` block (last column of the second table). Dev columns are the selection set; probe8 columns are held out and were scored once per configuration (engine v1/v2 duplicates of the Granite int8 S row excepted, identical replies).

| config | source files (dev_ / probe_ + config + .json) | measure-phase peak GB `blender2.meas.peak_proc_vram_GB` | whole-run peak GB `blender2.proc_vram_peak_all_GB` | KV+state GB @2,560 `meas.kv_cache_GB_at_ctx` | pinned host GB | prefill tok/s | decode tok/s | ppl `blender2.ppl.ppl` | dev faithful % | dev no-new % | dev hedge % | dev blend coherent % | dev chat decline % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| granite42_q8_S | dev_granite42_q8_S.json, probe_granite42_q8_S.json | 1.46 | 1.46 | 0.1081 | 3.34 | 756.8 | 10.54 | 7.511 | 24.7 | 93.5 | 96.2 | 99.0 | 0.0 |
| granite42_q8_M | dev_granite42_q8_M.json, probe_granite42_q8_M.json | 0.90 | 0.90 | 0.1081 | 3.34 | 752.0 | 7.15 | 7.524 | 24.7 | 93.5 | 96.2 | 98.1 | 0.0 |
| granite42_q4g_S | dev_granite42_q4g_S.json, probe_granite42_q4g_S.json | 1.38 | 1.38 | 0.1081 | 1.67 | 863.2 | 13.86 | 7.735 | 20.8 | 93.5 | 100.0 | 97.1 | 0.0 |
| granite42_q4g_M | dev_granite42_q4g_M.json, probe_granite42_q4g_M.json | 0.82 | 0.82 | 0.1081 | 1.67 | 881.1 | 8.51 | 7.745 | 22.1 | 93.5 | 100.0 | 96.1 | 0.0 |
| minicpm5_q8_S | dev_minicpm5_q8_S.json, probe_minicpm5_q8_S.json | 1.32 | 1.41 | 0.0559 | 2.11 | 1178.5 | 13.41 | 9.611 | 39.0 | 87.0 | 96.2 | 100.0 | 6.7 |
| minicpm5_q8_M | dev_minicpm5_q8_M.json, probe_minicpm5_q8_M.json | 0.70 | 0.70 | 0.0559 | 2.11 | 1151.3 | 8.13 | 9.622 | 40.3 | 87.0 | 96.2 | 100.0 | 6.7 |
| minicpm5_q4g_S | dev_minicpm5_q4g_S.json, probe_minicpm5_q4g_S.json | 1.27 | 1.36 | 0.0559 | 1.05 | 1316.6 | 15.56 | 9.896 | 48.0 | 89.6 | 57.7 | 97.1 | 20.0 |
| minicpm5_q4g_M | dev_minicpm5_q4g_M.json, probe_minicpm5_q4g_M.json | 0.65 | 0.65 | 0.0559 | 1.05 | 1084.6 | 7.17 | 9.902 | 45.5 | 87.0 | 61.5 | 97.1 | 20.0 |
| qwen35_2b_q8_S | dev_qwen35_2b_q8_S.json, probe_qwen35_2b_q8_S.json | 1.46 | 1.71 | 0.0159 | 1.46 | 83.0 | 13.19 | 8.216 | 22.1 | 36.4 | 100.0 | 100.0 | 0.0 |
| qwen35_2b_q8_M | dev_qwen35_2b_q8_M.json, probe_qwen35_2b_q8_M.json | 0.89 | 0.94 | 0.0159 | 1.46 | 114.4 | 5.84 | 8.202 | 26.0 | 39.0 | 100.0 | 100.0 | 0.0 |
| qwen35_2b_q4g_S | dev_qwen35_2b_q4g_S.json, probe_qwen35_2b_q4g_S.json | 1.36 | 1.61 | 0.0356 | 0.73 | 93.3 | 25.51 | 8.591 | 28.6 | 62.3 | 100.0 | 100.0 | 0.0 |
| qwen35_2b_q4g_M | dev_qwen35_2b_q4g_M.json, probe_qwen35_2b_q4g_M.json | 0.82 | 0.88 | 0.0356 | 0.73 | 96.7 | 5.33 | 8.586 | 35.1 | 64.9 | 100.0 | 100.0 | 0.0 |
| qwen35_4b_q8_S | dev_qwen35_4b_q8_S.json, probe_qwen35_4b_q8_S.json | 2.02 | 2.10 | 0.0942 | 3.79 | 85.3 | 10.3 | 6.132 | 40.3 | 59.7 | 96.2 | 100.0 | 0.0 |
| qwen35_4b_q8_M | dev_qwen35_4b_q8_M.json, probe_qwen35_4b_q8_M.json | 1.30 | 1.30 | 0.0942 | 3.79 | 82.8 | 4.88 | 6.134 | 40.3 | 59.7 | 96.2 | 100.0 | 0.0 |
| qwen35_4b_q4g_S | dev_qwen35_4b_q4g_S.json, probe_qwen35_4b_q4g_S.json | 1.86 | 1.94 | 0.0942 | 1.90 | 93.3 | 17.01 | 6.269 | 36.4 | 50.6 | 88.5 | 100.0 | 0.0 |
| qwen35_4b_q4g_M | dev_qwen35_4b_q4g_M.json, probe_qwen35_4b_q4g_M.json | 1.18 | 1.18 | 0.0942 | 1.90 | 90.9 | 5.35 | 6.266 | 36.4 | 50.6 | 84.6 | 100.0 | 0.0 |

| config | probe8 faithful % `summary.blend_probe.faithful_para.p` | no-new % | hedge % | blend coherent % | chat answered % | chat coherent % | decline % | emergency-text FP % | distinct-2 | opener-3 repeat % | same process as the VRAM row |
|---|---|---|---|---|---|---|---|---|---|---|---|
| granite42_q8_S | 62.0 | 64.2 | 93.3 | 100.0 | 79.4 | 100.0 | 2.9 | 0.0 | 0.625 | 48.5 | True |
| granite42_q8_M | 61.3 | 65.7 | 95.6 | 100.0 | 79.4 | 100.0 | 2.9 | 0.0 | 0.6239 | 48.5 | True |
| granite42_q4g_S | 65.0 | 72.3 | 97.8 | 91.2 | 77.9 | 100.0 | 2.9 | 0.0 | 0.6376 | 48.5 | True |
| granite42_q4g_M | 63.5 | 73.0 | 97.8 | 91.8 | 80.9 | 97.1 | 2.9 | 0.0 | 0.6323 | 48.5 | True |
| minicpm5_q8_S | 43.1 | 56.2 | 91.1 | 97.2 | 42.6 | 95.6 | 25.0 | 0.0 | 0.6137 | 45.6 | True |
| minicpm5_q8_M | 43.1 | 55.5 | 91.1 | 96.7 | 39.7 | 95.6 | 26.5 | 0.0 | 0.6135 | 45.6 | True |
| minicpm5_q4g_S | 51.8 | 65.7 | 73.3 | 94.0 | 42.6 | 95.6 | 17.6 | 0.0 | 0.5773 | 47.1 | True |
| minicpm5_q4g_M | 51.8 | 65.0 | 71.1 | 92.9 | 36.8 | 95.6 | 22.1 | 0.0 | 0.562 | 42.6 | True |
| qwen35_2b_q8_S | 7.3 | 10.2 | 82.2 | 97.8 | 76.5 | 98.5 | 0.0 | 0.0 | 0.5777 | 64.7 | True |
| qwen35_2b_q8_M | 8.0 | 10.9 | 80.0 | 97.2 | 72.1 | 100.0 | 0.0 | 0.0 | 0.577 | 64.7 | True |
| qwen35_2b_q4g_S | 18.2 | 21.2 | 86.7 | 99.5 | 75.0 | 100.0 | 0.0 | 0.0 | 0.5977 | 67.7 | True |
| qwen35_2b_q4g_M | 19.0 | 21.9 | 88.9 | 98.9 | 77.9 | 98.5 | 0.0 | 0.0 | 0.6029 | 66.2 | True |
| qwen35_4b_q8_S | 57.7 | 62.0 | 95.6 | 95.6 | 85.3 | 100.0 | 2.9 | 0.0 | 0.6423 | 58.8 | True |
| qwen35_4b_q8_M | 56.2 | 60.6 | 95.6 | 96.2 | 79.4 | 100.0 | 2.9 | 0.0 | 0.6213 | 58.8 | True |
| qwen35_4b_q4g_S | 52.5 | 55.5 | 97.8 | 100.0 | 77.9 | 100.0 | 5.9 | 0.0 | 0.5911 | 61.8 | True |
| qwen35_4b_q4g_M | 53.3 | 56.2 | 93.3 | 100.0 | 77.9 | 100.0 | 5.9 | 0.0 | 0.5828 | 60.3 | True |

Notes on reading the rows:
- **Whole-run vs measure-phase peak.** The two differ only where the load or the 64-token warm-up left cached blocks before the first `empty_cache()` (MiniCPM5 S and Qwen3.5 rows, by up to 0.25 GB). The CPU-table (M) rows never exceed the measure peak by more than 0.06 GB.
- **Where the VRAM goes (M rows).** Granite 4.2 int4 M: torch allocated at most 0.52 GB (ring 2 x ~45 MB, int8 KV 0.11 GB at 2,560 tokens, 256-token prefill activations) and the rest of the 0.82 GB is the HIP context and kernel images. Qwen3.5-2B int8 M: only 0.38 GB allocated (DeltaNet state plus the 6 full-attention layers' KV is 0.04 GB, against 0.11 GB for Granite) but 0.89 GB whole-process: its non-allocator overhead (context, kernel images, library code objects and workspaces) (whole-process peak minus torch reserved, `meas.peak_proc_vram_GB` - `meas.torch_max_reserved_GB`) is 0.48 GB against 0.28 GB for Granite int8 M. The likely source is the extra libraries the torch DeltaNet fallback loads (`torch.linalg.solve_triangular` goes through rocBLAS/rocSOLVER); this was not isolated further.
- **Speed.** M rows decode 5 to 9 tokens/s (the lm_head runs on the CPU), S rows 10 to 26. Qwen3.5 prefill is 80 to 130 tokens/s against 750 to 1,300 for the dense models: the chunked DeltaNet recurrence runs as many small torch ops.
- **Qwen3.5-2B int8 KV+state field.** `qwen35_2b_q8_S` and `qwen35_2b_q8_M` show 0.0159 GB because they ran before the dict-held DeltaNet states were counted (amendment 4); the measure-only re-runs on the fixed engine give 0.0356 GB (`measonly_qwen35_2b_q8_{S,M}.json`, same field). Their VRAM peaks are fdinfo measurements and unaffected.

## 9. Selection (dev set only) and the VRAM x quality Pareto

`python3 scripts/hybrid1_report.py select <16 rows>` applies the gates and the three versions of the rule to the dev rows (all inputs from the `dev_*.json` files: `summary.blend_probe.*.p`, `summary.chat_probe.all.decline.p`, `blender2.ppl.ppl`, `blender2.meas.peak_proc_vram_GB`):

| config | dev faithful % | dev no-new % | sum | dev hedge % | dev blend coherent % | dev chat decline % | ppl / own int8 S | passes gates | eligible: plan as written | eligible: amendment 3 | eligible: amendment 5 | measure-phase peak GB | whole-run peak GB | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| granite42_q8_S | 24.7 | 93.5 | 118.2 | 96.2 | 99.0 | 0.0 | 1.000 | True | False | False | True | 1.463 | 1.463 | 10.54 |
| granite42_q8_M | 24.7 | 93.5 | 118.2 | 96.2 | 98.1 | 0.0 | 1.002 | True | False | False | True | 0.903 | 0.903 | 7.15 |
| granite42_q4g_S | 20.8 | 93.5 | 114.3 | 100.0 | 97.1 | 0.0 | 1.030 | True | False | False | True | 1.380 | 1.380 | 13.86 |
| granite42_q4g_M | 22.1 | 93.5 | 115.6 | 100.0 | 96.1 | 0.0 | 1.031 | True | False | False | True | 0.819 | 0.819 | 8.51 |
| minicpm5_q8_S | 39.0 | 87.0 | 126.0 | 96.2 | 100.0 | 6.7 | 1.000 | False | False | False | False | 1.318 | 1.410 | 13.41 |
| minicpm5_q8_M | 40.3 | 87.0 | 127.3 | 96.2 | 100.0 | 6.7 | 1.001 | False | False | False | False | 0.703 | 0.703 | 8.13 |
| minicpm5_q4g_S | 48.0 | 89.6 | 137.7 | 57.7 | 97.1 | 20.0 | 1.030 | False | False | False | False | 1.267 | 1.360 | 15.56 |
| minicpm5_q4g_M | 45.5 | 87.0 | 132.5 | 61.5 | 97.1 | 20.0 | 1.030 | False | False | False | False | 0.653 | 0.653 | 7.17 |
| qwen35_2b_q8_S | 22.1 | 36.4 | 58.4 | 100.0 | 100.0 | 0.0 | 1.000 | True | False | False | False | 1.462 | 1.709 | 13.19 |
| qwen35_2b_q8_M | 26.0 | 39.0 | 64.9 | 100.0 | 100.0 | 0.0 | 0.998 | True | False | False | False | 0.887 | 0.944 | 5.84 |
| qwen35_2b_q4g_S | 28.6 | 62.3 | 90.9 | 100.0 | 100.0 | 0.0 | 1.046 | True | False | False | False | 1.363 | 1.611 | 25.51 |
| qwen35_2b_q4g_M | 35.1 | 64.9 | 100.0 | 100.0 | 100.0 | 0.0 | 1.045 | True | False | False | False | 0.824 | 0.881 | 5.33 |
| qwen35_4b_q8_S | 40.3 | 59.7 | 100.0 | 96.2 | 100.0 | 0.0 | 1.000 | True | False | False | False | 2.017 | 2.097 | 10.3 |
| qwen35_4b_q8_M | 40.3 | 59.7 | 100.0 | 96.2 | 100.0 | 0.0 | 1.000 | True | False | False | False | 1.304 | 1.304 | 4.88 |
| qwen35_4b_q4g_S | 36.4 | 50.6 | 87.0 | 88.5 | 100.0 | 0.0 | 1.022 | True | False | False | False | 1.864 | 1.939 | 17.01 |
| qwen35_4b_q4g_M | 36.4 | 50.6 | 87.0 | 84.6 | 100.0 | 0.0 | 1.022 | True | False | False | False | 1.180 | 1.180 | 5.35 |

Best dev faithful over all rows 48.0, over gate-passing rows 40.3; best no-new over all 93.5, over gate-passing 93.5; best summed score over gate-passing 118.2.

- Plan as written selects: **none**
- Amendment 3 selects: **none**
- Amendment 5 selects: **granite42_q4g_M**

How the rule evolved (all timestamped in `blender2_eval.md`): the plan as written and amendment 3 select nothing (explained in section 12); amendment 5, written after the Granite, MiniCPM5 and Qwen3.5-2B dev results and before any Qwen3.5-4B result, selects **granite42_q4g_M**. The Qwen3.5-4B rows that came afterwards did not change it (best summed score 100.0 against the 113.2 threshold).

**Pareto on held-out probe8 (report only; one process per point, VRAM = measure-phase peak, whole-run peak in brackets where it differs):**

| VRAM GB | configuration | probe8 strict faithful | probe8 no-new | probe8 hedge | probe8 chat decline | decode tok/s |
|---|---|---|---|---|---|---|
| 0.65 | MiniCPM5-2B int4 M | 51.8% | 65.0% | 71.1% | 22.1% | 7.2 |
| 0.70 | MiniCPM5-2B int8 M | 43.1% | 55.5% | 91.1% | 26.5% | 8.1 |
| **0.82** | **Granite 4.2 3B int4 M (recommended)** | **63.5%** | **73.0%** | **97.8%** | **2.9%** | **8.5** |
| 0.82 (0.88) | Qwen3.5-2B int4 M | 19.0% | 21.9% | 88.9% | 0.0% | 5.3 |
| 0.89 (0.94) | Qwen3.5-2B int8 M | 8.0% | 10.9% | 80.0% | 0.0% | 5.8 |
| 0.90 | Granite 4.2 3B int8 M | 61.3% | 65.7% | 95.6% | 2.9% | 7.2 |
| 1.18 | Qwen3.5-4B int4 M | 53.3% | 56.2% | 93.3% | 5.9% | 5.4 |
| 1.27 (1.36) | MiniCPM5-2B int4 S | 51.8% | 65.7% | 73.3% | 17.6% | 15.6 |
| 1.30 | Qwen3.5-4B int8 M | 56.2% | 60.6% | 95.6% | 2.9% | 4.9 |
| 1.32 (1.41) | MiniCPM5-2B int8 S | 43.1% | 56.2% | 91.1% | 25.0% | 13.4 |
| 1.36 (1.61) | Qwen3.5-2B int4 S | 18.2% | 21.2% | 86.7% | 0.0% | 25.5 |
| 1.38 | Granite 4.2 3B int4 S | 65.0% | 72.3% | 97.8% | 2.9% | 13.9 |
| 1.46 | Granite 4.2 3B int8 S | 62.0% | 64.2% | 93.3% | 2.9% | 10.5 |
| 1.46 (1.71) | Qwen3.5-2B int8 S | 7.3% | 10.2% | 82.2% | 0.0% | 13.2 |
| 1.86 (1.94) | Qwen3.5-4B int4 S | 52.5% | 55.5% | 97.8% | 5.9% | 17.0 |
| 2.02 (2.10) | Qwen3.5-4B int8 S | 57.7% | 62.0% | 95.6% | 2.9% | 10.3 |

On probe8 the recommended point is also the Pareto front's knee: nothing smaller has comparable grounding (MiniCPM5 is smaller but declines one greeting or venting message in four or five and hedges only 71% of uncovered questions at int4), and nothing larger beats its strict faithful by more than the 4-point binomial SE (Granite int4 S, 65.0%, at 1.38 GB). HYBRID-1's Qwen3-4B int8 S (86.9% strict faithful, 1.63 GB on engine v3) would sit above this front on quality, but it is excluded by the CEO and its two numbers come from different processes.

## 10. Recommended configuration

**Granite 4.2 3B, GPTQ int4 group 128, streamed, CPU tables, int8 KV, 256-token chunks, frozen `split4` prompt.**

- Weights: `~/ai/hybrid1/bakes/granite42_q4g128_gptq/` (2.70 GB; every file's sha256 in `bakes/blender2_granite42_3b_q4g128_gptq/bake_files.sha256`); manifest and NOTICE in `bakes/blender2_granite42_3b_q4g128_gptq/`; attribution in `NOTICE`.
- Flag (default off): `AMNI_BLENDER=atex:/home/anmire/ai/hybrid1/bakes/granite42_q4g128_gptq` and `AMNI_BLENDER_OPTS='{"sys":"split4","stream":true,"head":"cpu","embed":"cpu","kv":8,"chunk":256}'`. End-to-end check through `adam_t_client.client()` (`tests/blender_flag_smoke.py`, output `smoke_amni_blender.json`): one cached `Blender` instance, load 2.9 s, grounded blend reply, warm chat reply, "I'm not sure about that." on the uncovered question; with the variables unset, `blender_atex.active()` is False.
- Measured together in one process (files `dev_granite42_q4g_M.json`, `probe_granite42_q4g_M.json`): whole-process peak 0.819 GB (measure phase and whole run; torch allocated at most 0.522 GB; KV 0.108 GB at 2,560 tokens), 8.51 tokens/s decode, 881 tokens/s prefill, perplexity 7.745; dev: faithful 22.1%, no-new 93.5%, hedge 100%, blend coherent 96.1%, chat decline 0%; probe8: faithful 63.5%, no-new 73.0%, hedge 97.8%, blend coherent 91.8%, chat answered 80.9%, chat coherent 97.1%, decline 2.9%, emergency-text FP 0%, distinct-2 0.632, opener-3 repeat 48.5%.
- Host RAM: 1.67 GB pinned (layers) + about 1.03 GB (bf16 embedding and lm_head on the CPU).
- If speed matters more than 0.56 GB: the same bake in S mode (int8 tables on the GPU) is 1.38 GB at 13.9 tokens/s with the same quality class (`*_granite42_q4g_S.json`).
- Known quality costs of int4 vs int8 on probe8 (reported, not used to choose): blend coherent 91.8% vs 100% (int4 says "I don't know about X." on some covered items), hedge 97.8% vs 95.6%, strict faithful 63.5% vs 61.3%.

## 11. Where this left the paradigm, and why

- **Not GF(17), and lossy.** The bakes are affine group-quantised bit-planes in safetensors, 0.5% (int8) and about 12% (GPTQ int4) relative weight error; only the code pack/unpack is exact. A lossless Reffelt container for a third-party bf16 model would be 16 bits per weight, which defeats the VRAM goal. GPTQ calibration is an extra lossy, data-dependent step (calibration text: ADAM-T SFT v8 train split, never probe8 or the dev set).
- **Matmul, not TMU.** The ATEX kernel uses `tl.dot`; attention uses SDPA; in M mode the lm_head is a CPU bf16 matmul; GPTQ uses Cholesky factorisations and matmuls at bake time; the Qwen3.5 DeltaNet recurrence runs on transformers' torch fallbacks (matmul, `solve_triangular`, `conv1d`). Only the embedding rows use `index_select`.
- **RAM streaming, not SSD streaming.** Layers stream from pinned RAM (1.7 to 3.9 GB host RAM depending on the bake) through a two-slot ring; NVMe tiering was not built.
- **Third-party weights as the blender** (CEO decision, blender role only); the system prompt forbids new facts and the harness gate is unchanged.

## 12. What failed or fell short

- **The pre-registered selection rule selected nothing, twice.** As written (F* over all rows) the only rows near F* were MiniCPM5 rows that fail the decline gate; with F* over gate-passing rows (amendment 3), no row was within 5 points of both the best strict-faithful row (Qwen3.5-2B int4 M) and the best no-new-content row (Granite). The recommendation comes from amendment 5 (summed score), which was written after the dev results of three candidates were known. It is flagged as post hoc; probe8 (held out, not used) points the same way (Granite 4.2 61 to 65% strict faithful against 7 to 19% for Qwen3.5-2B and 43 to 52% for MiniCPM5 with 18 to 27% declines).
- **Nothing reaches HYBRID-1's Qwen3-4B on the strict gate.** Qwen3-4B-Instruct-2507 int8 scored 86.9% strict faithful on probe8; the best current-generation row here is Granite 4.2 int4 S at 65.0%. The CEO ruled Qwen3 out as too old, so it is not a candidate, but the quality gap on this metric is real.
- **Granite 4.2 answers short.** Its dev strict-faithful is 21 to 25% while no-new-content is 93.5%: correct, concise answers that the strict gate rejects for being under 8 words or restating under 60% of the source. The frozen prompt was chosen on Granite 4.1 int4 in HYBRID-1; it was not re-tuned.
- **Qwen3.5 small models follow the notes prompt poorly.** Qwen3.5-2B int8 adds new content words in 61 to 64% of dev paraphrase replies, says "I am not sure" on covered items, and once invented facts about a place with no notes (probe8 item on Garam Chashma). GPTQ int4 behaved better than int8 on this (dev no-new 62 to 65% vs 36 to 39%), an unexplained but measured difference.
- **The DeltaNet port does not save whole-process VRAM yet.** KV plus recurrent state at 2,560 tokens is 3 to 5 times smaller for Qwen3.5-2B than for the dense models (about equal to Granite for the 4B, whose fixed fp32 state is 50 MB), but about 0.2 GB more fixed overhead (most likely the libraries behind the torch fallback's `solve_triangular`; not isolated) cancels the saving, and prefill runs at 80 to 130 tokens/s.
- **Engine had to be re-measured twice** (amendments 1 and 2): allocator slack from `torch.cat` KV growth and chunked-prefill fragmentation; the v1/v2 rows are kept but not used.
- **A long-running chain was killed by the agent's 2-hour background limit** mid-way through the MiniCPM5 int4 M probe; that row was re-run from scratch in a detached process (both of its files come from the re-run).
- **Decode speed is noisy** (Qwen3.5-2B int8 S: 13.2 tokens/s in its row, 21.8 in the measure-only re-run 3 hours later): the decode loop is CPU-launch-bound and another agent (STORE-1) shares the CPU. VRAM figures are not affected.
- **Not done:** the locked E2E suite (by instruction); NVMe streaming; a GF(17) container for the codes; Triton DeltaNet kernels.

## 13. Next steps

1. **Fix the strict gate or the prompt, then re-select.** The strict `gok` gate fails correct short answers ("Ostrander Lake is in the Pellam Range."), which is most of Granite 4.2's dev misses; no-new-content is 93.5%. A Granite-specific notes prompt chosen on the dev set (the frozen `split4` was picked on Granite 4.1 in HYBRID-1) is the cheapest lever; then one E2E run by the CTO.
2. **Smaller fixed overhead.** In the M rows the HIP context, kernel images and library code objects are 0.3 to 0.5 GB of the 0.65 to 0.9 GB. Candidates: avoid rocBLAS/rocSOLVER code objects (a Triton DeltaNet kernel instead of `solve_triangular`), avoid hipBLASLt workspaces in the prefill, and try `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (memory-efficient SDPA; would remove the materialised attention scores in the 256-token prefill chunks) behind an equivalence check.
3. **Prefill chunk below 256** for the M configuration (activation peak scales with the chunk; chunking is not numerics-neutral, so it needs its own row).
4. **Table quantisation offline** (bake the int8 tables) to drop the load-time quantisation spike and the warm-up residue that makes some whole-run peaks exceed the measure-phase peak.
5. **Speed:** CUDA-graph capture of the decode step, a fused dequant-GEMV, and a CPU lm_head in int8 for M mode (M decodes at 6 to 9 tokens/s).
6. **Qwen3.5 kernels:** a Triton chunked Gated DeltaNet prefill (prefill is 80 to 130 tokens/s on the torch fallback against 750 to 1,300 for the dense models).
7. **GF(17) container** for the bit-plane codes so the bake lives in PTEX pages (lossless for the codes; the weights stay lossy).

## 14. Reproduce

```
python scripts/hybrid1_fetch.py ibm-granite/granite-4.2-3b e459acceac81e5fe67c07d9cfc72329a332e7eb1 raw/granite42
python scripts/bake_hybrid1.py --src raw/granite42 --out bakes/granite42_q4g128_gptq --bits 4 --gs 128 --calib calib_sft8_s11_3000.txt --nseq 128 --seqlen 512
python scripts/hybrid1_equiv.py --bake bakes/granite42_q4g128_gptq --cfgs '{"R_M":{"head":"cpu","embed":"cpu","kv":8,"chunk":256},"M":{"stream":true,"head":"cpu","embed":"cpu","kv":8,"chunk":256}}' --set dev --long 8 --out equiv.json
python scripts/hybrid1_run.py --bake bakes/granite42_q4g128_gptq --opts '{"sys":"split4","stream":true,"head":"cpu","embed":"cpu","kv":8,"chunk":256}' --set dev,probe --meas --ppl --name granite42_q4g_M --out '{set}_granite42_q4g_M.json'
HYBRID1_RUNS=<dir> python scripts/hybrid1_report.py rows granite42_q4g_M
```
Calibration text: 3,000 examples sampled with numpy seed 11 from `data/adam_t/sft_v8/train.bpe4096.u16`, decoded with `data/clean2/bpe4096.json` (sha256 `c7f0fa6b0e80eef45ba4d7c411ae1ce3f426b498e6e76c44f809b61991ef83de`). Environment: `docs/adam_t/blender2_env.txt`.
