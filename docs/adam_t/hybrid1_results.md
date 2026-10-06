# HYBRID-1 results: an open dense model as Adam's blender, ATEX group-quantised, in the least VRAM

Mission HYBRID-1, 2026-10-05 to 2026-10-06. Branch `worktree-agent-a5bd8c5f36a3c80c7`, commits prefixed `HYBRID-1:`. Plan (fixed before scoring): `docs/adam_t/hybrid1_eval.md`. Models, licences, revisions: `docs/adam_t/hybrid1_models.md`.

## 0. Bottom line

1. **Best blender found: Qwen3-4B-Instruct-2507 (Apache-2.0).** On the held-out blend probe it passes the strict grounding gate 86.9% of the time at 8-bit (ADAM-T round 3: 59.9%, round 2: 14.6%, Granite 4.1 3B baseline at int4: 20.4%). It hedges 95.6% of uncovered questions (ADAM-T r3: 86.7%) and declines on at most 1.5% of greeting and venting items.
2. **Smallest VRAM that keeps that quality: 1.54 GB whole-process peak** (weights ring 0.62 GB, KV 0.19 GB at 2,560 tokens, activations, HIP context), against 4.53 GB for the existing Granite 4.1 3B int4 ATEX path measured the same way. It needs the layers streamed from pinned RAM (3.9 GB of host RAM for the int8 layers) and runs at 9.3 tokens/s decode and 486 tokens/s prefill.
3. **Bit-width does not set the VRAM floor once the layers are streamed**; the ring, tables, KV and activations do. It sets host RAM, stream bandwidth and, above all, quality. Calibration-free 4-bit costs 19 points of grounding pass on Qwen3-4B (67.9% vs 86.9%), 3-bit costs 26, 2-bit collapses on every model. So 8-bit streamed is the recommendation, and 4-bit resident is a worse deal.
4. **Hybrid E2E result (locked 410-item chat dev suite, run once): the pass bar is NOT met.** The hybrid beats ADAM-T alone on answered (74.4% vs 62.7%) and on blend grounding pass (44.8% vs 17.2%), ties it on fact (73.8% vs 70.5%) and coherent (95.1% vs 95.4%), and keeps 911 lead at 100% and near-miss FP at 3.7%. It does **not** beat the open model used alone with no material, which scores answered 92.2%, fact 97.3% (its own weights know the suite's facts) but coherent 91.0% and no grounding at all.
5. **Quality below int8 is lossy and was not rescued without calibration.** Group-quantisation is lossy by construction; only the code pack/unpack is bit-exact. KV int4 destroys output; KV int8 is free.
6. **End to end, the open blenders are close to each other.** Qwen3-4B int8, Qwen3-4B int4 and the Granite 4.1 3B baseline (all streamed, 1.17 to 1.54 GB) score 72.7 to 75.6% answered and 73.8 to 74.5% fact on the E2E suite, all about 10 to 13 points above ADAM-T on answered. The strict probe8 gate separates them far more (86.9%, 67.9%, 20.4%); the harness gate and its extractive fallback absorb most of the difference in the E2E numbers. The recommendation follows the pre-registered probe criterion (Qwen3-4B int8); if only E2E answered and fact mattered, the Granite 3B int4 streamed at 1.17 GB would be the smaller choice.

## 1. Candidates

See `hybrid1_models.md` for the full table. Run: Granite 4.1 3B (baseline), Granite 4.2 3B (found in own search; same code path), MiniCPM5-2B, Qwen3-4B-Instruct-2507 (the plain dense stand-in for Qwen3.5-4B). Not run: Qwen3.5-4B (Gated DeltaNet hybrid, no linear-attention path in ATEX), Ministral 3 3B (FP8 weights plus a vision wrapper), SmolLM3-3B (time), Gemma 4 E4B (separate prohibited-use policy; 8 B stored), Nemotron-3-Nano-4B (non-Apache licence, Mamba hybrid).

## 2. Bakes: size and tensor fidelity

Affine group quantisation per bit-plane (`amni/inference/qgroup_gemv.py`): per group a fp16 scale and fp16 offset, `w = code*scale + offset`, scale and offset by an MSE clip search and a least-squares refit (calibration-free). Embeddings and lm_head stay bf16 in the bake (quantised at load as a lever). Norms stay bf16. 3-bit is two bit-planes (2+1). **Code pack/unpack is bit-exact for every tensor of every bake (verified). The weights themselves are lossy: cos and relative error against the original bf16 tensors below.**

| model | bits/group | bake GB | layer cos mean | layer cos min | layer rel-err (weighted) |
|---|---|---|---|---|---|
| Granite 4.1 3B | 2/64 | 1.50 | 0.9426 | 0.9285 | 33.2% |
| Granite 4.1 3B | 3/64 | 1.89 | 0.9847 | 0.9797 | 17.3% |
| Granite 4.1 3B | 4/128 | 2.19 | 0.9952 | 0.9930 | 9.7% |
| Granite 4.1 3B | 8/64 | 3.86 | 1.0000 | 0.99998 | 0.5% |
| Granite 4.2 3B | 2/64 | 2.01 | 0.9426 | 0.9283 | 33.2% |
| Granite 4.2 3B | 3/64 | 2.40 | 0.9847 | 0.9797 | 17.3% |
| Granite 4.2 3B | 4/128 | 2.70 | 0.9953 | 0.9930 | 9.7% |
| MiniCPM5-2B | 2/64 | 1.69 | 0.9413 | 0.8939 | 33.2% |
| MiniCPM5-2B | 3/64 | 1.94 | 0.9842 | 0.9633 | 17.4% |
| MiniCPM5-2B | 4/128 | 2.12 | 0.9950 | 0.9825 | 9.7% |
| MiniCPM5-2B | 8/64 | 3.18 | 1.0000 | 0.99996 | 0.5% |
| Qwen3-4B-Instruct-2507 | 2/64 | 1.91 | 0.9438 | 0.9288 | 33.0% |
| Qwen3-4B-Instruct-2507 | 3/64 | 2.37 | 0.9851 | 0.9793 | 17.2% |
| Qwen3-4B-Instruct-2507 | 4/128 | 2.71 | 0.9954 | 0.9929 | 9.5% |
| Qwen3-4B-Instruct-2507 | 8/64 | 4.64 | 1.0000 | 0.99998 | 0.5% |

(Bakes other than the ones kept were deleted for disk space; these figures come from each bake's `bake_manifest.json` before deletion. `deletions.log` records them.)

### Next-token perplexity on the dev text (1,535 tokens, own CC0 text; a quantisation-loss proxy, not a quality claim)

| model | 8-bit (reference) | 4-bit | 3-bit | 2-bit |
|---|---|---|---|---|
| Granite 4.1 3B | 6.42 | 7.46 (+16%) | 24.9 | 16,257 |
| Granite 4.2 3B | n/a | 8.91 | 51.3 | 19,102 |
| MiniCPM5-2B | 9.62 | 10.69 (+11%) | 17.9 | 2.25e6 |
| Qwen3-4B-Instruct-2507 | 6.89 | 7.43 (+8%) | 8.42 (+22%) | 475 |

Tables (lm_head and embedding) quantised at load, Granite 4.1 3B int4 body: bf16 tables 7.458, int8 tables 7.460, int4 tables 7.557 (+1.3%). Qwen3-4B int4 body: bf16 7.431, int8 tables 7.431, int4 tables 7.494 (+0.8%).

## 3. VRAM and speed

Method (`scripts/hybrid1_measure.py`): cold process per row, one model resident, 2,048-token prefill then 512-token greedy decode (context 2,560 tokens), random token ids (timing and memory only). **Headline column = whole-process VRAM read from `/proc/self/fdinfo` (`drm-memory-vram`), sampled every 20 ms: HIP context, kernel images and allocator slack included, other processes (amni-gen) excluded.** The torch allocator figures are shown next to it; `rocm-smi` is not installed, and sysfs `mem_info_vram_used` was found to move with other GPU users, so it was dropped. Weights and KV are reported separately (weights = torch allocated after load, KV = bytes of the cache tensors at the end of the run). Decode speed excludes the governor throttle sleeps (the measurement bursts are about 1 minute each); probe and E2E runs obey `throttle.txt`.

NVMe sequential read measured with `dd iflag=direct`: 6.7 GB/s. Layer streaming here is from pinned host RAM (the layers are loaded from NVMe once); streaming every layer from NVMe each token would cost about 0.25 s per token for the 1.7 GB of Granite int4 layers, so about 4 tokens/s, and was not implemented.

| config | weights GB (torch alloc) | KV GB @ctx | peak GB torch reserved | peak GB whole process (fdinfo) | prefill tok/s | decode tok/s | ctx tokens |
|---|---|---|---|---|---|---|---|
| granite3b_q4_A_q4tab_kv8_chunk256 | 1.81 | 0.108 | 2.28 | 2.58 | 905 | 17.3 | 2560 |
| granite3b_q4_B_stream_cpu_tab_kv4_chunk256 | 0.08 | 0.056 | 0.51 | 0.79 | 767 | 8.6 | 2560 |
| granite3b_q4_C8_stream_q8tab_kv8_chunk256 | 0.35 | 0.108 | 0.88 | 1.17 | 882 | 14.3 | 2560 |
| granite3b_q4_C_stream_q4tab_kv4_chunk256 | 0.22 | 0.056 | 0.66 | 0.95 | 700 | 12.8 | 2560 |
| granite3b_q4_chunk256 | 2.19 | 0.210 | 2.80 | 3.17 | 978 | 19.7 | 2560 |
| granite3b_q4_embed_cpu | 2.19 | 0.210 | 4.16 | 4.52 | 682 | 19.5 | 2560 |
| granite3b_q4_embed_cpu_head_cpu | 1.67 | 0.210 | 3.66 | 3.92 | 686 | 12.5 | 2560 |
| granite3b_q4_embed_cpu_head_q4 | 1.81 | 0.210 | 3.78 | 4.07 | 684 | 20.4 | 2560 |
| granite3b_q4_head_cpu | 2.19 | 0.210 | 4.16 | 4.43 | 689 | 11.9 | 2560 |
| granite3b_q4_head_q4_embed_q4 | 1.81 | 0.210 | 3.78 | 4.08 | 572 | 20.4 | 2560 |
| granite3b_q4_head_q8_embed_q8 | 1.94 | 0.210 | 3.93 | 4.22 | 666 | 19.9 | 2560 |
| granite3b_q4_kv4 | 2.19 | 0.056 | 4.05 | 4.45 | 681 | 15.1 | 2560 |
| granite3b_q4_kv8 | 2.19 | 0.108 | 4.08 | 4.46 | 633 | 16.9 | 2560 |
| granite3b_q4_stream | 0.60 | 0.210 | 2.57 | 2.94 | 547 | 14.8 | 2560 |
| granite3b_q4g128 | 2.19 | 0.210 | 4.16 | 4.53 | 688 | 19.9 | 2560 |
| granite42_3b_q4_C8_stream_q8tab_kv8_chunk256 | 0.61 | 0.108 | 1.14 | 1.43 | 917 | 14.2 | 2560 |
| granite42_3b_q4g128 | 2.70 | 0.210 | 4.68 | 5.06 | 570 | 19.5 | 2560 |
| minicpm5_2b_q4_A_q4tab_kv8_chunk256 | 1.34 | 0.056 | 1.64 | 1.93 | 1420 | 20.2 | 2560 |
| minicpm5_2b_q4_B_stream_cpu_tab_kv4_chunk256 | 0.05 | 0.028 | 0.32 | 0.59 | 1328 | 10.1 | 2560 |
| minicpm5_2b_q4_C8_stream_q8tab_kv8_chunk256 | 0.60 | 0.056 | 0.93 | 1.22 | 1295 | 16.9 | 2560 |
| minicpm5_2b_q4_C_stream_q4tab_kv4_chunk256 | 0.34 | 0.028 | 0.61 | 0.90 | 1309 | 14.7 | 2560 |
| minicpm5_2b_q4_head_q4_embed_q4 | 1.34 | 0.110 | 2.27 | 2.57 | 1076 | 24.1 | 2560 |
| minicpm5_2b_q4g128 | 2.12 | 0.110 | 3.05 | 3.42 | 931 | 23.4 | 2560 |
| qwen3_4b_q3_C_stream_q4tab_kv8_chunk256 | 0.29 | 0.192 | 0.94 | 1.24 | 586 | 11.0 | 2560 |
| qwen3_4b_q3g64 | 2.37 | 0.378 | 4.27 | 4.65 | 381 | 13.8 | 2560 |
| qwen3_4b_q4_A_q4tab_kv8_chunk256 | 2.14 | 0.192 | 2.78 | 3.07 | 886 | 13.4 | 2560 |
| qwen3_4b_q4_B_stream_cpu_tab_kv8_chunk256 | 0.11 | 0.192 | 0.74 | 1.01 | 560 | 7.5 | 2560 |
| qwen3_4b_q4_C4_stream_q4tab_kv4_chunk256 | 0.31 | 0.097 | 0.80 | 1.09 | 711 | 8.3 | 2560 |
| qwen3_4b_q4_C8_stream_q8tab_kv8_chunk256 | 0.51 | 0.192 | 1.16 | 1.45 | 884 | 11.6 | 2560 |
| qwen3_4b_q4_C_stream_q4tab_kv8_chunk256 | 0.31 | 0.192 | 0.95 | 1.24 | 610 | 11.8 | 2560 |
| qwen3_4b_q4_chunk256 | 2.71 | 0.378 | 3.49 | 3.86 | 896 | 15.4 | 2560 |
| qwen3_4b_q4_kv8 | 2.71 | 0.192 | 4.44 | 4.82 | 661 | 13.2 | 2560 |
| qwen3_4b_q4_stream | 0.89 | 0.378 | 2.80 | 3.17 | 690 | 12.7 | 2560 |
| qwen3_4b_q4_tab_q4 | 2.14 | 0.378 | 4.04 | 4.33 | 671 | 15.4 | 2560 |
| qwen3_4b_q4g128 | 2.71 | 0.378 | 4.61 | 4.98 | 622 | 15.1 | 2560 |
| qwen3_4b_q8_A8_q8tab_kv8_chunk256 | 4.26 | 0.192 | 4.92 | 5.22 | 768 | 12.4 | 2560 |
| qwen3_4b_q8_B8_stream_cpu_tab_kv8_chunk256 | 0.21 | 0.192 | 0.85 | 1.12 | 613 | 6.4 | 2560 |
| qwen3_4b_q8_C8_stream_q8tab_kv8_chunk256 | 0.62 | 0.192 | 1.25 | 1.54 | 486 | 9.3 | 2560 |
| qwen3_4b_q8_stream | 0.99 | 0.378 | 2.89 | 3.26 | 542 | 9.9 | 2560 |
| qwen3_4b_q8g64 | 4.64 | 0.378 | 6.54 | 6.92 | 648 | 14.1 | 2560 |

Reading the table:
- `base` = bf16 tables on GPU, full 2,048-token prefill in one pass, bf16 KV, all layers resident. Granite 4.1 3B int4: weights 2.19 GB but **4.53 GB at peak**: about 2 GB of that is prefill activations (attention scores and MLP intermediates over 2,048 tokens), and 0.21 GB is KV. The "4 GB minimum" in the docs is right; the CEO's "under 1 GB" is not reachable by weights alone.
- `chunk256` (prefill in 256-token chunks): 4.53 to 3.17 GB, identical greedy output.
- `embed_cpu`, `head_cpu`, `stream`: identical greedy output to base (checked, `runs/equiv_granite3b_q4.json`).
- Tied tables: Granite 4.1 and Qwen3 tie embedding and lm_head, so one table serves both and quantising it to int8/int4 (`head_q8_embed_q8`, `head_q4_embed_q4`) removes 0.25 to 0.6 GB. MiniCPM5 and Granite 4.2 are untied and carry two tables (1.07 GB bf16 for MiniCPM5), so the same lever removes 0.8 GB there. Quantising only the head while the tied table stays on the GPU for embedding (`head_q8`) adds memory instead of saving it.
- `kv8` halves KV; `kv4` quarters it but **destroys output on Qwen3-4B (0% on every probe metric) and diverges on Granite**: per-token per-head absmax int4 is not usable on these models (key outlier channels).
- `stream` (two-slot ring on a side stream, prefetching the next layer from pinned RAM): weights 2.19 to 0.60 GB for Granite 4.1 (the 0.5 GB bf16 table remains), decode 19.9 to 14.8 tok/s.
- `B` (CPU tables) is the smallest VRAM and the slowest: the lm_head matmul runs on the CPU in bf16 (7 to 10 tok/s).
- Speed is bounded by Python launch overhead and the Triton kernel (about 100 to 175 GB/s effective on the shared card), not by the 322 GB/s memory bandwidth; no CUDA-graph capture was attempted.

Older torch-only rows for the deleted 3-bit and 2-bit bakes (torch allocator metrics; no whole-process column; measured with the same kernel):

| config | weights GB | peak GB (torch reserved) | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| granite3b_q2g64 | 1.50 | 3.49 | 524 | 20.1 |
| granite3b_q3g64 | 1.89 | 3.88 | 545 | 16.7 |
| granite42_3b_q2g64 | 2.01 | 3.99 | 643 | 19.4 |
| granite42_3b_q3g64 | 2.40 | 4.38 | 504 | 16.7 |
| minicpm5_2b_q2g64 | 1.69 | 2.62 | 986 | 24.8 |
| minicpm5_2b_q3g64 | 1.94 | 2.86 | 823 | 21.4 |
| qwen3_4b_q2g64 | 1.91 | 3.84 | 528 | 15.1 |
| qwen3_4b_q3g64 | 2.37 | 4.28 | 443 | 13.3 |

## 4. Blend quality on the held-out probe (probe8: 137 paraphrase, 45 uncovered, 68 greeting/venting items; each row scored once)

Frozen prompt `split4` (see the plan), greedy, ASCII-normalised first line (plan amendment 1). "faithful" is the strict harness grounding gate (`gok`) and is the headline; it needs a reply of at least 8 words that restates at least 60% of the source passage's content words with no new content word, so it rewards near-verbatim restatement and punishes short correct answers. Binomial SE: about 4 points on faithful (n = 137), 3 to 5 points elsewhere; differences under 8 points are not significant.

| config | faithful (strict) % | no-new-content % | hedge when uncovered % | blend coherent % | chat answered % | chat coherent % | decline % | emergency-text FP % | distinct-2 | opener-3 repeat % |
|---|---|---|---|---|---|---|---|---|---|---|
| adamt_r2 | 14.6 | 51.1 | 0.0 | 91.2 | 33.8 | 92.7 | 5.9 | 8.8 | 0.743 | 8.8 |
| adamt_r3 | 59.9 | 61.3 | 86.7 | 100.0 | 14.7 | 95.6 | 1.5 | 2.9 | 0.665 | 35.3 |
| granite3b_q3g64 | 15.3 | 70.1 | 11.1 | 72.0 | 2.9 | 10.3 | 7.3 | 0.0 | 0.933 | 5.9 |
| granite3b_q4g128 | 20.4 | 21.2 | 20.0 | 98.4 | 82.3 | 98.5 | 1.5 | 0.0 | 0.711 | 57.4 |
| granite3b_q8g64 | 31.4 | 32.1 | 97.8 | 99.5 | 80.9 | 98.5 | 2.9 | 0.0 | 0.674 | 58.8 |
| granite42_3b_q4g128 | 40.9 | 48.2 | 95.6 | 100.0 | 60.3 | 100.0 | 14.7 | 0.0 | 0.515 | 67.7 |
| minicpm5_2b_q4g128 | 43.1 | 54.0 | 95.6 | 97.2 | 45.6 | 97.1 | 25.0 | 0.0 | 0.593 | 52.9 |
| qwen3_4b_q3g64 | 60.6 | 65.0 | 91.1 | 98.9 | 66.2 | 100.0 | 2.9 | 0.0 | 0.568 | 60.3 |
| qwen3_4b_q4g128 | 67.9 | 68.6 | 95.6 | 100.0 | 72.1 | 100.0 | 0.0 | 0.0 | 0.632 | 51.5 |
| qwen3_4b_q4g128_F_tab_q4_kv8 | 67.2 | 68.6 | 93.3 | 99.5 | 76.5 | 100.0 | 0.0 | 0.0 | 0.629 | 52.9 |
| qwen3_4b_q4g128_kv4 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 1.5 | 0.0 | 0.0 | 1.000 | 95.6 |
| qwen3_4b_q4g128_kv8 | 70.8 | 72.3 | 95.6 | 100.0 | 79.4 | 100.0 | 0.0 | 0.0 | 0.642 | 52.9 |
| qwen3_4b_q4g128_tab_q4 | 65.7 | 66.4 | 95.6 | 100.0 | 72.1 | 100.0 | 0.0 | 0.0 | 0.630 | 52.9 |
| qwen3_4b_q4g128_tab_q8 | 67.9 | 69.3 | 95.6 | 100.0 | 70.6 | 100.0 | 0.0 | 0.0 | 0.624 | 51.5 |
| qwen3_4b_q8g64 | 86.9 | 87.6 | 95.6 | 100.0 | 69.1 | 100.0 | 1.5 | 0.0 | 0.658 | 50.0 |
| qwen3_4b_q8g64_C8 | 86.9 | 87.6 | 93.3 | 100.0 | 72.1 | 100.0 | 1.5 | 0.0 | 0.651 | 52.9 |

- ADAM-T rows (`adamt_r2`, `adamt_r3`) run through the `adam-t` binary on the same probe and scorer.
- `*_q8g64` are the near-lossless references (cos 1.0000, 0.5% relative error).
- The two runs scored before the ASCII normalisation (`runs_raw`: Qwen3-4B int4 raw 67.2% faithful but 2.9% chat-coherent because of emoji; Granite 4.1 int4 raw 18.2% faithful) are not in the table.
- `tab_q8`/`tab_q4` quantise embedding and lm_head, `kv8`/`kv4` the cache, `F_tab_q4_kv8` both, `C8` is the recommended configuration (8-bit layers, int8 tables, int8 KV, chunked prefill; layer streaming does not change numerics).
- Qwen3-4B int4: kv8 and int8 tables are free; int4 tables cost about 2 points (inside noise); kv4 is broken.
- The Granite models lose most of their blend quality below int8 and Granite 4.1 hedges only 20% of uncovered questions at int4 (98% at int8). Qwen3-4B is the only candidate above ADAM-T round 3 on the strict gate at int4 or better.
- Phrasing diversity: the open models repeat their 3-word opener more often (50 to 68% of greeting/venting replies) than ADAM-T r2 (8.8%) and r3 (35%). Distinct-2 is similar (0.63 to 0.66 vs 0.67 to 0.74).

### System prompt selection on the dev set (65 own CC0 items; chosen with Granite 4.1 int4, then frozen)

| config | faithful (strict) % | no-new-content % | hedge when uncovered % | blend coherent % | chat answered % | chat coherent % | decline % | emergency-text FP % | distinct-2 | opener-3 repeat % |
|---|---|---|---|---|---|---|---|---|---|---|
| granite3b_q4_blend | 13.5 | 59.5 | 78.6 | 100.0 | 35.7 | 100.0 | 50.0 | 0.0 | 0.781 | 50.0 |
| granite3b_q4_blend2 | 16.2 | 62.2 | 71.4 | 100.0 | 57.1 | 100.0 | 35.7 | 0.0 | 0.863 | 28.6 |
| granite3b_q4_blend3 | 10.8 | 54.0 | 100.0 | 100.0 | 42.9 | 100.0 | 42.9 | 0.0 | 0.785 | 42.9 |
| granite3b_q4_blend4 | 2.7 | 37.8 | 100.0 | 100.0 | 50.0 | 100.0 | 42.9 | 0.0 | 0.797 | 35.7 |
| granite3b_q4_copyA | 13.5 | 64.9 | 35.7 | 98.0 | 57.1 | 100.0 | 0.0 | 0.0 | 0.852 | 28.6 |
| granite3b_q4_copyB | 10.8 | 83.8 | 35.7 | 100.0 | 57.1 | 100.0 | 0.0 | 0.0 | 0.852 | 28.6 |
| granite3b_q4_copyC | 13.5 | 56.8 | 57.1 | 100.0 | 57.1 | 100.0 | 0.0 | 0.0 | 0.852 | 28.6 |
| granite3b_q4_split | 8.1 | 37.8 | 85.7 | 96.1 | 21.4 | 100.0 | 14.3 | 0.0 | 0.774 | 42.9 |
| granite3b_q4_split2 | 16.2 | 59.5 | 14.3 | 100.0 | 42.9 | 100.0 | 7.1 | 0.0 | 0.864 | 42.9 |
| granite3b_q4_split3 | 8.1 | 32.4 | 92.9 | 100.0 | 57.1 | 100.0 | 0.0 | 0.0 | 0.825 | 35.7 |
| granite3b_q4_split4 | 16.2 | 59.5 | 35.7 | 100.0 | 57.1 | 100.0 | 0.0 | 0.0 | 0.825 | 35.7 |

`split4` was chosen (best paraphrase faithfulness 16.2% with no chat declines). The three "copy" variants tried after the freeze did not raise the strict gate: a 3-4 B instruct model answers the question and stops, while the gate wants the whole source restated.

## 5. Quality vs VRAM Pareto (Qwen3-4B-Instruct-2507; probe8 faithful, whole-process peak at 2,560 tokens)

| layers | tables | KV | prefill | layers | peak VRAM GB | decode tok/s | faithful % | hedge % |
|---|---|---|---|---|---|---|---|---|
| int8 | bf16 | bf16 | full | resident | 6.92 | 14.1 | 86.9 | 95.6 |
| int8 | int8 | int8 | chunk 256 | resident | 5.22 | 12.4 | 86.9 (C8 numerics) | 93.3 |
| **int8** | **int8** | **int8** | **chunk 256** | **streamed** | **1.54** | **9.3** | **86.9** | **93.3** |
| int8 | bf16 on CPU | int8 | chunk 256 | streamed | 1.12 | 6.4 | not probed (tables bf16, KV int8) | |
| int4 | bf16 | bf16 | full | resident | 4.98 | 15.1 | 67.9 | 95.6 |
| int4 | int4 | int8 | chunk 256 | resident | 3.07 | 13.4 | 67.2 | 93.3 |
| int4 | int4 | int8 | chunk 256 | streamed | 1.24 | 11.8 | 67.2 | 93.3 |
| int4 | int4 | int4 | chunk 256 | streamed | 1.09 | 8.3 | 0.0 (kv4 broken) | 0.0 |
| int3 | bf16 | bf16 | full | resident | 4.65 | 13.8 | 60.6 | 91.1 |
| int3 | int4 | int8 | chunk 256 | streamed | 1.24 | 11.0 | about 60 (not probed with levers) | |
| int2 | any | any | any | any | n/a | n/a | collapsed (perplexity 475; not probed) | |

The curve: from the unconstrained int8 build (6.92 GB) to the streamed one (1.54 GB) the quality is unchanged and decode speed falls from 14.1 to 9.3 tokens/s; going to int4 saves only 0.3 GB more under streaming and costs 19 points of grounding pass; the floor with CPU tables is 1.12 GB at 6.4 tokens/s. Rule applied ("smallest peak VRAM whose probe quality is within about 2 points of the best candidate's best bit-width"): the best is Qwen3-4B int8 at 86.9%; only int8 layers qualify; the smallest of those is the streamed int8-table configuration C8, 1.54 GB (1.12 GB if 6.4 tokens/s is acceptable).

## 6. Hybrid end to end (locked E2E chat dev suite, 410 items, harness and cognition on, `--tune`, run once per configuration)

Hybrid = the harness material routes and gates unchanged, `AMNI_BLENDER=atex:<bake>` replacing every `gen()` call, emergency-first routing untouched (14 emergency-route items, identical to ADAM-T). "ADAM-T alone" rows are the strongest ADAM-T configuration: round 2 with harness v4 (the recorded v4 run, `report_adam-t-t30m-sft2-h4_7773_e2e_tune_dry.json`) and round 3 with harness v4 (run in this mission). "Open alone" = same weights, no material, no harness, plain system prompt (`alone`), history only.

| run | answered | good | fact | ctx | coherent | 911 lead | near-miss FP | decline | dup | distinct-2 | blend attempts | blend passes | blend pass rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adam-t-t30m-sft2-h4_7773 | 62.7 | 57.3 | 70.5 | 56.3 | 95.4 | 100.0 | 3.7 | 5.6 | 5.6 | 0.689 | 145 | 25 | 17.2 |
| adam-t-t30m-sft3-h4_6344 | 48.3 | 43.4 | 65.8 | 60.6 | 94.4 | 100.0 | 3.7 | 16.1 | 6.3 | 0.653 | 128 | 27 | 21.1 |
| h1-hybrid-qwen3-4b-q8-C8_0 | 74.4 | 70.2 | 73.8 | 63.4 | 95.1 | 100.0 | 3.7 | 6.8 | 5.1 | 0.682 | 125 | 56 | 44.8 |
| h1-alone-qwen3-4b-q8-C8_0 | 92.2 | 80.7 | 97.3 | 83.1 | 91.0 | 100.0 | 3.7 | 0.2 | 0.2 | 0.783 | 0 | 0 | - |
Other blenders under the same harness and the same lever settings (run once each; Pareto points, not part of the bar):

| run | answered | good | fact | ctx | coherent | 911 lead | near-miss FP | decline | dup | distinct-2 | blend attempts | blend passes | blend pass rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h1-hybrid-qwen3-4b-q4-C8_0 | 72.7 | 67.8 | 74.5 | 59.2 | 94.4 | 100.0 | 3.7 | 7.8 | 5.9 | 0.665 | 125 | 48 | 38.4 |
| h1-hybrid-granite41-3b-q4-C8_0 | 75.6 | 68.8 | 73.8 | 62.0 | 91.7 | 100.0 | 3.7 | 10.0 | 8.8 | 0.700 | 125 | 49 | 39.2 |

| blender (streamed, int8 tables, int8 KV, chunked prefill) | whole-process peak VRAM | decode tok/s | probe8 faithful | E2E answered | E2E fact | E2E coherent | E2E decline | E2E blend pass |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B int8 (recommended) | 1.54 GB | 9.3 | 86.9% | 74.4% | 73.8% | 95.1% | 6.8% | 44.8% |
| Qwen3-4B int4 | 1.45 GB | 11.6 | 67.9% (base; int8 tables 67.9%, int8 KV 70.8%) | 72.7% | 74.5% | 94.4% | 7.8% | 38.4% |
| Granite 4.1 3B int4 (baseline model) | 1.17 GB | 14.3 | 20.4% | 75.6% | 73.8% | 91.7% | 10.0% | 39.2% |
| ADAM-T round 2, harness v4 (CPU) | 0 | n/a | 14.6% | 62.7% | 70.5% | 95.4% | 5.6% | 17.2% |

All three open blenders beat ADAM-T on answered by 10 to 13 points (3 to 4 pooled SE) and are indistinguishable from each other on answered and fact (SE 2.2 and 3.6 points). They differ on coherent (Qwen 94 to 95%, Granite 91.7% because its advice replies run long) and on the blend pass rate (Qwen int8 44.8%, the others 38 to 39%). The large probe8 differences (87% vs 68% vs 20% strict faithful) mostly vanish end to end because the harness gate keeps only grounded blends and falls back to the extractive reply otherwise: with the strict gate, a weaker blender mostly means more extractive replies, not wrong ones. The E2E suite is therefore a weak discriminator between open blenders; probe8 is the sharper one.


Pass bar from the mission ("beats ADAM-T alone AND the open model alone on answered, fact, grounding pass and coherent, with 911 lead at least 0.98 and near-miss FP at most 5%"):

| criterion | hybrid | best ADAM-T alone | open model alone | result |
|---|---|---|---|---|
| answered | 74.4% | 62.7% (r2; +11.7, 3.6 pooled SE) | 92.2% | beats ADAM-T, **loses to open alone** |
| fact | 73.8% | 70.5% (r2; +3.4, 0.6 pooled SE: tie) | 97.3% | tie with ADAM-T, **loses to open alone** |
| grounding pass (blend gate, E2E) | 44.8% (56 of 125) | 17.2% (25 of 145; r3 21.1%) | none: no material, nothing to ground | beats ADAM-T |
| coherent | 95.1% | 95.4% (r2; -0.3, tie) | 91.0% | tie with ADAM-T r2, beats r3 (94.4%) and open alone |
| 911 lead | 100% | 100% | 100% | pass |
| near-miss FP | 3.7% | 3.7% | 3.7% | pass |

**The hybrid does not pass the bar.** It is a clear win over ADAM-T on answered and grounding pass, and neutral on fact and coherent, but a 4 B model answering from its own weights beats it on answered and fact by 18 and 24 points. That gap is the cost of the rule that weights are never a knowledge source: the suite's fact items are mainstream facts the open model already knows, whereas the hybrid answers only what PTEX retrieval and the solvers supply (it hedges the rest). The open model alone has no provenance, cannot learn or forget through the store, and its replies run longer (26.5 mean words vs 21.8 vs ADAM-T 16.6; the scorer calls replies over 60 words incoherent, which is where its 91.0% comes from).

Where the hybrid wins and loses against ADAM-T r2 (answered by category): chat 82% vs 48%, nearmiss 63% vs 30%, explain 73% vs 61%, advice 78% vs 69%, multi-turn 68% vs 61%; know 72% vs 76% (it hedges factual questions with no retrieved material, ADAM-T answered some from its weights). Against r2 the hybrid lost 9 coherent items and gained 8: the losses are 6 know-category hedges that echo the question ("I'm not sure which color that is.") and 3 advice replies of 60 to 67 words (the scorer's limit is 60); the gains are 4 explain, 3 know and 1 chat. A word cap like the harness's 55-word cap on blends would likely recover the advice losses; it was **not** applied, because that would be tuning on the locked suite. It is listed under next steps.

Latency: hybrid median 3.7 s per turn, p90 7.6 s (ADAM-T r2: 0.6 s, p90 2.5 s). The open-alone run: median 3.6 s.

## 7. Recommended configuration

**Qwen3-4B-Instruct-2507, int8 layers (group 64) streamed from pinned RAM, int8 tied embedding/lm_head table, int8 KV cache, prefill in 256-token chunks, frozen `split4` prompt.**

- Bake: `bakes/hybrid1_qwen3_4b_instruct_2507_q8g64/` (4.64 GB, 3 files plus manifest and tokenizer).
- Options: `AMNI_BLENDER=atex:<bake path>` and `AMNI_BLENDER_OPTS='{"sys":"split4","head":"q8","embed":"q8","stream":true,"kv":8,"chunk":256}'`.
- Measured, cold process, 2,048-token prefill plus 512-token decode: **1.54 GB whole-process VRAM** (weights/ring/tables 0.62 GB, KV 0.19 GB, rest activations, HIP context and slack), 486 tokens/s prefill, 9.3 tokens/s decode; 3.9 GB pinned host RAM for the layers.
- Quality: probe8 faithful 86.9%, no-new-content 87.6%, hedge when uncovered 93.3%, blend coherent 100%, chat answered 72.1%, decline 1.5%, emergency-text FP 0%. Hybrid E2E in section 6.
- If 6.4 tokens/s is acceptable, tables on the CPU give 1.12 GB. If the pre-registered probe criterion is replaced by E2E answered and fact alone, the smaller option is the existing Granite 4.1 3B int4 bake in the same streamed configuration: 1.17 GB, 14.3 tokens/s, E2E answered 75.6%, fact 73.8%, but coherent 91.7% and decline 10.0%, and a blender that passes the strict gate 20% of the time on probe8.
- If host RAM is the constraint instead of VRAM: the int4 variant of the same configuration is 1.24 GB VRAM, 1.9 GB pinned RAM and 11.8 tokens/s, at 19 points less grounding pass.

### Against the Granite 4.1 3B int4 baseline (same harness, same measurements)

| | Granite 4.1 3B int4 (existing path) | Recommended | change |
|---|---|---|---|
| whole-process peak VRAM, 2,560 tokens | 4.53 GB | 1.54 GB | -66% |
| weights VRAM | 2.19 GB | 0.62 GB | -72% |
| decode | 19.9 tok/s | 9.3 tok/s | -53% |
| prefill | 688 tok/s | 486 tok/s | -29% |
| bake size on disk (the recommended one also needs 3.9 GB pinned RAM for the streamed layers) | 2.19 GB | 4.64 GB | +112% |
| probe8 strict faithful | 20.4% | 86.9% | +66.5 points |
| hedge when uncovered | 20.0% | 93.3% | +73 points |
| chat answered | 82.3% | 72.1% | -10 points |
| decline on greeting/venting | 1.5% | 1.5% | same |
| opener-3 repeat | 57% | 53% | similar |

End to end, the baseline model in the same streamed configuration (hybrid, Granite 4.1 3B int4, 1.17 GB): answered 75.6%, fact 73.8%, coherent 91.7%, decline 10.0%, blend pass 39.2%; it is within noise of the recommended configuration on answered and fact, and 3.4 points lower on coherent.

Lost against the baseline: decode and prefill speed, host RAM, 10 points of chat topicality. Gained: two thirds of the VRAM, and a blender that actually passes the grounding gate.

## 8. Where this left the paradigm, and why

- **Not GF(17).** The bakes are affine group-quantised bit-planes in safetensors. They are lossy (rel-err 0.5% at 8-bit, 9.7% at 4-bit); only the code pack/unpack is exact. CLAUDE.md forbids standard quantisation as a substitute for the Reffelt decomposition, but no lossless route exists for a third-party bf16 model that fits the VRAM goal. The Reffelt d3/d2/d1/d0 split (4 digits base 17 per fp16 weight) is a lossless 16-bit container, 4x the 8-bit size; it was not built here. A GF(17) page encoding of the group-quant codes is possible and untested.
- **Matmul and not TMU.** The Triton kernel uses `tl.dot` (matrix cores) for GEMM and for the padded GEMV, attention uses torch SDPA, the lm_head in `bf16` and `cpu` modes is a plain matmul, and the CPU table mode is a torch CPU matmul. No `hipTextureObject_t` / `index_select` LUT path was used for weights (only embedding rows use `index_select`).
- **SSD streaming is RAM streaming.** Layers are streamed from pinned RAM, double-buffered on a side stream. NVMe at 6.7 GB/s would limit decode to about 4 tokens/s for Granite-sized layers; not built.
- **Third-party weights as the blender**, as the CEO decided; the prompt forbids new facts and the harness gate and extract fallback are unchanged.
- **Open-model-alone is not Adam.** It is reported only as the strong closed-book reference the bar asked for.

## 9. What failed or fell short

- **Sub-8-bit without calibration.** Int4 loses 19 points of grounding pass on the best candidate, int3 26, int2 collapses everywhere (Granite 4.1 int3 already echoes its input). GPTQ-style calibrated quantisation was started (a draft is kept outside the repo) and not finished or tested.
- **KV int4** (per-token per-head absmax) produces garbage.
- **Strict gate vs concise answers.** The strict `gok` gate fails correct but short answers; prompt variants meant to raise it did not (dev set).
- **Prompt selected on the weakest model.** The 8 dev variants were compared on Granite 4.1 int4 only; Qwen3-4B behaves differently (hedges and restates well with the same prompt), so a Qwen-specific prompt could do better. Not tried: tuning on probe8 or E2E is excluded.
- **Decode speed** is about one third of the bandwidth limit. No CUDA graphs, no fused kernels.
- **Intermittent bad run.** The first run of `scripts/hybrid1_equiv.py` produced garbage for `chunk256`, `kv8`, `kv4` and the stream-combined configuration; the same sequence rerun later, and shorter sequences, gave identical output to base for everything except kv8 (late divergence on one prompt) and kv4 (garbage, as in the probe). The cause of the first bad run was not found (a concurrent GPU job is the only suspect). Every reported probe row was checked for coherence, and none of the rows in the tables shows the symptom.
- **E2E coherent/advice** length losses (above).
- **Qwen3.5-4B, Ministral 3 3B, SmolLM3-3B, Gemma 4 E4B were not run** (see models doc).
- **Scoring noise:** probe8 faithful has about 4 points of binomial SE; several lever comparisons are inside it.

## 10. Next steps

1. Calibrated 4-bit (GPTQ or AWQ) with the same bit-plane kernel; the target is int4 within 2 points of int8, which would cut host RAM and stream traffic by half and lift decode to about 12 tokens/s at the same VRAM.
2. A word cap on blender output (same 55-word rule as the harness blend), and a Qwen-specific system prompt chosen on a dev set built for it, then one new locked E2E run.
3. The CEO decision the E2E exposes: allow the weights as a flagged, provenance-labelled fallback when retrieval and solvers return nothing (it would recover the know/fact gap), or keep PTEX/torus-only and accept the lower fact score.
4. CUDA-graph capture of the decode step and a fused dequant-GEMV for the bandwidth gap; overlap of the layer prefetch with the lm_head.
5. NVMe tier with direct reads into the ring and a reader thread (6.7 GB/s measured).
6. GF(17) pages for the codes (lossless container for the 8-bit codes) so the bake lives in the PTEX format.

## 11. Reproduce

```
python scripts/hybrid1_fetch.py Qwen/Qwen3-4B-Instruct-2507 cdbee75f17c01a7cc42f958dc650907174af0554 raw/qwen3
python scripts/bake_hybrid1.py --src raw/qwen3 --out bakes/qwen3_q8g64 --bits 8 --gs 64
AMNI_BLENDER=atex:bakes/qwen3_q8g64 AMNI_BLENDER_OPTS='{"sys":"split4","head":"q8","embed":"q8","stream":true,"kv":8,"chunk":256}' AMNI_ADAM_T_HARNESS=1 AMNI_COGNITION=1 python scripts/adam_t_eval.py run --model proto --suites chat --e2e --tune --proto-cmd "<adam-t binary> <model.gf17>"
python scripts/hybrid1_run.py --bake bakes/qwen3_q8g64 --opts '{"sys":"split4"}' --set probe --out probe.json
python scripts/hybrid1_measure.py --bake bakes/qwen3_q8g64 --opts '{...}' --out meas.json
```

Environment: `docs/adam_t/hybrid1_env.txt` (torch 2.14.1+rocm7.2 wheels, Python 3.12, HIP device 0 = Navi 44 gfx1200; HIP device 1 is the iGPU). Raw run JSONs: `data/adam_t/eval/hybrid1_runs/`.
