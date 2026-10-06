# HYBRID-1 evaluation plan (fixed before any held-out scoring)

Written 2026-10-05, committed before probe8 or the E2E chat suite was scored by any open model. Results go in `hybrid1_results.md`; this file is the plan and is not edited after the first probe8 run except by dated amendments at the bottom.

## What is being measured

An open dense model (Apache-2.0 or MIT licence only), baked with the ATEX group-quant path, is the BLENDER: it receives the harness material and turns it into a coherent, non-canned reply. Knowledge stays in PTEX and the torus; the open weights are never a knowledge source. The comparison is quality vs VRAM.

## Frozen elements

- **Blend prompt format:** the harness format, `User: <Label>: <content> <Label>: <content> ... <user message>\nAdam:`, material items joined by one space on one User line. The harness `adam_t_client.template()` produces it; the open-model backend (`amni/serve/blender_atex.py`) parses it back into chat turns.
- **System instruction:** one fixed pair, `SYSM['split4']` in `blender_atex.py`: a notes prompt (restate what the notes say in two or three sentences using their wording, add nothing new, say you are not sure when the notes do not answer) used when the last user turn starts with `Label: text`, and a chat prompt (warm one or two sentences; no outside facts; "I'm not sure about that." for factual questions) used otherwise. Chosen on the dev set below, frozen at the commit that carries this file.
- **"Open model alone" condition:** the same weights with `SYSM['alone']` ("warm and concise, reply briefly"), no material and the harness off. It answers closed-book from its own weights. This is the strongest honest "no material" baseline.
- **Decoding:** greedy, no sampling, first line only (stop at newline), at most 133 new tokens (400 bytes / 3).
- **Thinking modes:** disabled (`enable_thinking=False` where the template supports it).
- **Open weights are used as released**, only group-quantised by our baker; no fine-tuning, no distillation.

## Dev set used to choose settings (not scored for claims)

`data/adam_t/eval/hybrid1_dev/` (`scripts/build_hybrid1_dev.py`): 51 blend items (37 paraphrase, 14 uncovered) and 14 greeting/venting items, all written for this mission, CC0-1.0, zero shared 6-grams with probe8, the chat suites, the h4/harness/cognition sets and the concept eval files (checked with `amni.compute.decontam`). Used for: the system prompt (8 prompt variants tried, `split4` chosen: best paraphrase faithfulness 16%, no chat declines), kernel tile sizes, and nothing else. Quantiser settings (group size, clip grid) were set on tensor reconstruction error, not on the probes.

## Held-out sets, each scored once per frozen configuration

1. **probe8 blend** (182 items: 137 paraphrase, 45 uncovered). Never trained on by ADAM-T.
2. **probe8 greeting/venting** (68 items).
3. **Locked E2E chat dev suite** (410 items; `adam_t_eval.py run --model proto --suites chat --e2e --tune`), dev split only. The test split is not touched.

## Metrics (all computed by the existing scorers, unchanged)

From `scripts/adam_t_probe8.py`:
- **faithful_para:** the strict harness grounding check (`gok`: at least 8 words, no new content word of 4 or more letters, no new digit, no colon, covers at least 60% of the source line's content words, at least 60% of reply trigrams occur in material or question, ends with `.` or `!`). This is the headline blend pass rate. It is strict: a correct short answer can fail it by being short.
- **no_new_content_para:** no new content words or digits vs material and question (a looser, purely extractive-faithfulness check; reported next to the strict one, never instead of it).
- **hedge_uncov:** on uncovered items, a decline phrase and no new digits or names.
- **coherent**, **decline rate**, **emergency-text false positives**, **canned rate**, **answered** on greeting/venting.
- **Phrasing diversity:** distinct-2 and repeated 3-word opener rate on greeting/venting.

From the E2E suite (`adam_t_eval.py` `chat_sum`): answered, good, fact, ctx, coherent, 911 lead, near-miss FP, decline, dup rate, per-route.

Binomial standard errors are reported; differences under about 2 pooled SE are called ties.

## Reference baselines (same probes, same scorers)

ADAM-T round 2 (`exports/adam_t/t30m_sft2/model.gf17`) and round 3 (`exports/adam_t/t30m_sft3/model.gf17`, step 6344) through `crates/target/release/adam-t serve`, driven by `adam_t_probe8.py run`. E2E: round 3 with the harness on (the strongest ADAM-T configuration), and the recorded v3/v4 harness numbers for round 2.

## Configuration grid

Candidates: Granite 4.1 3B (baseline), MiniCPM5-2B, Qwen3-4B-Instruct-2507 (the dense stand-in for Qwen3.5-4B, whose hybrid linear-attention layers the ATEX serve path cannot run; see `hybrid1_models.md`). SmolLM3-3B only if time allows.

Bake grid (affine group quantisation, per-group scale and offset, MSE clip search plus least-squares refit):
- 4-bit group 128 (the ATEX int4-group baseline);
- 3-bit group 64 (two bit-planes, 2+1);
- 2-bit group 64.

Lever grid on the best 4-bit candidate (VRAM measured cold, quality scored on probe8 once per setting that changes numerics):
- embedding table off VRAM (CPU, gathered by token row);
- lm_head: bf16 GPU, int8 ATEX, int4 ATEX, CPU;
- layer streaming with a two-slot prefetch ring (progressive tiers from pinned RAM);
- KV cache int8 and int4;
- tied embeddings (a property of the model, reported; not altered);
- chunked prefill (activation peak).

Settings that do not change numerics (embedding placement, streaming, chunked prefill) are checked for identical greedy output on a fixed prompt set rather than rescored on probe8.

## Speed and VRAM protocol

Cold process per row. Peak VRAM sampled from `/sys/class/drm/card*/device/mem_info_vram_used` every 20 ms (this includes the HIP context, and any other process on the GPU, so the idle baseline is subtracted) and `torch.cuda.max_memory_allocated`. A 2048-token prefill, then a 512-token decode. Reported: weights VRAM, KV VRAM, peak, prefill tok/s, decode tok/s, context used. GPU etiquette: governor throttle file honoured by a sleep every 8 decoded tokens; one model resident at a time.

## Pass bar for "hybrid beats both"

The hybrid (open model as blender, harness material routes on) must beat ADAM-T alone AND the open model alone (no material) on E2E answered, fact, grounding pass (probe8 faithful_para for the blend step) and coherent, with 911 lead at least 0.98 and near-miss FP at most 5%. The result is reported plainly either way. Selection rule for the recommended configuration: the smallest peak VRAM whose probe8 faithful_para and no_new_content_para are within 2 percentage points of the best candidate's best bit-width.

## Paradigm notes (declared up front)

- Group quantisation below 16 bits is lossy by construction. The "roundtrip cos = 1.0" statement applies only to the code-pack/unpack step (verified exact for every tensor); the tensor-level cos and relative error against the original weights are reported honestly.
- The Triton kernels use `tl.dot` for prefill and for the batched path. Attention uses torch SDPA. The lm_head in the bf16 and CPU modes is a plain matmul. These are departures from "TMU first" and are listed in the results.
- Weights are not stored as GF(17) RGBA PTEX pages here; they are stored as group-quant bit-planes in safetensors. A GF(17) page encoding of the codes is possible but not done in this mission.
