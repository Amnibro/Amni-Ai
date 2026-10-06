# BLENDER-2 evaluation plan (fixed before any BLENDER-2 dev or probe8 scoring)

Written 2026-10-06, before any configuration of this mission was scored on the dev set or on probe8. Results go in `blender2_results.md`. Changes after the first scoring run are dated amendments at the bottom, each with its effect stated.

## What changed from HYBRID-1 and why

The CTO audit of HYBRID-1 found that the recommended pairing (1.54 GB at 86.9% grounding) joined a VRAM number and a quality number from different process configurations, that streamed-vs-resident equivalence was checked on one model with three short prompts, that an intermittent garbage run was never explained, and that the sub-1 GB configuration had no quality measurement. This plan removes each of those gaps:

1. **One process per configuration row.** `scripts/hybrid1_run.py --meas --ppl --set dev,probe` loads the engine once with the frozen options, then in the same process: measures VRAM and speed (2,048-token prefill plus 512-token greedy decode on fixed random ids), computes perplexity through the engine's own prefill path (same cache, chunking and tables), scores the dev set, then scores probe8. A whole-process VRAM sampler (`/proc/self/fdinfo`, `drm-memory-vram`, 20 ms) runs from before the load to the end and reports the peak per phase. Every quality number in the results cites the JSON of the same process as its VRAM number.
2. **Equivalence before claims.** For every reported model, the deployed configuration is compared token for token with the resident configuration of identical numerics (`scripts/hybrid1_equiv.py --set dev`, 136 prompts, 48 new tokens, greedy), reporting identical-prompt count, each divergence step and the logit max-abs-diff up to and including the divergence step.
3. **Determinism.** The HYBRID-1 config sequence is run in at least 5 fresh processes before and after the fix.

## Frozen elements (unchanged from HYBRID-1)

Blend prompt format, the `split4` system prompt pair, greedy decoding, first line only, at most 133 new tokens, thinking disabled (`enable_thinking=False`), ASCII normalisation of the reply (HYBRID-1 amendment 1), the scorers in `scripts/adam_t_probe8.py`, unchanged.

## Configurations (fixed now)

Per candidate model and per bake (int8 group 64 calibration-free; int4 group 128 GPTQ-calibrated on 128 x 512 tokens of the ADAM-T SFT v8 train split, seed 11, damp 0.01):

- **S (streamed):** layers streamed from pinned RAM (two-slot ring), embedding and lm_head int8 on the GPU, int8 KV, prefill in 256-token chunks. `{"stream":true,"head":"q8","embed":"q8","kv":8,"chunk":256}`
- **M (minimal VRAM):** layers streamed, embedding and lm_head on the CPU, int8 KV, 256-token chunks. `{"stream":true,"head":"cpu","embed":"cpu","kv":8,"chunk":256}`
- **R (resident reference, equivalence only):** S or M numerics without streaming.
- **Sub-1 GB:** every S/M row whose measured peak is below 1 GB is reported with its quality. If a model has no row below 1 GB, one extra row is run with the further levers that HYBRID-1 used for its 0.79 GB figure (CPU tables, int4 KV), `{"stream":true,"head":"cpu","embed":"cpu","kv":4,"chunk":256}`, and reported with its quality or marked failing.

## Dev set (selection only)

`data/adam_t/eval/hybrid1_dev/`: the 65 HYBRID-1 items plus 68 written for BLENDER-2 (40 paraphrase, 12 uncovered, 16 greeting/venting), CC0-1.0, 6-gram clean against the shared tree's eval files (`scripts/build_hybrid1_dev.py`, `meta.json`). Totals: 77 paraphrase, 26 uncovered, 30 greeting/venting.

## Selection rule (dev set only; probe8 is never used to choose)

Let F* be the best dev `faithful_para` over all configurations of all candidates. A configuration is eligible if, on the dev set: `faithful_para` >= F* - 5 points; `no_new_content_para` >= (best) - 5 points; `hedge_uncov` >= 80%; blend `coherent` >= 95%; chat `decline` <= 5%; and its perplexity is at most 1.15 x the perplexity of the same model's int8 S row. The recommended configuration is the eligible one with the smallest measure-phase whole-process peak VRAM; ties within 0.05 GB go to the higher decode speed. The probe8 numbers of every configuration are reported, scored once each, and are never used to choose.

## Reported per row

Whole-process peak VRAM (measure phase, and the maximum over the whole run), torch allocator peak, KV/state bytes at 2,560 tokens, pinned host RAM, prefill and decode tokens/s, perplexity (dev text, 1,535 tokens, through the engine), dev and probe8: faithful (strict), no-new-content, hedge when uncovered, blend coherent, chat answered, chat coherent, decline, emergency-text false positives, distinct-2, opener-3 repeat. The greeting/venting probe is the probe8 `greet_vent_probe.jsonl` (68 items) and the dev `greet_vent_probe.jsonl` (30 items).

## Not run

The locked E2E suite (the CTO runs it after STORE-1 lands).

## Amendments

- **2026-10-06 11:15, amendment 1 (engine v2, numerics unchanged).** The first row (Granite 4.2 int8 S, engine v1) peaked at 1.774 GB in the measure phase with only 1.138 GB allocated by torch: about 0.34 GB of caching-allocator slack, caused by the int8 KV layer re-allocating its whole cache with `torch.cat` on every decoded token, plus a 1.7 GB load-time spike from quantising the tables in 8,192-row chunks. Changes: the KV layer now grows in 256-token blocks and writes in place; tables are quantised in 1,024-row chunks. Effect on numerics: none; Granite int8 S/M/resident produce token-identical output to engine v1 on all 136 dev prompts and the streamed configurations stay bit-identical to resident on 144 prompts including 8 multi-chunk prompts of up to 664 tokens (`equiv_granite42_q8_A2.json`, `equiv_granite42_q8_B2.json`, field `toks_sha1`). Effect on reporting: every row is re-run on engine v2; the v1 row is kept as `v1_dev_granite42_q8_S.json` / `v1_probe_granite42_q8_S.json` and not used in the tables. Its probe8 numbers were seen before this change; since the numerics are identical, the v2 rerun of the same configuration is a consistency check, not a new choice.
