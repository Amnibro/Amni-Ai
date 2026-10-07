# Guardian Council: BLENDER-2 (current-generation open blender, VRAM and quality measured together)

Question: which current (2026) dense open model of at most about 4B, Apache-2.0 or MIT, gives Adam's blender the least VRAM without heavy quality loss, and how do we make every VRAM number and every quality number come from the same process configuration?

## Architect
Keep one engine (`amni/serve/blender_atex.py`) and one kernel (`amni/inference/qgroup_gemv.py`). Extend them; do not fork a second backend for the hybrid linear-attention models. The layer ring should move one flat byte block per layer (one copy, any layer shape), which serves the uniform dense models and the mixed DeltaNet/attention stack alike. Linear-attention state lives in the same cache object as the KV so VRAM is measured the same way.

## Sentinel
The audit found pairing errors last time. Every reported quality row must be produced by the exact options string that produced its VRAM row, in the same process (the probe runner records whole-process peak VRAM from fdinfo while it scores). Nothing is tuned on probe8; prompt or option choices come from the dev set only. The flag stays off by default. No `gguf_runtime.py` changes. Licences read from each card; MIT/Apache only.

## Scholar
Calibration-free 4-bit lost 19 points on Qwen3-4B. Calibrated GPTQ (Hessian from dev-text activations, same affine group format so the kernel is unchanged) is the standard fix and must be measured, not assumed. Perplexity on own text is a loss proxy only. The intermittent garbage must be explained by a mechanism that is shown to produce it, not by a suspect.

## Engineer
Reproduce first: the full HYBRID-1 config sequence in fresh processes, at least five times, with prompts long enough that chunked prefill actually chunks (the old prompts were shorter than one 256-token chunk, so `chunk256` was never exercised). Audit the stream ring for wrap-around and event ordering, the allocator config, and KV slot reuse. One model resident at a time; short GPU bursts; throttle file honoured.

## Pathfinder
Do Granite 4.2 3B first (CEO priority, no port), then MiniCPM5-2B, then the Qwen3.5 port inside a hard one-day box. The DeltaNet state is fixed-size, so Qwen3.5's cache should be far smaller than a full KV; if the port cannot be finished in the box, report exactly what is missing. Qwen3.8 has no dense model of 4B or less on the Qwen org (checked), so the port's value is the 2B/4B Qwen3.5 pair and future Qwen layouts.

Decision (5 of 5): verification first (determinism root cause, equivalence harness, same-process VRAM+quality recorder), then the candidates in CEO priority order, GPTQ for int4, the flat-block layer ring as the one streaming path for all layer types.
