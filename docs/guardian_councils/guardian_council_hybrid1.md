# Guardian Council: HYBRID-1 (open dense blender, ATEX group-quant, minimum VRAM)

Question: how do we serve an Apache-2.0 dense open model as Adam's blender in the least VRAM without MoE and without heavy quality loss, while staying as close to the paradigm as the hardware and time allow?

## Architect
Keep ATEX as the storage and serve layer: per-tensor bit-plane codes plus per-group affine parameters, decoded inside the kernel, never materialised as a bf16 weight tensor. Generalise the existing packed int4 kernel into one kernel parameterised by bit-planes (2, 3 = 2+1, 4, 8) instead of one kernel per width. Reuse HF module graphs for attention and norms so any dense Llama-family model drops in.

## Sentinel
No third-party weight may become a knowledge source: the blender prompt forbids new facts, the grounding gate stays the judge, and `AMNI_BLENDER` is off by default. Licences are read from each card and recorded; only Apache-2.0 and MIT are accepted. The backend lives behind a flag and never touches `gguf_runtime.py`. Emergency-first routing is unchanged because the blender only replaces the `gen()` call.

## Scholar
Group quantisation below 16 bits is lossy, so GF(17) "cos = 1.0" can only be claimed for the code roundtrip, not for the weights. Use asymmetric per-group scale/offset with an MSE clip search and a least-squares refit: it is calibration-free and cheap, and is the strongest option short of GPTQ/AWQ, which need activation statistics. Report relative error per tensor class. Lower than 4 bits is expected to hurt a 2-4B model; measure instead of assuming.

## Engineer
One model resident at a time, honouring the GPU governor. Measure VRAM from sysfs plus torch's allocator in a cold process. Levers: embedding rows gathered from pinned CPU RAM; lm_head as int8/int4 ATEX or on CPU; a two-slot layer ring prefetched on a side stream from pinned RAM (progressive tiers); int8/int4 KV via a custom cache layer; chunked prefill. Tune Triton tile sizes on a microbenchmark before running quality sweeps so sweeps are not decode-bound.

## Pathfinder
Do the practical candidates first (a Llama-architecture model, a plain Qwen3, the Granite baseline) rather than porting the hybrid linear-attention Qwen3.5 or the multimodal Gemma 4 E4B. If lower bits fail, the honest path is GPTQ-style calibrated quantisation or mixed precision; record it as the next step. Keep the hybrid's quality gate identical to the existing harness gate so results are comparable with the 17% ADAM-T number.

Decision (majority, 4 of 5, Pathfinder dissenting on scope only): generalised bit-plane ATEX kernel, calibration-free affine quantiser, levers measured one at a time, practical candidates first.
