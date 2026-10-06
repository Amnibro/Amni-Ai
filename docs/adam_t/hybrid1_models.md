# HYBRID-1 candidate models: licences, revisions, bakeability

Verified 2026-10-05 from each model card, `config.json` and the Hugging Face API (official repos only; downloads pinned to the revision hashes below, no HF cache, no `trust_remote_code`). The CEO rule for this mission: Apache-2.0 or MIT, no use/naming/distillation restrictions, weights never a knowledge source. Attribution is in `NOTICE`.

## Candidate table

| model | params | licence (as stated) | architecture | released | revision | bakeable by the current ATEX path? | decision |
|---|---|---|---|---|---|---|---|
| ibm-granite/granite-4.1-3b | 3.40 B | Apache-2.0 (card + README) | dense GQA (40 layers, 8 KV heads), tied embeddings | 2026-04-06 | `c0650403e44e78ec0262dab1c90914c65b196c4e` | yes (baseline) | baked, measured, scored |
| ibm-granite/granite-4.2-3b | 3.66 B | Apache-2.0 | same `GraniteForCausalLM` as 4.1, untied lm_head, thinking template (disabled) | 2026-08-07 | `e459acceac81e5fe67c07d9cfc72329a332e7eb1` | yes, unchanged code path | baked, measured, scored (found in own search) |
| openbmb/MiniCPM5-2B | 2.52 B | Apache-2.0 (card header + "License" section: repository and weights) | plain `LlamaForCausalLM` (42 layers, 2 KV heads, head_dim 128), untied | 2026-09-06 (card edited 2026-09-29) | `f97400052a43d642bbc6e9975e2397e3ae6a6b52` | yes | baked, measured, scored |
| Qwen/Qwen3-4B-Instruct-2507 | 4.02 B | Apache-2.0 (LICENSE file is the Apache text) | dense `Qwen3ForCausalLM` (36 layers, 8 KV heads), tied | 2025-08-05 | `cdbee75f17c01a7cc42f958dc650907174af0554` | yes | baked, measured, scored (dense stand-in for Qwen3.5-4B) |
| Qwen/Qwen3.5-4B | 4.66 B (incl. vision tower) | Apache-2.0 | hybrid: 3 of every 4 layers are Gated DeltaNet linear attention, 1 gated full attention; image-text-to-text | 2026-02-27 | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | no | skipped, see below |
| mistralai/Ministral-3-3B-Instruct-2512 | 3.85 B (0.82 B bf16 + 3.03 B FP8-E4M3) | Apache-2.0 | `Mistral3ForConditionalGeneration` (vision tower), FP8 weights, tied | 2025-10-31 | `b35d4dfe56c142746f54dbd64f579faab2744308` | not without work | skipped, see below |
| HuggingFaceTB/SmolLM3-3B | 3.08 B | Apache-2.0 | dense, GQA, NoPE every 4th layer, tied, thinking template | 2025-07-08 | `a07cc9a04f16550a088caea529712d1d335b0ac1` | probably yes (HF `smollm3` class) | not run: time; 2025 model, behind the three above on recency |
| google/gemma-4-E4B-it | 7.996 B stored (4 B "effective" via per-layer embeddings) | Apache-2.0 on the card, but the licence page points to a separate Prohibited Use Policy and Intended Use Statement | `Gemma4ForConditionalGeneration`, sliding + full attention, audio/vision towers | 2026-03-02 | `ee0ef6023621cff504d758262d4e04895a5af4a2` | no | excluded: use restrictions beyond Apache-2.0 and an 8 B file |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 3.97 B | NVIDIA Nemotron Open Model License (not Apache/MIT) | Mamba2/attention hybrid (`NemotronHForCausalLM`) | 2026-03-07 | `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` | no | excluded: licence and architecture |

## Why the skipped ones are skipped

- **Qwen3.5-4B:** needs a Gated DeltaNet recurrent-state kernel (the HF torch fallback is slow and the ATEX serve path has no linear-attention cache), conv state, a gated-attention variant with head_dim 256, and a multimodal wrapper to strip. Rough cost: a new decode kernel plus cache layer plus parity tests, several days, to serve a model that is only about 0.6 B larger than Qwen3-4B-2507 and whose quality edge for a blend task is unmeasured. Not worth it before the plain dense models are measured.
- **Ministral 3 3B:** the weights are FP8-E4M3 with per-tensor scales plus a vision tower in a `Mistral3` wrapper; baking means dequantising FP8 to float, re-quantising to int4 (a second lossy step), and loading only the text sub-model. About a day. Deferred; Granite 4.2 and MiniCPM5 cover the 2-4 B range with no port.
- **SmolLM3-3B:** likely works through the same loader (class exists in transformers 5.18), but it is a mid-2025 model; time went to Granite 4.2 (newer, same code path).
- **Gemma 4 E4B:** the card says Apache-2.0, but the licence page also links a Prohibited Use Policy that this mission cannot accept without reading and clearing it; the stored file is 8 B parameters because of per-layer embedding tables.

## Licence text recorded

- Granite 4.1 3B and 4.2 3B: model cards read "License: Apache 2.0" with the standard Apache link (https://www.apache.org/licenses/LICENSE-2.0). The 4.1 card states the model is "finetuned from Granite-4.1-3B-Base using a combination of open source instruction datasets with permissive license and internally collected synthetic datasets".
- MiniCPM5-2B: "This repository and MiniCPM model weights are released under the Apache-2.0 License." (README, section License.)
- Qwen3-4B-Instruct-2507: the repository `LICENSE` file is "Apache License, Version 2.0, January 2004".
- No model above restricts naming, use or distillation in the files read. Apache-2.0 requires keeping the licence and notices (done in `NOTICE`).

## Raw weight hashes (for the deletion log)

| model | file | size B | sha256 |
|---|---|---|---|
| granite-4.1-3b | model-00001-of-00002.safetensors | 4991538344 | 895bf5f2d7c8b06ca902499567d3c3d9ed30061e4c5ad94bf8216286ca67e2fd |
| granite-4.1-3b | model-00002-of-00002.safetensors | 1814176448 | de8c9efdaa6f669d595bda8b949213cba92ea69689fd2a27fbceed3d1ebeb2f7 |
| granite-4.2-3b | model-00001-of-00002.safetensors | 5358533712 | db39eb2776b5404dbfd0696cf7fe5ec671f0dad0e0918c62c5aea9e2eb5825d7 |
| granite-4.2-3b | model-00002-of-00002.safetensors | 1960983408 | 8fba6ea17551d5d129eac519dbd70237139221c7fd0b89623ef54d9711062505 |
| MiniCPM5-2B | model-00000-of-00001.safetensors | 5033557096 | 14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c |
| Qwen3-4B-Instruct-2507 | model-00001-of-00003.safetensors | 3957900840 | 75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6 |
| Qwen3-4B-Instruct-2507 | model-00002-of-00003.safetensors | 3987450520 | 0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1 |
| Qwen3-4B-Instruct-2507 | model-00003-of-00003.safetensors | 99630640 | 7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d |

Hashes are the Hugging Face LFS sha256 values for the pinned revisions; the Granite 4.1 pair was also recomputed locally from the downloaded files before deletion and matched.
