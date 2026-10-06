# BLENDER-2 candidate models: licences, dates, architectures, revisions

Verified 2026-10-06 from each model card (README), `config.json`, the repository `LICENSE` file where present, and the Hugging Face API (`/api/models/<repo>`: `sha`, `createdAt`, `lastModified`, `safetensors.total`). Official repositories only. Downloads are pinned to the revision hashes below, fetched with `scripts/hybrid1_fetch.py` (direct `resolve/<rev>` URLs, no HF cache, no `trust_remote_code`). CEO rules: dense (no MoE), at most about 4B, Apache-2.0 or MIT, blender role only (never a knowledge source); attribution in `NOTICE`. CEO priority update 2026-10-06: Granite 4.2 3B first, MiniCPM5-2B second, the Qwen3.5 Gated DeltaNet port third.

## Candidates run

| model | params (from the HF API) | licence (as read) | architecture | dates (HF created / last modified; card release date) | revision | ATEX path | status |
|---|---|---|---|---|---|---|---|
| ibm-granite/granite-4.2-3b | 3.66 B | Apache-2.0 (card front matter and "License" row) | dense `GraniteForCausalLM`, 40 layers, GQA 40 q / 8 KV heads, head_dim 64, untied lm_head, `<think>` template (disabled) | 2026-08-07 / 2026-09-02; card "Release Date: August 25, 2026" | `e459acceac81e5fe67c07d9cfc72329a332e7eb1` | yes, unchanged | priority 1 |
| openbmb/MiniCPM5-2B | 2.52 B | Apache-2.0 ("This repository and MiniCPM model weights are released under the Apache-2.0 License.") | dense `LlamaForCausalLM`, 42 layers, 16 q / 2 KV heads, head_dim 128, untied | 2026-09-06 / 2026-09-29 | `f97400052a43d642bbc6e9975e2397e3ae6a6b52` | yes, unchanged | priority 2 |
| Qwen/Qwen3.5-2B | 2.27 B incl. vision tower | Apache-2.0 (`LICENSE` is the Apache 2.0 text; card `license: apache-2.0`) | hybrid: 24 layers, 3 of every 4 Gated DeltaNet (16 QK / 16 V linear heads, head 128, conv 4), 1 of 4 gated full attention (8 q / 2 KV heads, head_dim 256, partial rotary 0.25), tied; `Qwen3_5ForConditionalGeneration` with a 24-block vision tower and one MTP layer | 2026-02-28 / 2026-03-02 | `15852e8c16360a2fea060d615a32b45270f8a8fc` | after the BLENDER-2 port (text-only bake) | priority 3 |
| Qwen/Qwen3.5-4B | 4.66 B incl. vision tower | Apache-2.0 (same) | hybrid: 32 layers, same 3:1 layout (16 QK / 32 V linear heads), full attention 16 q / 4 KV heads, head_dim 256, tied | 2026-02-27 / 2026-03-02 | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | after the port | priority 3 |

## Searched, not run (newer than 2026-06 or otherwise considered)

Search: the Hugging Face API sorted by creation date for the Qwen, google, microsoft, HuggingFaceTB, ibm-granite, mistralai, allenai, openbmb, swiss-ai, baidu, zai-org, nvidia, arcee-ai, ServiceNow-AI, tencent, internlm, deepseek-ai, stepfun-ai, meta-llama, LiquidAI, XiaomiMiMo, inclusionAI, moonshotai, amd, apple, JetBrains, PleIAs, tiiuae, CohereLabs and stabilityai organisations, keeping releases from 2026-06 on.

| model | released (HF) | licence | why not run |
|---|---|---|---|
| Qwen3.8 (any dense <= 4B) | none exists | n/a | The Qwen org lists only Qwen3.8-27B (+FP8, Apache-2.0, dense hybrid with vision, 2026-08-05), Qwen3.8-Flash-Next (+FP8, licence "other") and Qwen3.8-2.4T-A95B (+FP8, MoE, licence "other"). No Qwen3.8 model of 4B or less. The 27B is out of scope (VRAM and disk). |
| ibm-granite/granite-swash-2b (rev `af1e3227100b`) | 2026-07-01 (card: July 7, 2026) | Apache-2.0 | Base model, not instruction-tuned ("a broad, general-purpose ... base model ... early exploration and small-scale preview"); sliding-window + learnable sinks that SDPA cannot express (card: "SDPA ... is not a supported backend"). |
| ibm-granite/granite-swash-3b-a600m | 2026-07-01 | Apache-2.0 | MoE (excluded). |
| microsoft/FrogNano-4B-2609 (rev `b90468c11a91`) | 2026-09-17 (card: 22-SEP-2026) | MIT on the repo tag, "Apache License 2.0" in the card table | Qwen3.5-4B fine-tune for repository-level coding agents (RL on SWE tasks); a chat blender is outside its trained use, and the licence statements disagree. Same architecture as Qwen3.5-4B, so the port would serve it. |
| allenai/tmax-2b / tmax-4b | 2026-06-17 | Apache-2.0 | Qwen3.5 fine-tunes for terminal agents (DPPO). Not general chat. |
| internlm/Intern-Decision-0.8B/2B/4B | 2026-09-26 | Apache-2.0 | Qwen3.5-derived vision-language decision models. |
| Qwen/Qwen-Drive-1.0-4B | 2026-08-27 | Apache-2.0 | Driving-planning VLM (`QwenDriveForPlanning`). |
| microsoft/Fara1.5-4B | 2026-07-17 | MIT | Computer-use VLM (Qwen3.5-4B based). |
| mistralai/Shieldstral-1.0-3B | 2026-07-16 | Apache-2.0 | Safety classifier on a Mistral3 multimodal wrapper. |
| LiquidAI/LFM2.5-2.6B | 2026-07-28 | "other" (LFM licence) | Licence. |
| CohereLabs/tiny-aya-* | 2026-09 | CC-BY-NC-4.0 | Licence. |
| google/gemma-4-E2B/E4B QAT variants | 2026-06-01 to 06-05 | Apache-2.0 tag | Re-releases of the March 2026 Gemma 4 E-series (separate use policy, per-layer embedding tables; excluded in HYBRID-1 for the same reasons). |
| PleIAs/Pleias-SLM-RAG | 2026-07-08 | Apache-2.0 | A 300M GGUF RAG server, not a 2-4B blender. |

No newer dense, instruction-tuned, general-chat model of at most 4B under Apache-2.0 or MIT was found beyond the four above.
