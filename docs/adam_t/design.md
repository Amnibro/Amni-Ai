# Adam-T: homegrown ternary language core with PTEX lookup memory

Status: phase 1 (ADAM-T-b, 2026-09-25). This restarts ADAM-T, which was paused because its GPU runs slowed Amni-Gen.

- **Code:** `crates/adam-t`.
- **Outputs:** `exports/adam_t/`.
- **Eval rules:** `docs/adam_t/eval_suite.md`.

## 0. Summary

- **Core:** a pre-norm causal transformer with every linear map ternary. Weights are in {-1, 0, +1} × one scale per matrix, trained by straight-through estimation (STE) on latent float weights.
- **Input:** raw bytes. A hashed byte n-gram memory adds its rows into the residual stream.
- **Memory storage:** the memory lives in GF(17) PTEX pages, and every lookup is addressed by the n-gram hash.
- **Export:** the trained model exports to one `GF17PTX1` file.
  - Trits are packed 10 per pixel: 3^10 = 59049 ≤ 17^4 = 83521, so 10 trits fit the four base-17 digits of one RGBA texel.
  - The GF(17) roundtrip is bit-exact.
- **Inference:** a CPU path that reads only that file needs no multiplies in the linear layers. It uses 243-entry sum tables per 5 inputs, indexed by the stored trit codes. It reproduces the trainer's bpb to within 0.0013.
- **Smoke result** (6.49M-param core, 49.2M training bytes, clean2 hold, 2,560 windows):

| Model | Hold bpb |
|---|---|
| ternary + PTEX memory | 1.840 |
| float, same size | 1.869 |
| ternary | 1.977 |
| Adam hash-field v18 base, same bytes | 2.243 |

- **Where it stands against the references:**
  - SmolLM2-135M scores 1.146 on the same windows. Adam-T is **0.7 bpb behind** it at 1/20 of its size (core only) and about 1/160,000 of its training data. SmolLM2-135M saw about 2T tokens, roughly 8 TB of text.
  - The v18 base needs **4.06 GB** of training to reach 1.783. Adam-T with memory reaches 1.840 after 49 MB.
- **Throughput:**
  - On this shared GPU, Adam-T trains at about **17–25 KB/s** of text under the 60% governor.
  - That is about **1/80 to 1/120 of v18's throughput**. v18 runs at about 3.1 MB/s unthrottled.
  - Compute is the binding constraint for the 120M and 300M sizes (section 8).

## 1. Architecture: bytes or a small vocabulary

### What phase 1 uses

- **Model:** byte input (256 symbols) and a 256-way byte head.
- **Shape:** d=256, L=8, H=4, ctx=256, FF=4d with ReLU², learned absolute positions, no biases.
- **Normalisation:** a parameter-free RMSNorm at the input of every linear map.
- **Size:** 6.49M core parameters.

### Evidence

Evidence 1: homegrown BPE on clean2. We trained on a 60 MB sample of `clean2/train.bin` and measured on 8 MB of `hold.bin`. Results are in `exports/adam_t/vocab_bytes_per_token.json`.

| vocab | 512 | 1024 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|
| bytes/token | 1.94 | 2.39 | 3.25 | 3.68 | 4.08 | 4.38 |

Evidence 2: compute per byte. With the same body, a 4096 vocabulary cuts body FLOPs per byte by 3.25×. The embedding and head cost 2·d·V per token, which matters at small sizes. Share of parameters spent on the vocabulary (embedding plus head) at V=4096:

| Size | d | Vocabulary share |
|---|---|---|
| 10M | 320 | 26% |
| 30M | 512 | 14% |
| 120M | 768 | 5% |
| 300M | 1024 | 3% |

Net FLOPs per byte at the same body:

| Size | Ratio of a 4096 vocabulary to bytes |
|---|---|
| 10M | 0.35× |
| 30M | 0.33× |
| 120M | 0.32× |
| 300M | 0.31× |

So the vocabulary is worth 2.9–3.2× more text per GPU-hour.

Evidence 3: the PTEX memory already gives the byte model multi-byte units without extra FLOPs.
- **Gain:** at the same core, compute and bytes, it takes the ternary model from 1.977 to 1.840 bpb (section 7). That is most of the gain a subword vocabulary gives a small model.
- **Confound:** the memory also adds 50M lookup parameters. The equal-parameter control `smoke_tern_mem12` (6.3M memory parameters) still gains −0.054 bpb, and scaling to 50M gains −0.137 (section 7).

Evidence 4: context. A 256-byte window holds about 45 English words. A 4096 vocabulary at the same ctx sees 3.25× more text, and attention cost stays the same.

### Decision (A/B run 2026-09-25): BPE-4096 from 30M up

We ran the A/B at matched GPU time. Both arms used the same 6.5M-class core (d=256, L=8, ternary + PTEX memory), 3,000 steps × 64 × 256 symbols, and the same schedule.

| arm | memory | training bytes seen | hold bpb | GPU min |
|---|---|---|---|---|
| bytes (`smoke_tern_mem`) | 6 × 2^15 byte n-gram rows | 49.2M | 1.840 | about 26 |
| BPE-4096 (`smoke_bpe_tern_mem`) | 4 × 2^15 token n-gram rows (n = 2, 3, 4, 6) | 162.7M | **1.740** | 26.8 |

**How to read the A/B:**
- **The token model wins by 0.100 bpb at the same compute.** Its 256-token window also spans about 830 bytes, against 256 for the byte model, and reading comprehension needs that span.
- **The eval stays comparable.** It uses the same sampler over hold tokens and bpb = Σ nats / Σ bytes of the target tokens.
- **The export still checks out.** The token model's `.gf17` roundtrip is bit-exact. The LUT bpb is 1.73650 against a GPU bpb of 1.73662, with argmax agreement of 98.7%.

**The tokenizer:**
- **Merges:** 3,840 merges on 60 MB of `clean2/train.bin`, seed 17, using a homegrown regex pre-split: `` ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+``. HF `tokenizers` was used only as a trainer library.
- **Encoder:** our own Rust encoder, `adam-t tokenize` / `bpe.rs`. It is lossless (verified byte-exact).
- **Storage:** the merges are stored in the model's `.gf17` file as the `bpe` table.
- **Compression:** clean2 comes out at 3.25 bytes/token and the ADAM-T stream at 3.47. clean2 train is 1.29B tokens.

**Plan by size:**
- **10M:** stays a reference only.
- **30M:** BPE-4096 plus token n-gram PTEX memory.
- **120M and 300M:** BPE-4096, or BPE-8192 if the 30M run shows the head is not the bottleneck.

## 2. PTEX memory: how it enters the residual stream

- **Tables:** K hashed n-gram tables. Phase 1 uses K=6 with n ∈ {2,3,4,5,6,8}, and each table holds 2^R rows of d values (R=15).
- **Addressing:** at position p, table k is read at row `k·2^R + top_R(hash64(x[p-n+1..=p], n))`. The n-gram is truncated at the window start. The 64-bit hash is keyed by n and length, with a murmur-style finaliser (`hsh`/`hrow` in `crates/adam-t/src/lib.rs`).
  - **The coordinate is the hash**, so nothing is looked up by value: the address alone says which page, texel and channel to read.
- **Injection:** `x_0 = emb[x_p] + pos[p] + Σ_k mem[row_k(p)]`, before block 0. Memory therefore acts like a learned, content-addressed extension of the byte embedding. It costs no FLOPs: K gathers and K−1 adds of d values per byte.
- **Training:** the table is a separate module with its own AdamW, no weight decay and `LR_MEM`. Only the rows that were touched get gradient.
  - The phase 1 optimiser still keeps dense moments.
  - A sparse (row-lazy) optimiser is needed above 2^18 rows per order.
- **Inference:**
  - Rows are read with `Store::get` from the GF(17) file by coordinate (table, row·2d + j), and never loaded as a tensor.
  - The store streams pages from SSD and keeps a byte budget (`Store::open(p, budget)`).
  - In phase 1 each f32 row value is 2 u16 words, or 8 base-17 digits, stored exactly.
- **Planned after phase 1:**
  1. **GF(17)-grid memory.** Rows are trained by STE on a 289-level grid (2 base-17 digits, value = s·(d0 + 17·d1 − 144)). That makes pages 4× smaller than f32. Tiers then become real progressive refinement: d1 (coarse, VRAM) first, then d0 (NVMe).
  2. **Scale-out.** Use 2^18 to 2^20 rows per order on SSD pages. Add a second injection point at layer L/2 with a learned per-channel gate, and add a context-hash table keyed on a document-level rolling hash.
  3. **v18 as a memory channel.** Adam's hash-field v18 base (1.783 bpb) becomes an extra memory row source: its 256-way byte distribution goes through a ternary 256→d projection into the residual. It is homegrown, it is already trained, and it costs one field lookup per byte.

## 3. Ternary STE recipe (as implemented)

For every linear map (qkv, o, ff1, ff2 and the head):

1. **Normalise:** `xn = x / rms(x)`, with no learned gain.
2. **Activation grid:** `m = max|xn|` per row, then `xi = round(clip(127·xn/m))` with STE: `xs + (round(xs) − xs).detach()`.
3. **Ternary weights:** `γ = mean|W|` per matrix. Then `t = ((W − γ/2).sign + (W + γ/2).sign)/2`, which is a threshold at γ/2. STE `a + (t − a).detach()` is applied on `a = W/γ`.
4. **Output:** `y = xi · t · (m·γ/127)`. This is exact integer arithmetic followed by one scale per row.

Training settings and the resulting weights:
- **Optimiser:** AdamW with β2 0.95, weight decay 0.05, grad-clip 1.0, lr 2e-3, 200 warm-up steps, then cosine decay to 0.1×. Batch 64×256 bytes.
- **Latent weights:** they stay float in the trainer only. What gets exported is t and γ.
- **Trits:** 36% come out as zero. The 6.36M trits take 1.29 MB.
- **Float parts:** the embedding, position table and memory are lookup tables and stay float (stored exactly as GF(17)). Attention has no weights, and it runs on the ternary projections.

## 4. Lookup-only LUT / TMU inference

`crates/adam-t/src/lut.rs::Lm` reads only the `.gf17` file:

- **Linear layer:**
  - Activations are put on the same 255-level grid as in training.
  - For each group of 5 inputs, a 243-entry table holds every signed sum Σ±x. Each output is the sum of the table entries indexed by its trit codes, two lookups per u16 code.
  - There are no weight multiplies. The only multiply is the one scale per row.
- **Attention, softmax and RMSNorm:** these remain float arithmetic.
- **Parity against the trainer's GPU logits** on 8 hold windows, 2,048 positions:

| run | GPU bpb | LUT bpb | max \|Δlogit\| | mean \|Δlogit\| | argmax agree |
|---|---|---|---|---|---|
| smoke_tern_mem | 1.83773 | 1.83752 | 0.044 | 0.0040 | 99.90% |
| smoke_tern | 1.92309 | 1.92184 | 0.255 | 0.0143 | 99.27% |

  - **Why the paths differ:** float summation order differs between the GPU and the CPU. That occasionally flips the rounding of one activation by one grid step, and the flip spreads through the 8 layers.
  - **Size of the effect:** the bpb difference is at most 0.0013, far below the 0.005 promotion gate.
- **TMU mapping, next:** each 243-entry table is one texel row, and the trit code is the texel coordinate. A WGSL kernel that uses `textureLoad` over an R32Sint LUT texture is the GPU form. The hand-written WGSL GEMM already runs 20% faster than burn's (section 6), so custom kernels are the inference path.
- **Speed:** the CPU reference runs at 1.2 KB/s on 4 threads. It recomputes whole windows and has no KV cache, so it is a correctness path only. Incremental decoding with a KV cache removes about 256× of that work.

## 5. GF(3) / GF(17) page layout

The container is `adam_gf17` `GF17PTX1`:

- **Pages:** 4096×4096 RGBA pages.
- **Values:** every u16 word becomes 4 base-17 digits (K4 = [1, 17, 289, 4913]), one digit per channel. Digit planes are rANS-coded in 128-px tiles.
- **Table streams:** an `exact` table is one stream. A non-exact table is stored as 4 progressive digit tiers, d3 then d0.

| table | content | encoding |
|---|---|---|
| `cfg` | d, L, H, ctx, FF, R, K, n-gram orders | u32 → 2 words, exact |
| `g` | γ per linear map (33) | f32 → 2 words, exact |
| `w0..w32` | trit codes, row-major per output column | 10 trits per u16, `Σ 3^j (t_j + 1)`, exact; **1.62 bits/trit** against the log2(3) = 1.585 floor, 97.8% efficient before rANS |
| `emb`, `pos` | byte and position embeddings | f32 → 2 words → 8 digits, tiered |
| `mem` | K·2^R × d memory rows | f32 → 8 digits, tiered (becomes a 2-digit GF(17) grid in phase 2) |

**Measured files:**

| run | file size | contents |
|---|---|---|
| smoke_tern | 2.36 MB | tables 1.81 MB, of which trit codes are 1.29 MB |
| smoke_tern_mem | 178 MB | the f32 memory rows dominate, which is why the memory needs the GF(17) grid |

**Roundtrip checks** at export (`lut::finish`) show every table **bit-exact** (`roundtrip_bytes_exact`) and every trit exact after unpacking (`roundtrip_trits_exact`).

## 6. Framework decision: burn (wgpu/Vulkan) for training, hand-written WGSL and CPU LUT for inference

**GEMM microbenchmark** (`adam-t mm`, M=16384, GFLOP/s):

| shape (N×K) | hand-written WGSL f32 | burn wgpu f32 | burn vulkan f16 |
|---|---|---|---|
| 1024×256 | 4895 | 4015 | 4067 |
| 256×1024 | 4823 | 4158 | 4148 |
| 768×256 | 4871 | 3936 | 3929 |

The run was throttled and sampled during the SMART-LOAD window.

**Full training step** (`adam-t bench`, 6.49M core, batch 64×256, GPU time per step, throttled):

| backend | float | ternary | ternary + memory |
|---|---|---|---|
| burn wgpu (Vulkan, fusion, autotune) | 403 ms / 40.6 KB/s | 416 ms / 39.4 KB/s | 536 ms / 30.6 KB/s |
| burn ndarray (CPU) | 2124 ms | 1950 ms | – |
| burn candle (CPU) | 1158 ms | 1762 ms | – |

The CPU rows come from the first agent's `exports/adam_t/cpubench.log`, 4 steps each.

**Batch scaling for the ternary model:**

| batch | time per step | throughput |
|---|---|---|
| 16 | 111 ms | 36.8 KB/s |
| 64 | 416 ms | 39.4 KB/s |
| 128 | 1768 ms | 18.5 KB/s |

At batch 128, naive attention materialises B·H·T² scores, and VRAM is shared with Amni-Gen. The working point is batch 64.

**Decision:**
- **burn 0.21 + wgpu/Vulkan for training.** It gives autodiff, kernel fusion and autotune on RADV today, and it reaches about 85% of hand-written GEMM.
- **candle is rejected.** It has no Vulkan or ROCm backend, so on this AMD card it is CPU-only and 3–4× slower.
- **A custom wgpu trainer is rejected for now.** It would mean writing our own autodiff for a 20% GEMM gain.
- **Hand-written WGSL** is used where burn cannot express the operation: the TMU/LUT inference kernels, and later a fused ternary-STE linear and a flash-style attention for training.
- **burn's f16 path gives no speed-up** here, because it does not use cooperative-matrix instructions. It is also against the lossless rule for stored weights, so it is not used.

## 7. Smoke results (matched training bytes)

**Setup:**
- **Budget:** every Adam-T arm uses 3,000 steps × 64 × 256 = **49.15M training bytes** from `clean2/train.bin`.
- **Sampling:** the same deterministic sampler and seed 0.
- **Schedule:** the same LR schedule.
- **Eval:** 20 batches × 128 windows × 256 bytes = 655,360 scored positions of `clean2/hold.bin` (seed 1234, the adam-train sampler).
- **v18:** uses its own production settings (L22, 21 levels, 4 hops, cell-ternary, bs 128) at 1,500 steps × 32,768 = the same 49.15M bytes. Its eval uses the identical windows.

| arm | core params | memory params | bpb @8.2M | @16.4M | @24.6M | @32.8M | @41.0M | **@49.2M** | exported file |
|---|---|---|---|---|---|---|---|---|---|
| `smoke_tern_mem` (ternary + PTEX memory) | 6.49M | 50.3M | 2.157 | 2.029 | 1.955 | 1.901 | 1.859 | **1.840** | 178 MB |
| `smoke_tern_mem12` (ternary + memory at 2^12 rows, equal parameters) | 6.49M | 6.29M | 2.268 | 2.123 | 2.045 | 1.985 | 1.949 | **1.923** | 24.3 MB |
| `smoke_float` (same core, float) | 6.49M | 0 | 2.788 | 2.278 | 2.065 | 1.966 | 1.901 | **1.869** | – |
| `smoke_tern` (ternary, no memory) | 6.49M | 0 | 3.013 | 2.378 | 2.219 | 2.097 | 2.018 | **1.977** | 2.36 MB |
| v18 hash field, same bytes (`v18_match`) | 352M stored values (L22, 21 levels) | – | 3.176 | 2.569 | 2.385 | 2.302 | 2.262 | **2.243** | – |

**Reading the table:**
- **Ternary costs 0.108 bpb at this size.** The float model reaches 1.869 and the ternary one 1.977, which is the known small-model ternary penalty.
- **Memory more than recovers it.** Ternary + PTEX memory reaches 1.840, 0.029 better than float, and keeps lookup-only inference.
- **The memory helps a lot.** At equal compute, the PTEX memory is worth −0.137 bpb (1.977 → 1.840).
- **The gain isn't just extra parameters.** Memory at 2^12 rows, which adds about as many parameters as the core itself (6.3M), still gives −0.054 (1.923). The gain keeps growing with the number of rows, at no extra FLOPs.
- **Adam-T beats v18 per byte.** At the same bytes it is 0.27–0.40 bpb ahead of the v18 hash field.
- **v18 is far cheaper per byte.** It trains about 100× faster per byte (1,500 steps took 72 s). Its full clean2 base reached 1.783 after 4.06 GB (124k steps), about 45 minutes of wall time. At matched GPU time, v18 is still the better model today. Adam-T's case rests on quality per parameter and per byte, which matters once the bottleneck is how much the model generalises, not how many bytes it has seen.
- **Third-party references on the same windows** (5 batches, 640 windows; `ref_bpb.py`, eval only):

| Model | Hold bpb |
|---|---|
| SmolLM2-135M | 1.146 |
| SmolLM2-360M | 1.070 |
| Qwen2.5-0.5B | 1.136 |

**GPU load during these runs** (1 Hz samples of `/sys/class/drm/card1/device/gpu_busy_percent`; this is total card load, including Amni-Gen and other users):

| run | samples | mean busy | max | fraction >60% | self duty |
|---|---|---|---|---|---|
| b_bench float / ternary / ternary + memory (65 steps each) | 38 / 67 / 65 | 63.9 / 50.6 / 51.0 | 100 | 0.63 / 0.48 / 0.49 | 0.39 / 1.0 / 0.52 |
| b_bench batch 16 / 128 | 17 / 129 | 38.4 / 65.4 | 100 | 0.35 / 0.64 | 0.91 / 0.73 |
| mm GEMM bench | 5 | 21.6 | 72 | 0.20 | – |
| v18_match (adam-train, 78 s) | 79 | 25.2 | 99 | 0.20 | – |
| smoke_tern resume 2500→3000 (50-step throttle reads) | 491 | 48.1 | 100 | 0.45 | 0.33 |
| smoke_float segment 1, 0→1760 (50-step reads) | 1136 | 48.7 | 100 | 0.46 | 0.33–0.37 |
| smoke_float segment 2, 1760→3000 (reads every step) | 747 | 59.3 | 100 | 0.54 | 0.57–0.62 |
| smoke_tern_mem12, 500→3000 (cap raised to 95%, Amni-Gen offline) | 1228 | 91.7 | 100 | 0.96 | 0.91–0.97 |

The first agent's first 2,500 steps of `smoke_tern` and all of `smoke_tern_mem` ran before the fix, with a flat 0.1 s sleep. That gave about 80% self duty, which is the load that slowed Amni-Gen.

## 7b. Phase 2 (30M): what the first bench showed, and the fix

**The first 30M bench stalled.** The config was d=512, L=10, H=8, ctx=640 tokens, bs 24, with 4 × 2^16 memory rows. It ran at **6.2 s/step**, which is about 2.5K tokens/s and 2.6× slower than FLOPs predict. The CEO also paused the machine for lag at the same time.

**The cause is attention memory:**
- burn's autodiff backend uses its naive attention fallback. It materialises B·H·T² scores, the mask and the softmax, and saves them for backward.
- That comes to about 4 × 24 × 8 × 640² floats per layer, about 1.6 GB × 10 layers, which overflows the 16 GB card into GTT spill.

**The fixes (commit 6a55df4):**
- **Micro-batch gradient accumulation (`MICRO`).** It gives an exact match: a 20-step run with MICRO=2 matches MICRO=1 bit-for-bit once the backend RNG is seeded (`B::seed`).
- **A throttle read and sleep after every micro-step**, so the governor's cap applies at a finer grain.
- **ctx 512 tokens (about 1.7 KB)**, with micro-batches of 8 windows. The attention tensors come to about 2.7 GB in total.

**The next kernel step is a hand-written WGSL flash-attention forward/backward.** It removes the T² memory, and it is needed for 120M at ctx ≥ 1K.

**Running the 30M:**
- It runs through `crates/adam-t/campaign.sh`.
- Segments are `SEG` minutes long and each ends with a checkpoint.
- Between segments, the ADAM-T-EVAL harness scores the checkpoint through `adam-t serve <run dir>` (proto 1). The trainer is paused while it runs, so there is still only one GPU job.

## 8. Scaling plan, data plan and compute budget

**Byte-level configurations**, where the core is 12·L·d² plus embeddings:

| size | d | L | H | ctx | FF | core |
|---|---|---|---|---|---|---|
| 10M | 320 | 8 | 5 | 512 | 1280 | 9.9M |
| 30M | 512 | 10 | 8 | 512 | 2048 | 31.5M |
| 120M | 768 | 16 | 12 | 1024 | 3072 | 113M |
| 300M | 1024 | 24 | 16 | 1024 | 4096 | 302M |

Memory at each size is 6–8 orders × 2^16 to 2^20 rows, on SSD pages. It is lookup-only, so it adds no FLOPs.

**Data:**
- **10M and 30M:** `data/clean2` (4.21 GB train, 86.7 MB hold; licences apache/bsd/cc-by-sa/cc0/isc/mit/odc-by/own/us-pd, listed per source in `meta.json`).
- **120M and above:** add FineWeb-Edu from `data/ks/store_clean/fineweb_edu`, which is ODC-By 1.0.
  - It holds 56 KS-coded shards: 87.1 GB of source text, 16.25M documents and 18.7 GB of codes.
  - It is decoded through `adam-ks`.
  - Every shard must pass the 13-gram hold/benchmark contamination filter first. clean2 already contains 18% FineWeb, so overlap with `hold.bin` is a real risk.
  - Mix for 120M: clean2 at 25% and FineWeb-Edu at 75%. For 300M: 15% and 85%.
  - Never used: big2, big3, chat1–4e or Grokipedia.
- **Byte budgets** at about 20 tokens per parameter, with a 3.25 bytes/token equivalent:

| size | budget | data |
|---|---|---|
| 10M | 0.65 GB | one pass of a clean2 subset |
| 30M | 2 GB | clean2 |
| 120M | 8 GB | clean2 + FineWeb-Edu |
| 300M | 20 GB | clean2 + FineWeb-Edu |

**Compute on the shared RX 9060 XT** (16 GB, Vulkan/RADV, governor cap 60%):
- **Measured rate:** 6.49M byte-level ternary, 20–25 KB/s wall under the governor, which is about 1.9 GB/day. Throughput scales as 1/N.

| size | burn today, byte-level | + BPE-4096 (×2.9–3.2) | + custom kernels (goal ×5: fused ternary STE, flash attention, coop-matrix WGSL) | days for the byte budget, best case |
|---|---|---|---|---|
| 10M | 1.25 GB/day | 3.6 GB/day | – | 0.2 d (byte-level 0.52 d) |
| 30M | 0.39 GB/day | 1.13 GB/day | 5.6 GB/day | 0.36 d (burn + BPE: 1.8 d) |
| 120M | 0.11 GB/day | 0.32 GB/day | 1.6 GB/day | 5 d (burn + BPE: 25 d) |
| 300M | 0.041 GB/day | 0.12 GB/day | 0.6 GB/day | 33 d (burn + BPE: 167 d) |

These rates scale only the 6N FLOPs. Longer ctx (512–1024) adds about 15–30% attention cost on top.

**What the budget means:**
- **10M and 30M:** can go ahead now with burn, overnight, under the governor.
- **120M:** needs BPE and the custom kernels, and even then it takes about 5 days of shared GPU.
- **300M:** is not realistic on this shared card. It needs a dedicated GPU window from the CEO. The M6 Mac's Metal backend through the same burn/wgpu code is an alternative, but it is also shared, so that is a CEO decision.
- **Every run follows the same rules:** throttle file, ≤20 minutes per process, checkpoint and resume, and one GPU job at a time.

## 9. Expected quality (honest), protocol-A bpb on clean2 hold

| Adam-T size | bytes trained | expected bpb | vs references |
|---|---|---|---|
| 6.5M smoke (done) | 49 MB | 1.84 | – |
| 10M | 0.65 GB | 1.50–1.60 | beats v18 base (1.78) |
| 30M | 2 GB | 1.32–1.42 | about v18 + 0.4 bpb better |
| 120M | 8 GB | 1.15–1.25 | about SmolLM2-135M (1.146), which trained on 2T tokens |
| 300M | 20 GB | 1.05–1.15 | between SmolLM2-360M (1.070) and Qwen2.5-0.5B (1.136) |

**How these estimates were built:**
- They are extrapolations from one smoke point and from published ternary and byte-level scaling:
  - BitNet b1.58 is close to float only at ≥1B.
  - Below 100M, ternary costs about 1.5–2× effective parameters, and this smoke already shows ternary 1.977 vs float 1.869.
- They assume the byte budgets above get met. With burn alone, the 120M and 300M budgets are not met, and quality stops at the data that fits in the time.

**What to expect on benchmarks and downstream:**
- **Zero-shot:** below 120M, expect chance-level ARC-C, MMLU and OBQA. PIQA should be about 58–62% and ARC-E about 38–45% by 120M. At 360M, the SmolLM2 models score about 70% PIQA and 60% ARC-E. Adam-T at 300M should land 5–10 points under them, because it has 100–1000× less data.
- **Chat and NQ:** these depend on the memory and retrieval stack more than on the core. The core should mainly buy fluency and fewer malformed answers.

## 10. Paradigm notes and open decisions for the CEO

1. **Training uses GEMM on latent float weights.** Autodiff through STE needs that. The exported model and the inference path are lookup-only for weights, and no float weight tensor ships.
2. **The activation grid has 255 levels.** It is the LUT index and not stored compression. Switching it to 289 levels, two GF(17) digits (±144), is a one-line change, planned for the 10M run.
3. **Attention, softmax and norms stay float arithmetic at inference.** A GF(17) exp/log table form of softmax is future work.
4. **Memory rows are f32** in phase 1, stored exactly. The GF(17)-grid memory in section 2 turns them into true coordinates.
5. **The governor contract:**
   - The trainer reads `THROTTLE_FILE` every step, which is inside the ≤50-step rule.
   - It syncs the GPU every step, so the sleep really idles the card.
   - It sleeps `throttle × max(1, gpu_ms/50)` per step, so a 400 ms step backs off like eight adam-train steps.
   - It refuses to train when the file is missing.
   - It stops after `MAX_MIN=19` minutes with a checkpoint, and a rerun resumes.
