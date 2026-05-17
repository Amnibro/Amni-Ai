# Adam-1

A texture-native, lossless, federated learning architecture for transformer LLMs.

Adam-1 takes any standard HuggingFace transformer (Qwen2.5, Llama, Mistral, etc.) and re-stores its weights as **GF(17) digit planes** — a bit-exact lossless representation that opens three capabilities the standard safetensors+VRAM path can't:

1. **Tier-protected weight authority.** Every tensor is classified into one of five layers (Asimov / Commandments / Ascension / Foundation / Wisdom) with explicit write-authority rules. Foundational layers never get touched by any learning process; only Wisdom-tier weights accept residual updates.
2. **Subject-routed residual learning.** Each tensor can carry multiple subject-tagged residual planes on disk (`<tensor>.<subject>.gf17res`). At inference time you select which subjects' overlays apply via `(d_base + l_residual) mod 17`. Math residuals never pollute code residuals; subjects can be cleared independently or shared between instances.
3. **PrismTex federation.** Residual planes and experience atlases serialize to portable `.prismtex` and `.expatlas` bundles. Multiple Adam instances can swap learnings without sharing raw weights.

The entire stack runs on consumer hardware. A 1.5B-parameter Qwen2.5-Instruct bake fits in 8 GB VRAM and produces real benchmark growth on commodity GPUs (validated on RX 7800 XT).

---

## What you get

| Capability | Status | Validated |
|---|---|---|
| Lossless GF(17) bake of any safetensors model | shipped | bit-exact roundtrip on 1.5B model (338 tensors) |
| Streaming inference with VRAM cache management | shipped | 8 GB budget holds 1.5B + residuals + activations |
| Asimov / Foundation tier protection | shipped | 142/338 tensors auto-locked on Qwen2.5-1.5B |
| Subject-tagged residual storage | shipped | multiple subjects coexist on same tensor |
| Subject-aware inference overlay | shipped | bit-perfect `(d + l) mod 17` per query |
| Auto-learning via frozen-base SFT → residuals | shipped | training reproducible (loss minimum hit 3/3 runs); single-corpus SFT lift is on-domain category-specific (n=16 mean +3.33pp MMLU; n=32 cross-category mean −0.83pp; HS +0.5pp consistent across N). Multi-cycle locks gains under LR decay 10×/cycle, never regresses |
| PrismTex residual federation | shipped | fp16-averaged bundle merge; linearity validated at N=2 (predicted +1.25, actual +1.30 MMLU) and N=3 (gap 0.83pp within sample noise) |
| Multi-subject inference overlay | shipped | fp16-averaged decode for `subjects=[X, Y, ...]`; multi-subject co-activation no longer collapses (was −25pp pre-v5.5.46) |
| Subject-tagged bundle export / merge / apply | shipped | bundles carry `subject` in header; round-trip preserves it; `merge_fp16_avg` requires shared subject |
| ExperienceAtlas memory federation | shipped | export / import bit-perfect |
| Atlas-driven distillation (queries → smarter Adam) | shipped | mechanics PASS; pass `subject='global'` to `train_from_atlas` for global-bench visibility |

Full benchmarks against `Qwen2.5-0.5B-Base` and `Qwen2.5-1.5B-Instruct` baselines in [`changelog.md`](changelog.md).

---

## Quickstart

### Install
```bash
git clone https://github.com/<your-org>/adam1
cd adam1
python -m venv .venv && source .venv/Scripts/activate    # or .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

### One-command lifecycle (recommended)

```bash
python scripts/adam1.py auto \
    --hf-id Qwen/Qwen2.5-1.5B-Instruct \
    --code-root /path/to/your/codebase
```

That's it. The orchestrator handles everything:
- Auto-bakes the HF model into GF(17) form (if no bake exists yet).
- Auto-detects the local snapshot path under `downloaded_models/`.
- Initial-ingests your codebase into a PTEX-encoded ExperienceAtlas (subject='global').
- Spawns the **grow daemon** (continuous distillation: polls atlas, trains when 50+ new records, encodes residuals).
- Spawns the **serve API** on port 8000 (OpenAI-compatible, with `X-Adam-Subject: auto` per-query routing).
- Periodically re-ingests the codebase (every 1 hour by default).
- Optionally pulls federation bundles from a peer HF repo (`--federation-repo`).
- Optionally streams a HuggingFace dataset for additional training data (`--hf-stream-dataset hellaswag`).
- Cleans up all child processes on Ctrl-C.

Logs land in `logs/adam1_auto/{grow,serve}.log`. Then point Cursor / Continue / OpenWebUI at `http://localhost:8000/v1` with any API key.

### Manual mode (advanced — individual commands)

If you want to run the components separately (debugging, custom orchestration, cron-driven workflows):

```bash
# Bake
python scripts/adam1.py bake --hf-id Qwen/Qwen2.5-1.5B-Instruct --out bakes/qwen25_1_5b_gf17

# Inference (any OpenAI-compatible IDE/tool)
python scripts/adam1.py serve --bake bakes/qwen25_1_5b_gf17 --port 8000

# Direct Python API
from amni.inference.streaming_chat import StreamingChatService
svc = StreamingChatService('bakes/qwen25_1_5b_gf17', 'downloaded_models/.../Qwen2.5-1.5B-Instruct', budget_mb=8000)
response, _ = svc.chat("What is GF(17) arithmetic?", max_new_tokens=128, subject='auto')
print(response)
```

Per-request subject overlay via `X-Adam-Subject` header — values: explicit subject (`math`, `code`, etc.), `global` (no overlay), or **`auto`** (SubjectClassifier picks per query, response includes `adam1.subject` showing what was chosen).

### Federate residuals across N Adams
```bash
# Each contributor exports a PrismTex bundle from their local bake
python -c "from amni.learning.prismtex import PrismTexBundle; \
    PrismTexBundle.export_from_bake('bakes/local', subject='math', \
    contributor_id='node-a').write('node_a_math.prismtex')"

# Aggregator merges N contributor bundles into one consensus bundle
python scripts/adam1.py federate \
    --bundles node_a_math.prismtex node_b_math.prismtex node_c_math.prismtex \
    --base-bake bakes/qwen25_1_5b_gf17 \
    --out merged_math.prismtex \
    --apply                          # also writes merged residual to base bake's math subject
```

For automated federation, pass `--federation-repo my-org/adam1-shared` to `adam1 auto` and it'll periodically pull and apply published bundles.

`adam1_federate` validates (a) all bundles share the same `source_sha256` (same starting Adam) and (b) all bundles share the same `subject`. The merge uses `merge_fp16_avg` — decode each contributor's residual to fp16 deltas, average, re-encode — which is the only correct N>1 merge primitive. Naive digit-sum collapses for N>1; the CLI's deprecated `--legacy-additive` opt-in still produces it for backward compatibility but with a clear warning.

### Auto-learn from your own data
```python
from amni.learning import ResidualSFTLearner, ExperienceAtlas

# Log queries+responses as Adam runs
atlas = ExperienceAtlas('experiences/', subject='my-domain')
atlas.append(prompt, response, outcome=1, system='...')

# Periodically distill into subject residuals
learner = ResidualSFTLearner('bakes/qwen25_1_5b_gf17', '<hf-source-path>', trainable_layer_min=20)
learner.load_model()
learner.train_from_atlas(atlas, lr=2e-5, max_len=384)
learner.shutdown()

# Next inference call uses the new residuals via overlay (no model reload needed)
```

The base weights never change. Asimov / Commandments / Ascension / Foundation tiers refuse residual writes. Wisdom-tier weights get the deltas as `(d_base + l_residual) mod 17`. Roll back any time by clearing the residual files.

---

## Architecture in 60 seconds

```
                         Adam-1 architecture

     ┌─────────────────────────────────────────────────────────┐
     │          GF(17) digit-plane bake (immutable)            │
     │  Each fp16 weight = 4 base-17 digits stored as RGBA.    │
     │  Tier-classified: Asimov, Commandments, Ascension,      │
     │  Foundation (locked) / Wisdom (writable).               │
     └─────────────────────────────────────────────────────────┘
                              │
                              │  (mod 17 overlay per active subject)
                              ▼
     ┌─────────────────────────────────────────────────────────┐
     │      TensorRegistry (VRAM cache, budget-bounded)        │
     │  set_active_subjects(['math']) → only math overlay on.  │
     │  Reads are bit-exact: u16 = (d_base + l) mod 17 sums.   │
     └─────────────────────────────────────────────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  Inference   │ ─── response ─────┐
                       └──────────────┘                   │
                              ▲                           │
                              │                           ▼
        ┌─────────────────────┴─────────────────┐    ┌────────────┐
        │    LearningWriter (writes residuals)  │ ◄──┤ Experience │
        │  encode_target_array_as_residuals     │    │   Atlas    │
        │  (subject='wisdom' or per-domain)     │    │  (per-     │
        └────────────┬──────────────────────────┘    │  subject)  │
                     │                               └────────────┘
                     ▼                                     │
               ┌──────────────┐         ┌────────────┐    │
               │   PrismTex   │ ◄─────► │ Adam Node  │    │
               │  federation  │  share  │   #2..N    │    │
               └──────────────┘         └────────────┘    │
                                                          │
                                                          ▼
                                              (federated atlas
                                               sharing across
                                               Adam instances)
```

Five storage primitives, two federation primitives, one inference engine, one learning loop.

---

## Federation between Adams

```python
from amni.learning import PrismTexBundle, ExperienceAtlas

# Adam-A: train, export
bundle_a = PrismTexBundle.export_from_bake('bakes/A', contributor_id='node-A')
bundle_a.write('a.prismtex')

# Adam-B: train independently, export
bundle_b = PrismTexBundle.export_from_bake('bakes/B', contributor_id='node-B')
bundle_b.write('b.prismtex')

# Server: merge via GF(17) digit-wise sum (works within a subject)
swarm = PrismTexBundle.merge([
    PrismTexBundle.read('a.prismtex'),
    PrismTexBundle.read('b.prismtex'),
])
swarm.apply_to_bake('bakes/local')   # Asimov-protected tensors auto-rejected

# Or share experiences instead of weight deltas (more robust, less collapse-prone)
atlas_bundle = atlas.export_bundle('shared.expatlas')
remote_atlas.import_bundle('shared.expatlas')
remote_learner.train_from_atlas(remote_atlas)   # each node distills locally
```

---

## Why GF(17)?

Prime 17 was chosen for three reasons: (1) `17^4 = 83521 > 65536 = 2^16`, so four base-17 digits losslessly encode any uint16 (and therefore any fp16 / bf16 by bit-pattern view), (2) 17 is small enough that `lut[d]` lookup tables fit easily in registers, (3) digit position carries semantic meaning — `d0` is the LSB (smallest weight perturbation per increment), `d3` is the MSB (coarsest). Federated learning naturally writes to `d0` for fine-grained updates; structural changes would only touch `d3`. Both decompose under the same modular arithmetic.

The digit storage is `H × W × 4` uint8 RGBA pages — at 5.9 GB for a 1.5B model, exactly 2× the fp16 baseline (3 bits wasted per byte for direct addressability without bitfield extraction). A 5-bit-packed format would recover that 37.5% but at the cost of LUT-friendly byte addressing; the current trade-off favors compute simplicity.

---

## Federation surface

End-to-end validated across all four orthogonal dimensions:

| Configuration | Mechanism | Validated |
|---|---|---|
| Single contributor, single subject | mod-17 fast path | bit-perfect roundtrip |
| Multiple contributors, single subject | bundle-merge fp16-avg | linearity at N=2 (gap 0.05pp) and N=3 (gap 0.83pp) |
| Single contributor, multiple subjects | inference-overlay fp16-avg | `subjects=[math, code]` lifts to +2.5pp MMLU (was −25pp under broken mod-17 stacking) |
| Multiple contributors × multiple subjects | both fp16-avg layers compose | full matrix runs without collapse |

Both layers — bundle merge (federation between Adams) and inference overlay (multi-subject composition within a single Adam) — use the same fp16-averaging algorithm: decode each contributor's residual to fp16 deltas, average the deltas, add to the base fp16 weight. The math is identical at both layers; the implementation is on disk for federation merge, in-memory at decode time for inference overlay.

The naive alternative — stacking residual digit planes via `(d_base + Σ r_i) mod 17` — collapses the model when more than one contributor is active. v5.5.36 introduced fp16-avg at the federation layer; v5.5.46 mirrored it at the inference layer.

VRAM held flat at ~5.5 GB across every federation experiment regardless of contributor count or subject count. Federation merge runs CPU-side as a vectorized numpy op against the base bake's GF(17) digit planes.

---

## Foundational tier hierarchy

| Tier | Authority | Writable | Default mapping |
|---|---|---|---|
| Asimov | system | ❌ | embed_tokens (token-meaning boundary; 5 immutable laws hash-locked here) |
| Commandments | anthony | ❌ | lm_head (output voice) |
| Ascension | anthony | ❌ | model.norm (final transformation, purpose) |
| Foundation | system | ❌ | layer norms + biases (structural stability) |
| Wisdom | swarm | ✅ | attention Q/K/V/O + MLP gate/up/down |

Auto-classification on Qwen2.5-1.5B: 1 / 0 / 1 / 140 / 196 of 338 tensors.

The `is_writable_by(name, requestor)` check enforces this on every write attempt. PrismTex apply respects it (counts violations as `refused`, not `applied`). LearningWriter raises `AsimovProtectedError` on direct writes.

---

## Benchmarks

On RX 7800 XT (16 GB VRAM), 8 GB streaming budget. Headline measurement uses deepeval at full bench scale (912 MMLU questions across all 57 subjects, 192 HellaSwag, 200 Winogrande).

### Headline: Adam-1 specialist bakes outperform Qwen2.5-1.5B-Instruct at full scale

The substrate enables training cheap (~30-50 min) specialist LoRA adapters that merge back into the lossless GF(17) bake. Specialists outperform the base Qwen on workload-matched benchmarks:

| Workload | Qwen2.5-1.5B baseline | Best Adam-1 specialist | Δ | Bake |
|---|---:|---:|---:|---|
| **MMLU full-57** (912q) | 62.5% | **64.3%** | **+1.75pp** | `qwen25_1_5b_mcq_lora_v5_5_125_gf17` |
| **HellaSwag** (192q) | 61.9% | **66.5%** | **+4.6pp** | `qwen25_1_5b_commonsense_lora_v5_5_129_gf17` |
| **Winogrande** (200q) | 57.0% | **61.5%** | **+4.5pp** | `qwen25_1_5b_commonsense_lora_v5_5_129_gf17` |

### Single-bake deployment (no routing)

Element-wise averaging of two specialist LoRA adapters before merging produces ONE bake that beats Qwen on all three workloads. Tunable knob via averaging weight:

| Avg weight (MCQ / csense) | MMLU | HellaSwag | Winogrande | Profile |
|---|---:|---:|---:|---|
| 70 / 30 | 63.4% (+0.9) | 62.5% (+0.6) | 56.5% (-0.5) | MMLU-leaning |
| 50 / 50 | 62.6% (+0.1) | 62.5% (+0.6) | 59.5% (+2.5) | always-positive |
| **30 / 70** | 62.0% (-0.5) | **64.2% (+2.3)** | **62.5% (+5.5)** | best aggregate lift |

### Bake roster

| Bake | Trained on | Best at | Notes |
|---|---|---|---|
| `qwen25_1_5b_instruct_gf17_v5_0_3` | (none — pure GF(17) of Qwen) | general | base lossless reference |
| `qwen25_1_5b_mcq_lora_v5_5_125_gf17` | 32K MCQs (MMLU aux + sciq + arc + obqa) | MMLU recall | LR=5e-5 LoRA rank=16 |
| `qwen25_1_5b_commonsense_lora_v5_5_129_gf17` | 29K commonsense (HellaSwag + Winogrande + OBQA) | HS / Wino | LR=5e-5 LoRA rank=16 |
| `qwen25_1_5b_avg_lora_v5_5_130_gf17` | average of MCQ + csense (50/50) | balanced | always-positive single bake |
| `qwen25_1_5b_avg73_lora_v5_5_131_gf17` | weighted avg (0.7 MCQ / 0.3 csense) | MMLU-leaning | |
| `qwen25_1_5b_avg37_lora_v5_5_131_gf17` | weighted avg (0.3 MCQ / 0.7 csense) | best aggregate | |

### Architectural property

All 6 bakes are **lossless GF(17) re-bakes** (cos=1.0 round-trip across all 338 tensors). The substrate fully supports gradient-derived LoRA distillation; the merged weight is bit-exact preservable in the GF(17) substrate. Specialists multiply cheaply (~30-50 min train + 30s bake each); choose specialist by workload, or use averaged single bake for always-positive lift without routing.

### Caveats / honest framing

- These are **specialist bakes**: pure MCQ-LoRA hurts HellaSwag/Winogrande by -1 to -5pp; pure CSense-LoRA hurts MMLU by -1.4pp. The averaged bakes are the no-routing-needed alternatives.
- KB attach (multi-KB retrieval): helps base bake +0.11pp on MMLU at gate=12, helps AVG bake +0.11pp, **HURTS pure specialists** by -0.66pp. Deploy KB only with non-specialist bakes.
- All numbers are deepeval against `Qwen2.5-1.5B-Instruct`. The base GF(17) bake reproduces Qwen exactly (lossless).

### Earlier residual-SFT era (v5.5.42 baseline)

Earlier Wisdom-tier residual-SFT path (pre-LoRA-distillation, characterized v5.5.42): single-corpus SFT lift was on-domain category-specific (n=16 mean +3.33pp MMLU; n=32 cross-category mean −0.83pp). Residual-SFT mechanism was sound but didn't transfer to scale. **The LoRA distillation path superseded it** as the production substrate's training method.

| Model | Bench | Direct PTEX | After 1-cycle Wisdom-tier residual SFT |
|---|---|---:|---:|
| Qwen2.5-1.5B-Instruct | MMLU (deepeval n=16) | 42.5% | 46.3% (+3.8pp at n=16; **see caveat below — at n=32 the mean is −0.83pp, this lift is n=16-category-specific**) |
| Qwen2.5-1.5B-Instruct | HellaSwag (deepeval n=16) | 64.4% | **66.7% (+2.3pp peak, +0.5–0.7pp mean across N — consistent across n=16 and n=32)** |
| Qwen2.5-1.5B-Instruct | GSM8K (deepeval n=20) | 5.0% | 5.0% |
| Qwen2.5-0.5B-Base | MMLU (deepeval n=16) | 17.5% | n/a |

**MMLU caveat (v5.5.52):** At deepeval n=16 the per-category sample variance is ~±10pp per single bench (the same model produces baselines that swing 12pp on `high_school_mathematics` between n=16 and n=32 sub-samples). This dominates typical SFT-induced shifts. The v5.5.42 +3.33pp mean was largely driven by one strongly-shifted category (`global_facts` +12.5pp at n=16, +7.3pp at n=32 — direction consistent). Of 5 categories sampled at both n levels, only 2 show direction-consistent shifts; the other 3 are within sample noise. HellaSwag is more uniform and shows a small consistent lift (+0.5–0.7pp). **Training mechanism is solid** (residuals encode losslessly, federation linearity validated, foundational tiers protected). **Honest MMLU measurement requires n=32+ evaluation with broader-corpus training**; the federation architecture is the path forward — many domain-specific Adams trained on category-targeted data, federated via fp16-avg consensus.

Cycle 1 is the **SFT peak** under uniform LR. With naive uniform `lr=2e-5` across cycles, cycle 2 partially regressed MMLU and cycle 3 returned to baseline — same training loss trajectory (1.50 → 1.24 → 1.24), but the descent direction stopped tracking benchmark gains.

**Multi-cycle LR decay solves this.** With per-cycle decay (`2e-5 → 2e-6 → 2e-7 → 2e-8 → 2e-9`), the cycle-1 lift **locks across all 5 cycles, never regresses**, and cycle 3 added an HS bonus (+2.3pp) on top. v5.5.37 validated 5 consecutive cycles with deepeval-n=16 between each — net delta +3.8pp MMLU / +2.3pp HS held through cycles 2-5 with no further regression. The federated multi-cycle policy is therefore: **cycle 1 lr=2e-5, cycle N>1 lr ÷= 10 each round**.

Variance of a single SFT round was characterized at n=16 in v5.5.42: same 500 records, 3 fresh runs, MMLU mean +3.33pp, σ=0.72pp, range [+2.5, +3.8]. Same training loss reached every run (1.50 ± 0.004); residuals reach the same point in loss space, with bench variance reflecting sample noise at n=16.

Architectural property: VRAM stays at 8 GB regardless of how many subject residuals exist on disk. Disk is the unbounded growth medium; VRAM is per-active-subject.

---

## What this is NOT (yet)

- A new pre-training method. Adam-1 starts from existing instruct models and adds a learning layer on top.
- A replacement for fine-tuning. The frozen-base residual SFT path is *one* learning algorithm; many others are compatible with the substrate.
- A serverless / cloud product. Single-machine deployment, optionally federated.
- Production-ready on every architecture. Tested on Qwen2.5 (0.5B and 1.5B). Llama / Mistral should work via the same safetensors → GF(17) path; not yet validated.

Open research questions and known limits are listed in [`docs/whitepaper/adam1.md`](docs/whitepaper/adam1.md).

---

## Repository layout

```
amni/                      # core package
  compute/reffelt4.py      # GF(17) digit-plane encoder/decoder
  inference/streaming_*.py # TensorRegistry + StreamingChatService
  learning/
    gf17_writer.py         # LearningWriter with tier + subject + residual API
    auto_learner.py        # ResidualSFTLearner (frozen-base SFT)
    prismtex.py            # residual-bundle federation
    experience_atlas.py    # PTEX-encoded memory texture maps

scripts/
  adam1.py                 # umbrella entry point: adam1 <subcommand> [args...]
  adam1_auto.py            # zero-manual-command orchestrator: bake + grow + serve + ingest
  adam1_bake.py            # one-command HF → GF(17) bake
  adam1_ingest_codebase.py # walk dir tree → atlas (training data from local source)
  adam1_ingest_hf.py       # stream HuggingFace dataset → atlas (no disk cache)
  adam1_grow.py            # continuous distillation daemon (polls atlas, trains)
  adam1_autotrain.py       # single-shot trainer (cron-friendly variant of grow)
  adam1_federate.py        # N-Adam consensus via merge_fp16_avg
  adam1_pull.py            # download bake or PrismTex bundles from HF Hub
  adam1_serve.py           # OpenAI-compatible API server (X-Adam-Subject: auto routing)
  adam1_publish.py         # upload bake to HF Hub
  v5_5_*.py                # research scripts (corpus mining, benchmarking)

bakes/                     # GF(17) bakes (gitignored)
downloaded_models/         # HF cache (gitignored)
docs/whitepaper/           # ArXiv-shaped whitepaper sources
examples/                  # quickstart scripts
tests/                     # smoke + integration tests
```

---

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

## Citation

If Adam-1 is useful in your work, please cite the whitepaper at `docs/whitepaper/adam1.md` (BibTeX entry inside).

## Contact

Issues, PRs, and federation experiments welcome. The architecture is intentionally swarm-shaped — a single deployment is fine; many federated deployments is the point.
