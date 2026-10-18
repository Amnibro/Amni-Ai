# Lane A FPA Arm A2 runbook

Hub-lean train patch so Antman can run the moment the PC reconnects.

**Not Done. Never PASS. Tidus owns SHIP.**

A3 is **not** implemented here. If the 5k kill fires, escalate A3 — do not
quietly lower the bar.

## What A2 is

| | v0 / A0 (in tree) | A2 (this arm) |
|---|---|---|
| Train tokens | `random_ids` (soft random vocab ids) | same real-prompt family as eval |
| `mse_weight` | default **0** | **1.0** (flag **0.5** only) |
| Eval | separate real-prompt script after train | same encoded prompt table as train |
| Score keys | plumbing `kl_init` / `kl_final` or `kl_freeze_init` / `kl_trained` | Kimahri pack below |
| Script language | v0 kill may say PASS/FAIL | A2 **refuses PASS** |

Hypothesis checked in tree before this patch: v0
`amni/training/fpa_distill_v0.py` trains with `random_ids(...)` and
`DistillConfig.mse_weight: float = 0.0`. The later
`scripts/lane_a_fpa_real_prompt_eval_v0.py` only **evaluates** on English
prompts. That split matches the failed 32×128 real-prompt KL
(freeze `0.451462` → trained `0.487534`, **+8% worse**).

A2 does not redefine the 5% bar. After 5k real-prompt steps:

* `delta_vs_freeze >= 0.05` → `improved_ge_5pct` / `hold_for_tidus_ship`
* else → `not_ge_5pct_better` / **`escalate_a3`** (exit 2)

`delta_vs_freeze = (freeze_kl - trained_kl) / freeze_kl`. Positive = better.

## Inventory (existing contract)

| Piece | Path |
|---|---|
| Forward math | `y = U@(Vᵀx) + pq + sparse` — never dense `W` |
| Format / packing | `gf17_fpa` / `factor_product_atlas_v1` |
| `FpaLinear` | `amni/inference/fpa_linear.py` |
| Bake load | `amni/inference/gf17_fpa.py` + `FpaBake` / `load_fpa_linear` |
| Distill v0 (random-id) | `amni/training/fpa_distill_v0.py` + `scripts/lane_a_fpa_distill_v0.py` |
| Real-prompt eval v0 | `amni/training/fpa_real_prompt_eval_v0.py` + `scripts/lane_a_fpa_real_prompt_eval_v0.py` |
| Shared prompt family | `amni/training/fpa_real_prompts.py` (`real_prompt_en_v1`, 128 sentences) |
| A2 distill | `amni/training/fpa_distill_a2.py` + `scripts/lane_a_fpa_distill_a2.py` |
| Kimahri schema | `amni/training/fpa_summary_a2.py` + `docs/A2_summary.schema.json` |
| Spec | `docs/FACTOR_PRODUCT_ATLAS.md` |

Print the same inventory from a machine with the repo:

```bash
python scripts/lane_a_fpa_distill_a2.py --inventory
```

Trainables (unchanged): `U`, `V`, `U_scale`, `V_scale`, `pq_scale`.
Frozen: `codebooks`, `pq_idx`, sparse sidecars.

Swap **only** Qwen3.5-4B `layers.15.mlp.up_proj`. GDN / `linear_attn` stay dense.

## Bake path

Expected one-tensor probe (Antman disk; may be absent in CI):

```
bakes/qwen35_4b_hc_fpa_onetensor_probe/
  bake_manifest.json          # format=gf17_fpa packing=factor_product_atlas_v1
  model.safetensors           # *.U *.V *.U_scale *.V_scale *.pq_idx *.pq_scale
  _fpa_codebooks.npy          # float32 (8, 256, 16)
  _fpa_sparse.npz             # indices i64, values i8, vmax, shape (out,inn)
  _fpa_report.json            # optional
```

Teacher weights: `downloaded_models/Qwen3.5-4B`.

If the bake folder is missing, A2 exits with `FPA bake not found` — do not
invent a dense-W fallback.

## Antman headless (HIP / ROCm)

Torch on Antman sees ROCm as `cuda`. Pin the compute GPU. Typical:

```bash
cd /path/to/Amni-Ai
export HIP_VISIBLE_DEVICES=0
# optional: keep the display GPU out of the process
# export AMNI_RESERVE_DISPLAY_GPU=1

python scripts/lane_a_fpa_distill_a2.py \
  --teacher downloaded_models/Qwen3.5-4B \
  --bake bakes/qwen35_4b_hc_fpa_onetensor_probe \
  --steps 5000 --lr 1e-4 --eval-every 500 \
  --mse-weight 1.0 \
  --n-prompts 32 --seq-len 128 \
  --out logs/lane_a_fpa_distill_a2/qwen35_l15_up_mse1
```

Second cell (`mse_weight=0.5`):

```bash
export HIP_VISIBLE_DEVICES=0
python scripts/lane_a_fpa_distill_a2.py \
  --teacher downloaded_models/Qwen3.5-4B \
  --bake bakes/qwen35_4b_hc_fpa_onetensor_probe \
  --steps 5000 --lr 1e-4 --eval-every 500 \
  --mse-weight 0.5 \
  --n-prompts 32 --seq-len 128 \
  --out logs/lane_a_fpa_distill_a2/qwen35_l15_up_mse05
```

Documented larger mix (still the same family, first 128 sentences):

```bash
python scripts/lane_a_fpa_distill_a2.py \
  --n-prompts 128 --seq-len 128 \
  --mse-weight 1.0 \
  --out logs/lane_a_fpa_distill_a2/qwen35_l15_up_mse1_128
```

Custom file (txt one-per-line or JSON list) — train and eval still share it:

```bash
python scripts/lane_a_fpa_distill_a2.py --prompts /path/to/prompts.txt --n-prompts 64
```

`--mse-weight 0` is rejected. That was v0 plumbing, not A2.

If VRAM is tight, `--teacher-device cpu` keeps the dense teacher on host RAM.

## How `summary.json` is produced

The A2 script writes `--out/summary.json` at the end of `run_distill_a2`.
Required Kimahri fields (score **only** these):

| key | meaning |
|---|---|
| `freeze_kl` | eval KL at freeze-init, real-prompt family |
| `trained_kl` | eval KL after train, **same** family |
| `delta_vs_freeze` | `(freeze_kl - trained_kl) / freeze_kl` |
| `teacher_self_kl` | KL(teacher \|\| teacher) on those batches |
| `bpw_allin` | stored bits / (`out*inn`), sidecars included |
| `layer_mse` | `{freeze, trained}` on L15 `up_proj` out vs teacher |
| `arm_id` | `"A2"` |
| `steps` | configured step count |
| `data_mix` | `real_prompt_en_v1:32x128` (or file / larger N) |

Also written (not the score contract): `kl_curve.jsonl`, `kl_curve.csv`,
`freeze_init_trainables.pt`, `trained_trainables.pt`, `trained_fpa.pt`,
`trained_fpa_repacked.pt`.

Schema file: [`A2_summary.schema.json`](A2_summary.schema.json).

`verdict` is one of `kill_not_armed` | `improved_ge_5pct` | `not_ge_5pct_better`.
**None of these is PASS.** `status` is `not Done`.

Exit codes: `0` informational / hold-for-Tidus; `2` when `next_action` is
`escalate_a3`.

## CI / no GPU

Do **not** start a 5k GPU train in CI.

```bash
python -m unittest tests.test_fpa_distill_a2 tests.test_gf17_fpa tests.test_fpa_distill_v0 -v

python scripts/lane_a_fpa_distill_a2.py --synthetic --skip-gpu-train \
  --steps 8 --eval-every 4 --n-prompts 8 --seq-len 32 --out /tmp/fpa_a2
```

`--skip-gpu-train` forces CPU and refuses a non-synthetic run.

## Out of scope

Relay, SEO, Haven: not touched. Full-model chat serve: not touched. A3: not
implemented. Quality / Done / ship claims: not made.
