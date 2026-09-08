#!/usr/bin/env python3
"""Lane A FPA distill v0 — one-Linear proof (NOT Done, not near-1).

Recipe (lane_a_fpa_distill_recipe_v1): replace ONLY Qwen3.5-4B L15 up_proj
with FpaLinear from the one-tensor probe bake. Train U/V/U_scale/V_scale/pq_scale.
Freeze codebooks + pq_idx + sparse. KL distill T=2. Never materialise dense W.
GDN / linear_attn stay dense.

Antman HIP (torch sees ROCm as cuda)::

    python scripts/lane_a_fpa_distill_v0.py \\
        --teacher downloaded_models/Qwen3.5-4B \\
        --bake bakes/qwen35_4b_hc_fpa_onetensor_probe \\
        --steps 5000 --lr 1e-4 --eval-every 500 \\
        --out logs/lane_a_fpa_distill_v0/qwen35_l15_up

CI / no weights::

    python scripts/lane_a_fpa_distill_v0.py --synthetic --steps 8 --eval-every 4 --out /tmp/fpa_v0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from amni.training.fpa_distill_v0 import DistillConfig, run_distill


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Lane A FPA distill v0 (one-Linear proof, not Done)")
    p.add_argument("--teacher", default="downloaded_models/Qwen3.5-4B")
    p.add_argument("--bake", default="bakes/qwen35_4b_hc_fpa_onetensor_probe")
    p.add_argument("--out", default="logs/lane_a_fpa_distill_v0")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--mse-weight", type=float, default=0.0, help="optional MSE on L15 up_proj output")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="", help="cuda / cpu; empty = HIP/CUDA if available")
    p.add_argument("--teacher-device", default="", help="optional CPU teacher if VRAM is tight")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--synthetic", action="store_true", help="tiny Linear pair (no 4B weights)")
    p.add_argument("--no-sparse", action="store_true")
    args = p.parse_args(argv)
    cfg = DistillConfig(
        teacher=args.teacher,
        bake=args.bake,
        out=args.out,
        steps=args.steps,
        lr=args.lr,
        eval_every=args.eval_every,
        temperature=args.temperature,
        mse_weight=args.mse_weight,
        layer=args.layer,
        batch=args.batch,
        seq_len=args.seq_len,
        eval_batches=args.eval_batches,
        seed=args.seed,
        device=args.device,
        teacher_device=args.teacher_device,
        dtype=args.dtype,
        synthetic=args.synthetic,
        use_sparse=not args.no_sparse,
    )
    print("Lane A gf17_fpa distill v0 — bootstrap, NOT Done; freeze-quantize REFUTED", flush=True)
    summary = run_distill(cfg)
    if summary.get("kill_armed") and summary["verdict"] == "FAIL":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
