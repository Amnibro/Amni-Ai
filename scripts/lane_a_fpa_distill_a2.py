#!/usr/bin/env python3
"""Lane A FPA Arm A2 — real-prompt train+eval + layer MSE (NOT Done, never PASS).

Same English prompt family for train and eval (default 32×128). mse_weight is
1.0 by default or 0.5 via flag. Freeze codebooks+pq_idx; train U,V,scales,pq_scale.

Antman HIP (torch sees ROCm as cuda)::

    export HIP_VISIBLE_DEVICES=0
    python scripts/lane_a_fpa_distill_a2.py \\
        --teacher downloaded_models/Qwen3.5-4B \\
        --bake bakes/qwen35_4b_hc_fpa_onetensor_probe \\
        --steps 5000 --lr 1e-4 --eval-every 500 \\
        --mse-weight 1.0 --n-prompts 32 --seq-len 128 \\
        --out logs/lane_a_fpa_distill_a2/qwen35_l15_up

mse_weight 0.5::

    python scripts/lane_a_fpa_distill_a2.py --mse-weight 0.5 ...same...

CI / no GPU train::

    python scripts/lane_a_fpa_distill_a2.py --synthetic --skip-gpu-train \\
        --steps 8 --eval-every 4 --out /tmp/fpa_a2

Writes Kimahri ``summary.json`` (freeze_kl, trained_kl, delta_vs_freeze,
teacher_self_kl, bpw_allin, layer_mse, arm_id, steps, data_mix). After 5k, if
KL is not ≥5% better than freeze → escalate A3 (A3 is not in this script).
Tidus owns SHIP. See docs/A2_runbook.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from amni.training.fpa_distill_a2 import config_from_args, parse_a2_args, run_distill_a2
from amni.training.fpa_summary_a2 import inventory


def main(argv=None) -> int:
    args = parse_a2_args(argv)
    if args.inventory:
        print(json.dumps(inventory(), indent=2))
        return 0
    cfg = config_from_args(args)
    print(
        "Lane A gf17_fpa Arm A2 — real-prompt train+eval + layer MSE; "
        "NOT Done; never PASS; Tidus owns SHIP",
        flush=True,
    )
    summary = run_distill_a2(cfg)
    if summary.get("kill", {}).get("next_action") == "escalate_a3":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
