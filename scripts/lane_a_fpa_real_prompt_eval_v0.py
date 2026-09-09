#!/usr/bin/env python3
"""Honest real-prompt KL eval for Lane A FPA distill v0 (NOT Done, not near-1).

Loads the dense Qwen3.5-4B teacher and a student with FpaLinear on L15 up_proj,
then scores freeze_init_trainables.pt vs trained_trainables.pt on short English
prompts. Also reports KL(teacher || teacher) (~0) as a control.

Antman (after a 5k distill run)::

    python scripts/lane_a_fpa_real_prompt_eval_v0.py \\
        --teacher downloaded_models/Qwen3.5-4B \\
        --bake bakes/qwen35_4b_hc_fpa_onetensor_probe \\
        --ckpt-dir logs/lane_a_fpa_distill_v0/qwen35_l15_up \\
        --out logs/lane_a_fpa_real_prompt_eval_v0

CI / no 4B (needs a prior --synthetic distill out dir)::

    python scripts/lane_a_fpa_distill_v0.py --synthetic --steps 8 --out /tmp/fpa_v0
    python scripts/lane_a_fpa_real_prompt_eval_v0.py --synthetic --ckpt-dir /tmp/fpa_v0 --out /tmp/fpa_eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from amni.training.fpa_real_prompt_eval_v0 import PromptEvalConfig, run_prompt_eval


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Lane A FPA real-prompt KL eval v0 (not Done)")
    p.add_argument("--teacher", default="downloaded_models/Qwen3.5-4B")
    p.add_argument("--bake", default="bakes/qwen35_4b_hc_fpa_onetensor_probe")
    p.add_argument("--ckpt-dir", default="logs/lane_a_fpa_distill_v0", help="dir with freeze_init_trainables.pt + trained_trainables.pt")
    p.add_argument("--out", default="logs/lane_a_fpa_real_prompt_eval_v0")
    p.add_argument("--prompts", default="", help="optional .txt (one/line) or .json list; default = built-in 48 English sentences")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--teacher-device", default="")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--no-sparse", action="store_true")
    args = p.parse_args(argv)
    cfg = PromptEvalConfig(
        teacher=args.teacher,
        bake=args.bake,
        ckpt_dir=args.ckpt_dir,
        out=args.out,
        prompts=args.prompts,
        layer=args.layer,
        seq_len=args.seq_len,
        batch=args.batch,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
        teacher_device=args.teacher_device,
        dtype=args.dtype,
        synthetic=args.synthetic,
        use_sparse=not args.no_sparse,
    )
    print("Lane A real-prompt eval v0 — NOT Done; freeze-quantize REFUTED; not near-1", flush=True)
    run_prompt_eval(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
