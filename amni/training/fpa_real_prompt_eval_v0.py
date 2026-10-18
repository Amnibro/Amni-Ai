"""Honest real-prompt KL eval for Lane A FPA distill v0.

Compares freeze-init vs trained ``FpaLinear`` L15 against the dense teacher
on tokenized English prompts. Also reports KL(teacher || teacher) (~0).

Not Done. Not near-1. One-Linear probe only.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from amni.inference.fpa_linear import RECIPE_V0_TRAIN
from amni.training.fpa_distill_v0 import (
    DEFAULT_BAKE,
    DEFAULT_LAYER,
    DEFAULT_T,
    DEFAULT_TEACHER,
    _dtype,
    _forward,
    freeze_except_fpa,
    kl_logits,
    load_causal_lm,
    load_trainables_pt,
    pick_device,
    swap_up_proj,
    build_synthetic_pair,
)
from amni.training.fpa_real_prompts import (
    DEFAULT_PROMPTS,
    batched,
    encode_prompts,
    load_prompts as _load_family,
    load_tokenizer,
)

DEFAULT_OUT = "logs/lane_a_fpa_real_prompt_eval_v0"


@dataclass
class PromptEvalConfig:
    teacher: str = DEFAULT_TEACHER
    bake: str = DEFAULT_BAKE
    ckpt_dir: str = "logs/lane_a_fpa_distill_v0"
    out: str = DEFAULT_OUT
    prompts: str = ""
    layer: int = DEFAULT_LAYER
    seq_len: int = 64
    batch: int = 4
    temperature: float = DEFAULT_T
    seed: int = 0
    device: str = ""
    teacher_device: str = ""
    dtype: str = "bfloat16"
    synthetic: bool = False
    use_sparse: bool = True
    n_prompts: int = 32


def load_prompts(path: str = "", limit: int = 32) -> List[str]:
    """v0 eval still slices the original 48-sentence core (first 32 by default)."""
    return _load_family(path, limit, family=DEFAULT_PROMPTS)


def mean_kl_pairs(
    left: nn.Module,
    right: nn.Module,
    batches: Sequence[torch.Tensor],
    T: float,
    left_device: torch.device,
    right_device: torch.device,
) -> Tuple[float, List[float]]:
    """Mean token-KL(right_logits → left_logits) i.e. KL(teacher=right || student=left)? 

    Convention here: ``kl_logits(student, teacher)`` = KL(softmax(teacher/T) || student).
    So pass student=left, teacher=right.
    """
    left.eval()
    right.eval()
    per: List[float] = []
    with torch.no_grad():
        for b in batches:
            sl = _forward(left, b.to(left_device))
            tl = _forward(right, b.to(right_device))
            if sl.device != tl.device:
                tl = tl.to(sl.device)
            per.append(float(kl_logits(sl, tl, T).item()))
    return sum(per) / max(1, len(per)), per


def run_prompt_eval(cfg: PromptEvalConfig) -> Dict[str, Any]:
    device = pick_device(cfg.device)
    tdev = pick_device(cfg.teacher_device) if cfg.teacher_device else device
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(cfg.prompts, limit=cfg.n_prompts)
    print(
        f"[fpa_prompt_eval] n_prompts={len(prompts)} seq_len={cfg.seq_len}  NOT Done / not near-1",
        flush=True,
    )

    if cfg.synthetic:
        teacher, student, path, fpa = build_synthetic_pair(seed=cfg.seed)
        teacher = teacher.to(tdev)
        student = student.to(device)
        fpa = student.up_proj
        vocab = int(getattr(teacher, "vocab_size", 128))
        tok = load_tokenizer("", True, vocab)
        ckpt = Path(cfg.ckpt_dir)
        if not (ckpt / "freeze_init_trainables.pt").is_file() or not (ckpt / "trained_trainables.pt").is_file():
            # allow eval-only synthetic: snapshot current as both if missing
            raise FileNotFoundError(
                f"need freeze_init_trainables.pt and trained_trainables.pt under {ckpt} "
                "(run distill --synthetic --out that dir first)"
            )
    else:
        if not Path(cfg.teacher).exists():
            raise FileNotFoundError(f"teacher not found: {cfg.teacher}")
        if not Path(cfg.bake).exists():
            raise FileNotFoundError(f"bake not found: {cfg.bake}")
        dt = _dtype(cfg.dtype)
        print(f"[fpa_prompt_eval] load teacher {cfg.teacher}", flush=True)
        teacher = load_causal_lm(cfg.teacher, tdev, dt)
        student = load_causal_lm(cfg.teacher, device, dt)
        path, fpa = swap_up_proj(student, cfg.bake, cfg.layer, device, cfg.use_sparse)
        freeze_except_fpa(student, fpa)
        tok = load_tokenizer(cfg.teacher, False, _vocab_from(teacher))
        print(f"[fpa_prompt_eval] student L{cfg.layer} → {path} {fpa.extra_repr()}", flush=True)

    ckpt = Path(cfg.ckpt_dir)
    freeze_p = ckpt / "freeze_init_trainables.pt"
    trained_p = ckpt / "trained_trainables.pt"
    if not freeze_p.is_file() or not trained_p.is_file():
        raise FileNotFoundError(f"missing snapshots in {ckpt} (need freeze_init_trainables.pt and trained_trainables.pt)")

    ids = encode_prompts(tok, prompts, cfg.seq_len, device)
    batches = batched(ids, cfg.batch)
    print(f"[fpa_prompt_eval] tokens {tuple(ids.shape)} batches={len(batches)}", flush=True)

    t0 = time.time()
    # dense teacher self-KL(logits, logits) — plumbing control, expect ~0
    kl_self, per_self = _teacher_self_kl(teacher, batches, cfg.temperature, tdev)
    print(f"[fpa_prompt_eval] self_kl_teacher={kl_self:.6e} (logits,logits control)", flush=True)

    fpa.restore_trainables(load_trainables_pt(freeze_p))
    kl_init, per_init = mean_kl_pairs(student, teacher, batches, cfg.temperature, device, tdev)
    print(f"[fpa_prompt_eval] kl_freeze_init={kl_init:.6f}", flush=True)

    fpa.restore_trainables(load_trainables_pt(trained_p))
    kl_trained, per_trained = mean_kl_pairs(student, teacher, batches, cfg.temperature, device, tdev)
    print(f"[fpa_prompt_eval] kl_trained={kl_trained:.6f}", flush=True)

    rel_vs_freeze = None if kl_init <= 0 else (kl_init - kl_trained) / kl_init
    summary = {
        "status": "not Done",
        "verdict": "plumbing_complete",
        "self_kl_teacher": kl_self,
        "kl_freeze_init": kl_init,
        "kl_trained": kl_trained,
        "rel_vs_freeze": rel_vs_freeze,
        "n_prompts": len(prompts),
        "seq_len": cfg.seq_len,
        "temperature": cfg.temperature,
        "layer_path": path if not cfg.synthetic else "up_proj",
        "trainable_keys": list(RECIPE_V0_TRAIN),
        "ckpt_dir": str(ckpt),
        "freeze_init_pt": str(freeze_p),
        "trained_pt": str(trained_p),
        "synthetic": cfg.synthetic,
        "device": str(device),
        "teacher_device": str(tdev),
        "wall_s": round(time.time() - t0, 3),
        "prompts_preview": prompts[:8],
        "per_batch": {
            "self_kl_teacher": per_self,
            "kl_freeze_init": per_init,
            "kl_trained": per_trained,
        },
        "note": "plumbing-only one-Linear KL numbers; not a quality claim; not Done",
        "cfg": asdict(cfg),
    }
    out_json = out_dir / "summary.json"
    out_jsonl = out_dir / "per_batch.jsonl"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, (a, b, c) in enumerate(zip(per_self, per_init, per_trained)):
            f.write(json.dumps({"batch": i, "self_kl_teacher": a, "kl_freeze_init": b, "kl_trained": c}) + "\n")
    print(f"[fpa_prompt_eval] wrote {out_json}  {out_jsonl}", flush=True)
    return summary


def _teacher_self_kl(
    teacher: nn.Module,
    batches: Sequence[torch.Tensor],
    T: float,
    device: torch.device,
) -> Tuple[float, List[float]]:
    """KL(softmax(logits/T) || log_softmax(same logits/T)) — dense self-check."""
    teacher.eval()
    per: List[float] = []
    with torch.no_grad():
        for b in batches:
            logits = _forward(teacher, b.to(device))
            per.append(float(kl_logits(logits, logits, T).item()))
    return sum(per) / max(1, len(per)), per


def _vocab_from(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "vocab_size"):
        return int(cfg.vocab_size)
    return 32000
