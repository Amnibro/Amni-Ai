"""Lane A FPA Arm A2 — real-prompt train+eval + layer MSE (not Done).

A2 = A1 data discipline (same real-prompt family for train and eval) plus
``mse_weight ∈ {0.5, 1.0}`` on L15 ``mlp.up_proj`` output vs the dense teacher.

v0 trained on random token ids with ``mse_weight=0`` (plumbing). The 32×128
real-prompt KL that followed (freeze 0.451462 → trained 0.487534, +8% worse)
is the reason A2 exists. This module does not redefine that 5% kill bar.

Freeze: codebooks + pq_idx (+ sparse). Train: U, V, U_scale, V_scale, pq_scale.
Forward stays ``U@(Vᵀx)+pq+sparse`` — never dense W.

Kill: after 5k real-prompt steps, if KL is not ≥5% better than freeze →
escalate A3. A3 is not implemented here. This script never writes PASS.
Tidus owns SHIP.

CLI: ``python scripts/lane_a_fpa_distill_a2.py``
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from amni.inference.fpa_linear import RECIPE_V0_TRAIN, FpaLinear
from amni.inference.gf17_fpa import bpw_allin
from amni.training.fpa_distill_v0 import (
    DEFAULT_BAKE,
    DEFAULT_EVAL_EVERY,
    DEFAULT_LAYER,
    DEFAULT_LR,
    DEFAULT_STEPS,
    DEFAULT_T,
    DEFAULT_TEACHER,
    _dtype,
    _forward,
    _vocab_size,
    build_synthetic_pair,
    freeze_except_fpa,
    kl_logits,
    load_causal_lm,
    pick_device,
    resolve_module,
    save_run_artifacts,
    swap_up_proj,
)
from amni.training.fpa_real_prompts import (
    FAMILY_ID,
    FAMILY_V1,
    PROMPT_CAP,
    cycle_batch,
    data_mix_id,
    encode_prompts,
    load_prompts,
    load_tokenizer,
    prompt_fingerprint,
    source_tag,
)
from amni.training.fpa_summary_a2 import (
    ARM_ID,
    build_summary,
    write_summary,
)

ALLOWED_MSE_WEIGHTS = (0.5, 1.0)
DEFAULT_MSE_WEIGHT = 1.0
DEFAULT_N_PROMPTS = 32
DEFAULT_SEQ_LEN = 128
DEFAULT_OUT = "logs/lane_a_fpa_distill_a2"


def validate_mse_weight(weight: float) -> float:
    w = float(weight)
    if w not in ALLOWED_MSE_WEIGHTS:
        raise ValueError(
            f"A2 mse_weight must be one of {ALLOWED_MSE_WEIGHTS} (got {weight}). "
            "mse_weight=0 was v0/A0 random-id plumbing and is not A2."
        )
    return w


@dataclass
class A2Config:
    teacher: str = DEFAULT_TEACHER
    bake: str = DEFAULT_BAKE
    out: str = DEFAULT_OUT
    steps: int = DEFAULT_STEPS
    lr: float = DEFAULT_LR
    eval_every: int = DEFAULT_EVAL_EVERY
    temperature: float = DEFAULT_T
    mse_weight: float = DEFAULT_MSE_WEIGHT
    layer: int = DEFAULT_LAYER
    batch: int = 2
    seq_len: int = DEFAULT_SEQ_LEN
    n_prompts: int = DEFAULT_N_PROMPTS
    prompts: str = ""
    seed: int = 0
    device: str = ""
    teacher_device: str = ""
    dtype: str = "bfloat16"
    synthetic: bool = False
    use_sparse: bool = True
    skip_gpu_train: bool = False

    def __post_init__(self) -> None:
        self.mse_weight = validate_mse_weight(self.mse_weight)
        self.n_prompts = max(1, min(int(self.n_prompts), PROMPT_CAP))
        self.seq_len = max(8, int(self.seq_len))


def _teacher_up(teacher: nn.Module, path: str, synthetic: bool) -> nn.Module:
    try:
        return resolve_module(teacher, path if not synthetic else "up_proj")
    except AttributeError:
        return teacher.up_proj


def _attach_layer_hooks(teacher: nn.Module, fpa: FpaLinear, path: str, synthetic: bool):
    acts: Dict[str, torch.Tensor] = {}

    def _thook(_m, _i, o):
        acts["t"] = o.detach() if torch.is_tensor(o) else o[0].detach()

    def _shook(_m, _i, o):
        acts["s"] = o if torch.is_tensor(o) else o[0]

    hooks = [
        _teacher_up(teacher, path, synthetic).register_forward_hook(_thook),
        fpa.register_forward_hook(_shook),
    ]
    return acts, hooks


def _mse_from_acts(acts: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if "s" not in acts or "t" not in acts:
        raise RuntimeError("layer MSE hooks did not fire")
    s = acts["s"]
    t = acts["t"].to(device=s.device, dtype=torch.float32)
    return F.mse_loss(s.float(), t)


def eval_kl_mse_self(
    teacher: nn.Module,
    student: nn.Module,
    fpa: FpaLinear,
    path: str,
    batches: Sequence[torch.Tensor],
    *,
    T: float,
    device: torch.device,
    tdev: torch.device,
    synthetic: bool,
) -> Tuple[float, float, float]:
    """Mean KL(student‖teacher), layer MSE, and teacher self-KL on the same batches."""
    teacher.eval()
    student.eval()
    kls: List[float] = []
    mses: List[float] = []
    selfs: List[float] = []
    with torch.no_grad():
        for ids in batches:
            s_ids = ids.to(device)
            t_ids = ids.to(tdev)
            acts, hooks = _attach_layer_hooks(teacher, fpa, path, synthetic)
            try:
                t_logits = _forward(teacher, t_ids)
                s_logits = _forward(student, s_ids)
                if t_logits.device != s_logits.device:
                    t_logits = t_logits.to(s_logits.device)
                kls.append(float(kl_logits(s_logits, t_logits, T).item()))
                mses.append(float(_mse_from_acts(acts).item()))
                selfs.append(float(kl_logits(t_logits, t_logits, T).item()))
            finally:
                for h in hooks:
                    h.remove()
    n = max(1, len(kls))
    return sum(kls) / n, sum(mses) / n, sum(selfs) / n


def run_distill_a2(cfg: A2Config) -> Dict[str, Any]:
    if cfg.skip_gpu_train and not cfg.synthetic:
        raise RuntimeError("skip_gpu_train is the CI latch; use --synthetic for no-GPU smoke")
    device = pick_device("cpu" if cfg.skip_gpu_train else cfg.device)
    if cfg.skip_gpu_train:
        device = torch.device("cpu")
    tdev = pick_device(cfg.teacher_device) if cfg.teacher_device else device
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(cfg.prompts, limit=cfg.n_prompts, family=FAMILY_V1)
    mix = data_mix_id(len(prompts), cfg.seq_len, source=source_tag(cfg.prompts))
    print(
        f"[fpa_distill_a2] arm={ARM_ID} data_mix={mix} mse_weight={cfg.mse_weight}  NOT Done",
        flush=True,
    )
    print(
        f"[fpa_distill_a2] family={source_tag(cfg.prompts)} n={len(prompts)} seq={cfg.seq_len} "
        f"sha={prompt_fingerprint(prompts)} (train and eval share this table)",
        flush=True,
    )

    if cfg.synthetic:
        teacher, student, path, fpa = build_synthetic_pair(seed=cfg.seed)
        teacher = teacher.to(tdev)
        student = student.to(device)
        fpa = student.up_proj
        vocab = int(getattr(teacher, "vocab_size", 128))
        tok = load_tokenizer("", True, vocab)
        print(f"[fpa_distill_a2] SYNTHETIC path={path} {fpa.extra_repr()}", flush=True)
    else:
        if not Path(cfg.teacher).exists():
            raise FileNotFoundError(f"teacher not found: {cfg.teacher} (pass --synthetic for CI)")
        if not Path(cfg.bake).exists():
            raise FileNotFoundError(f"FPA bake not found: {cfg.bake} (expected under bakes/)")
        dt = _dtype(cfg.dtype)
        print(f"[fpa_distill_a2] load teacher {cfg.teacher} -> {tdev} {cfg.dtype}", flush=True)
        teacher = load_causal_lm(cfg.teacher, tdev, dt)
        print(f"[fpa_distill_a2] load student + swap L{cfg.layer} up_proj from {cfg.bake}", flush=True)
        student = load_causal_lm(cfg.teacher, device, dt)
        path, fpa = swap_up_proj(student, cfg.bake, cfg.layer, device, cfg.use_sparse)
        print(f"[fpa_distill_a2] replaced {path} with {fpa.extra_repr()}", flush=True)
        vocab = _vocab_size(teacher)
        tok = load_tokenizer(cfg.teacher, False, vocab)

    freeze_except_fpa(student, fpa)
    names = [n for n, p in fpa.named_parameters() if p.requires_grad]
    print(f"[fpa_distill_a2] trainable {names}  (codebooks+pq_idx+sparse frozen)", flush=True)
    if set(names) != set(RECIPE_V0_TRAIN):
        print(f"[fpa_distill_a2] WARNING expected {RECIPE_V0_TRAIN} got {names}", flush=True)

    ids = encode_prompts(tok, prompts, cfg.seq_len, device)
    eval_batches = [ids[i : i + cfg.batch] for i in range(0, ids.shape[0], cfg.batch)]
    print(f"[fpa_distill_a2] prompt tokens {tuple(ids.shape)} eval_batches={len(eval_batches)}", flush=True)

    def _eval() -> Tuple[float, float, float]:
        return eval_kl_mse_self(
            teacher, student, fpa, path, eval_batches,
            T=cfg.temperature, device=device, tdev=tdev, synthetic=cfg.synthetic,
        )

    freeze_kl, mse_freeze, teacher_self = _eval()
    init_snap = fpa.snapshot_trainables()
    print(
        f"[fpa_distill_a2] freeze_kl={freeze_kl:.6f} layer_mse.freeze={mse_freeze:.6f} "
        f"teacher_self_kl={teacher_self:.6e}",
        flush=True,
    )

    opt = torch.optim.AdamW(list(fpa.trainable_parameters()), lr=cfg.lr)
    log_jsonl = out_dir / "kl_curve.jsonl"
    log_csv = out_dir / "kl_curve.csv"
    rows: List[Dict[str, Any]] = []
    fields = ["step", "split", "kl", "mse", "kl_init", "lr", "data_mix"]

    def _log(rec: Dict[str, Any]) -> None:
        rows.append(rec)
        with open(log_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    with open(log_csv, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    def _csv(rec: Dict[str, Any]) -> None:
        with open(log_csv, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({k: rec.get(k, "") for k in fields})

    rec0 = {
        "step": 0, "split": "eval", "kl": freeze_kl, "mse": mse_freeze,
        "kl_init": freeze_kl, "lr": cfg.lr, "data_mix": mix,
    }
    _log(rec0)
    _csv(rec0)

    t0 = time.time()
    last_kl, last_mse = freeze_kl, mse_freeze
    for step in range(1, cfg.steps + 1):
        batch_ids = cycle_batch(ids, cfg.batch, step)
        t_ids = batch_ids.to(tdev) if tdev != device else batch_ids
        acts, hooks = _attach_layer_hooks(teacher, fpa, path, cfg.synthetic)
        try:
            with torch.no_grad():
                t_logits = _forward(teacher, t_ids)
                if tdev != device:
                    t_logits = t_logits.to(device)
            s_logits = _forward(student, batch_ids)
            loss_kl = kl_logits(s_logits, t_logits, cfg.temperature)
            loss_mse = _mse_from_acts(acts)
            loss = loss_kl + cfg.mse_weight * loss_mse
        finally:
            for h in hooks:
                h.remove()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        rec = {
            "step": step, "split": "train", "kl": float(loss_kl.item()),
            "mse": float(loss_mse.item()), "kl_init": freeze_kl, "lr": cfg.lr, "data_mix": mix,
        }
        _log(rec)
        _csv(rec)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            last_kl, last_mse, _self = _eval()
            erec = {
                "step": step, "split": "eval", "kl": last_kl, "mse": last_mse,
                "kl_init": freeze_kl, "lr": cfg.lr, "data_mix": mix,
            }
            _log(erec)
            _csv(erec)
            print(
                f"[fpa_distill_a2] step {step}/{cfg.steps}  train_kl={rec['kl']:.6f}  "
                f"eval_kl={last_kl:.6f}  layer_mse={last_mse:.6f}  freeze={freeze_kl:.6f}",
                flush=True,
            )

    artifacts = save_run_artifacts(out_dir, fpa, init_snap)
    extras = {
        "layer_path": path if not cfg.synthetic else "up_proj",
        "trainable": names,
        "synthetic": cfg.synthetic,
        "device": str(device),
        "teacher_device": str(tdev),
        "wall_s": round(time.time() - t0, 3),
        "mse_weight": cfg.mse_weight,
        "prompt_sha256_12": prompt_fingerprint(prompts),
        "n_prompts": len(prompts),
        "seq_len": cfg.seq_len,
        "family_id": FAMILY_ID,
        "artifacts": {k: str(v) for k, v in artifacts.items()},
        "cfg": asdict(cfg),
        "choice": "U/V unpacked int4 codes; forward (U*U_scale)@((V*V_scale).T @ x)+pq+sparse; never dense W",
        "hypothesis": (
            "v0 fail consistent with random-id train + mse_weight=0; "
            "A2 retrains on the same real-prompt family used for eval"
        ),
        "prior_real_prompt_32x128": {
            "freeze_kl": 0.451462,
            "trained_kl": 0.487534,
            "note": "v0 random-id train then real-prompt eval; +8% worse; not this run",
        },
    }
    summary = build_summary(
        freeze_kl=freeze_kl,
        trained_kl=last_kl,
        teacher_self_kl=teacher_self,
        bpw_allin=float(bpw_allin(fpa)),
        layer_mse_freeze=mse_freeze,
        layer_mse_trained=last_mse,
        steps=cfg.steps,
        data_mix=mix,
        extras=extras,
    )
    write_summary(out_dir / "summary.json", summary)
    print(
        f"[fpa_distill_a2] verdict={summary['verdict']} next={summary['kill']['next_action']} "
        f"(not PASS, not Done)",
        flush=True,
    )
    print(f"[fpa_distill_a2] wrote {out_dir / 'summary.json'}", flush=True)
    return summary


def parse_a2_args(argv: Optional[Sequence[str]] = None):
    """Argparse for the A2 CLI — importable so tests do not need a GPU or a run."""
    import argparse

    p = argparse.ArgumentParser(
        description="Lane A FPA Arm A2 real-prompt distill (not Done; never PASS)"
    )
    p.add_argument("--teacher", default=DEFAULT_TEACHER)
    p.add_argument("--bake", default=DEFAULT_BAKE, help="FPA bake dir (Antman: bakes/qwen35_4b_hc_fpa_onetensor_probe)")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    p.add_argument("--temperature", type=float, default=DEFAULT_T)
    p.add_argument(
        "--mse-weight",
        type=float,
        default=DEFAULT_MSE_WEIGHT,
        help="layer MSE weight; A2 allows 1.0 (default) or 0.5 only",
    )
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS, help="32–128 from real_prompt_en_v1")
    p.add_argument("--prompts", default="", help="optional .txt/.json; default = real_prompt_en_v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="", help="cuda / cpu; empty = HIP/CUDA if available")
    p.add_argument("--teacher-device", default="")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--synthetic", action="store_true", help="tiny Linear pair (CI / no 4B / no GPU train)")
    p.add_argument("--no-sparse", action="store_true")
    p.add_argument(
        "--skip-gpu-train",
        action="store_true",
        help="CI latch: force CPU and require --synthetic (never start a GPU 5k)",
    )
    p.add_argument("--inventory", action="store_true", help="print FPA/A2 inventory JSON and exit")
    return p.parse_args(list(argv) if argv is not None else None)


def config_from_args(args) -> A2Config:
    return A2Config(
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
        n_prompts=args.n_prompts,
        prompts=args.prompts,
        seed=args.seed,
        device=args.device,
        teacher_device=args.teacher_device,
        dtype=args.dtype,
        synthetic=args.synthetic,
        use_sparse=not args.no_sparse,
        skip_gpu_train=args.skip_gpu_train,
    )
