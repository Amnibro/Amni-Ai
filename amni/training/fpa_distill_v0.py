"""Lane A FPA distill v0 — one-Linear proof (not Done, not near-1).

Authoritative recipe (hub: lane_a_fpa_distill_recipe_v1.md):

* Teacher: Qwen/Qwen3.5-4B (local ``downloaded_models/Qwen3.5-4B``)
* Init bake: ``bakes/qwen35_4b_hc_fpa_onetensor_probe`` (arm r64_K256_kf0.004)
* Replace ONLY ``model.language_model.layers.15.mlp.up_proj`` with FpaLinear
* Freeze: codebooks + pq_idx (+ sparse)
* Train: U, V, U_scale, V_scale, pq_scale (float scales; U/V = unpacked int4
  codes as float; forward is ``(U*U_scale) @ ((V*V_scale).T @ x)`` + pq + sparse)
* NEVER materialise dense W
* Loss: KL(teacher ‖ student) at T=2, optional MSE on that layer's output
* Steps 2k–5k, LR 1e-4, eval KL every 500 vs freeze-init baseline
* Kill: after 5k, KL not improved vs freeze-init beyond ±5% noise → FAIL
* GDN / linear_attn: leave dense untouched

CLI: ``python scripts/lane_a_fpa_distill_v0.py``
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from amni.inference.fpa_linear import FpaLinear, RECIPE_V0_TRAIN, load_fpa_linear

LAYER_PATHS = (
    "model.language_model.layers.{i}.mlp.up_proj",
    "model.model.language_model.layers.{i}.mlp.up_proj",
    "language_model.layers.{i}.mlp.up_proj",
    "model.layers.{i}.mlp.up_proj",
)
DEFAULT_TEACHER = "downloaded_models/Qwen3.5-4B"
DEFAULT_BAKE = "bakes/qwen35_4b_hc_fpa_onetensor_probe"
DEFAULT_LAYER = 15
DEFAULT_T = 2.0
DEFAULT_LR = 1e-4
DEFAULT_STEPS = 5000
DEFAULT_EVAL_EVERY = 500
KILL_NOISE = 0.05


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    p: Any = root
    for part in path.split("."):
        p = getattr(p, part)
    return p


def set_module(root: nn.Module, path: str, new: nn.Module) -> None:
    parts = path.split(".")
    parent = resolve_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    setattr(parent, parts[-1], new)


def find_up_proj(model: nn.Module, layer: int = DEFAULT_LAYER) -> str:
    for tmpl in LAYER_PATHS:
        path = tmpl.format(i=layer)
        try:
            resolve_module(model, path)
            return path
        except AttributeError:
            continue
    raise AttributeError(
        f"could not find layers.{layer}.mlp.up_proj under {type(model).__name__} "
        f"(tried {LAYER_PATHS})"
    )


def pick_device(explicit: str = "") -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def kl_logits(student: torch.Tensor, teacher: torch.Tensor, T: float = DEFAULT_T) -> torch.Tensor:
    """Token-mean KL(softmax(t/T) ‖ log_softmax(s/T)) * T² (standard KD)."""
    t = teacher.float().reshape(-1, teacher.shape[-1])
    s = student.float().reshape(-1, student.shape[-1])
    p = F.softmax(t / T, dim=-1)
    log_q = F.log_softmax(s / T, dim=-1)
    kl = (p * (p.clamp_min(1e-12).log() - log_q)).sum(-1).mean()
    return kl * (T * T)


def kill_verdict(kl_init: float, kl_final: float, noise: float = KILL_NOISE) -> Tuple[str, str]:
    """PASS iff KL dropped by more than ``noise`` relative to freeze-init."""
    if kl_init <= 0:
        return "FAIL", f"freeze-init KL non-positive ({kl_init})"
    rel = (kl_init - kl_final) / kl_init
    if rel > noise:
        return "PASS", f"KL improved {rel * 100:.2f}% vs freeze-init ({kl_init:.6f} → {kl_final:.6f})"
    return (
        "FAIL",
        f"KL not improved beyond ±{noise * 100:.0f}% noise "
        f"(rel={rel * 100:.2f}%; {kl_init:.6f} → {kl_final:.6f})",
    )


@dataclass
class DistillConfig:
    teacher: str = DEFAULT_TEACHER
    bake: str = DEFAULT_BAKE
    out: str = "logs/lane_a_fpa_distill_v0"
    steps: int = DEFAULT_STEPS
    lr: float = DEFAULT_LR
    eval_every: int = DEFAULT_EVAL_EVERY
    temperature: float = DEFAULT_T
    mse_weight: float = 0.0
    layer: int = DEFAULT_LAYER
    batch: int = 2
    seq_len: int = 64
    eval_batches: int = 4
    seed: int = 0
    device: str = ""
    teacher_device: str = ""
    dtype: str = "bfloat16"
    synthetic: bool = False
    use_sparse: bool = True


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}[name]


class _TinyUpLM(nn.Module):
    """Synthetic stand-in: embed → up_proj → head. Used when --synthetic (CI / no 4B)."""

    def __init__(self, inn: int, out: int, vocab: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab, inn)
        self.up_proj = nn.Linear(inn, out, bias=False)
        self.lm_head = nn.Linear(out, vocab, bias=False)
        self.vocab_size = vocab

    def forward(self, input_ids: torch.Tensor):
        h = self.up_proj(self.embed(input_ids))
        return type("Out", (), {"logits": self.lm_head(h), "last_hidden_state": h})()


def build_synthetic_pair(seed: int = 0, inn: int = 128, out: int = 64, rank: int = 8):
    g = torch.Generator().manual_seed(seed)
    teacher = _TinyUpLM(inn, out)
    with torch.no_grad():
        teacher.embed.weight.normal_(generator=g)
        teacher.up_proj.weight.copy_(torch.randn(out, inn, generator=g) / (inn ** 0.5))
        teacher.lm_head.weight.normal_(generator=g)
    student = _TinyUpLM(inn, out)
    student.load_state_dict(teacher.state_dict())
    fpa = FpaLinear.born(inn, out, rank=rank, trainable=True, seed=seed)
    fpa.apply_recipe_v0()
    student.up_proj = fpa
    return teacher, student, "up_proj", fpa


def load_causal_lm(path: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise RuntimeError("transformers is required to load the Qwen teacher") from e
    try:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True)
    except Exception:
        from transformers import AutoModel

        m = AutoModel.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True)
    return m.to(device).eval()


def _logits(out: Any) -> torch.Tensor:
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"model output has no logits: {type(out)}")


def _forward(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    try:
        out = model(input_ids=ids, use_cache=False)
    except TypeError:
        out = model(ids)
    return _logits(out)


def swap_up_proj(student: nn.Module, bake: str, layer: int, device: torch.device, use_sparse: bool) -> Tuple[str, FpaLinear]:
    path = find_up_proj(student, layer)
    dense = resolve_module(student, path)
    fpa = load_fpa_linear(bake, trainable=True, device=device).apply_recipe_v0()
    if not use_sparse and fpa.sparse_nnz:
        fpa.sp_indices = fpa.sp_indices[:0]
    inn = int(getattr(dense, "in_features", fpa.in_features))
    out = int(getattr(dense, "out_features", fpa.out_features))
    if fpa.in_features != inn or fpa.out_features != out:
        raise ValueError(
            f"FPA bake {fpa.in_features}×{fpa.out_features} != dense {path} {inn}×{out}"
        )
    set_module(student, path, fpa)
    return path, fpa


def freeze_except_fpa(student: nn.Module, fpa: FpaLinear) -> None:
    for p in student.parameters():
        p.requires_grad_(False)
    fpa.apply_recipe_v0()
    for p in fpa.trainable_parameters():
        p.requires_grad_(True)


def random_ids(vocab: int, batch: int, seq: int, device: torch.device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    return torch.randint(0, max(2, vocab), (batch, seq), device=device, generator=generator)


def _vocab_size(model: nn.Module) -> int:
    for attr in ("vocab_size", "config"):
        v = getattr(model, attr, None)
        if isinstance(v, int):
            return v
        if v is not None and hasattr(v, "vocab_size"):
            return int(v.vocab_size)
    emb = getattr(model, "get_input_embeddings", lambda: None)()
    if emb is not None and hasattr(emb, "num_embeddings"):
        return int(emb.num_embeddings)
    return 32000


def load_trainables_pt(path: Path) -> Dict[str, torch.Tensor]:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict):
        raise TypeError(f"{path} is not a trainables dict")
    return {k: v for k, v in blob.items() if torch.is_tensor(v)}


def save_trainables_pt(path: Path, snap: Dict[str, torch.Tensor]) -> Path:
    path = Path(path)
    torch.save({k: v.detach().cpu().contiguous() for k, v in snap.items()}, path)
    return path


def save_run_artifacts(out_dir: Path, fpa: FpaLinear, init_snap: Dict[str, torch.Tensor]) -> Dict[str, Path]:
    """Write freeze-init + trained snapshots. Live float codes are restored after repack export."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze_p = save_trainables_pt(out_dir / "freeze_init_trainables.pt", init_snap)
    trained_snap = fpa.snapshot_trainables()
    trained_p = save_trainables_pt(out_dir / "trained_trainables.pt", trained_snap)
    fpa_p = out_dir / "trained_fpa.pt"
    torch.save(fpa.state_dict(), fpa_p)
    live = fpa.snapshot_trainables()
    fpa.repack_int4()
    repack_p = out_dir / "trained_fpa_repacked.pt"
    torch.save(fpa.state_dict(), repack_p)
    fpa.restore_trainables(live)
    return {
        "freeze_init": freeze_p,
        "trained": trained_p,
        "trained_fpa": fpa_p,
        "trained_fpa_repacked": repack_p,
    }


def eval_kl(
    teacher: nn.Module,
    student: nn.Module,
    batches: Sequence[torch.Tensor],
    T: float,
) -> float:
    teacher.eval()
    student.eval()
    acc = []
    with torch.no_grad():
        for ids in batches:
            kt = _forward(teacher, ids)
            ks = _forward(student, ids)
            acc.append(float(kl_logits(ks, kt, T).item()))
    return sum(acc) / max(1, len(acc))


def run_distill(cfg: DistillConfig) -> Dict[str, Any]:
    device = pick_device(cfg.device)
    tdev = pick_device(cfg.teacher_device) if cfg.teacher_device else device
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = out_dir / "kl_curve.jsonl"
    log_csv = out_dir / "kl_curve.csv"

    if cfg.synthetic:
        teacher, student, path, fpa = build_synthetic_pair(seed=cfg.seed)
        teacher = teacher.to(tdev)
        student = student.to(device)
        fpa = student.up_proj
        print(f"[fpa_distill_v0] SYNTHETIC one-Linear proof path={path} {fpa.extra_repr()}", flush=True)
    else:
        if not Path(cfg.teacher).exists():
            raise FileNotFoundError(f"teacher not found: {cfg.teacher} (pass --synthetic for CI)")
        if not Path(cfg.bake).exists():
            raise FileNotFoundError(f"FPA bake not found: {cfg.bake}")
        dt = _dtype(cfg.dtype)
        print(f"[fpa_distill_v0] load teacher {cfg.teacher} → {tdev} {cfg.dtype}", flush=True)
        teacher = load_causal_lm(cfg.teacher, tdev, dt)
        print(f"[fpa_distill_v0] load student copy + swap L{cfg.layer} up_proj from {cfg.bake}", flush=True)
        student = load_causal_lm(cfg.teacher, device, dt)
        path, fpa = swap_up_proj(student, cfg.bake, cfg.layer, device, cfg.use_sparse)
        print(f"[fpa_distill_v0] replaced {path} with {fpa.extra_repr()}", flush=True)

    freeze_except_fpa(student, fpa)
    n_train = sum(p.numel() for p in fpa.trainable_parameters())
    names = [n for n, p in fpa.named_parameters() if p.requires_grad]
    print(f"[fpa_distill_v0] trainable {names}  n={n_train}  (atlas frozen)", flush=True)
    if set(names) != set(RECIPE_V0_TRAIN):
        print(f"[fpa_distill_v0] WARNING expected {RECIPE_V0_TRAIN} got {names}", flush=True)

    vocab = _vocab_size(teacher)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + 17)
    eval_set = [
        random_ids(vocab, cfg.batch, cfg.seq_len, device, gen) for _ in range(cfg.eval_batches)
    ]
    if tdev != device:
        eval_t = [b.to(tdev) for b in eval_set]
    else:
        eval_t = eval_set

    def _eval() -> float:
        if tdev != device:
            # student on device, teacher on tdev — move ids
            acc = []
            student.eval()
            teacher.eval()
            with torch.no_grad():
                for s_ids, t_ids in zip(eval_set, eval_t):
                    acc.append(float(kl_logits(_forward(student, s_ids), _forward(teacher, t_ids), cfg.temperature).item()))
            return sum(acc) / max(1, len(acc))
        return eval_kl(teacher, student, eval_set, cfg.temperature)

    kl_init = _eval()
    init_snap = fpa.snapshot_trainables()
    print(f"[fpa_distill_v0] freeze-init eval KL={kl_init:.6f}  T={cfg.temperature}", flush=True)

    opt = torch.optim.AdamW(list(fpa.trainable_parameters()), lr=cfg.lr)
    rows: List[Dict[str, Any]] = []

    def _log(rec: Dict[str, Any]) -> None:
        rows.append(rec)
        with open(log_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    with open(log_csv, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=["step", "split", "kl", "mse", "kl_init", "lr"]).writeheader()

    def _csv(rec: Dict[str, Any]) -> None:
        with open(log_csv, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=["step", "split", "kl", "mse", "kl_init", "lr"]).writerow(
                {k: rec.get(k, "") for k in ["step", "split", "kl", "mse", "kl_init", "lr"]}
            )

    rec0 = {"step": 0, "split": "eval", "kl": kl_init, "mse": 0.0, "kl_init": kl_init, "lr": cfg.lr}
    _log(rec0)
    _csv(rec0)

    t0 = time.time()
    train_gen = torch.Generator(device=device)
    train_gen.manual_seed(cfg.seed + 99)
    last_eval = kl_init
    for step in range(1, cfg.steps + 1):
        ids = random_ids(vocab, cfg.batch, cfg.seq_len, device, train_gen)
        t_ids = ids.to(tdev) if tdev != device else ids
        with torch.no_grad():
            t_logits = _forward(teacher, t_ids)
            if tdev != device:
                t_logits = t_logits.to(device)
        acts: Dict[str, torch.Tensor] = {}
        hooks = []
        if cfg.mse_weight:
            def _thook(_m, _i, o):
                acts["t"] = o.detach() if torch.is_tensor(o) else o[0].detach()

            def _shook(_m, _i, o):
                acts["s"] = o if torch.is_tensor(o) else o[0]

            try:
                t_mod = resolve_module(teacher, path if not cfg.synthetic else "up_proj")
            except AttributeError:
                t_mod = teacher.up_proj
            hooks.append(t_mod.register_forward_hook(_thook))
            hooks.append(fpa.register_forward_hook(_shook))
        try:
            s_logits = _forward(student, ids)
            loss_kl = kl_logits(s_logits, t_logits, cfg.temperature)
            loss_mse = s_logits.new_zeros(())
            if cfg.mse_weight and "s" in acts and "t" in acts:
                loss_mse = F.mse_loss(acts["s"].float(), acts["t"].float().to(acts["s"].device))
            loss = loss_kl + cfg.mse_weight * loss_mse
        finally:
            for h in hooks:
                h.remove()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        rec = {
            "step": step,
            "split": "train",
            "kl": float(loss_kl.item()),
            "mse": float(loss_mse.item()) if cfg.mse_weight else 0.0,
            "kl_init": kl_init,
            "lr": cfg.lr,
        }
        _log(rec)
        _csv(rec)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            last_eval = _eval()
            erec = {"step": step, "split": "eval", "kl": last_eval, "mse": 0.0, "kl_init": kl_init, "lr": cfg.lr}
            _log(erec)
            _csv(erec)
            print(
                f"[fpa_distill_v0] step {step}/{cfg.steps}  train_kl={rec['kl']:.6f}  eval_kl={last_eval:.6f}  init={kl_init:.6f}",
                flush=True,
            )

    verdict, reason = kill_verdict(kl_init, last_eval)
    kill_armed = cfg.steps >= DEFAULT_STEPS
    if not kill_armed and verdict == "FAIL":
        reason = reason + f" (kill armed only at steps>={DEFAULT_STEPS}; this run is informational)"
    summary = {
        "verdict": verdict,
        "kill_armed": kill_armed,
        "reason": reason,
        "kl_init": kl_init,
        "kl_final": last_eval,
        "steps": cfg.steps,
        "layer_path": path if not cfg.synthetic else "up_proj",
        "trainable": names,
        "synthetic": cfg.synthetic,
        "device": str(device),
        "wall_s": round(time.time() - t0, 3),
        "choice": "U/V are unpacked int4 codes (float); forward (U*U_scale)@(V*V_scale)^T x + pq + sparse; then repack_int4()",
        "status": "Lane A distill v0 bootstrap — NOT Done; freeze-quantize REFUTED; not near-1",
        "cfg": asdict(cfg),
    }
    artifacts = save_run_artifacts(out_dir, fpa, init_snap)
    summary["artifacts"] = {k: str(v) for k, v in artifacts.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[fpa_distill_v0] {verdict}: {reason}", flush=True)
    print(f"[fpa_distill_v0] logs {log_jsonl}  {log_csv}  {out_dir / 'summary.json'}", flush=True)
    print(f"[fpa_distill_v0] saved {artifacts['freeze_init']}  {artifacts['trained']}  {artifacts['trained_fpa']}", flush=True)
    return summary
