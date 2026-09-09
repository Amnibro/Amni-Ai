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

# Short English prompts — local, no wiki download. 48 sentences.
DEFAULT_PROMPTS: Tuple[str, ...] = (
    "The river flooded the valley after three days of rain.",
    "Please explain how a bicycle stays upright while moving.",
    "Paris is the capital of France and a major cultural center.",
    "What is seventeen times twenty-three?",
    "She left the keys on the kitchen table this morning.",
    "A compass points roughly toward magnetic north.",
    "Write a short poem about winter light on snow.",
    "The library closes at six on weekdays and four on Sunday.",
    "Why does ice float on water?",
    "He promised to call when the train reached Ottawa.",
    "Photosynthesis converts light into chemical energy in plants.",
    "Name three planets closer to the Sun than Jupiter.",
    "The bread needs twenty minutes more in the oven.",
    "If a car starts from rest and accelerates at three meters per second squared for four seconds, what is its speed?",
    "They argued about whether the map was upside down.",
    "Translate the word apple into Spanish.",
    "A week has seven days and a common year has three hundred sixty-five.",
    "The cat slept in the sunbeam until noon.",
    "Describe the water cycle in two sentences.",
    "Mercury is the closest planet to the Sun.",
    "I need a recipe for tomato soup without cream.",
    "Sound travels slower than light.",
    "The museum exhibit opens next Tuesday.",
    "How do you reverse a list in Python?",
    "The mountain trail is icy after sunset.",
    "Gold is denser than aluminum.",
    "Please summarize the plot of a typical fable.",
    "The battery lasted only four hours under load.",
    "What city is the capital of Japan?",
    "She planted beans along the fence in May.",
    "A triangle has three sides and three angles.",
    "The radio played a song I had not heard in years.",
    "Explain gravity to a ten-year-old.",
    "The harbor was quiet except for gulls.",
    "Two plus two equals four.",
    "He packed a raincoat because the forecast said storms.",
    "Iron rusts when exposed to water and air.",
    "Where does the Nile empty?",
    "The clock on the tower was five minutes fast.",
    "List the primary colors of light.",
    "They crossed the bridge before the fog arrived.",
    "A sonnet usually has fourteen lines.",
    "The server returned a 404 for the missing page.",
    "Why is the sky blue at noon?",
    "She measured the window before buying curtains.",
    "The Pacific is the largest ocean on Earth.",
    "Boil the water before adding the pasta.",
    "Can you spell the word necessary?",
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


def load_prompts(path: str = "", limit: int = 128) -> List[str]:
    texts: List[str] = []
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"prompts file not found: {path}")
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            data = json.loads(raw)
            if isinstance(data, list):
                texts = [str(x).strip() for x in data]
            elif isinstance(data, dict) and "prompts" in data:
                texts = [str(x).strip() for x in data["prompts"]]
            else:
                raise ValueError("JSON prompts must be a list or {prompts: [...]}")
        else:
            texts = [ln.strip() for ln in raw.splitlines()]
    else:
        texts = list(DEFAULT_PROMPTS)
    texts = [t for t in texts if t and not t.startswith("#")]
    if not texts:
        raise ValueError("no prompts after filtering")
    n = min(len(texts), max(1, min(int(limit), 128)))
    return texts[:n]


class _WordTokenizer:
    """Tiny fallback when transformers/teacher tokenizer is absent (synthetic)."""

    def __init__(self, vocab_size: int = 128, pad_id: int = 0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_id
        self.eos_token_id = 1

    def __call__(self, texts: Sequence[str], padding=True, truncation=True, max_length=64, return_tensors="pt"):
        rows = []
        for t in texts:
            ids = [(ord(c) % (self.vocab_size - 2)) + 2 for c in t[:max_length]]
            if not ids:
                ids = [2]
            if truncation:
                ids = ids[:max_length]
            rows.append(ids)
        if padding:
            m = max(len(r) for r in rows)
            m = min(m, max_length)
            rows = [r + [self.pad_token_id] * (m - len(r)) for r in rows]
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}


def load_tokenizer(teacher_path: str, synthetic: bool, vocab: int):
    if synthetic:
        return _WordTokenizer(vocab_size=vocab)
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers required for real-prompt eval") from e
    tok = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    return tok


def encode_prompts(tok: Any, prompts: Sequence[str], seq_len: int, device: torch.device) -> torch.Tensor:
    enc = tok(list(prompts), padding=True, truncation=True, max_length=seq_len, return_tensors="pt")
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    return ids.to(device)


def batched(ids: torch.Tensor, batch: int) -> List[torch.Tensor]:
    return [ids[i : i + batch] for i in range(0, ids.shape[0], batch)]


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

    prompts = load_prompts(cfg.prompts)
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
    kl_self, per_self = mean_kl_pairs(teacher, teacher, batches, cfg.temperature, tdev, tdev)
    print(f"[fpa_prompt_eval] teacher self-KL={kl_self:.6e} (expect ~0)", flush=True)

    fpa.restore_trainables(load_trainables_pt(freeze_p))
    kl_init, per_init = mean_kl_pairs(student, teacher, batches, cfg.temperature, device, tdev)
    print(f"[fpa_prompt_eval] freeze-init KL={kl_init:.6f}", flush=True)

    fpa.restore_trainables(load_trainables_pt(trained_p))
    kl_trained, per_trained = mean_kl_pairs(student, teacher, batches, cfg.temperature, device, tdev)
    print(f"[fpa_prompt_eval] trained KL={kl_trained:.6f}", flush=True)

    rel = None if kl_init <= 0 else (kl_init - kl_trained) / kl_init
    summary = {
        "status": "Lane A real-prompt eval v0 — NOT Done; not near-1; one-Linear probe only",
        "kl_teacher_self": kl_self,
        "kl_freeze_init": kl_init,
        "kl_trained": kl_trained,
        "rel_improvement": rel,
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
            "teacher_self": per_self,
            "freeze_init": per_init,
            "trained": per_trained,
        },
        "honesty": {
            "done": False,
            "near_1": False,
            "note": "full-vocab KL on a small English prompt list; not a quality or bpw claim",
        },
        "cfg": asdict(cfg),
    }
    out_json = out_dir / "summary.json"
    out_jsonl = out_dir / "per_batch.jsonl"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, (a, b, c) in enumerate(zip(per_self, per_init, per_trained)):
            f.write(json.dumps({"batch": i, "kl_teacher_self": a, "kl_freeze_init": b, "kl_trained": c}) + "\n")
    print(f"[fpa_prompt_eval] wrote {out_json}  {out_jsonl}", flush=True)
    return summary


def _vocab_from(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "vocab_size"):
        return int(cfg.vocab_size)
    return 32000
