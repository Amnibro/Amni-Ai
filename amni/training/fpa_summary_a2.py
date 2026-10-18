"""Kimahri-pack summary for Lane A Arm A2. Honest fields only. Never PASS.

Score these keys and no silent rename of the 5% kill bar:

    freeze_kl, trained_kl, delta_vs_freeze, teacher_self_kl,
    bpw_allin, layer_mse, arm_id, steps, data_mix

``layer_mse`` is ``{"freeze": float, "trained": float}`` on L15 ``mlp.up_proj``
output vs the dense teacher (same real-prompt family as KL).

``delta_vs_freeze`` = ``(freeze_kl - trained_kl) / freeze_kl``.
Positive means trained KL is lower (better). The kill bar stays ≥5% better
after 5k real-prompt steps. Missing that bar → escalate A3 (not implemented
here). Beating the bar is **not** a PASS / Done / ship claim — Tidus owns SHIP.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

ARM_ID = "A2"
KILL_BAR = 0.05  # same 5% relative bar as v0; not redefined
KILL_STEPS = 5000
KIMAHRI_KEYS: Tuple[str, ...] = (
    "freeze_kl",
    "trained_kl",
    "delta_vs_freeze",
    "teacher_self_kl",
    "bpw_allin",
    "layer_mse",
    "arm_id",
    "steps",
    "data_mix",
)


def relative_delta(freeze_kl: float, trained_kl: float) -> Optional[float]:
    if freeze_kl is None or freeze_kl <= 0:
        return None
    return (float(freeze_kl) - float(trained_kl)) / float(freeze_kl)


def kill_record(freeze_kl: float, trained_kl: float, steps: int) -> Dict[str, Any]:
    """Kill / escalate record. Never uses PASS/FAIL as the result token."""
    rel = relative_delta(freeze_kl, trained_kl)
    rec: Dict[str, Any] = {
        "armed": int(steps) >= KILL_STEPS,
        "bar": KILL_BAR,
        "bar_meaning": "trained_kl must be >=5% lower than freeze_kl after 5k real-prompt steps",
        "rel_improve": rel,
        "steps": int(steps),
        "kill_steps": KILL_STEPS,
    }
    if not rec["armed"]:
        rec["result"] = "kill_not_armed"
        rec["next_action"] = "informational"
        rec["note"] = "5k kill not armed; do not treat this run as a quality call"
        return rec
    if rel is not None and rel >= KILL_BAR:
        rec["result"] = "improved_ge_5pct"
        rec["next_action"] = "hold_for_tidus_ship"
        rec["note"] = "met the 5% kill bar; not Done; Tidus owns SHIP; do not mark PASS"
        return rec
    rec["result"] = "not_ge_5pct_better"
    rec["next_action"] = "escalate_a3"
    rec["note"] = "after 5k, real-prompt KL not >=5% better than freeze → escalate A3 (A3 is not in this patch)"
    return rec


def _is_pass_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().upper() == "PASS"


def build_summary(
    *,
    freeze_kl: float,
    trained_kl: float,
    teacher_self_kl: float,
    bpw_allin: float,
    layer_mse_freeze: float,
    layer_mse_trained: float,
    steps: int,
    data_mix: str,
    extras: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    delta = relative_delta(freeze_kl, trained_kl)
    kill = kill_record(freeze_kl, trained_kl, steps)
    summary: Dict[str, Any] = {
        "freeze_kl": float(freeze_kl),
        "trained_kl": float(trained_kl),
        "delta_vs_freeze": delta,
        "teacher_self_kl": float(teacher_self_kl),
        "bpw_allin": float(bpw_allin),
        "layer_mse": {"freeze": float(layer_mse_freeze), "trained": float(layer_mse_trained)},
        "arm_id": ARM_ID,
        "steps": int(steps),
        "data_mix": str(data_mix),
        "status": "not Done",
        "verdict": kill["result"],
        "kill": kill,
        "score_keys": list(KIMAHRI_KEYS),
        "note": (
            "Kimahri A2 pack. Score only the listed keys. "
            "Not Done. Not a quality PASS. Tidus owns SHIP."
        ),
    }
    if extras:
        for k, v in extras.items():
            if k in summary and k in KIMAHRI_KEYS:
                continue
            summary[k] = v
    refuse_pass_language(summary)
    validate_kimahri(summary)
    return summary


def validate_kimahri(summary: Mapping[str, Any]) -> None:
    missing = [k for k in KIMAHRI_KEYS if k not in summary]
    if missing:
        raise ValueError(f"Kimahri summary missing {missing}")
    lm = summary["layer_mse"]
    if not isinstance(lm, Mapping) or "freeze" not in lm or "trained" not in lm:
        raise ValueError("layer_mse must be {freeze, trained}")
    if summary.get("arm_id") != ARM_ID:
        raise ValueError(f"arm_id must be {ARM_ID!r}, got {summary.get('arm_id')!r}")
    refuse_pass_language(summary)


def refuse_pass_language(summary: Mapping[str, Any]) -> None:
    if _is_pass_token(summary.get("verdict")):
        raise ValueError(f"A2 refuses PASS in verdict ({summary.get('verdict')!r})")
    kill = summary.get("kill") or {}
    if isinstance(kill, Mapping) and _is_pass_token(kill.get("result")):
        raise ValueError("A2 refuses PASS in kill.result")
    status = str(summary.get("status", ""))
    if status.strip().lower() in ("done", "pass"):
        raise ValueError(f"A2 refuses Done/PASS status ({status!r})")


def write_summary(path: Path, summary: Mapping[str, Any]) -> Path:
    validate_kimahri(summary)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")
    return path


def inventory() -> Dict[str, Any]:
    return {
        "arm_id": ARM_ID,
        "status": "not Done",
        "format": "gf17_fpa / factor_product_atlas_v1",
        "forward": "y = U@(V.T @ x) + pq + sparse  (never dense W)",
        "FpaLinear": "amni/inference/fpa_linear.py",
        "gf17_fpa": "amni/inference/gf17_fpa.py",
        "distill_v0": "amni/training/fpa_distill_v0.py + scripts/lane_a_fpa_distill_v0.py",
        "real_prompt_eval_v0": "amni/training/fpa_real_prompt_eval_v0.py + scripts/lane_a_fpa_real_prompt_eval_v0.py",
        "distill_a2": "amni/training/fpa_distill_a2.py + scripts/lane_a_fpa_distill_a2.py",
        "runbook": "docs/A2_runbook.md",
        "bake_default": "bakes/qwen35_4b_hc_fpa_onetensor_probe",
        "teacher_default": "downloaded_models/Qwen3.5-4B",
        "layer": "model.language_model.layers.15.mlp.up_proj (Qwen3.5-4B)",
        "v0_train": "random token ids + mse_weight=0 (plumbing only)",
        "a2_train": "same real-prompt family as eval; mse_weight in {0.5, 1.0}",
        "trainables": ["U", "V", "U_scale", "V_scale", "pq_scale"],
        "frozen": ["codebooks", "pq_idx", "sparse"],
        "score_keys": list(KIMAHRI_KEYS),
        "kill": "after 5k, if real-prompt KL not >=5% better than freeze → escalate A3",
        "a3": "not implemented in this patch",
        "ship": "Tidus owns SHIP; this arm does not mark PASS or Done",
    }
