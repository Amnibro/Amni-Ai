"""Lane A distill v0 helpers + synthetic loop (no 4B weights)."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from amni.inference.fpa_linear import RECIPE_V0_TRAIN, FpaLinear
from amni.training.fpa_distill_v0 import (
    DistillConfig,
    find_up_proj,
    kill_verdict,
    kl_logits,
    run_distill,
)


class TestKillAndKl(unittest.TestCase):
    def test_kill_pass_and_fail(self):
        v, _ = kill_verdict(1.0, 0.90)
        self.assertEqual(v, "PASS")
        v, _ = kill_verdict(1.0, 0.96)
        self.assertEqual(v, "FAIL")
        v, _ = kill_verdict(1.0, 1.02)
        self.assertEqual(v, "FAIL")

    def test_kl_zero_when_equal(self):
        z = torch.randn(2, 4, 8)
        self.assertLess(float(kl_logits(z, z, T=2.0)), 1e-6)


class TestFindLayer(unittest.TestCase):
    def test_language_model_path(self):
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                layers = nn.ModuleList()
                for _ in range(16):
                    blk = nn.Module()
                    blk.mlp = nn.Module()
                    blk.mlp.up_proj = nn.Linear(8, 16, bias=False)
                    layers.append(blk)
                self.model.language_model.layers = layers

        path = find_up_proj(M(), 15)
        self.assertEqual(path, "model.language_model.layers.15.mlp.up_proj")


class TestSyntheticDistill(unittest.TestCase):
    def test_loop_writes_logs(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = DistillConfig(
                synthetic=True,
                steps=6,
                eval_every=3,
                lr=1e-3,
                batch=2,
                seq_len=8,
                eval_batches=2,
                mse_weight=0.1,
                out=td,
                seed=0,
                device="cpu",
            )
            summary = run_distill(cfg)
            self.assertIn(summary["verdict"], ("PASS", "FAIL"))
            self.assertTrue((Path(td) / "kl_curve.jsonl").is_file())
            self.assertTrue((Path(td) / "kl_curve.csv").is_file())
            self.assertTrue((Path(td) / "summary.json").is_file())
            with open(Path(td) / "kl_curve.jsonl", encoding="utf-8") as jf:
                recs = [json.loads(l) for l in jf if l.strip()]
            splits = {r["split"] for r in recs}
            self.assertIn("eval", splits)
            self.assertIn("train", splits)
            self.assertEqual(summary["trainable"], list(RECIPE_V0_TRAIN) or summary["trainable"])
            self.assertTrue(set(summary["trainable"]) == set(RECIPE_V0_TRAIN))
            for name in ("freeze_init_trainables.pt", "trained_trainables.pt", "trained_fpa.pt", "trained_fpa_repacked.pt"):
                self.assertTrue((Path(td) / name).is_file(), name)
            from amni.training.fpa_distill_v0 import load_trainables_pt

            init = load_trainables_pt(Path(td) / "freeze_init_trainables.pt")
            trained = load_trainables_pt(Path(td) / "trained_trainables.pt")
            self.assertEqual(set(init), set(RECIPE_V0_TRAIN))
            self.assertEqual(set(trained), set(RECIPE_V0_TRAIN))
            self.assertTrue(any(not torch.equal(init[k], trained[k]) for k in RECIPE_V0_TRAIN))

    def test_real_prompt_eval_synthetic(self):
        from amni.training.fpa_real_prompt_eval_v0 import PromptEvalConfig, run_prompt_eval

        with tempfile.TemporaryDirectory() as td:
            ddir = Path(td) / "distill"
            edir = Path(td) / "eval"
            run_distill(DistillConfig(synthetic=True, steps=4, eval_every=2, out=str(ddir), seed=0, device="cpu", batch=2, seq_len=8, eval_batches=1))
            summary = run_prompt_eval(PromptEvalConfig(
                synthetic=True,
                ckpt_dir=str(ddir),
                out=str(edir),
                seed=0,
                device="cpu",
                batch=8,
                seq_len=32,
            ))
            self.assertLess(abs(summary["self_kl_teacher"]), 1e-5)
            self.assertIn("kl_freeze_init", summary)
            self.assertIn("kl_trained", summary)
            self.assertIn("rel_vs_freeze", summary)
            self.assertEqual(summary["status"], "not Done")
            self.assertEqual(summary["verdict"], "plumbing_complete")
            self.assertNotIn("PASS", str(summary["verdict"]).upper())
            self.assertGreaterEqual(summary["n_prompts"], 32)
            self.assertTrue((edir / "summary.json").is_file())

    def test_uv_still_matches_dense_during_recipe(self):
        lin = FpaLinear.born(128, 32, rank=8, trainable=True, seed=1)
        x = torch.randn(5, 128)
        W = lin.U_deq() @ lin.V_deq().T
        self.assertTrue(torch.allclose(lin.uv_gemv(x), x @ W.T, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
