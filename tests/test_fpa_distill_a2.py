"""A2 smoke: import, argparse, Kimahri schema, tiny CPU forward. No GPU train."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from amni.inference.fpa_linear import RECIPE_V0_TRAIN, FpaLinear
from amni.inference.gf17_fpa import bpw_allin, write_synthetic_bake
from amni.training.fpa_distill_a2 import (
    ALLOWED_MSE_WEIGHTS,
    DEFAULT_MSE_WEIGHT,
    A2Config,
    config_from_args,
    parse_a2_args,
    run_distill_a2,
    validate_mse_weight,
)
from amni.training.fpa_real_prompts import (
    FAMILY_V1,
    FAMILY_V1_CORE,
    cycle_batch,
    data_mix_id,
    encode_prompts,
    load_prompts,
    prompt_fingerprint,
)
from amni.training.fpa_summary_a2 import (
    ARM_ID,
    KIMAHRI_KEYS,
    build_summary,
    inventory,
    kill_record,
    refuse_pass_language,
    validate_kimahri,
)


class TestImportAndArgs(unittest.TestCase):
    def test_import_contract(self):
        from amni.inference.fpa_linear import FpaBake, load_fpa_linear
        from amni.training import fpa_distill_a2, fpa_distill_v0, fpa_real_prompts

        self.assertTrue(callable(load_fpa_linear))
        self.assertEqual(fpa_distill_a2.DEFAULT_MSE_WEIGHT, 1.0)
        self.assertEqual(fpa_distill_v0.DistillConfig().mse_weight, 0.0)
        self.assertEqual(len(fpa_real_prompts.FAMILY_V1), 128)
        self.assertEqual(fpa_real_prompts.FAMILY_V1[:48], FAMILY_V1_CORE)
        self.assertIs(FpaBake, FpaBake)

    def test_argparse_defaults_and_mse_flag(self):
        args = parse_a2_args([])
        self.assertEqual(args.mse_weight, 1.0)
        self.assertEqual(args.n_prompts, 32)
        self.assertEqual(args.seq_len, 128)
        self.assertEqual(args.bake, "bakes/qwen35_4b_hc_fpa_onetensor_probe")
        cfg = config_from_args(parse_a2_args(["--mse-weight", "0.5", "--synthetic", "--skip-gpu-train"]))
        self.assertEqual(cfg.mse_weight, 0.5)
        self.assertTrue(cfg.synthetic)
        self.assertTrue(cfg.skip_gpu_train)

    def test_mse_weight_rejects_zero(self):
        with self.assertRaises(ValueError) as ctx:
            validate_mse_weight(0.0)
        self.assertIn("0.5", str(ctx.exception))
        self.assertEqual(ALLOWED_MSE_WEIGHTS, (0.5, 1.0))
        with self.assertRaises(ValueError):
            A2Config(mse_weight=0.0)
        self.assertEqual(validate_mse_weight(DEFAULT_MSE_WEIGHT), 1.0)


class TestKimahriSchema(unittest.TestCase):
    def test_required_keys_and_no_pass(self):
        s = build_summary(
            freeze_kl=0.45,
            trained_kl=0.40,
            teacher_self_kl=1e-8,
            bpw_allin=1.2,
            layer_mse_freeze=0.3,
            layer_mse_trained=0.2,
            steps=5000,
            data_mix="real_prompt_en_v1:32x128",
        )
        for k in KIMAHRI_KEYS:
            self.assertIn(k, s)
        self.assertEqual(s["arm_id"], ARM_ID)
        self.assertEqual(s["status"], "not Done")
        self.assertNotEqual(s["verdict"].upper(), "PASS")
        self.assertGreater(s["delta_vs_freeze"], 0.05)
        self.assertEqual(s["kill"]["result"], "improved_ge_5pct")
        self.assertEqual(s["kill"]["next_action"], "hold_for_tidus_ship")
        self.assertEqual(s["layer_mse"]["freeze"], 0.3)
        validate_kimahri(s)
        refuse_pass_language(s)

    def test_kill_escalates_a3_without_pass(self):
        rec = kill_record(0.451462, 0.487534, 5000)
        self.assertEqual(rec["result"], "not_ge_5pct_better")
        self.assertEqual(rec["next_action"], "escalate_a3")
        self.assertTrue(rec["armed"])
        self.assertEqual(rec["bar"], 0.05)
        s = build_summary(
            freeze_kl=0.451462,
            trained_kl=0.487534,
            teacher_self_kl=0.0,
            bpw_allin=1.0,
            layer_mse_freeze=1.0,
            layer_mse_trained=1.1,
            steps=5000,
            data_mix="real_prompt_en_v1:32x128",
        )
        self.assertEqual(s["verdict"], "not_ge_5pct_better")
        self.assertNotIn("PASS", s["verdict"].upper())

    def test_refuse_pass_verdict(self):
        s = build_summary(
            freeze_kl=1.0, trained_kl=0.9, teacher_self_kl=0.0, bpw_allin=1.0,
            layer_mse_freeze=1.0, layer_mse_trained=1.0, steps=8, data_mix="x",
        )
        s["verdict"] = "PASS"
        with self.assertRaises(ValueError):
            refuse_pass_language(s)

    def test_inventory_paths(self):
        inv = inventory()
        self.assertEqual(inv["arm_id"], "A2")
        self.assertIn("fpa_linear.py", inv["FpaLinear"])
        self.assertIn("qwen35_4b_hc_fpa_onetensor_probe", inv["bake_default"])
        self.assertEqual(inv["status"], "not Done")


class TestPromptFamily(unittest.TestCase):
    def test_train_eval_share_table(self):
        a = load_prompts("", 32)
        b = load_prompts("", 32)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 32)
        self.assertEqual(a, list(FAMILY_V1[:32]))
        self.assertEqual(data_mix_id(32, 128), "real_prompt_en_v1:32x128")
        ids = encode_prompts(
            __import__("amni.training.fpa_real_prompts", fromlist=["WordTokenizer"]).WordTokenizer(128),
            a,
            32,
            torch.device("cpu"),
        )
        self.assertEqual(tuple(ids.shape), (32, 32))
        b0 = cycle_batch(ids, 2, 1)
        b1 = cycle_batch(ids, 2, 1 + 16)
        self.assertEqual(tuple(b0.shape), (2, 32))
        self.assertTrue(torch.equal(b0, b1))
        self.assertEqual(len(prompt_fingerprint(a)), 12)


class TestTinyForwardAndSyntheticLoop(unittest.TestCase):
    def test_fake_bake_forward_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            write_synthetic_bake(td, out=32, inn=128, rank=8, seed=3)
            from amni.inference.fpa_linear import load_fpa_linear

            lin = load_fpa_linear(td, trainable=True)
            lin.apply_recipe_v0()
            y = lin(torch.randn(4, 128))
            self.assertEqual(tuple(y.shape), (4, 32))
            self.assertGreater(float(bpw_allin(lin)), 0.0)
            names = {n for n, p in lin.named_parameters() if p.requires_grad}
            self.assertEqual(names, set(RECIPE_V0_TRAIN))
            self.assertFalse(lin.codebooks.requires_grad)
            self.assertFalse(lin.pq_idx.requires_grad)

    def test_born_forward_cpu(self):
        lin = FpaLinear.born(128, 32, rank=8, trainable=True, seed=2)
        y = lin(torch.randn(3, 128))
        self.assertEqual(tuple(y.shape), (3, 32))

    def test_synthetic_a2_writes_kimahri(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = A2Config(
                synthetic=True,
                skip_gpu_train=True,
                steps=6,
                eval_every=3,
                lr=1e-3,
                batch=2,
                seq_len=32,
                n_prompts=8,
                mse_weight=1.0,
                out=td,
                seed=0,
                device="cpu",
            )
            summary = run_distill_a2(cfg)
            for k in KIMAHRI_KEYS:
                self.assertIn(k, summary, k)
            self.assertEqual(summary["arm_id"], "A2")
            self.assertEqual(summary["data_mix"], "real_prompt_en_v1:8x32")
            self.assertEqual(summary["status"], "not Done")
            self.assertNotEqual(str(summary["verdict"]).upper(), "PASS")
            self.assertEqual(summary["kill"]["result"], "kill_not_armed")
            self.assertIn("freeze", summary["layer_mse"])
            self.assertIn("trained", summary["layer_mse"])
            self.assertGreater(summary["bpw_allin"], 0.0)
            self.assertLess(abs(summary["teacher_self_kl"]), 1e-5)
            self.assertTrue((Path(td) / "summary.json").is_file())
            disk = json.loads((Path(td) / "summary.json").read_text(encoding="utf-8"))
            for k in KIMAHRI_KEYS:
                self.assertIn(k, disk)
            self.assertTrue((Path(td) / "freeze_init_trainables.pt").is_file())
            self.assertTrue((Path(td) / "trained_trainables.pt").is_file())
            self.assertEqual(set(summary["trainable"]), set(RECIPE_V0_TRAIN))
            splits = set()
            with open(Path(td) / "kl_curve.jsonl", encoding="utf-8") as jf:
                for line in jf:
                    if line.strip():
                        splits.add(json.loads(line)["split"])
            self.assertEqual(splits, {"train", "eval"})

    def test_mse_weight_half_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            summary = run_distill_a2(A2Config(
                synthetic=True, skip_gpu_train=True, steps=4, eval_every=2,
                mse_weight=0.5, out=td, device="cpu", n_prompts=4, seq_len=16, batch=2,
            ))
            self.assertEqual(summary["mse_weight"], 0.5)
            self.assertEqual(summary["arm_id"], "A2")


class TestCliSmoke(unittest.TestCase):
    def test_cli_inventory_and_synthetic(self):
        import subprocess
        import sys

        script = Path(__file__).resolve().parents[1] / "scripts" / "lane_a_fpa_distill_a2.py"
        inv = subprocess.run(
            [sys.executable, str(script), "--inventory"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(inv.returncode, 0, inv.stderr)
        payload = json.loads(inv.stdout)
        self.assertEqual(payload["arm_id"], "A2")
        self.assertEqual(payload["status"], "not Done")
        with tempfile.TemporaryDirectory() as td:
            run = subprocess.run(
                [
                    sys.executable, str(script),
                    "--synthetic", "--skip-gpu-train",
                    "--steps", "4", "--eval-every", "2",
                    "--n-prompts", "4", "--seq-len", "16", "--out", td,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue((Path(td) / "summary.json").is_file())
            self.assertNotIn("\nPASS", run.stdout)


if __name__ == "__main__":
    unittest.main()
