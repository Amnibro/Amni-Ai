"""Unit tests for Lane A gf17_fpa / factor_product_atlas_v1 bootstrap.

Covers: int4 nibble pack/unpack, UV gemv vs materialised U@Vᵀ, product-VQ
reconstruction, synthetic bake load, trainable factor hook. Not a quality claim.
"""
import json
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from amni.inference.fpa_linear import FpaBake, FpaLinear, load_fpa_linear
from amni.inference.gf17_fpa import (
    FORMAT,
    PACKING,
    dequant_int4_cols,
    is_atex_codes_bake,
    is_fpa_manifest,
    pack_int4_nibbles,
    pq_gemv,
    pq_reconstruct_groups,
    quantize_int4_cols,
    sparse_gemv,
    unpack_int4_nibbles,
    uv_gemv,
    write_synthetic_bake,
)


class TestInt4Nibbles(unittest.TestCase):
    def test_known_bytes(self):
        # even → lo nibble, odd → hi; stored value = signed + 8
        q = torch.tensor([[-8, -1, 0, 7]], dtype=torch.int16)
        packed = pack_int4_nibbles(q)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(tuple(packed.shape), (1, 2))
        self.assertEqual(int(packed[0, 0]), (0x0) | ((-1 + 8) << 4))  # 0x70
        self.assertEqual(int(packed[0, 1]), (8) | (15 << 4))  # 0xF8
        back = unpack_int4_nibbles(packed)
        self.assertTrue(torch.equal(back, q))

    def test_roundtrip_random(self):
        torch.manual_seed(1)
        q = torch.randint(-8, 8, (17, 32), dtype=torch.int16)
        self.assertTrue(torch.equal(unpack_int4_nibbles(pack_int4_nibbles(q)), q))

    def test_matches_int4_linear_layout(self):
        # Int4GroupLinear: packed = qf[:,0::2] | (qf[:,1::2]<<4) with qf = q+8
        q = torch.tensor([3, -4, 7, -8, 0, 1], dtype=torch.int16)
        qf = (q + 8).to(torch.uint8)
        ref = (qf[0::2] | (qf[1::2] << 4))
        self.assertTrue(torch.equal(pack_int4_nibbles(q), ref))

    def test_per_col_dequant(self):
        torch.manual_seed(2)
        W = torch.randn(21, 8)
        packed, scale = quantize_int4_cols(W)
        rec = dequant_int4_cols(packed, scale)
        self.assertEqual(tuple(rec.shape), (21, 8))
        # reconstruction of the quantized grid, not the original W
        q = unpack_int4_nibbles(packed).float()
        self.assertTrue(torch.allclose(rec, q * scale, atol=0, rtol=0))


class TestUvGemv(unittest.TestCase):
    def test_uv_matches_materialised_uvt(self):
        torch.manual_seed(3)
        out, inn, rank = 32, 64, 8
        U = torch.randn(out, rank)
        V = torch.randn(inn, rank)
        x = torch.randn(inn)
        y = uv_gemv(x, U, V)
        W = U @ V.T
        self.assertEqual(tuple(W.shape), (out, inn))
        ref = F.linear(x, W)
        self.assertTrue(torch.allclose(y, ref, atol=1e-5, rtol=1e-5))
        batched = torch.randn(5, 3, inn)
        yb = uv_gemv(batched, U, V)
        refb = batched @ W.T
        self.assertTrue(torch.allclose(yb, refb, atol=1e-5, rtol=1e-5))

    def test_fpalinear_uv_vs_dense(self):
        torch.manual_seed(4)
        lin = FpaLinear.born(128, 48, rank=8, gs=128, M=8, K=32, trainable=False, seed=4)
        x = torch.randn(128)
        y = lin.uv_gemv(x)
        ref = F.linear(x, lin.U @ lin.V.T)
        self.assertTrue(torch.allclose(y, ref, atol=1e-5, rtol=1e-5))


class TestPqPath(unittest.TestCase):
    def test_reconstruct_then_gemv(self):
        torch.manual_seed(5)
        out, gs, M, K = 16, 128, 8, 32
        inn, G, D = gs, 1, gs // M
        codebooks = torch.randn(M, K, D)
        pq_idx = torch.randint(0, K, (out, G, M)).to(torch.uint8)
        pq_scale = torch.randn(out, G)
        x = torch.randn(inn)
        groups = pq_reconstruct_groups(pq_idx, pq_scale, codebooks, gs)
        self.assertEqual(tuple(groups.shape), (out, G, gs))
        W_pq = groups.reshape(out, inn)
        ref = W_pq @ x
        y = pq_gemv(x, pq_idx, pq_scale, codebooks, gs)
        self.assertTrue(torch.allclose(y, ref, atol=1e-5, rtol=1e-5))
        # sanity: a single codebook lookup * scale reconstructs the group
        o, g = 3, 0
        chunks = [codebooks[m, int(pq_idx[o, g, m])] for m in range(M)]
        rec = pq_scale[o, g] * torch.cat(chunks, 0)
        self.assertTrue(torch.allclose(groups[o, g], rec, atol=1e-6, rtol=1e-6))

    def test_pq_batched(self):
        torch.manual_seed(6)
        out, gs, M, K = 8, 128, 8, 16
        codebooks = torch.randn(M, K, gs // M)
        pq_idx = torch.randint(0, K, (out, 1, M)).to(torch.uint8)
        pq_scale = torch.rand(out, 1)
        x = torch.randn(4, 128)
        W = pq_reconstruct_groups(pq_idx, pq_scale, codebooks, gs).reshape(out, 128)
        y = pq_gemv(x, pq_idx, pq_scale, codebooks, gs)
        self.assertTrue(torch.allclose(y, x @ W.T, atol=1e-5, rtol=1e-5))


class TestSparseGemv(unittest.TestCase):
    def test_coo(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        rows = torch.tensor([0, 2, 2])
        cols = torch.tensor([1, 0, 3])
        vals = torch.tensor([0.5, 1.0, -1.0])
        y = sparse_gemv(x, rows, cols, vals, 3)
        ref = torch.zeros(3)
        ref[0] += 0.5 * 2.0
        ref[2] += 1.0 * 1.0
        ref[2] += -1.0 * 4.0
        self.assertTrue(torch.allclose(y, ref))


class TestBakeLoadAndForward(unittest.TestCase):
    def test_synthetic_folder_roundtrip(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as td:
            write_synthetic_bake(td, out=32, inn=128, rank=8, gs=128, M=8, K=64, sparse_nnz=6, seed=7)
            with open(os.path.join(td, "bake_manifest.json"), encoding="utf-8") as mf:
                man = json.load(mf)
            self.assertEqual(man["format"], FORMAT)
            self.assertEqual(man["packing"], PACKING)
            self.assertTrue(is_fpa_manifest(man))
            self.assertFalse(is_atex_codes_bake(man))
            lin = load_fpa_linear(td, trainable=False)
            self.assertEqual(lin.in_features, 128)
            self.assertEqual(lin.out_features, 32)
            self.assertEqual(lin.rank, 8)
            self.assertEqual(tuple(lin.U_packed.shape), (32, 4))
            self.assertEqual(tuple(lin.V_packed.shape), (128, 4))
            self.assertEqual(tuple(lin.pq_idx.shape), (32, 1, 8))
            x = torch.randn(128)
            y = lin(x)
            self.assertEqual(tuple(y.shape), (32,))
            ref = lin.uv_gemv(x) + lin.pq_gemv(x) + lin.sparse_gemv(x)
            self.assertTrue(torch.allclose(y.float(), ref.float(), atol=1e-5, rtol=1e-5))
            W_uv = lin.U @ lin.V.T
            self.assertTrue(torch.allclose(lin.uv_gemv(x), F.linear(x, W_uv), atol=1e-5, rtol=1e-5))
            self.assertGreater(float(y.float().norm()), 0.0)
            bake = FpaBake(td)
            self.assertEqual(len(bake.names()), 1)
            self.assertIn("pq", bake.report or {})

    def test_rejects_atex_codes_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            man = {"format": "gf17_atex", "gs": 128, "tensors": {"w": {"q": 1, "shape": [4, 4]}}}
            with open(os.path.join(td, "bake_manifest.json"), "w", encoding="utf-8") as mf:
                json.dump(man, mf)
            with self.assertRaises(ValueError) as ctx:
                FpaBake(td)
            self.assertIn("Gf17Atex", str(ctx.exception))

    def test_contract_probe_shapes(self):
        # Qwen L15 up_proj [9216,2560] r=64 mirrored at reduced size that keeps ratios
        # (out=72=9216/128, inn=160=2560/16, rank=8, gs=32, M=8 → pq_idx (72, 5, 8))
        with tempfile.TemporaryDirectory() as td:
            write_synthetic_bake(td, out=72, inn=160, rank=8, gs=32, M=8, K=32, name="model.layers.15.mlp.up_proj.weight", seed=8)
            lin = load_fpa_linear(td)
            self.assertEqual(tuple(lin.U_packed.shape), (72, 4))
            self.assertEqual(tuple(lin.V_packed.shape), (160, 4))
            self.assertEqual(tuple(lin.pq_idx.shape), (72, 5, 8))
            self.assertEqual(tuple(lin.pq_scale.shape), (72, 5))
            y = lin(torch.randn(2, 160))
            self.assertEqual(tuple(y.shape), (2, 72))


class TestTrainableHook(unittest.TestCase):
    def test_parameters_and_grad(self):
        lin = FpaLinear.born(128, 24, rank=8, trainable=True, seed=9)
        names = {n for n, _ in lin.named_parameters()}
        self.assertTrue({"U", "V", "pq_scale", "codebooks"} <= names)
        x = torch.randn(3, 128, requires_grad=True)
        y = lin(x)
        y.square().mean().backward()
        self.assertIsNotNone(lin.U.grad)
        self.assertIsNotNone(lin.V.grad)
        self.assertIsNotNone(lin.pq_scale.grad)
        self.assertIsNotNone(lin.codebooks.grad)
        # discrete idx / packed codes stay non-trainable buffers
        bufs = {n for n, _ in lin.named_buffers()}
        self.assertIn("U_packed", bufs)
        self.assertIn("pq_idx", bufs)
        self.assertFalse(lin.U_packed.requires_grad)

    def test_serve_mode_has_no_factor_params(self):
        lin = FpaLinear.born(128, 16, rank=8, trainable=False, seed=10)
        self.assertEqual(list(lin.parameters()), [])


class TestCodebookLayouts(unittest.TestCase):
    def test_normalise_permutations(self):
        from amni.inference.gf17_fpa import normalize_codebooks

        M, K, D = 8, 16, 4
        canon = torch.arange(M * K * D).float().reshape(M, K, D)
        for arr in (
            canon,
            canon.permute(1, 0, 2).contiguous(),
            canon.permute(0, 2, 1).contiguous(),
            canon.reshape(-1),
        ):
            got = normalize_codebooks(arr.numpy() if isinstance(arr, torch.Tensor) else arr, M, K, D)
            self.assertEqual(tuple(got.shape), (M, K, D))
            self.assertTrue(torch.equal(got, canon))


if __name__ == "__main__":
    unittest.main()
