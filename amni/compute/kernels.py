import taichi as ti
import numpy as np
_initialized = False
_MUL_LUT = None
_ADD_LUT = None
_INV_LUT = None
_CUBE_LUT_GPU = None
_DLOG_LUT = None
_EXP_LUT = None
def ensure_init(arch: str = "gpu"):
    global _initialized, _MUL_LUT, _ADD_LUT, _INV_LUT, _CUBE_LUT_GPU, _DLOG_LUT, _EXP_LUT
    if not _initialized:
        arch_map = {"gpu": ti.gpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu, "metal": ti.metal}
        ti.init(arch=arch_map.get(arch, ti.gpu), default_fp=ti.f32)
        _initialized = True
        mul = np.zeros((17, 17), dtype=np.int32)
        add = np.zeros((17, 17), dtype=np.int32)
        for a in range(17):
            for b in range(17):
                mul[a, b] = (a * b) % 17
                add[a, b] = (a + b) % 17
        _MUL_LUT = mul
        _ADD_LUT = add
        inv = np.zeros(17, dtype=np.int32)
        for a in range(1, 17): inv[a] = pow(a, 15, 17)
        _INV_LUT = inv
        _CUBE_LUT_GPU = np.array([pow(int(x), 3, 17) for x in range(17)], dtype=np.int32)
        g = 3
        exp_t = np.array([pow(g, k, 17) for k in range(16)], dtype=np.int32)
        dlog_t = np.zeros(17, dtype=np.int32)
        for k in range(16): dlog_t[exp_t[k]] = k
        _DLOG_LUT = dlog_t
        _EXP_LUT = exp_t
@ti.kernel
def _matmul_kernel(a: ti.types.ndarray(dtype=ti.f32, ndim=2),
                   b: ti.types.ndarray(dtype=ti.f32, ndim=2),
                   c: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in ti.ndrange(a.shape[0], b.shape[1]):
        acc = 0.0
        for k in range(a.shape[1]):
            acc += a[i, k] * b[k, j]
        c[i, j] = acc
@ti.kernel
def _matmul_bias_kernel(a: ti.types.ndarray(dtype=ti.f32, ndim=2),
                        b: ti.types.ndarray(dtype=ti.f32, ndim=2),
                        bias: ti.types.ndarray(dtype=ti.f32, ndim=1),
                        c: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in ti.ndrange(a.shape[0], b.shape[1]):
        acc = bias[j]
        for k in range(a.shape[1]):
            acc += a[i, k] * b[k, j]
        c[i, j] = acc
@ti.kernel
def _relu_kernel(x: ti.types.ndarray(dtype=ti.f32, ndim=2),
                 out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in ti.ndrange(x.shape[0], x.shape[1]):
        out[i, j] = ti.max(x[i, j], 0.0)
@ti.kernel
def _gelu_kernel(x: ti.types.ndarray(dtype=ti.f32, ndim=2),
                 out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in ti.ndrange(x.shape[0], x.shape[1]):
        v = x[i, j]
        out[i, j] = 0.5 * v * (1.0 + ti.tanh(0.7978845608 * (v + 0.044715 * v * v * v)))
@ti.kernel
def _silu_kernel(x: ti.types.ndarray(dtype=ti.f32, ndim=2),
                 out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in ti.ndrange(x.shape[0], x.shape[1]):
        v = x[i, j]
        out[i, j] = v / (1.0 + ti.exp(-v))
@ti.kernel
def _softmax_exp_kernel(x: ti.types.ndarray(dtype=ti.f32, ndim=2),
                        row_max: ti.types.ndarray(dtype=ti.f32, ndim=1),
                        out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in ti.ndrange(x.shape[0], x.shape[1]):
        out[i, j] = ti.exp(x[i, j] - row_max[i])
@ti.kernel
def _layer_norm_kernel(x: ti.types.ndarray(dtype=ti.f32, ndim=2),
                       mean: ti.types.ndarray(dtype=ti.f32, ndim=1),
                       rstd: ti.types.ndarray(dtype=ti.f32, ndim=1),
                       out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in ti.ndrange(x.shape[0], x.shape[1]):
        out[i, j] = (x[i, j] - mean[i]) * rstd[i]
@ti.kernel
def _add_vectors_kernel(a: ti.types.ndarray(dtype=ti.f32, ndim=2),
                        b: ti.types.ndarray(dtype=ti.f32, ndim=2),
                        out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in ti.ndrange(a.shape[0], a.shape[1]):
        out[i, j] = a[i, j] + b[i, j]
        
@ti.kernel
def _csr_matmul_kernel(x:ti.types.ndarray(dtype=ti.f16,ndim=2),
                       rp:ti.types.ndarray(dtype=ti.i32,ndim=1),
                       ci:ti.types.ndarray(dtype=ti.i32,ndim=1),
                       v:ti.types.ndarray(dtype=ti.f16,ndim=1),
                       out:ti.types.ndarray(dtype=ti.f16,ndim=2)):
    for b,i in ti.ndrange(x.shape[0],rp.shape[0]-1):
        acc=ti.f16(0.0)
        for k in range(rp[i],rp[i+1]):
            acc+=x[b,ci[k]]*v[k]
        out[b,i]=acc

@ti.kernel
def _matmul_quant_b_kernel(a: ti.types.ndarray(dtype=ti.f32, ndim=2),
                           b_quant: ti.types.ndarray(dtype=ti.u8, ndim=2),
                           out: ti.types.ndarray(dtype=ti.f32, ndim=2),
                           scale: ti.f32,
                           bias_w: ti.f32):
    # Dequantize B on the fly: b = b_quant * scale + bias_w
    for i, j in ti.ndrange(a.shape[0], b_quant.shape[1]):
        acc = 0.0
        for k in range(a.shape[1]):
            val = ti.cast(b_quant[k, j], ti.f32) * scale + bias_w
            acc += a[i, k] * val
        out[i, j] = acc

@ti.kernel
def _matmul_quant_b_bias_kernel(a: ti.types.ndarray(dtype=ti.f32, ndim=2),
                                b_quant: ti.types.ndarray(dtype=ti.u8, ndim=2),
                                bias_layer: ti.types.ndarray(dtype=ti.f32, ndim=1),
                                out: ti.types.ndarray(dtype=ti.f32, ndim=2),
                                scale: ti.f32,
                                bias_w: ti.f32):
    for i, j in ti.ndrange(a.shape[0], b_quant.shape[1]):
        acc = bias_layer[j]
        for k in range(a.shape[1]):
            val = ti.cast(b_quant[k, j], ti.f32) * scale + bias_w
            acc += a[i, k] * val
        out[i, j] = acc
@ti.kernel
def _gf17_matmul_t_kernel(a: ti.types.ndarray(dtype=ti.i32, ndim=2),
                          w: ti.types.ndarray(dtype=ti.i32, ndim=2),
                          mul_lut: ti.types.ndarray(dtype=ti.i32, ndim=2),
                          add_lut: ti.types.ndarray(dtype=ti.i32, ndim=2),
                          out: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    for i, j in ti.ndrange(a.shape[0], w.shape[0]):
        s = ti.i32(0)
        for k in range(a.shape[1]):
            prod = mul_lut[a[i, k], w[j, k]]
            s = add_lut[s, prod]
        out[i, j] = s
@ti.kernel
def _gf17_matmul_t_alu_kernel(a: ti.types.ndarray(dtype=ti.i32, ndim=2),
                               w: ti.types.ndarray(dtype=ti.i32, ndim=2),
                               out: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    for i, j in ti.ndrange(a.shape[0], w.shape[0]):
        s = ti.i32(0)
        for k in range(a.shape[1]):
            s += a[i, k] * w[j, k]
        out[i, j] = s % 17
@ti.kernel
def _gf17_add_kernel(a: ti.types.ndarray(dtype=ti.i32, ndim=2),
                     b: ti.types.ndarray(dtype=ti.i32, ndim=2),
                     add_lut: ti.types.ndarray(dtype=ti.i32, ndim=2),
                     out: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    for i, j in ti.ndrange(a.shape[0], a.shape[1]):
        out[i, j] = add_lut[a[i, j], b[i, j]]
@ti.kernel
def _gf17_mul_kernel(a: ti.types.ndarray(dtype=ti.i32, ndim=2),
                     b: ti.types.ndarray(dtype=ti.i32, ndim=2),
                     mul_lut: ti.types.ndarray(dtype=ti.i32, ndim=2),
                     out: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    for i, j in ti.ndrange(a.shape[0], a.shape[1]):
        out[i, j] = mul_lut[a[i, j], b[i, j]]
@ti.kernel
def _gf17_activate_kernel(x: ti.types.ndarray(dtype=ti.i32, ndim=2),
                          act_lut: ti.types.ndarray(dtype=ti.i32, ndim=1),
                          out: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    for i, j in ti.ndrange(x.shape[0], x.shape[1]):
        out[i, j] = act_lut[x[i, j]]
@ti.kernel
def _gf17_rms_norm_kernel(x: ti.types.ndarray(dtype=ti.i32, ndim=2),
                          inv_lut: ti.types.ndarray(dtype=ti.i32, ndim=1),
                          mul_lut: ti.types.ndarray(dtype=ti.i32, ndim=2),
                          add_lut: ti.types.ndarray(dtype=ti.i32, ndim=2),
                          out: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    for i in range(x.shape[0]):
        sq = ti.i32(0)
        for k in range(x.shape[1]):
            sq = add_lut[sq, mul_lut[x[i, k], x[i, k]]]
        iv = inv_lut[sq] if sq > 0 else ti.i32(1)
        for j in range(x.shape[1]):
            out[i, j] = mul_lut[x[i, j], iv]
@ti.kernel
def _gf17_fused_mlp_kernel(x: ti.types.ndarray(dtype=ti.i32, ndim=2),
                           gw: ti.types.ndarray(dtype=ti.i32, ndim=2),
                           uw: ti.types.ndarray(dtype=ti.i32, ndim=2),
                           dw: ti.types.ndarray(dtype=ti.i32, ndim=2),
                           cube_lut: ti.types.ndarray(dtype=ti.i32, ndim=1),
                           mul_lut: ti.types.ndarray(dtype=ti.i32, ndim=2),
                           add_lut: ti.types.ndarray(dtype=ti.i32, ndim=2),
                           gate_buf: ti.types.ndarray(dtype=ti.i32, ndim=2),
                           up_buf: ti.types.ndarray(dtype=ti.i32, ndim=2),
                           out: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    S = x.shape[0]
    D = x.shape[1]
    inter = gw.shape[0]
    for i, j in ti.ndrange(S, inter):
        s = ti.i32(0)
        for k in range(D):
            s = add_lut[s, mul_lut[x[i, k], gw[j, k]]]
        gate_buf[i, j] = cube_lut[s]
    for i, j in ti.ndrange(S, inter):
        s = ti.i32(0)
        for k in range(D):
            s = add_lut[s, mul_lut[x[i, k], uw[j, k]]]
        up_buf[i, j] = s
    for i, j in ti.ndrange(S, D):
        s = ti.i32(0)
        for k in range(inter):
            prod = mul_lut[gate_buf[i, k], up_buf[i, k]]
            s = add_lut[s, mul_lut[prod, dw[j, k]]]
        out[i, j] = s
@ti.kernel
def _gf17_norm_matmul_t_kernel(x: ti.types.ndarray(dtype=ti.i32, ndim=2),
                                w: ti.types.ndarray(dtype=ti.i32, ndim=2),
                                inv_buf: ti.types.ndarray(dtype=ti.i32, ndim=1),
                                inv_lut: ti.types.ndarray(dtype=ti.i32, ndim=1),
                                mul_lut: ti.types.ndarray(dtype=ti.i32, ndim=2),
                                add_lut: ti.types.ndarray(dtype=ti.i32, ndim=2),
                                out: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    S = x.shape[0]
    for i in range(S):
        sq = ti.i32(0)
        for k in range(x.shape[1]):
            sq = add_lut[sq, mul_lut[x[i, k], x[i, k]]]
        inv_buf[i] = inv_lut[sq] if sq > 0 else ti.i32(1)
    for i, j in ti.ndrange(S, w.shape[0]):
        s = ti.i32(0)
        iv = inv_buf[i]
        for k in range(x.shape[1]):
            nv = mul_lut[x[i, k], iv]
            s = add_lut[s, mul_lut[nv, w[j, k]]]
        out[i, j] = s
@ti.kernel
def _gf17_residual_add_kernel(x: ti.types.ndarray(dtype=ti.i32, ndim=2),
                               res: ti.types.ndarray(dtype=ti.i32, ndim=2),
                               add_lut: ti.types.ndarray(dtype=ti.i32, ndim=2),
                               out: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    for i, j in ti.ndrange(x.shape[0], x.shape[1]):
        out[i, j] = add_lut[x[i, j], res[i, j]]
