import numpy as np, time
from typing import Tuple, Dict, Optional
P = 17
P_SQ = P * P
P4 = P ** 4
_INV_TAB = np.zeros(P, dtype=np.uint8)
for _a in range(1, P):
    _INV_TAB[_a] = pow(int(_a), P - 2, P)
GF17_INV = _INV_TAB
_CUBE_LUT = np.array([pow(int(x), 3, P) for x in range(P)], dtype=np.uint8)
_POW5_LUT = np.array([pow(int(x), 5, P) if x > 0 else 0 for x in range(P)], dtype=np.uint8)
_LEG_LUT = np.array([pow(int(x), (P - 1) // 2, P) if x > 0 else 0 for x in range(P)], dtype=np.uint8)
_RELU_LUT = np.array([x if x <= 8 else 0 for x in range(P)], dtype=np.uint8)
_ABS_LUT = np.array([x if x <= 8 else P - x for x in range(P)], dtype=np.uint8)
_CUBE_INV_LUT = np.zeros(P, dtype=np.uint8)
for _x in range(P):
    _CUBE_INV_LUT[_CUBE_LUT[_x]] = _x
_SWISH_LUT = np.array([(int(x) * int(_CUBE_LUT[x])) % P for x in range(P)], dtype=np.uint8)
ACTIVATION_LUTS = {"cube": _CUBE_LUT, "power5": _POW5_LUT, "legendre": _LEG_LUT,
                   "relu": _RELU_LUT, "abs": _ABS_LUT, "swish": _SWISH_LUT}
_MUL_LUT = np.zeros((P, P), dtype=np.uint8)
_ADD_LUT = np.zeros((P, P), dtype=np.uint8)
for _i in range(P):
    for _j in range(P):
        _MUL_LUT[_i, _j] = (_i * _j) % P
        _ADD_LUT[_i, _j] = (_i + _j) % P
_HIP_ENGINE=None
_HIP_TRIED=False
def _try_hip():
    global _HIP_ENGINE,_HIP_TRIED
    if _HIP_TRIED:return _HIP_ENGINE
    _HIP_TRIED=True
    try:
        from amni.compute.gf17_engine import is_available,get_engine
        _HIP_ENGINE=get_engine()if is_available()else None
    except Exception:_HIP_ENGINE=None
    return _HIP_ENGINE
def gf17_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _ADD_LUT[a.ravel(), b.ravel()].reshape(a.shape)
def gf17_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _ADD_LUT[a.ravel(), ((P - b.ravel()) % P).astype(np.uint8)].reshape(a.shape)
def gf17_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _MUL_LUT[a.ravel(), b.ravel()].reshape(a.shape)
def gf17_neg(a: np.ndarray) -> np.ndarray:
    return ((P - a.astype(np.uint8)) % P).astype(np.uint8)
def gf17_inv(a: np.ndarray) -> np.ndarray:
    return GF17_INV[a.ravel()].reshape(a.shape)
def gf17_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _MUL_LUT[a.ravel(), GF17_INV[b.ravel()]].reshape(a.shape)
def gf17_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float64) @ b.astype(np.float64) % P).astype(np.uint8)
def gf17_matmul_t(a:np.ndarray,w:np.ndarray)->np.ndarray:eng=_try_hip();o=a.shape;return eng.download(eng.matmul_t(eng.upload(a.reshape(-1,o[-1])),eng.upload(w))).reshape(*o[:-1],w.shape[0])if eng else((a.reshape(-1,a.shape[-1]).astype(np.float32)@w.astype(np.float32).T%P).astype(np.uint8)).reshape(*a.shape[:-1],w.shape[0])
def gf17_activate(x: np.ndarray, fn: str = "cube") -> np.ndarray:
    return ACTIVATION_LUTS.get(fn, _CUBE_LUT)[x.ravel()].reshape(x.shape)
def gf17_rms_norm(x: np.ndarray) -> np.ndarray:
    x2 = x.reshape(-1, x.shape[-1])
    sq = _MUL_LUT[x2.ravel(), x2.ravel()].reshape(x2.shape)
    sq_sum = sq.astype(np.uint32).sum(axis=-1) % P
    sq_safe = np.where(sq_sum == 0, 1, sq_sum).astype(np.uint8)
    iv = GF17_INV[sq_safe]
    return _MUL_LUT[x2.ravel(), np.repeat(iv, x2.shape[-1])].reshape(x.shape)
def gf17_fused_mlp(x:np.ndarray,gw:np.ndarray,uw:np.ndarray,dw:np.ndarray)->np.ndarray:
    eng=_try_hip()
    if eng:
        try:
            sh=x.shape;dx=eng.upload(x.reshape(-1,sh[-1]));dgw,duw,ddw=eng.upload(gw),eng.upload(uw),eng.upload(dw);r=eng.fused_mlp(dx,dgw,duw,ddw);eng.free(dx);eng.free(dgw);eng.free(duw);eng.free(ddw);out=eng.download(r);eng.free(r);return out.reshape(sh)
        except Exception:pass
    return gf17_matmul_t(gf17_mul(gf17_activate(gf17_matmul_t(x,gw),"cube"),gf17_matmul_t(x,uw)),dw)
def gf17_norm_matmul_t(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return gf17_matmul_t(gf17_rms_norm(x), w)
def gf17_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return np.minimum(d, P - d).astype(np.uint8)
def gf17_hamming(pred: np.ndarray, target: np.ndarray) -> int:
    return int(np.sum(pred != target))
def gf17_init_weights(shape: Tuple[int, ...], method: str = "uniform") -> np.ndarray:
    if method == "uniform":
        return np.random.randint(0, P, size=shape, dtype=np.uint8)
    elif method == "centered":
        w = np.random.normal(8.0, 3.0, size=shape)
        return np.clip(np.round(w), 0, 16).astype(np.uint8)
    elif method == "sparse":
        w = np.zeros(shape, dtype=np.uint8)
        mask = np.random.random(shape) < 0.3
        w[mask] = np.random.randint(1, P, size=int(mask.sum()), dtype=np.uint8)
        return w
    return np.random.randint(0, P, size=shape, dtype=np.uint8)
def float_to_gf17(x: np.ndarray, scale: float = 8.0) -> np.ndarray:
    scaled = np.clip(x / scale, -1.0, 1.0)
    return np.clip(np.round((scaled + 1.0) * 8.0), 0, 16).astype(np.uint8)
def gf17_to_float(q: np.ndarray, scale: float = 8.0) -> np.ndarray:
    return (q.astype(np.float32) / 8.0 - 1.0) * scale
def dq_matmul_t(x,w,s=2.0):xf=x.reshape(-1,x.shape[-1]).astype(np.float32);return(xf@((w.astype(np.float32)/8.0-1.0)*np.float32(s)).T).astype(np.float32)
def bf16_to_b17(x:np.ndarray)->np.ndarray:
    u16=x.view(np.uint16).ravel() if x.dtype==np.float16 else ((np.float32(x).view(np.uint32)>>16)&0xFFFF).astype(np.uint16).ravel() if x.dtype==np.float32 else x.view(np.uint16).ravel()
    out=np.empty((len(u16),4),dtype=np.uint8);v=u16.astype(np.uint32)
    out[:,0]=(v%17).astype(np.uint8);v//=17;out[:,1]=(v%17).astype(np.uint8);v//=17;out[:,2]=(v%17).astype(np.uint8);out[:,3]=(v//17).astype(np.uint8)
    return out
def b17_to_f32(b:np.ndarray)->np.ndarray:
    d=b.reshape(-1,4).astype(np.uint32)
    u16=d[:,0]+d[:,1]*17+d[:,2]*289+d[:,3]*4913
    return((u16<<16).view(np.float32))
def f32_rms_norm(x,eps=1e-6):x2=x.reshape(-1,x.shape[-1]).astype(np.float32);return(x2/np.sqrt((x2*x2).mean(axis=-1,keepdims=True)+np.float32(eps))).reshape(x.shape).astype(np.float32)
class GF17Tensor:
    __slots__ = ('data', 'shape')
    def __init__(self, data: np.ndarray):
        self.data = data if data.dtype == np.uint8 else data.astype(np.uint8)
        self.shape = data.shape
    def matmul(self, other: 'GF17Tensor') -> 'GF17Tensor':
        return GF17Tensor(gf17_matmul(self.data, other.data))
    def matmul_t(self, w: 'GF17Tensor') -> 'GF17Tensor':
        return GF17Tensor(gf17_matmul_t(self.data, w.data))
    def activate(self, fn: str = "cube") -> 'GF17Tensor':
        return GF17Tensor(gf17_activate(self.data, fn))
    def add(self, other: 'GF17Tensor') -> 'GF17Tensor':
        return GF17Tensor(gf17_add(self.data, other.data))
    def norm(self) -> 'GF17Tensor':
        return GF17Tensor(gf17_rms_norm(self.data))
def verify_field_axioms(n_tests: int = 1000) -> Dict:
    rng = np.random.default_rng(42)
    a = rng.integers(0, P, n_tests, dtype=np.uint8)
    b = rng.integers(0, P, n_tests, dtype=np.uint8)
    c = rng.integers(0, P, n_tests, dtype=np.uint8)
    assoc_add = np.all(gf17_add(gf17_add(a, b), c) == gf17_add(a, gf17_add(b, c)))
    assoc_mul = np.all(gf17_mul(gf17_mul(a, b), c) == gf17_mul(a, gf17_mul(b, c)))
    comm_add = np.all(gf17_add(a, b) == gf17_add(b, a))
    comm_mul = np.all(gf17_mul(a, b) == gf17_mul(b, a))
    id_add = np.all(gf17_add(a, np.zeros_like(a)) == a)
    id_mul = np.all(gf17_mul(a, np.ones_like(a)) == a)
    nz = a[a > 0]
    inv_ok = np.all(gf17_mul(nz, gf17_inv(nz)) == 1)
    distrib = np.all(gf17_mul(a, gf17_add(b, c)) == gf17_add(gf17_mul(a, b), gf17_mul(a, c)))
    cube_bij = len(set(_CUBE_LUT.tolist())) == P
    results = {"assoc_add": bool(assoc_add), "assoc_mul": bool(assoc_mul),
               "comm_add": bool(comm_add), "comm_mul": bool(comm_mul),
               "id_add": bool(id_add), "id_mul": bool(id_mul),
               "inv_mul": bool(inv_ok), "distrib": bool(distrib), "cube_bij": cube_bij}
    results["all_pass"] = all(results.values())
    return results
def benchmark_gf17_ops(dim: int = 512, batch: int = 32) -> Dict:
    rng = np.random.default_rng(42)
    a = rng.integers(0, P, (batch, dim), dtype=np.uint8)
    w = rng.integers(0, P, (dim, dim), dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = gf17_matmul_t(a, w)
    mm_ms = (time.perf_counter() - t0) / 100 * 1000
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = gf17_activate(a, "cube")
    act_ms = (time.perf_counter() - t0) / 1000 * 1000
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = gf17_rms_norm(a)
    norm_ms = (time.perf_counter() - t0) / 1000 * 1000
    return {"dim": dim, "batch": batch, "matmul_ms": round(mm_ms, 3),
            "activate_ms": round(act_ms, 3), "norm_ms": round(norm_ms, 3),
            "matmul_gops": round(2 * batch * dim * dim / mm_ms / 1e6, 2)}
_PRIM_G = 3
_GF17_EXP = np.array([pow(_PRIM_G, k, P) for k in range(P - 1)], dtype=np.uint8)
_GF17_DLOG = np.zeros(P, dtype=np.uint8)
for _k in range(P - 1): _GF17_DLOG[_GF17_EXP[_k]] = _k
GF17_EXP, GF17_DLOG = _GF17_EXP, _GF17_DLOG
_ALT_ENCODE = np.zeros((P, 4), dtype=np.int8)
_ALT_DECODE = np.zeros(16, dtype=np.uint8)
for _code in range(16):
    _bits = np.array([1 if (_code >> (3 - i)) & 1 else -1 for i in range(4)], dtype=np.int8)
    _val = (8 * _bits[0] + 4 * _bits[1] + 2 * _bits[2] + _bits[3]) % P
    _ALT_ENCODE[_val] = _bits
    _ALT_DECODE[_code] = _val
GF17_ALT_ENCODE, GF17_ALT_DECODE = _ALT_ENCODE, _ALT_DECODE
_PRIMITIVE_MASK = np.zeros(P, dtype=np.bool_)
for _x in range(1, P):
    _ord = 1
    _v = _x
    while _v != 1 or _ord == 0: _v = (_v * _x) % P; _ord += 1
    _PRIMITIVE_MASK[_x] = (_ord == P - 1)
GF17_PRIMITIVE = _PRIMITIVE_MASK
def gf17_dlog(a: np.ndarray) -> np.ndarray:
    return GF17_DLOG[np.clip(a.ravel(), 1, P - 1).astype(np.uint8)].reshape(a.shape)
def gf17_exp(e: np.ndarray) -> np.ndarray:
    return GF17_EXP[(e.ravel().astype(np.uint8) % (P - 1))].reshape(e.shape)
def gf17_mul_via_log(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    za = (a == 0); zb = (b == 0)
    la, lb = GF17_DLOG[a.ravel()].reshape(a.shape), GF17_DLOG[b.ravel()].reshape(b.shape)
    r = GF17_EXP[((la.astype(np.uint16) + lb.astype(np.uint16)) % (P - 1)).ravel()].reshape(a.shape)
    return np.where(za | zb, np.uint8(0), r)
def gf17_alt_encode(a: np.ndarray) -> np.ndarray:
    return GF17_ALT_ENCODE[a.ravel()].reshape(a.shape + (4,))
def gf17_alt_decode(bits: np.ndarray) -> np.ndarray:
    b = bits.reshape(-1, 4).astype(np.int16)
    vals = (8 * b[:, 0] + 4 * b[:, 1] + 2 * b[:, 2] + b[:, 3]) % P
    return vals.reshape(bits.shape[:-1]).astype(np.uint8)
def gf17_alt_mul(a_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    a, b = a_bits.astype(np.int16), b_bits.astype(np.int16)
    c3 = a[..., 0]*b[..., 3] + a[..., 1]*b[..., 2] + a[..., 2]*b[..., 1] + a[..., 3]*b[..., 0]
    c2 = a[..., 1]*b[..., 3] + a[..., 2]*b[..., 2] + a[..., 3]*b[..., 1] - a[..., 0]*b[..., 0]
    c1 = a[..., 2]*b[..., 3] + a[..., 3]*b[..., 2] - a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
    c0 = a[..., 3]*b[..., 3] - a[..., 0]*b[..., 2] - a[..., 1]*b[..., 1] - a[..., 2]*b[..., 0]
    return np.stack([c3, c2, c1, c0], axis=-1)
def gf17_alt_score(q_bits: np.ndarray, k_bits: np.ndarray) -> np.ndarray:
    prod = gf17_alt_mul(q_bits, k_bits)
    return ((8*prod[...,0]+4*prod[...,1]+2*prod[...,2]+prod[...,3]) % P).astype(np.uint8)
