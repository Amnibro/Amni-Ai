import numpy as ti
import numpy as np
# import numpy as ti  # temp mock
from amni.core.texture_mgr import TextureManager
try:from amni.compute.ops import matmul, matmul_quant, activate
except ImportError:matmul=matmul_quant=activate=None

class TextureLinear:
    def __init__(self, name: str, in_features: int, out_features: int,
                 tex_mgr: TextureManager, activation: str = "relu", use_bias: bool = True):
        self.name = name
        self.in_f = in_features
        self.out_f = out_features
        self.tex_mgr = tex_mgr
        self.activation = activation
        self.use_bias = use_bias
        self._w_cache = None
        self._b_cache = None
        self._w_meta = {}
        self._is_quant = False
        self._w_scale = 1.0
        self._w_bias = 0.0

    def _load_weights(self):
        if self._w_cache is None:
            w_raw, self._w_meta = self.tex_mgr.get_full_tensor(self.name, "weights")
            if w_raw is None:
                raise RuntimeError(f"Failed to load weights for layer {self.name}")
                
            self._is_quant = self._w_meta.get("quantized", False)
            
            # Prepare for GPU
            if self._is_quant:
                self._w_scale = self._w_meta.get("scale", 1.0)
                self._w_bias = self._w_meta.get("bias", 0.0)
                w_np = np.ascontiguousarray(w_raw.T)
                self._w_cache = ti.ndarray(dtype=ti.u8, shape=w_np.shape)
                self._w_cache.from_numpy(w_np)
            else:
                w_np = np.ascontiguousarray(w_raw.T)
                self._w_cache = ti.ndarray(dtype=ti.f32, shape=w_np.shape)
                self._w_cache.from_numpy(w_np)

        if self.use_bias and self._b_cache is None:
            b_raw, _ = self.tex_mgr.get_full_tensor(self.name, "bias")
            if b_raw is not None:
                b_np = np.ascontiguousarray(b_raw)
                self._b_cache = ti.ndarray(dtype=ti.f32, shape=b_np.shape)
                self._b_cache.from_numpy(b_np)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._load_weights()
        
        if self._is_quant:
            out = matmul_quant(x, self._w_cache, self._w_scale, self._w_bias, bias_layer=self._b_cache)
        else:
            out = matmul(x, self._w_cache, bias=self._b_cache)
            
        return activate(out, self.activation)

    def evict(self):
        self._w_cache = None
        self._b_cache = None
        self.tex_mgr.evict_layer(self.name)

    @property
    def param_count(self) -> int:
        return self.in_f * self.out_f + (self.out_f if self.use_bias else 0)

    def __repr__(self) -> str:
        q_tag = " [Q]" if self._is_quant else ""
        return f"TextureLinear({self.name}: {self.in_f}->{self.out_f}, act={self.activation}{q_tag})"
