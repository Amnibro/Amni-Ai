import numpy as np
from typing import List
from amni.model.layer import TextureLinear
from amni.core.texture_mgr import TextureManager
from amni.utils.config import ModelConfig
class Network:
    def __init__(self, cfg: ModelConfig, tex_mgr: TextureManager):
        self.cfg = cfg
        self.tex_mgr = tex_mgr
        self.layers: List[TextureLinear] = []
        for lc in cfg.layers:
            self.layers.append(TextureLinear(
                name=lc.name, in_features=lc.in_features,
                out_features=lc.out_features, tex_mgr=tex_mgr,
                activation=lc.activation, use_bias=lc.bias
            ))
    def forward(self, x: np.ndarray) -> np.ndarray:
        current = x.astype(np.float32)
        for i, layer in enumerate(self.layers):
            if i + 1 < len(self.layers):
                upcoming = [self.layers[j].name for j in range(i + 1, min(i + 3, len(self.layers)))]
                self.tex_mgr.prefetch_layers(upcoming)
                self.tex_mgr.execute_prefetch()
            current = layer.forward(current)
        return current
    def evict_all(self):
        for layer in self.layers:
            layer.evict()
    @property
    def total_params(self) -> int:
        return sum(l.param_count for l in self.layers)
    def summary(self) -> str:
        lines = [f"Network: {self.cfg.name} ({self.total_params:,} params)"]
        for l in self.layers:
            lines.append(f"  {l}")
        lines.append(f"  Cache: {self.tex_mgr.stats()}")
        return "\n".join(lines)
    @classmethod
    def from_config(cls, cfg: ModelConfig, tex_mgr: TextureManager) -> "Network":
        return cls(cfg, tex_mgr)
