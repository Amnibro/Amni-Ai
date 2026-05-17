from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
@dataclass
class EngineConfig:
    vram_budget_mb: int = 2048
    page_width: int = 256
    page_height: int = 256
    max_cached_pages: int = 512
    prefetch_depth: int = 2
    async_io: bool = True
    storage_dir: Path = field(default_factory=lambda: Path("weights"))
    image_format: str = "png"
    ti_arch: str = "gpu"
    @property
    def page_size(self) -> int:
        return self.page_width * self.page_height
    @property
    def page_bytes(self) -> int:
        return self.page_size * 4
@dataclass
class LayerConfig:
    name: str
    in_features: int
    out_features: int
    bias: bool = True
    activation: str = "relu"
@dataclass
class ModelConfig:
    name: str
    layers: List[LayerConfig] = field(default_factory=list)
    engine: EngineConfig = field(default_factory=EngineConfig)
    def total_params(self) -> int:
        total = 0
        for lc in self.layers:
            total += lc.in_features * lc.out_features
            total += lc.out_features if lc.bias else 0
        return total
    def add_layer(self, name: str, in_f: int, out_f: int, bias: bool = True, act: str = "relu") -> "ModelConfig":
        self.layers.append(LayerConfig(name, in_f, out_f, bias, act))
        return self
