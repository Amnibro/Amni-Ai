import json
from pathlib import Path
from typing import Optional
class TextureCatalog:
    def __init__(self, catalog_path: str = None):
        self.path = Path(catalog_path) if catalog_path else None
        self.entries = {}
    def register(self, layer_name: str, manifest: dict):
        self.entries[layer_name] = manifest
    def get(self, layer_name: str) -> Optional[dict]:return self.entries.get(layer_name)
    def get_page_path(self, layer_name: str, tensor: str, page_idx: int) -> Optional[str]:entry=self.entries.get(layer_name);t_info=entry.get(tensor)if entry else None;return t_info["files"][page_idx] if t_info and page_idx<len(t_info.get("files",[])) else None
    def get_page_count(self, layer_name: str, tensor: str) -> int:
        entry = self.entries.get(layer_name)
        return entry[tensor]["num_pages"] if entry and entry.get(tensor) else 0
    def save(self, path: str = None):
        out = Path(path) if path else self.path
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.entries, f, indent=2)
    def load(self, path: str = None):
        src = Path(path) if path else self.path
        with open(src, "r") as f:
            data = json.load(f)
            if "layers" in data and isinstance(data["layers"], dict):
                self.entries = data["layers"]
            else:
                self.entries = data
    @property
    def layer_names(self) -> list:
        return list(self.entries.keys())
    def summary(self) -> dict:
        total_pages, total_params = 0, 0
        for name, entry in self.entries.items():
            for tensor_key in ["weights", "bias"]:
                t = entry.get(tensor_key)
                if not t:
                    continue
                total_pages += t["num_pages"]
                shape = t["shape"]
                p = 1
                for s in shape:
                    p *= s
                total_params += p
        return {"layers": len(self.entries), "total_pages": total_pages, "total_params": total_params}
