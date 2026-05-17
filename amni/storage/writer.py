import numpy as np
from pathlib import Path
from PIL import Image
from amni.core.codec import weights_to_rgba, weights_to_u8_image, partition_weights, quantize_f32
from amni.utils.config import EngineConfig

class WeightWriter:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.storage_dir = Path(cfg.storage_dir)

    def write_layer(self, name: str, weights: np.ndarray, bias: np.ndarray = None, 
                   quantize: bool = False, weight_nonces: list = None) -> dict:
        layer_dir = self.storage_dir / name
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        w_manifest = self._write_pages(layer_dir, "w", weights, quantize=quantize, nonces=weight_nonces)
        
        # Bias is rarely quantized due to sensitivity, keep it as RGBA float32 by default
        b_manifest = self._write_pages(layer_dir, "b", bias, quantize=False) if bias is not None else None
        
        return {"name": name, "weights": w_manifest, "bias": b_manifest}

    def _write_pages(self, layer_dir: Path, prefix: str, data: np.ndarray, 
                    quantize: bool = False, nonces: list = None) -> dict:
        # Calculate global quantization parameters if needed
        global_scale, global_bias = None, None
        if quantize:
            _, global_scale, global_bias = quantize_f32(data)
            
        pages = partition_weights(data, self.cfg.page_size)
        pg_w, pg_h = self.cfg.page_width, self.cfg.page_height
        paths = []
        
        for i, page in enumerate(pages):
            if quantize:
                # Quantized (L mode)
                # Pass global scale/bias to ensure consistency across pages
                img_data, _, _ = weights_to_u8_image(page, pg_w, pg_h, scale=global_scale, bias=global_bias)
                img = Image.fromarray(img_data, mode="L")
            else:
                # Standard (RGBA float packing)
                img_data = weights_to_rgba(page, pg_w, pg_h)
                img = Image.fromarray(img_data, mode="RGBA")
                
            fname = f"{prefix}_{i:04d}.png"
            fpath = layer_dir / fname
            img.save(fpath)
            paths.append(str(fpath))
            
        manifest = {
            "shape": list(data.shape),
            "dtype": "uint8" if quantize else str(data.dtype),
            "num_pages": len(pages),
            "page_dims": [pg_w, pg_h],
            "last_page_count": len(pages[-1]) if pages else 0,
            "files": paths,
            "quantized": quantize
        }
        
        if quantize:
            manifest["scale"] = float(global_scale)
            manifest["bias"] = float(global_bias)
            
        if nonces is not None:
            # Ensure nonce list matches page count (truncate or pad with 0 if needed)
            if len(nonces) < len(pages):
                nonces = nonces + [0] * (len(pages) - len(nonces))
            elif len(nonces) > len(pages):
                nonces = nonces[:len(pages)]
            manifest["page_nonces"] = nonces
            
        return manifest

    def write_model(self, layers: dict, quantize_weights: bool = False) -> list:
        manifests = []
        for name, tensors in layers.items():
            w = tensors["weight"]
            b = tensors.get("bias")
            nonces = tensors.get("nonces") # Retrieve optional nonces
            manifests.append(self.write_layer(name, w, b, quantize=quantize_weights, weight_nonces=nonces))
        return manifests
