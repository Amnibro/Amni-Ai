import numpy as np
from pathlib import Path
from PIL import Image
from amni.core.codec import rgba_to_weights

class WeightReader:
    def __init__(self):
        pass

    def read_page(self, path: str) -> np.ndarray:
        img = Image.open(path)
        if img.mode == 'L':
             # Quantized uint8 data - keep as uint8
             return np.array(img, dtype=np.uint8).flatten()
        else:
             # Standard packed float (RGBA) - convert to float32
             img = img.convert("RGBA")
             rgba = np.array(img, dtype=np.uint8)
             return rgba_to_weights(rgba)

    def read_layer_weights(self, manifest: dict) -> np.ndarray:
        pages = [self.read_page(f) for f in manifest["files"]]
        combined = np.concatenate(pages)
        total = 1
        for s in manifest["shape"]:
            total *= s
        return combined[:total].reshape(manifest["shape"])

    def read_single_page(self, path: str, count: int = None) -> np.ndarray:
        raw = self.read_page(path)
        return raw[:count] if count is not None else raw

    async def read_page_async(self, path: str) -> np.ndarray:
        import aiofiles
        async with aiofiles.open(path, "rb") as f:
            data = await f.read()
        from io import BytesIO
        img = Image.open(BytesIO(data))
        if img.mode == 'L':
             return np.array(img, dtype=np.uint8).flatten()
        else:
             img = img.convert("RGBA")
             return rgba_to_weights(np.array(img, dtype=np.uint8))
