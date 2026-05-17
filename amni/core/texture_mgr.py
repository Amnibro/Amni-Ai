import numpy as np
import asyncio
from typing import Optional, List, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from amni.core.vram_cache import VRAMCache
from amni.storage.reader import WeightReader
from amni.storage.catalog import TextureCatalog
from amni.utils.config import EngineConfig

class TextureManager:
    def __init__(self, catalog: TextureCatalog, cfg: EngineConfig):
        self.catalog = catalog
        self.cfg = cfg
        self.cache = VRAMCache(max_pages=cfg.max_cached_pages, page_size=cfg.page_size)
        self.reader = WeightReader()
        self._prefetch_queue = deque()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._prefetch_tasks = {}
        self.active_nonces: List[int] = [] # List of active semantic concepts
        self.nonce_threshold = 2000 # Distance threshold for loading

    def set_active_nonces(self, nonces: List[int]):
        self.active_nonces = nonces

    def _page_key(self, layer: str, tensor: str, idx: int) -> str:
        return f"{layer}:{tensor}:{idx}"

    def get_page(self, layer: str, tensor: str, page_idx: int) -> Optional[np.ndarray]:
        # Sparsity Check
        if self.active_nonces:
            manifest = self.catalog.get(layer)
            if manifest and tensor in manifest:
                t_info = manifest[tensor]
                page_nonces = t_info.get("page_nonces")
                if page_nonces and page_idx < len(page_nonces):
                    p_nonce = page_nonces[page_idx]
                    
                    # 0 means "Global/General", always load
                    if p_nonce != 0:
                        # Check distance to nearest active concept
                        min_dist = min([abs(p_nonce - an) for an in self.active_nonces])
                        
                        if min_dist > self.nonce_threshold:
                            # SKIP LOAD - Return Zeros
                            # Determine correct shape/size
                            pg_w, pg_h = t_info['page_dims']
                            page_size = pg_w * pg_h
                            if page_idx == t_info['num_pages'] - 1 and t_info['last_page_count'] > 0:
                                page_size = t_info['last_page_count']
                            
                            dtype = np.uint8 if t_info.get("quantized") else np.float32
                            return np.zeros(page_size, dtype=dtype)
        
        key = self._page_key(layer, tensor, page_idx)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
            
        path = self.catalog.get_page_path(layer, tensor, page_idx)
        if not path:
            return None
            
        data = self.reader.read_page(path)
        self.cache.put(key, data)
        return data

    def get_full_tensor(self, layer: str, tensor: str) -> Tuple[Optional[np.ndarray], dict]:
        manifest = self.catalog.get(layer)
        if not manifest or not manifest.get(tensor):
            return None, {}
            
        t_info = manifest[tensor]
        pages = []
        for i in range(t_info["num_pages"]):
            page = self.get_page(layer, tensor, i)
            if page is None:
                return None, {}
            pages.append(page)
            
        combined = np.concatenate(pages)
        total = 1
        for s in t_info["shape"]:
            total *= s
            
        # Ensure correct reshape
        final_tensor = combined[:total].reshape(t_info["shape"])
        return final_tensor, t_info

    def schedule_prefetch(self, layer: str, tensor: str = "weights"):
        n_pages = self.catalog.get_page_count(layer, tensor)
        for i in range(n_pages):
            # Also check sparsity here to avoid queueing useless loads?
            # Yes, duplicate logic or verify?
            # Ideally verify. But get_page does the check cheaply.
            # If we prefetch, we call get_page (via execute_prefetch -> reader).
            # If get_page returns zeros, we shouldn't read disk.
            # The current get_page implementation avoids disk read if skipped.
            # So prefetch will just fill cache with zeros. Fast.
            key = self._page_key(layer, tensor, i)
            if not self.cache.contains(key):
                self._prefetch_queue.append((layer, tensor, i))

    def prefetch_layers(self, layer_names: List[str]):
        for name in layer_names[:self.cfg.prefetch_depth]:
            self.schedule_prefetch(name, "weights")
            self.schedule_prefetch(name, "bias")

    def execute_prefetch(self):
        while self._prefetch_queue:
            layer, tensor, idx = self._prefetch_queue.popleft()
            key = self._page_key(layer, tensor, idx)
            if self.cache.contains(key):
                continue
            
            # Use get_page logic to handle sparsity skipping automatically
            # Instead of manually reading reader
            data = self.get_page(layer, tensor, idx) 
            # get_page puts it in cache.
            # But get_page returns data, we don't need it here.

    async def execute_prefetch_async(self):
        # Async version is harder to unify with get_page because get_page is sync.
        # For now, disable async prefetch or make get_page support async?
        # Let's leave async prefetch as legacy/unsupported for sparsity for now.
        pass

    async def _async_load(self, key: str, path: str, loop):
        data = await loop.run_in_executor(self._executor, self.reader.read_page, path)
        self.cache.put(key, data)

    def evict_layer(self, layer: str):
        keys_to_evict = [k for k in self.cache.warm_keys() if k.startswith(f"{layer}:")]
        for k in keys_to_evict:
            self.cache.evict(k)

    def stats(self) -> dict:
        return {**self.cache.stats(), "prefetch_pending": len(self._prefetch_queue)}

    def shutdown(self):
        self._executor.shutdown(wait=False)
