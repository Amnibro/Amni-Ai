import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple
class VRAMCache:
    def __init__(self, max_pages: int = 512, page_size: int = 65536):
        self.max_pages = max_pages
        self.page_size = page_size
        self._cache = OrderedDict()
        self._hits = 0
        self._misses = 0
    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    def put(self, key: str, data: np.ndarray) -> Optional[Tuple[str, np.ndarray]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = data
            return None
        evicted = None
        if len(self._cache) >= self.max_pages:
            evicted = self._cache.popitem(last=False)
        self._cache[key] = data
        return evicted
    def contains(self, key: str) -> bool:
        return key in self._cache
    def evict(self, key: str) -> Optional[np.ndarray]:
        return self._cache.pop(key, None)
    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    @property
    def size(self) -> int:
        return len(self._cache)
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    @property
    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._cache.values())
    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)
    def stats(self) -> dict:
        return {
            "cached_pages": self.size,
            "max_pages": self.max_pages,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "memory_mb": round(self.memory_mb, 6)
        }
    def warm_keys(self) -> list:
        return list(self._cache.keys())
