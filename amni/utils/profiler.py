import time
from collections import defaultdict
class Profiler:
    def __init__(self):
        self._timers = defaultdict(list)
        self._counters = defaultdict(int)
        self._active = {}
    def start(self, name: str):
        self._active[name] = time.perf_counter()
    def stop(self, name: str) -> float:
        elapsed = time.perf_counter() - self._active.pop(name, time.perf_counter())
        self._timers[name].append(elapsed)
        return elapsed
    def count(self, name: str, n: int = 1):
        self._counters[name] += n
    def avg_ms(self, name: str) -> float:
        vals = self._timers.get(name, [])
        return (sum(vals) / len(vals) * 1000) if vals else 0.0
    def total_ms(self, name: str) -> float:
        return sum(self._timers.get(name, [])) * 1000
    def report(self) -> dict:
        r = {}
        for name, vals in self._timers.items():
            r[name] = {
                "calls": len(vals),
                "total_ms": f"{sum(vals) * 1000:.2f}",
                "avg_ms": f"{sum(vals) / len(vals) * 1000:.2f}",
                "min_ms": f"{min(vals) * 1000:.2f}",
                "max_ms": f"{max(vals) * 1000:.2f}"
            }
        for name, cnt in self._counters.items():
            r.setdefault(name, {})["count"] = cnt
        return r
    def reset(self):
        self._timers.clear()
        self._counters.clear()
        self._active.clear()
