from collections import defaultdict
import threading


class RouterMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters = defaultdict(int)

    def increment(self, key: str):
        with self._lock:
            self.counters[key] += 1

    def snapshot(self):
        with self._lock:
            return dict(self.counters)