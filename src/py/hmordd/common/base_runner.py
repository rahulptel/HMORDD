"""Common runner utilities for experiment runners."""

from __future__ import annotations

import multiprocessing as mp
from typing import Optional, Sequence

try:  # pragma: no cover - Windows compatibility
    import resource
except ImportError:  # pragma: no cover - Windows compatibility
    resource = None  # type: ignore[assignment]


class BaseRunner:
    """Shared runner functionality for experiment runners."""

    _MEMORY_LIMIT_ATTRS: Sequence[str] = ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS")

    def __init__(self, cfg):
        self.cfg = cfg
        self._memory_limit_gb: Optional[float] = getattr(cfg, "memory_limit_gb", 16)

    def worker(self, rank: int) -> None:  # pragma: no cover - interface method
        """Process a shard of work for the given rank."""

        raise NotImplementedError

    def _set_memory_limit(self) -> None:
        """Apply memory usage limits when supported by the platform."""

        if resource is None:
            print(
                "Warning: 'resource' module not available; unable to enforce memory limits."
            )
            return

        if self._memory_limit_gb is None:
            return

        try:
            limit_bytes = int(float(self._memory_limit_gb) * (1024**3))
        except (TypeError, ValueError):
            print(f"Invalid memory limit configuration: {self._memory_limit_gb}")
            return

        limit_tuple = (limit_bytes, limit_bytes)
        for attr_name in self._MEMORY_LIMIT_ATTRS:
            limit_const = getattr(resource, attr_name, None)
            if limit_const is None:
                continue
            try:
                resource.setrlimit(limit_const, limit_tuple)
            except (ValueError, OSError) as exc:
                print(f"Unable to enforce {attr_name} limit of {self._memory_limit_gb} GB: {exc}")

    def run(self) -> None:
        """Execute the runner with optional multiprocessing."""

        if getattr(self.cfg, "n_processes", 1) == 1:
            self.worker(0)
            return

        pool = mp.Pool(processes=self.cfg.n_processes)
        try:
            results = [
                pool.apply_async(self.worker, args=(rank,))
                for rank in range(self.cfg.n_processes)
            ]
            for result in results:
                result.get()
        finally:
            pool.close()
            pool.join()
