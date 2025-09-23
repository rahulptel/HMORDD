import signal
import time
from typing import Optional

import numpy as np

from hmordd.common.dd import DDManager
from hmordd.common.utils import handle_timeout
from hmordd.tsp.utils import get_env


class TSPDDManager(DDManager):
    def reset(self, inst: dict, order: Optional[list] = None) -> None:
        signal.signal(signal.SIGALRM, handle_timeout)
        self.env = get_env(self.cfg.prob.n_objs)
        self.env.reset()

        dists = inst["dists"]
        if isinstance(dists, np.ndarray):
            objs = dists.astype(int).tolist()
        else:
            objs = [[list(map(int, row)) for row in matrix] for matrix in dists]

        self.env.set_inst(inst["n_vars"], inst["n_objs"], objs)
        self.env.initialize_dd_constructor()

    def build_dd(self) -> None:
        start = time.perf_counter()
        self.env.generate_dd()
        self.time_build = time.perf_counter() - start

    def compute_frontier(self, pf_enum_method=None, time_limit: int = 1800) -> None:
        start = time.perf_counter()
        try:
            signal.alarm(time_limit)
            self.env.compute_pareto_frontier()
            frontier = self.env.get_frontier()
            self.frontier = {
                "x": np.asarray(frontier.get("x", []), dtype=int),
                "z": np.asarray(frontier.get("z", []), dtype=int),
            }
        except Exception:
            self.frontier = None
            self.time_frontier = time_limit
        else:
            self.time_frontier = time.perf_counter() - start
        finally:
            signal.alarm(0)

    def get_decision_diagram(self):
        return self.env.get_dd()


class TSPExactDDManager(TSPDDManager):
    pass


class DDManagerFactory:
    _managers = {
        "exact": TSPExactDDManager,
    }

    @classmethod
    def create_dd_manager(cls, cfg):
        manager_class = cls._managers.get(cfg.dd.type)
        if manager_class is None:
            raise ValueError(f"Unknown dd.type '{cfg.dd.type}' for tsp")
        return manager_class(cfg)


__all__ = [
    "TSPDDManager",
    "TSPExactDDManager",
    "DDManagerFactory",
]
