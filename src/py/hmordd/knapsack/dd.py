import signal
from typing import Optional

import numpy as np

from hmordd.common.dd import DDManager
from hmordd.common.utils import CONST, handle_timeout
from hmordd.knapsack.utils import get_env


class KnapsackDDManager(DDManager):
    def reset(self, inst: dict, order: Optional[list] = None) -> None:
        signal.signal(signal.SIGALRM, handle_timeout)
        self.env = get_env(self.cfg.prob.n_objs)

        order = [] if order is None else list(order)
        self.env.reset(
            self.cfg.prob.preprocess,
            self.cfg.prob.pf_enum_method,
            self.cfg.prob.maximization,
            self.cfg.prob.dominance,
            self.cfg.prob.bdd_type,
            self.cfg.prob.maxwidth,
            order,
        )

        obj_coeffs = list(map(list, zip(*inst["value"])))
        self.env.set_inst(
            inst["n_vars"],
            inst["n_cons"],
            inst["n_objs"],
            obj_coeffs,
            inst["cons_coeffs"],
            inst["rhs"],
        )
        if getattr(self.cfg.prob, "preprocess", False):
            self.env.preprocess_inst()
        self.env.initialize_dd_constructor()

    def build_dd(self) -> None:
        self.env.generate_dd()
        try:
            self.time_build = self.env.get_time(CONST.TIME_COMPILE)
        except Exception:
            self.time_build = None

    def compute_frontier(self, pf_enum_method=None, time_limit: int = 1800) -> None:
        try:
            signal.alarm(time_limit)
            self.env.compute_pareto_frontier()
            frontier = self.env.get_frontier()
            self.frontier = {
                "x": np.asarray(frontier.get("x", []), dtype=int),
                "z": np.asarray(frontier.get("z", []), dtype=int),
            }
            try:
                self.time_frontier = self.env.get_time(CONST.TIME_PARETO)
            except Exception:
                self.time_frontier = None
        except Exception:
            self.frontier = None
            self.time_frontier = time_limit
        finally:
            signal.alarm(0)

    def get_decision_diagram(self):
        return self.env.get_dd()


class KnapsackExactDDManager(KnapsackDDManager):
    pass


class DDManagerFactory:
    _managers = {
        "exact": KnapsackExactDDManager,
    }

    @classmethod
    def create_dd_manager(cls, cfg):
        manager_class = cls._managers.get(cfg.dd.type)
        if manager_class is None:
            raise ValueError(f"Unknown dd.type '{cfg.dd.type}' for knapsack")
        return manager_class(cfg)


__all__ = [
    "KnapsackDDManager",
    "KnapsackExactDDManager",
    "DDManagerFactory",
]
