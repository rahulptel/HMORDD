import signal

import numpy as np
from hmordd.common.dd import NOSH, DDManager
from hmordd.common.utils import CONST, handle_timeout
from hmordd.knapsack.utils import get_env


class NOSHRule(NOSH):
    def __init__(self, rule):
        super().__init__()
        self.rule = rule

    def score_nodes(self, layer):
        if self.rule == "min_weight":
            idx_score = [(i, n["s"][0]) for i, n in enumerate(layer)]
            idx_score = sorted(idx_score, key=lambda x: x[1])

        elif self.rule == "max_weight":
            idx_score = [(i, n["s"][0]) for i, n in enumerate(layer)]
            idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)

        return [i[0] for i in idx_score]
    
class NOSHFE(NOSH):
    def __init__(self):
        super().__init__()

    def get_node_limit(self, layer_idx: int, total_layers: int) -> int:
        return -1  # No node limit
    
class KnapsackDDManager(DDManager):
    def reset(self, inst, order=None):
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

    def build_dd(self):
        raise NotImplementedError

    def compute_frontier(self, pf_enum_method=None, time_limit=1800):
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
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def build_dd(self):
        self.env.generate_dd()        
        self.env.reduce_dd()
        self.time_build = self.env.get_time(CONST.TIME_COMPILE)

class KnapsackRestrictedDDManager(KnapsackDDManager):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.env = None
        self.scorer = None
        
        if self.cfg.dd.nosh in ["max_weight", "min_weight"]:
            self.scorer = NOSHRule(self.cfg.dd.nosh)
        elif self.cfg.dd.nosh == "FE":
            self.scorer = NOSHFE()
        else:
            raise ValueError(f"Unknown NOSH '{self.cfg.dd.nosh}' for knapsack")
        
    def reset(self, inst, order=None):
        super().reset(inst, order)
        if self.scorer is not None:
            self.scorer.reset_inst(inst)
            
    def build_dd(self):
        # Set the variable used to generate the next layer
        lid = 0
        self.set_var_layer(self.env)

        # Restrict and build
        while lid < self.cfg.prob.n_vars - 1:
            self.env.generate_next_layer()
            self.set_var_layer(self.env)
            lid += 1

            layer = self.env.get_layer(lid)
            if len(layer) > self.cfg.dd.width:
                ranked_idxs = self.scorer.score_nodes(layer)                
                removed_idxs = ranked_idxs[self.cfg.dd.width:]
                if len(removed_idxs):
                    self.env.approximate_layer(lid, 
                                               CONST.RESTRICT, 
                                               1, 
                                               removed_idxs)

        # Generate terminal layer
        self.env.generate_next_layer()
        
class DDManagerFactory:
    _managers = {
        "exact": KnapsackExactDDManager,
        "restricted": KnapsackRestrictedDDManager,
    }

    @classmethod
    def create_dd_manager(cls, cfg):
        manager_class = cls._managers.get(cfg.dd.type)
        if manager_class is None:
            raise ValueError(f"Unknown dd.type '{cfg.dd.type}' for knapsack")
        return manager_class(cfg)
