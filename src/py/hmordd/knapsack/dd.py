import signal
import time

import numpy as np
import xgboost as xgb
from hmordd.common.dd import NOSH, DDManager
from hmordd.common.utils import CONST, handle_timeout
from hmordd.knapsack.featurizer import (
    FeaturizerConfig,
    KnapsackFeaturizer,
    get_bdd_node_features_gbt_rank,
)
from hmordd.knapsack.trainer import XGBTrainer
from hmordd.knapsack.utils import get_env


class BDDLayerToXGBConverter:
    def __init__(self, with_parent=False):
        self.featurizer = KnapsackFeaturizer(cfg=FeaturizerConfig())
        self.with_parent = with_parent
        self.inst_data = None
        self.order = None
        self.features = None
        self.inst_features = None
        self.var_features = None
        self.num_var_features = 0
        self.layer_norm_const = 100
        self.state_norm_const = 1000

    def set_features(self, inst_data, order):
        self.inst_data = inst_data
        self.order = order

        self.features = self.featurizer.get(inst_data)
        # Instance features
        self.inst_features = self.features["inst"][0]
        # Variable features. Reordered features based on ordering
        self.var_features = self.features["var"][order]
        self.num_var_features = self.var_features.shape[1]

    def convert_bdd_layer(self, lidx, layer):
        """lidx starts from 0 for layer 1"""
        features_lst = []

        # Parent variable features
        var_feat = self.var_features[lidx]
        for node in layer:
            node_feat = get_bdd_node_features_gbt_rank(lidx, node, self.inst_data["capacity"],
                                                       self.layer_norm_const, self.state_norm_const)

            features_lst.append(np.concatenate((self.inst_features,
                                                var_feat,
                                                node_feat)))

        return np.array(features_lst)

class NOSHRule(NOSH):
    def __init__(self, rule):
        super().__init__()
        self.rule = rule

    def score_nodes(self, layer, lid=None):
        if self.rule == "Scal-":
            idx_score = [(i, n[0]) for i, n in enumerate(layer)]
            idx_score = sorted(idx_score, key=lambda x: x[1])

        elif self.rule == "Scal+":
            idx_score = [(i, n[0]) for i, n in enumerate(layer)]
            idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)

        return [i[0] for i in idx_score]
    
class NOSHFE(NOSH):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.converter = BDDLayerToXGBConverter(with_parent=False)
        self.trainer = XGBTrainer(self.cfg)
        self.trainer.setup_predict()
        
    def score_nodes(self, layer, lid=0):
        print("Restricting...")
        features = self.converter.convert_bdd_layer(lid, layer)
        scores = self.trainer.predict(xgb.DMatrix(np.array(features)))
        idx_score = [(i, s) for i, s in enumerate(scores)]
        idx_score = sorted(idx_score, key=lambda x: (x[1], -x[0]), reverse=True)
        
        return [i[0] for i in idx_score]
    
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
        
        if self.cfg.dd.nosh in ["Scal+", "Scal-"]:
            self.scorer = NOSHRule(self.cfg.dd.nosh)
        elif self.cfg.dd.nosh == "FE":
            self.scorer = NOSHFE(self.cfg)
        else:
            raise ValueError(f"Unknown NOSH '{self.cfg.dd.nosh}' for knapsack")
        
    def reset(self, inst, order=None):
        super().reset(inst, order)
        if self.cfg.dd.nosh == "FE":
            self.scorer.converter.set_features(inst, order)
                        
    def build_dd(self):
        # Set the variable used to generate the next layer
        start = time.time()
        lid = 0

        # Restrict and build
        while lid < self.cfg.prob.n_vars - 1:
            self.env.generate_next_layer()
            lid += 1

            layer = self.env.get_layer(lid)
            if len(layer) > self.cfg.dd.width:
                ranked_idxs = self.scorer.score_nodes(layer, lid=lid)                
                removed_idxs = ranked_idxs[self.cfg.dd.width:]
                if len(removed_idxs):
                    self.env.approximate_layer(lid, 
                                               CONST.RESTRICT, 
                                               removed_idxs)

        # Generate terminal layer
        self.env.generate_next_layer()
        self.env.reduce_dd()
        self.time_build = time.time() - start

        
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
