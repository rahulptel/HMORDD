import signal
import time

import numpy as np
from hmordd.common.dd import NOSH, DDManager
from hmordd.common.utils import handle_timeout
from hmordd.tsp.utils import get_env
from scipy.stats import rankdata

RESTRICT = 1

class NOSHRule(NOSH):
    def __init__(self, rule):
        super().__init__()
        self.rule = rule
        self.edge_agg, self.next_node = None, None
        if self.rule == "OrdMeanHigh":
            self.edge_agg, self.next_node = "mean", "min"
        elif self.rule == "OrdMaxHigh":
            self.edge_agg, self.next_node = "max", "min"
        elif self.rule == "OrdMinHigh":
            self.edge_agg, self.next_node = "min", "min"
        elif self.rule == "OrdMeanLow":
            self.edge_agg, self.next_node = "mean", "max"
        elif self.rule == "OrdMaxLow":
            self.edge_agg, self.next_node = "max", "max"
        elif self.rule == "OrdMinLow":
            self.edge_agg, self.next_node = "min", "max"
        else:
            raise ValueError(f"Unknown nosh rule '{self.rule}' for tsp restricted DD")
            
    def score_nodes(self, layer):
        # Rank edges based on distances
        edge_rank = []
        for obj in self.inst["dists"]:
            obj_flatten = obj.flatten()
            obj_flatten = rankdata(obj_flatten, method="dense")
            obj = obj_flatten.reshape(obj.shape)
            edge_rank.append(obj)
        edge_rank = np.array(edge_rank)

        if self.edge_agg == "mean":
            edge_rank = np.mean(edge_rank, axis=0)
        elif self.edge_agg == "max":
            edge_rank = np.max(edge_rank, axis=0)
        elif self.edge_agg == "min":
            edge_rank = np.min(edge_rank, axis=0)
        else:
            raise ValueError("Edge agg must be either 'mean', 'max', 'min'")

        scores = []
        for node in layer:
            last_visit = node[-1]
            to_visit = [i for i, n in enumerate(node[:-1]) if n == 0]
            if len(to_visit):
                if self.next_node == "min":
                    score = np.min([edge_rank[last_visit][tv] for tv in to_visit])
                elif self.next_node == "max":
                    score = np.max([edge_rank[last_visit][tv] for tv in to_visit])
                else:
                    raise ValueError(
                        "Node selection must be either 'greedy' or 'robust'"
                    )
                scores.append(score)
            else:
                # Static score for all nodes as we cannot discriminate between them
                scores.append(1)

        return scores

class NOSHE2E(NOSH):
    def __init__(self):
        super().__init__()
        self.model = None

    def get_variable_embeddings(self):
        pass
    
    def score_nodes(self, layer):
        pass

class TSPDDManager(DDManager):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def reset(self, inst):
        self.inst = inst
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

    def build_dd(self, *args, **kwargs):
        raise NotImplementedError("Build method must be implemented in subclasses.")
        
    def compute_frontier(self, pf_enum_method=None, time_limit=1800):
        start = time.time()
        try:
            signal.alarm(time_limit)
            self.env.compute_pareto_frontier()
            frontier = self.env.get_frontier()
            self.frontier = {
                "x": np.asarray(frontier.get("x", []), dtype=int),
                "z": np.asarray(frontier.get("z", []), dtype=int),
            }
        except:
            self.frontier = None
            self.time_frontier = time_limit
        else:
            self.time_frontier = time.time() - start
        finally:
            signal.alarm(0)

    def get_decision_diagram(self):
        return self.env.get_dd()


class TSPExactDDManager(TSPDDManager):
    def __init__(self, cfg):
        super().__init__(cfg)
                
    def build_dd(self):
        start = time.time()
        self.env.generate_dd()
        self.time_build = time.time() - start

                
class TSPRestrictedDDManager(TSPDDManager):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scorer = None        
        if self.cfg.dd.nosh in ["OrdMeanHigh", "OrdMaxHigh", "OrdMinHigh",
                                "OrdMeanLow", "OrdMaxLow", "OrdMinLow"]:            
            self.scorer = NOSHRule(self.cfg.dd.nosh)                
        elif self.cfg.dd.nosh == "E2E":
            self.scorer = NOSHE2E()
        else:
            raise ValueError(f"Unknown nosh '{self.cfg.dd.nosh}' for tsp restricted DD")
        
    def reset(self, inst):
        super().reset(inst)
        if self.scorer is not None:
            self.scorer.reset_inst(inst)
    
    def build_dd(self):
        self.time_build = time.time()
        
        # Restrict and build
        lid = 2
        while True:
            print("Building layer: ", lid)
            is_done = self.env.generate_next_layer()
            layer = self.env.get_layer(lid)
            print("Size: ", len(layer))
            if len(layer) > self.cfg.dd.width:
                print("\tRestricting")
                # Sort nodes in ascending order of scores and remove the last ones
                scores = self.scorer.score_nodes(layer)
                idx_scores = [(i, s) for i, s in enumerate(scores)]
                idx_scores.sort(key=lambda x: x[1])
                nodes_to_remove = [i for i, _ in idx_scores[self.cfg.dd.width:]]
                # nodes_to_remove.sort()
                # print(nodes_to_remove)
                self.env.approximate_layer(lid, RESTRICT, nodes_to_remove)
                print("\tSize: ", len(self.env.get_layer(lid)))
            lid += 1
            if is_done:
                break
                    
        self.time_build = self.time_build - time.time()

class DDManagerFactory:
    _managers = {
        "exact": TSPExactDDManager,
        "restricted": TSPRestrictedDDManager,
    }

    @classmethod
    def create_dd_manager(cls, cfg):
        manager_class = cls._managers.get(cfg.dd.type)
        if manager_class is None:
            raise ValueError(f"Unknown dd.type '{cfg.dd.type}' for tsp")
        return manager_class(cfg)
