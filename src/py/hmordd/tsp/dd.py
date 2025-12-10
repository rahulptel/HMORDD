import signal
import time

import numpy as np
import torch
from hmordd import Paths
from hmordd.common.dd import NOSH, DDManager
from hmordd.common.utils import handle_timeout
from hmordd.tsp import PROB_PREFIX
from hmordd.tsp.model import ParetoNodePredictor
from hmordd.tsp.utils import compute_stat_features, get_env
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
            
    def score_nodes(self, layer, lid=None):
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
    GRID_DIM = 1000
    MAX_DIST_ON_GRID = ((GRID_DIM ** 2) + (GRID_DIM ** 2)) ** 0.5

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.model = self._load_model()
        self.node_emb = None
        self.n_vars = None

    def _checkpoint_path(self):
        return (
            Paths.resources
            / "checkpoints"
            / PROB_PREFIX
            / self.cfg.prob.size
            / f"{self.cfg.model.type}_best_model.pt"
        )

    def _load_model(self):
        model = ParetoNodePredictor(self.cfg.model).to(self.device)
        ckpt_path = self._checkpoint_path()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = (
            checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def reset_inst(self, inst):
        super().reset_inst(inst)
        self.node_emb = None
        self.n_vars = int(inst.get("n_vars"))
        self._prepare_embeddings(inst)

    @torch.no_grad()
    def _prepare_embeddings(self, inst):
        coords = (torch.from_numpy(inst["coords"]) / self.GRID_DIM).float().to(self.device)
        dists = (torch.from_numpy(inst["dists"]) / self.MAX_DIST_ON_GRID).float().to(self.device)
        node_feat = torch.cat((coords, compute_stat_features(dists)), dim=-1)
        node_emb, edge_emb = self.model.token_encoder(
            node_feat.unsqueeze(0),
            dists.unsqueeze(0),
        )
        node_emb = self.model.graph_encoder(node_emb, edge_emb)
        self.node_emb = node_emb.squeeze(0)  # n_vars x d_emb

    @torch.no_grad()
    def score_nodes(self, layer, lid=None):
        if self.node_emb is None:
            raise RuntimeError("Variable embeddings not initialized; call reset_inst first.")
        if lid is None:
            raise ValueError("Layer id is required for the E2E scorer.")

        layer = torch.from_numpy(np.array(layer)).float().to(self.device)
        B = layer.shape[0]
        node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)

        last_visit = layer[:, -1].long()
        visit_mask = layer[:, :-1]
        visit_mask[torch.arange(B), last_visit] = 2
        visit_enc = self.model.visit_encoder(visit_mask.long())

        node_visit = self.model.node_visit_encoder2(
            self.model.node_visit_encoder1((node_emb + visit_enc)).sum(1)
        )
        customer_enc = node_emb[torch.arange(B), last_visit]
        n_vars = float(self.n_vars if self.n_vars is not None else self.node_emb.shape[0])
        l_tensor = torch.full((B,), float(lid), device=self.device)
        l_enc = self.model.layer_encoder(((n_vars - l_tensor) / n_vars).unsqueeze(-1))

        if getattr(self.model, "concat_emb", False):
            preds = self.model.pareto_predictor(
                torch.cat((node_visit, customer_enc, l_enc), dim=-1)
            )
        else:
            preds = self.model.pareto_predictor(node_visit + customer_enc + l_enc)
        preds = torch.softmax(preds, dim=-1)
        return preds[:, -1].cpu().numpy()

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
            self.scorer = NOSHE2E(cfg)
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
                scores = self.scorer.score_nodes(layer, lid - 1)
                idx_scores = [(i, s) for i, s in enumerate(scores)]
                if isinstance(self.scorer, NOSHE2E):
                    idx_scores.sort(key=lambda x: x[1], reverse=True)
                else:
                    idx_scores.sort(key=lambda x: x[1])
                nodes_to_remove = [i for i, _ in idx_scores[self.cfg.dd.width:]]
                # nodes_to_remove.sort()
                # print(nodes_to_remove)
                self.env.approximate_layer(lid, RESTRICT, nodes_to_remove)
                print("\tSize: ", len(self.env.get_layer(lid)))
            lid += 1
            if is_done:
                break
                    
        self.time_build = time.time() - self.time_build

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
