import hashlib
from pprint import pprint

import xgboost as xgb
from hmordd import Paths


class XGBTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.raw_train = None
        self.raw_val = None
        self.dtrain = None
        self.dval = None
        self.dtest = None
        self.pid_lid_map_train = None
        self.pid_lid_map_val = None
        self.evals = []
        self.bst = None
        self.params = {}
        self.evals_result = {}

        # Get model name
        self.mdl_path = Paths.pretrained / f"{self.cfg.model.type}/{self.cfg.prob.name}/{self.cfg.prob.size}"
        self.mdl_path.mkdir(parents=True, exist_ok=True)
        self.mdl_name = self.get_trainer_str()

        # Convert to hex
        h = hashlib.blake2s(digest_size=32)
        h.update(self.mdl_name.encode("utf-8"))
        self.mdl_hex = h.hexdigest()


    def set_mdl_param(self):
        self.param = {"max_depth": self.cfg.model.max_depth,
                      "eta": self.cfg.model.eta,
                      "min_child_weight": self.cfg.model.min_child_weight,
                      "subsample": self.cfg.model.subsample,
                      "colsample_bytree": self.cfg.model.colsample_bytree,
                      "objective": self.cfg.model.objective,
                      "device": self.cfg.device,
                      "eval_metric": list(self.cfg.model.eval_metric),
                      "nthread": self.cfg.model.nthread,
                      "seed": self.cfg.model.seed,
                      "random_state": self.cfg.model.seed}
        pprint(self.param)

    def set_model(self):
        self.set_mdl_param()
        mdl_path = self.mdl_path.joinpath(f"model_{self.mdl_name}.json")
        print("Loading model: ", mdl_path, ", Exists: ", mdl_path.exists())
        if mdl_path.exists():
            self.bst = xgb.Booster(self.param)
            self.bst.load_model(mdl_path)

    def get_trainer_str(self):
        name = ""
        if self.cfg.model.max_depth != 5:
            name += f"d{self.cfg.model.max_depth}-"
        if self.cfg.model.eta != 0.3:
            name += f"eta{self.cfg.model.eta}-"
        if self.cfg.model.min_child_weight != 100:
            name += f"mcw{self.cfg.model.min_child_weight}-"
        if self.cfg.model.subsample != 1:
            name += f"ss{self.cfg.model.subsample}-"
        if self.cfg.model.colsample_bytree != 1:
            name += f"csbt{self.cfg.model.colsample_bytree}-"
        if self.cfg.model.type == "gbt" and self.cfg.model.objective != "binary:logistic":
            name += f"{self.cfg.model.objective}-"
        elif self.cfg.model.type == "gbt_rank" and self.cfg.model.objective != "rank:pairwise":
            name += f"{self.cfg.model.objective}-"

        if self.cfg.model.num_round != 250:
            name += f"ep{self.cfg.model.num_round}-"
        if self.cfg.model.early_stopping_rounds != 20:
            name += f"es{self.cfg.model.early_stopping_rounds}-"
        if self.cfg.model.evals[-1] != "val":
            name += f"eval{self.cfg.model.evals[-1]}-"
        if self.cfg.model.type == "gbt" and self.cfg.model.eval_metric[-1] != "logloss":
            name += f"l{self.cfg.model.eval_metric[-1]}"
        elif self.cfg.model.type == "gbt_rank" and self.cfg.model.eval_metric[-1] != "ndcg":
            name += f"l{self.cfg.model.eval_metric[-1]}"
        if self.cfg.model.seed != 789541:
            name += f"seed{self.cfg.model.seed}"

        if self.cfg.dataset.train.from_pid != 0:
            name += f"trs{self.cfg.dataset.train.from_pid}-"
        if self.cfg.dataset.train.to_pid != 1000:
            name += f"tre{self.cfg.dataset.train.to_pid}-"
        if self.cfg.dataset.val.from_pid != 1000:
            name += f"vls{self.cfg.dataset.val.from_pid}-"
        if self.cfg.dataset.val.to_pid != 1100:
            name += f"vle{self.cfg.dataset.val.to_pid}-"

        if self.cfg.bdd_data.neg_to_pos_ratio != 1:
            name += f"npr{self.cfg.bdd_data.neg_to_pos_ratio}-"
        if self.cfg.bdd_data.layer_penalty != "exponential":
            name += f"lp{self.cfg.bdd_data.layer_penalty}-"
        if self.cfg.bdd_data.flag_importance_penalty is False:
            name += "nfimp-"
        if self.cfg.bdd_data.penalty_aggregation != "sum":
            name += f"{self.cfg.bdd_data.penalty_aggregation}-"
        if self.cfg.device != "cpu":
            name += f"dev{self.cfg.device}"
        return name

    def setup_predict(self):
        self.set_model()

    def predict(self, features):
        return self.bst.predict(features, iteration_range=(0, self.bst.best_iteration + 1))
