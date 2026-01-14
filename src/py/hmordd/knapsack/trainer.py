"""Training utilities for the XGBoost-based node scorer."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xgboost as xgb
from hmordd import Paths
from hmordd.knapsack.utils import get_dataset_prefix
from omegaconf import OmegaConf


@dataclass
class _SplitSource:
    kind: str  # "file", "zip", "dir"
    path: Path


class _NpyDataIter(xgb.DataIter):
    """Stream `.npy` shards into XGBoost without loading everything into RAM."""

    def __init__(
        self,
        files: Sequence[str],
        load_fn: Callable[[str], np.ndarray],
        split_rows: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    ):
        self.files = list(files)
        self.load_fn = load_fn
        self.split_rows = split_rows
        self._idx = 0
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def next(self, input_data):
        while self._idx < len(self.files):
            data = self.load_fn(self.files[self._idx])
            self._idx += 1
            if data is None or data.size == 0:
                continue

            features, labels, weights = self.split_rows(data)
            input_data(data=features, label=labels, weight=weights)
            return 1
        return 0

    def reset(self):
        self._idx = 0


class XGBTrainer:
    """Trainer for the XGBoost classifier used to score DD nodes."""

    def __init__(self, cfg):
        self.cfg = cfg

        # Dataset path components
        self.dataset_root = Paths.dataset / self.cfg.prob.name / self.cfg.model.type / self.cfg.prob.size
        self.data_cfg = getattr(cfg, "data", getattr(cfg, "bdd_data", None))
        self.bdd_cfg = getattr(cfg, "bdd_data", getattr(cfg, "data", None))
        self.dataset_prefix = get_dataset_prefix(
            getattr(self.data_cfg, "with_parent", False),
            getattr(self.data_cfg, "layer_weight", None),
            getattr(self.data_cfg, "neg_to_pos_ratio", 1.0),
        )

        # Model name and path
        self.mdl_path = Paths.pretrained / f"{self.cfg.model.type}/{self.cfg.prob.name}/{self.cfg.prob.size}"
        self.mdl_path.mkdir(parents=True, exist_ok=True)
        self.mdl_name = self._build_model_name()
        h = hashlib.blake2s(digest_size=32)
        h.update(self.mdl_name.encode("utf-8"))
        self.mdl_hex = h.hexdigest()

        # Training artifacts
        self.dtrain: Optional[xgb.DMatrix] = None
        self.dval: Optional[xgb.DMatrix] = None
        self.evals: List[Tuple[xgb.DMatrix, str]] = []
        self.bst: Optional[xgb.Booster] = None
        self.params: dict = {}
        self.evals_result: dict = {}

        # Optional weighting
        self.layer_weights = self._build_layer_weights(self.cfg.prob.n_vars)

    def _build_layer_weights(self, n_vars: int) -> Optional[np.ndarray]:
        if not getattr(self.bdd_cfg, "flag_layer_penalty", False):
            return None

        penalty = getattr(self.bdd_cfg, "layer_penalty", "exponential")
        n_layers = n_vars + 1
        if penalty == "exponential":
            decay = np.linspace(0, 2, n_layers)
            weights = np.exp(-decay)
        elif penalty == "linear":
            weights = np.linspace(1.0, 0.2, n_layers)
        else:
            weights = np.ones(n_layers)
        return weights.astype(np.float32)

    # ---------------------------------------------------------------------- #
    # Data loading helpers
    def _resolve_split_source(self, split: str) -> _SplitSource:
        split_root = self.dataset_root / split
        file_path = split_root / f"{self.dataset_prefix}-{split}.npy"
        zip_path = split_root / f"{self.dataset_prefix}.zip"
        dir_path = split_root / self.dataset_prefix

        if file_path.exists():
            return _SplitSource("file", file_path)
        if zip_path.exists():
            return _SplitSource("zip", zip_path)
        if dir_path.exists():
            return _SplitSource("dir", dir_path)
        raise FileNotFoundError(
            f"No dataset found for split '{split}'. "
            f"Looked for {file_path.name}, {zip_path.name}, and directory {dir_path}."
        )

    @staticmethod
    def _filter_pid_names(
        names: Iterable[str], start: int, end: int, stem_fn: Callable[[str], int]
    ) -> List[str]:
        filtered = []
        for name in names:
            try:
                pid = stem_fn(name)
            except (TypeError, ValueError):
                continue
            if start <= pid < end:
                filtered.append(name)
        return sorted(filtered, key=lambda n: stem_fn(n))

    def _zip_members(self, zip_file: zipfile.ZipFile, start: int, end: int) -> List[str]:
        prefix = f"{self.dataset_prefix}/"
        names = [n for n in zip_file.namelist() if n.startswith(prefix) and n.endswith(".npy")]
        return self._filter_pid_names(names, start, end, lambda n: int(Path(n).stem))

    def _dir_members(self, dir_path: Path, start: int, end: int) -> List[Path]:
        paths = dir_path.glob("*.npy")
        return [
            p for p in self._filter_pid_names(paths, start, end, lambda p: int(Path(p).stem))
        ]

    @staticmethod
    def _load_npy_from_zip(zip_file: zipfile.ZipFile, member: str) -> np.ndarray:
        with zip_file.open(member, "r") as fp:
            return np.load(io.BytesIO(fp.read()))

    @staticmethod
    def _load_npy(path: Path) -> np.ndarray:
        return np.load(path)

    def _split_columns(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        data = data.astype(np.float32, copy=False)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"Unexpected dataset shape {data.shape}; expected at least 3 columns.")

        layer_idx = data[:, -2].astype(int)
        scores = data[:, -1].astype(np.float32)
        labels = scores if self.bdd_cfg.label == "score" else (scores > 0).astype(np.float32)

        features = data[:, :-2]
        
        weights_importance = None
        if self.bdd_cfg.flag_importance_penalty:
            weights_importance = scores.copy()
        
        weights_layer = None
        if self.layer_weights is not None:
            weights_layer = self.layer_weights[np.clip(layer_idx, 0, self.layer_weights.shape[0] - 1)]
            
        weights = None
        if self.bdd_cfg.penalty_aggregation == "sum":
            if weights_importance is not None and weights_layer is not None:
                weights = weights_importance + weights_layer
            elif weights_importance is not None:
                weights = weights_importance
            else:
                weights = weights_layer
        elif self.bdd_cfg.penalty_aggregation == "prod":
            if weights_importance is not None and weights_layer is not None:
                weights = weights_importance * weights_layer
            elif weights_importance is not None:
                weights = weights_importance
            else:
                weights = weights_layer
        else:
            raise ValueError(f"Unknown penalty aggregation method: {self.bdd_cfg.penalty_aggregation}")

        return features, labels, weights

    def _dmatrix_from_file(self, path: Path) -> xgb.DMatrix:
        data = self._load_npy(path)
        features, labels, weights = self._split_columns(data)
        return xgb.DMatrix(data=features, label=labels, weight=weights)

    def _dmatrix_from_zip(self, source: _SplitSource, start: int, end: int) -> xgb.DMatrix:
        zip_file = zipfile.ZipFile(source.path)
        members = self._zip_members(zip_file, start, end)
        if not members:
            zip_file.close()
            raise ValueError(f"No files found in {source.path} for PID range [{start}, {end}).")
        iterator = _NpyDataIter(
            members,
            lambda name: self._load_npy_from_zip(zip_file, name),
            self._split_columns,
        )
        try:
            return xgb.DMatrix(iterator)
        finally:
            zip_file.close()

    def _dmatrix_from_dir(self, source: _SplitSource, start: int, end: int) -> xgb.DMatrix:
        members = self._dir_members(source.path, start, end)
        if not members:
            raise ValueError(f"No .npy files found under {source.path} for PID range [{start}, {end}).")
        iterator = _NpyDataIter(
            members,
            lambda path: self._load_npy(Path(path)),
            self._split_columns,
        )
        return xgb.DMatrix(iterator)

    def _load_split(self, split: str) -> xgb.DMatrix:
        source = self._resolve_split_source(split)
        start, end = self.cfg.dataset[split].from_pid, self.cfg.dataset[split].to_pid
        if source.kind == "file":
            return self._dmatrix_from_file(source.path)
        if source.kind == "zip":
            return self._dmatrix_from_zip(source, start, end)
        return self._dmatrix_from_dir(source, start, end)

    # ---------------------------------------------------------------------- #
    # Model configuration and persistence
    def _build_params(self) -> None:
        self.params = {
            "max_depth": self.cfg.model.max_depth,
            "eta": self.cfg.model.eta,
            "min_child_weight": self.cfg.model.min_child_weight,
            "subsample": self.cfg.model.subsample,
            "colsample_bytree": self.cfg.model.colsample_bytree,
            "objective": self.cfg.model.objective,
            "device": self.cfg.device,
            "eval_metric": list(self.cfg.model.eval_metric),
            "nthread": self.cfg.model.nthread,
            "seed": self.cfg.model.seed,
            "random_state": self.cfg.model.seed,
        }

    def _load_model(self) -> None:
        self._build_params()
        mdl_path = self.mdl_path / f"model_{self.mdl_name}.json"
        print(f"Loading model from {mdl_path} (exists={mdl_path.exists()})")
        if mdl_path.exists():
            self.bst = xgb.Booster(self.params)
            self.bst.load_model(mdl_path)

    def _build_model_name(self) -> str:
        name = ""
        if getattr(self.cfg.model, "max_depth", None) not in (None, 5):
            name += f"d{self.cfg.model.max_depth}-"
        if getattr(self.cfg.model, "eta", None) not in (None, 0.3):
            name += f"eta{self.cfg.model.eta}-"
        if getattr(self.cfg.model, "min_child_weight", None) not in (None, 100):
            name += f"mcw{self.cfg.model.min_child_weight}-"
        if getattr(self.cfg.model, "subsample", None) not in (None, 1):
            name += f"ss{self.cfg.model.subsample}-"
        if getattr(self.cfg.model, "colsample_bytree", None) not in (None, 1):
            name += f"csbt{self.cfg.model.colsample_bytree}-"
        if getattr(self.cfg.model, "objective", None) not in (None, "binary:logistic"):
            name += f"{self.cfg.model.objective}-"
        if getattr(self.cfg.model, "num_round", None) not in (None, 250):
            name += f"ep{self.cfg.model.num_round}-"
        if getattr(self.cfg.model, "early_stopping_rounds", None) not in (None, 20):
            name += f"es{self.cfg.model.early_stopping_rounds}-"
        if getattr(self.cfg.model, "evals", None):
            if self.cfg.model.evals[-1] != "val":
                name += f"eval{self.cfg.model.evals[-1]}-"
        if getattr(self.cfg.model, "eval_metric", None):
            if self.cfg.model.eval_metric[-1] != "logloss":
                name += f"l{self.cfg.model.eval_metric[-1]}"
        if getattr(self.cfg.model, "seed", None) not in (None, 789541):
            name += f"seed{self.cfg.model.seed}"

        dataset_cfg = getattr(self.cfg, "dataset", None)
        train_cfg = getattr(dataset_cfg, "train", None) if dataset_cfg else None
        val_cfg = getattr(dataset_cfg, "val", None) if dataset_cfg else None
        if train_cfg:
            if getattr(train_cfg, "from_pid", 0) != 0:
                name += f"trs{train_cfg.from_pid}-"
            if getattr(train_cfg, "to_pid", 1000) != 1000:
                name += f"tre{train_cfg.to_pid}-"
        if val_cfg:
            if getattr(val_cfg, "from_pid", 1000) != 1000:
                name += f"vls{val_cfg.from_pid}-"
            if getattr(val_cfg, "to_pid", 1100) != 1100:
                name += f"vle{val_cfg.to_pid}-"

        if getattr(self.bdd_cfg, "neg_to_pos_ratio", 1) != 1:
            name += f"npr{self.bdd_cfg.neg_to_pos_ratio}-"
        if getattr(self.bdd_cfg, "layer_penalty", "exponential") != "exponential":
            name += f"lp{self.bdd_cfg.layer_penalty}-"
        if getattr(self.bdd_cfg, "flag_importance_penalty", True) is False:
            name += "nfimp-"
        if getattr(self.bdd_cfg, "penalty_aggregation", "sum") != "sum":
            name += f"{self.bdd_cfg.penalty_aggregation}-"
        if getattr(self.cfg, "device", "cpu") != "cpu":
            name += f"dev{self.cfg.device}"
        return name

    # ---------------------------------------------------------------------- #
    # Public API
    def setup(self) -> None:
        print("Preparing datasets...")
        self.dtrain = self._load_split("train")
        self.dval = self._load_split("val")
        print(f"Training samples: {self.dtrain.num_row()}, Validation samples: {self.dval.num_row()}")

        self.evals = []
        for eval_name in getattr(self.cfg.model, "evals", []):
            if eval_name == "train" and self.dtrain is not None:
                self.evals.append((self.dtrain, "train"))
            elif eval_name == "val" and self.dval is not None:
                self.evals.append((self.dval, "val"))

        self._load_model()

    def train(self) -> None:
        if self.dtrain is None or self.dval is None:
            raise RuntimeError("Datasets not loaded. Call setup() before train().")

        print("Starting training...")
        self.bst = xgb.train(
            self.params,
            self.dtrain,
            num_boost_round=self.cfg.model.num_round,
            evals=self.evals,
            early_stopping_rounds=self.cfg.model.early_stopping_rounds,
            evals_result=self.evals_result,
        )
        self._save_config()
        self._save_model()
        self._save_metrics()
        self._print_stats(self.bst)

    def setup_predict(self) -> None:
        self._load_model()
        if self.bst is None:
            raise FileNotFoundError("No trained model found; run training before prediction.")

    def predict(self, features: xgb.DMatrix) -> np.ndarray:
        if self.bst is None:
            raise RuntimeError("Model not loaded. Call setup_predict() first.")
        best_iter = getattr(self.bst, "best_iteration", None)
        iteration_range = (0, None if best_iter is None else best_iter + 1)
        return self.bst.predict(features, iteration_range=iteration_range)

    # ---------------------------------------------------------------------- #
    # Persistence helpers
    def _save_config(self) -> None:
        cfg_path = self.mdl_path / f"config_{self.mdl_name}.yaml"
        OmegaConf.save(config=self.cfg, f=str(cfg_path))

    def _save_model(self) -> None:
        if self.bst is None:
            return
        mdl_path = self.mdl_path / f"model_{self.mdl_name}.json"
        self.bst.save_model(mdl_path)

    def _save_metrics(self) -> None:
        metrics_path = self.mdl_path / f"metrics_{self.mdl_name}.json"
        with open(metrics_path, "w") as fp:
            json.dump(self.evals_result, fp)

        summary_path = self.mdl_path / "summary.json"
        best_iter = getattr(self.bst, "best_iteration", None)
        summary_obj = {
            "mdl_hex": self.mdl_hex,
            "mdl_name": self.mdl_name,
            "best_iteration": best_iter,
            "eval_metric": list(self.cfg.model.eval_metric)[-1],
        }
        for em in self.cfg.model.eval_metric:
            for split_name in ("train", "val"):
                if split_name in self.evals_result and em in self.evals_result[split_name]:
                    key = f"{split_name}_{em}"
                    summary_obj[key] = self.evals_result[split_name][em][
                        best_iter if best_iter is not None else -1
                    ]

        existing = []
        if summary_path.exists():
            try:
                with open(summary_path, "r") as fp:
                    existing = json.load(fp)
            except Exception:
                existing = []
        existing.append(summary_obj)
        with open(summary_path, "w") as fp:
            json.dump(existing, fp)

    @staticmethod
    def _print_stats(bst: xgb.Booster) -> None:
        gain = bst.get_score(importance_type="gain")
        weight = bst.get_score(importance_type="weight")
        cover = bst.get_score(importance_type="cover")

        def _sorted_scores(scores: dict) -> List[Tuple[str, float]]:
            if not scores:
                return []
            mean_score = np.mean(list(scores.values()))
            return sorted(scores.items(), key=lambda x: x[1] / mean_score, reverse=True)

        print("Gain Scores:", _sorted_scores(gain))
        print("Weight Scores:", _sorted_scores(weight))
        print("Cover Scores:", _sorted_scores(cover))
