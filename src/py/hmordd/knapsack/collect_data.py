"""Collect node-level datasets for training XGBoost models on knapsack DDs."""

import shutil
import zipfile
from pathlib import Path

import hydra
import numpy as np

from hmordd import Paths
from hmordd.common.base_runner import BaseRunner
from hmordd.common.utils import append_pf_dom_path
from hmordd.knapsack.dd import DDManagerFactory
from hmordd.knapsack.featurizer import (
    FeaturizerConfig,
    KnapsackFeaturizer,
    get_bdd_node_features,
    get_bdd_node_features_gbt_rank,
)
from hmordd.knapsack.utils import (
    get_dataset_path,
    get_instance_data,
    get_static_order,
)


class DataCollector(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset_path = get_dataset_path(cfg)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    def _load_frontier(self, pid):
        """Load the saved exact Pareto frontier (expects solutions with decision vectors)."""
        frontier_dir = Paths.sols / self.cfg.prob.name / self.cfg.prob.size / self.cfg.split / "exact"
        frontier_dir = append_pf_dom_path(frontier_dir, self.cfg, include_dominance=True)
        npz_path = frontier_dir / f"{pid}.npz"
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                return {"x": data.get("x"), "z": data.get("z")}
            except Exception as exc:
                print(f"Error loading Pareto frontier from {npz_path}: {exc}")
                return None

        npy_path = frontier_dir / f"{pid}.npy"
        if npy_path.exists():
            try:
                # npy files from older runs may only store objective vectors; skip if no decisions.
                arr = np.load(npy_path, allow_pickle=True)
                if isinstance(arr, np.ndarray) and arr.dtype.names and "x" in arr.dtype.names:
                    payload = {"x": arr["x"], "z": arr["z"] if "z" in arr.dtype.names else None}
                    return payload
                if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                    maybe_dict = arr.item()
                    if isinstance(maybe_dict, dict) and "x" in maybe_dict:
                        return {"x": maybe_dict["x"], "z": maybe_dict.get("z")}
                print(f"Frontier at {npy_path} missing decision vectors; skipping PID {pid}")
            except Exception as exc:
                print(f"Error loading Pareto frontier from {npy_path}: {exc}")
        else:
            print(f"Pareto frontier not found for PID {pid} in {frontier_dir}")
        return None

    def _build_tagged_dd(self, pid, inst_data, order, pareto_x):
        """Rebuild the exact DD, tag Pareto nodes, and keep it in memory."""
        dd_manager = DDManagerFactory.create_dd_manager(self.cfg)
        try:
            dd_manager.reset(inst_data, order=order)
            dd_manager.build_dd()
            dd = dd_manager.get_decision_diagram()
        except Exception as exc:
            print(f"Failed to build DD for PID {pid}: {exc}")
            return None

        if dd is None:
            print(f"Empty DD for PID {pid}")
            return None

        pareto_state_scores = self._get_pareto_state_scores(inst_data, pareto_x, order=order)
        return self._tag_dd_nodes(dd, pareto_state_scores)

    def _get_pareto_state_scores(self, data, x, order=None):
        assert order is not None
        weight = np.array(data["weight"])[order]
        x = np.array(x)
        pareto_state_scores = []
        for i in range(1, x.shape[1]):
            x_partial = x[:, :i].reshape(-1, i)
            w_partial = weight[:i].reshape(i, 1)
            wt_dist = np.dot(x_partial, w_partial)
            pareto_state, pareto_score = np.unique(wt_dist, return_counts=True)
            pareto_score = pareto_score / pareto_score.sum()
            pareto_state_scores.append((pareto_state, pareto_score))

        return pareto_state_scores

    def _tag_dd_nodes(self, dd, pareto_state_scores):
        assert len(pareto_state_scores) == len(dd)

        for l in range(len(dd)):
            pareto_states, pareto_scores = pareto_state_scores[l]

            for n in dd[l]:
                node_state = n["s"][0]
                index = np.where(pareto_states == node_state)[0]
                if len(index):
                    n["pareto"] = 1
                    n["score"] = pareto_scores[index[0]]
                else:
                    n["pareto"] = 0
                    n["score"] = 0

        return dd

    def _get_bdd_node_dataset_gbt(self, inst_data, order, bdd, rng):
        featurizer = KnapsackFeaturizer(
            FeaturizerConfig(
                norm_const=self.cfg.prob.state_norm_const,
                raw=False,
                context=True,
            )
        )
        features = featurizer.get(inst_data)
        inst_features = features["inst"][0]
        var_features = features["var"][order]
        samples = []
        for lidx, layer in enumerate(bdd):
            actual_lidx = lidx + 1  # DD layers are 1..n-1
            if actual_lidx >= len(var_features):
                continue

            var_feat = var_features[actual_lidx]
            parent_var_feat = var_features[actual_lidx - 1] if self.cfg.data.with_parent else None
            prev_layer = bdd[lidx - 1] if self.cfg.data.with_parent and lidx > 0 else None

            pos_ids = [node_id for node_id, node in enumerate(layer) if node.get("pareto") == 1]
            neg_ids = list(set(range(len(layer))).difference(set(pos_ids)))
            num_pos_samples = len(pos_ids)
            if num_pos_samples == 0:
                continue

            if self.cfg.data.neg_to_pos_ratio < 1:
                num_neg_samples = len(neg_ids)
            else:
                num_neg_samples = int(self.cfg.data.neg_to_pos_ratio * num_pos_samples)
                num_neg_samples = np.min([num_neg_samples, len(neg_ids)])
                rng.shuffle(neg_ids)
            neg_ids = neg_ids[:num_neg_samples]

            node_ids = pos_ids[:]
            node_ids.extend(neg_ids)
            for node_id in node_ids:
                node = layer[node_id]
                node_feat = get_bdd_node_features(
                    actual_lidx,
                    node,
                    prev_layer,
                    inst_data["capacity"],
                    layer_norm_const=self.cfg.prob.layer_norm_const,
                    state_norm_const=self.cfg.prob.state_norm_const,
                    with_parent=self.cfg.data.with_parent,
                )
                if self.cfg.data.with_parent:
                    samples.append(
                        np.concatenate(
                            (
                                inst_features,
                                parent_var_feat,
                                var_feat,
                                node_feat,
                                [actual_lidx, node["score"]],
                            )
                        )
                    )
                else:
                    samples.append(
                        np.concatenate(
                            (inst_features, var_feat, node_feat, [actual_lidx, node["score"]])
                        )
                    )
        if not samples:
            return None
        return np.array(samples)

    def _get_bdd_node_dataset_gbt_rank(self, inst_data, order, bdd, rng):
        featurizer = KnapsackFeaturizer(
            FeaturizerConfig(
                norm_const=self.cfg.prob.state_norm_const,
                raw=False,
                context=True,
            )
        )
        features = featurizer.get(inst_data)
        inst_features = features["inst"][0]
        var_features = features["var"][order]
        samples = []
        for lidx, layer in enumerate(bdd):
            actual_lidx = lidx + 1
            if actual_lidx >= len(var_features):
                continue

            var_feat = var_features[actual_lidx]
            pos_ids = [node_id for node_id, node in enumerate(layer) if node.get("pareto") == 1]
            if len(pos_ids) == 0:
                continue
            threshold = 1 / len(pos_ids)
            neg_ids = list(set(range(len(layer))).difference(set(pos_ids)))
            num_pos_samples = len(pos_ids)
            if self.cfg.data.neg_to_pos_ratio < 1:
                num_neg_samples = len(neg_ids)
            else:
                num_neg_samples = int(self.cfg.data.neg_to_pos_ratio * num_pos_samples)
                num_neg_samples = np.min([num_neg_samples, len(neg_ids)])
                rng.shuffle(neg_ids)
            neg_ids = neg_ids[:num_neg_samples]

            node_ids = pos_ids[:]
            node_ids.extend(neg_ids)
            scores = np.array([layer[node_id]["score"] for node_id in node_ids])
            labels = np.zeros(len(node_ids))
            labels[scores > 0] = 2
            labels[scores > 1.25 * threshold] = 3
            labels[scores > 1.5 * threshold] = 4
            labels[scores > 2 * threshold] = 5
            labels[scores > 4 * threshold] = 10

            for i, node_id in enumerate(node_ids):
                node = layer[node_id]
                node_feat = get_bdd_node_features_gbt_rank(
                    actual_lidx,
                    node,
                    inst_data["capacity"],
                    layer_norm_const=self.cfg.prob.layer_norm_const,
                    state_norm_const=self.cfg.prob.state_norm_const,
                )
                samples.append(
                    np.concatenate(
                        (
                            inst_features,
                            var_feat,
                            node_feat,
                            [actual_lidx, node["score"], labels[i]],
                        )
                    )
                )
        if not samples:
            return None
        return np.array(samples)

    def _get_bdd_node_dataset(self, pid, inst_data, order, bdd, rng):
        if self.cfg.model.type == "gbt":
            dataset = self._get_bdd_node_dataset_gbt(inst_data, order, bdd, rng)
        elif self.cfg.model.type == "gbt_rank":
            dataset = self._get_bdd_node_dataset_gbt_rank(inst_data, order, bdd, rng)
            if dataset is not None:
                dataset = np.hstack(((np.ones(dataset.shape[0]) * pid).reshape(-1, 1), dataset))
        else:
            raise ValueError(f"Unsupported model type '{self.cfg.model.type}' for data collection")
        return dataset

    def _concat_dataset(self):
        mats = []
        for path in sorted(self.dataset_path.glob("*.npy")):
            mats.append(np.load(path))
        if not mats:
            print(f"No per-instance datasets found under {self.dataset_path}")
            return
        combined = np.concatenate(mats, axis=0)
        prefix = self.dataset_path.stem
        out_path = self.dataset_path.parent / f"{prefix}-{self.cfg.split}.npy"
        np.save(out_path, combined)
        print(f"Saved concatenated dataset to {out_path} with shape {combined.shape}")

    def _zip_dataset(self):
        zip_path = Path(str(self.dataset_path) + ".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in self.dataset_path.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(self.dataset_path.parent))
        shutil.rmtree(self.dataset_path)
        print(f"Zipped dataset to {zip_path} and removed directory {self.dataset_path}")

    def worker(self, rank):
        self._set_memory_limit()
        rng = np.random.RandomState(self.cfg.seed_dataset + rank)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"Collecting data for PID {pid} on rank {rank}")
            frontier = self._load_frontier(pid)
            if frontier is None or frontier.get("x") is None:
                continue

            inst_data = get_instance_data(self.cfg.prob.size, self.cfg.split, self.cfg.seed, pid)
            order = get_static_order(self.cfg.prob.order_type, inst_data)
            dd = self._build_tagged_dd(pid, inst_data, order, frontier["x"])
            if dd is None:
                continue

            dataset = self._get_bdd_node_dataset(pid, inst_data, order, dd, rng)
            if dataset is None or dataset.size == 0:
                print(f"No samples generated for PID {pid}")
                continue

            np.save(self.dataset_path / f"{pid}.npy", dataset)

    def run(self):
        super().run()
        if getattr(self.cfg.data, "concat", False):
            self._concat_dataset()
        if getattr(self.cfg.data, "zip", False):
            self._zip_dataset()


@hydra.main(config_path="./configs", config_name="collect_data.yaml", version_base="1.2")
def main(cfg):
    collector = DataCollector(cfg)
    collector.run()


if __name__ == "__main__":
    main()
