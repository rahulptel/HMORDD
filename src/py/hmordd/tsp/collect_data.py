"""Collect node-level datasets for training the TSP Pareto node predictor."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import hydra
import numpy as np

from hmordd import Paths
from hmordd.common.base_runner import BaseRunner
from hmordd.tsp import PROB_NAME, PROB_PREFIX


class DataCollector(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.root_path = Paths.dataset / PROB_NAME / self.cfg.prob.size
        self.shard_path = self.root_path / self.cfg.split
        self.shard_path.mkdir(parents=True, exist_ok=True)

    def _load_frontier(self, pid):
        frontier_dir = (
            Paths.sols
            / self.cfg.prob.name
            / self.cfg.prob.size
            / self.cfg.split
            / "exact"
        )
        npz_path = frontier_dir / f"{pid}.npz"
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                return {"x": data.get("x"), "z": data.get("z")}
            except Exception as exc:
                print(f"Error loading Pareto frontier from {npz_path}: {exc}")
        else:
            print(f"Pareto frontier not found for PID {pid} in {frontier_dir}")
        return None

    def _load_dd(self):
        dd_path = getattr(self.cfg, "dd_path", None)
        if dd_path:
            dd_path = Path(dd_path)
        else:
            dd_path = Paths.dds / PROB_NAME / self.cfg.prob.size / f"{PROB_PREFIX}_dd.json"
        if not dd_path.exists():
            raise FileNotFoundError(f"DD JSON not found at {dd_path}")
        with open(dd_path, "r") as fp:
            return json.load(fp)

    def _get_pareto_state_scores(self, x):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] < 2:
            return []
        x = x[:, 1:]
        n_pareto_sol = x.shape[0]

        pareto_state_scores = []
        for i in range(1, x.shape[1] + 1):
            x_partial = x[:, :i]
            states = np.zeros((x.shape[0], self.cfg.prob.n_vars), dtype=int)
            ind = np.arange(states.shape[0])
            for j in range(x_partial.shape[1]):
                states[ind, x_partial[:, j]] = 1
            last_city = x_partial[:, -1]
            states = np.hstack((states, last_city.reshape(-1, 1)))
            states_uq, cnt = np.unique(states, axis=0, return_counts=True)
            cntn = cnt / n_pareto_sol
            pareto_state_scores.append((states_uq, cntn))

        return pareto_state_scores

    def _tag_dd_nodes(self, pid, dd, pareto_state_scores, rng):
        if len(pareto_state_scores) != len(dd):
            raise ValueError("Pareto score layers do not match DD layers.")

        result = []
        total, pareto = 0, 0
        neg_ratio = getattr(self.cfg.data, "neg_to_pos_ratio", 1.0)

        for lidx, layer in enumerate(dd):
            pareto_states, pareto_scores = pareto_state_scores[lidx]
            pos = []
            for pareto_state, score in zip(pareto_states, pareto_scores):
                for nid, node in enumerate(layer):
                    if np.array_equal(pareto_state, node):
                        pos.append([pid, lidx, nid, float(score), 1])
                        break

            all_ids = list(range(len(layer)))
            pos_ids = [p[2] for p in pos]
            neg_ids = list(set(all_ids).difference(set(pos_ids)))
            pareto += len(pos_ids)
            total += len(neg_ids) + len(pos_ids)

            neg = [[pid, lidx, nid, 0.0, 0] for nid in neg_ids]
            rng.shuffle(neg)
            if neg_ratio < 1:
                neg_upto = len(neg)
            else:
                neg_upto = int(neg_ratio * len(pos))
            if neg_upto > 0:
                result.extend(neg[: min(len(neg), neg_upto)])
            result.extend(pos)

        if total > 0:
            self.pareto_frac_lst.append((pareto / total) * 100)

        if not result:
            return None
        return np.array(result, dtype=np.float32)

    def _save_shard(self, pid, dataset):
        if dataset is None or dataset.size == 0:
            return
        np.save(self.shard_path / f"{pid}.npy", dataset)

    def _concat_dataset(self):
        mats = []
        for path in sorted(self.shard_path.glob("*.npy")):
            mats.append(np.load(path))
        if not mats:
            print(f"No shard datasets found under {self.shard_path}")
            return
        combined = np.concatenate(mats, axis=0)
        out_path = self.root_path / f"{self.cfg.split}.npz"
        np.savez(out_path, combined)
        print(f"Saved concatenated dataset to {out_path} with shape {combined.shape}")

    def _zip_dataset(self):
        zip_path = Path(str(self.shard_path) + ".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in self.shard_path.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(self.shard_path.parent))
        shutil.rmtree(self.shard_path)
        print(f"Zipped dataset to {zip_path} and removed directory {self.shard_path}")

    def worker(self, rank):
        self._set_memory_limit()
        rng = np.random.RandomState(self.cfg.seed_dataset + rank)
        self.shard_path.mkdir(parents=True, exist_ok=True)
        self.pareto_frac_lst = []
        dd = self._load_dd()

        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"Collecting data for PID {pid} on rank {rank}")
            frontier = self._load_frontier(pid)
            if frontier is None or frontier.get("x") is None:
                continue

            pareto_state_scores = self._get_pareto_state_scores(frontier["x"])
            if not pareto_state_scores:
                print(f"No Pareto state scores for PID {pid}")
                continue

            dataset = self._tag_dd_nodes(pid, dd, pareto_state_scores, rng)
            if dataset is None:
                print(f"No samples generated for PID {pid}")
                continue

            self._save_shard(pid, dataset)

        if self.pareto_frac_lst:
            print(self.pareto_frac_lst)
            print(np.mean(self.pareto_frac_lst))

    def run(self):
        super().run()
        if getattr(self.cfg.data, "concat", True):
            self._concat_dataset()
        if getattr(self.cfg.data, "zip", False):
            self._zip_dataset()


@hydra.main(config_path="./configs", config_name="collect_data.yaml", version_base="1.2")
def main(cfg):
    collector = DataCollector(cfg)
    collector.run()


if __name__ == "__main__":
    main()
