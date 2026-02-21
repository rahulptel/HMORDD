"""Dataset helpers for training the TSP node classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from hmordd import Paths
from hmordd.tsp import PROB_NAME, PROB_PREFIX
from hmordd.tsp.utils import compute_stat_features, get_instance_data


class TSPNodeDataset:
    GRID_DIM = 1000
    MAX_DIST_ON_GRID = ((GRID_DIM**2) + (GRID_DIM**2)) ** 0.5
    MAX_INSTS_PER_SPLIT = {"train": 1000, "val": 100, "test": 100}
    PID_OFFSET = {"train": 0, "val": 1000, "test": 1100}
    COORD_DIM = 2

    def __init__(
        self,
        n_objs: int,
        n_vars: int,
        split: str,
        device: torch.device,
        n_insts: Optional[int] = None,
        pos_ratio: float = 1.0,
        neg_to_pos_ratio: float = 1.0,
        subsample: float = 1.0,
        resample: bool = False,
        generator: Optional[torch.Generator] = None,
        seed: int = 7,
        dataset_root: Optional[Path] = None,
        dd_path: Optional[Path] = None,
    ):
        self.n_objs = n_objs
        self.n_vars = n_vars
        self.split = split
        self.device = device
        self.n_insts = (
            self.MAX_INSTS_PER_SPLIT.get(split) if n_insts is None else n_insts
        )
        self.subsample = subsample
        self.pos_ratio = pos_ratio
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.resample = resample
        self.generator = generator
        self.seed = seed

        self.n_samples_pos = None
        self.n_samples_neg = None

        self.size = f"{n_objs}_{n_vars}"
        self.pid_offset = self.PID_OFFSET[split]
        self.dataset_path = (
            dataset_root
            if dataset_root is not None
            else Paths.dataset / PROB_NAME / self.size
        )

        # Load DD nodes to GPU.
        self.dd_flat = None
        self.nodes_in_layer_prefix = None
        self.dd_path = (
            Path(dd_path)
            if dd_path is not None
            else Paths.dds / PROB_NAME / self.size / f"{PROB_PREFIX}_dd.json"
        )
        self._set_dd_flat()

        # Instance data.
        self.coords = None
        self.dists = None
        self._set_instance_data()

        # Node data.
        node_np = np.load(self.dataset_path / f"{split}.npz")["arr_0"]
        node_np = node_np[node_np[:, 0] < self.pid_offset + self.n_insts]
        n_nodes = node_np.shape[0]

        self.node_data = torch.from_numpy(node_np).float().to(device)
        self.ids = torch.arange(n_nodes, device=device)

        # Positive samples.
        self.pos_ids = self.ids[self.node_data[:, -1] == 1]
        if self.pos_ids.numel() > 0:
            perm = torch.randperm(self.pos_ids.shape[0], generator=self.generator)
            self.pos_ids = self.pos_ids[perm.to(self.pos_ids.device)]
            self.pos_ids = self.pos_ids[: int(self.pos_ids.shape[0] * self.pos_ratio)]

        # Negative samples.
        self.neg_ids = self.ids[self.node_data[:, -1] != 1]
        if self.neg_ids.numel() > 0:
            perm = torch.randperm(self.neg_ids.shape[0], generator=self.generator)
            self.neg_ids = self.neg_ids[perm.to(self.neg_ids.device)]
            self.neg_ids = self.neg_ids[
                : min(int(self.pos_ids.shape[0] * self.neg_to_pos_ratio), self.neg_ids.shape[0])
            ]

        self.pos_ids_dataset = TensorDataset(self.pos_ids.long())
        self.neg_ids_dataset = TensorDataset(self.neg_ids.long())

        self.n_samples_pos = int(self.pos_ids.shape[0] * subsample)
        self.n_samples_neg = int(self.neg_ids.shape[0] * subsample)

        self.pos_ids_loader = DataLoader(
            self.pos_ids_dataset,
            batch_size=max(1, self.n_samples_pos),
            shuffle=True,
            drop_last=True,
        )
        self.pos_ids_iter = iter(self.pos_ids_loader)

        self.neg_ids_loader = DataLoader(
            self.neg_ids_dataset,
            batch_size=max(1, self.n_samples_neg),
            shuffle=True,
            drop_last=True,
        )
        self.neg_ids_iter = iter(self.neg_ids_loader)
        self._set_epoch_node_ids()

        self.sample_weight = (
            torch.from_numpy(
                np.array(
                    [
                        self.pos_ids.shape[0] / max(1, self.node_data.shape[0]),
                        self.neg_ids.shape[0] / max(1, self.node_data.shape[0]),
                    ]
                )
            )
            .float()
            .to(device)
        )

        self.node_dataset = TensorDataset(
            self.node_data[:, 0:-1],
            self.node_data[:, -1],
        )

    @staticmethod
    def get_layer_weights_exponential(lid: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * lid)

    def _set_dd_flat(self) -> None:
        if not self.dd_path.exists():
            raise FileNotFoundError(f"DD JSON not found at {self.dd_path}")

        dd = json.load(open(self.dd_path, "r"))
        nodes_in_layer = [len(layer) for layer in dd]
        self.nodes_in_layer_prefix = [0] * len(nodes_in_layer)
        for i in range(1, len(nodes_in_layer)):
            self.nodes_in_layer_prefix[i] = sum(nodes_in_layer[0:i])

        self.nodes_in_layer_prefix = (
            torch.from_numpy(np.array(self.nodes_in_layer_prefix))
            .long()
            .to(self.device)
        )

        flat = []
        for layer in dd:
            for node in layer:
                flat.append(node)
        self.dd_flat = torch.from_numpy(np.array(flat)).float().to(self.device)

    def _set_instance_data(self) -> None:
        self.coords = torch.zeros(
            (self.n_insts, self.n_objs, self.n_vars, self.COORD_DIM)
        )
        self.dists = torch.zeros(
            (self.n_insts, self.n_objs, self.n_vars, self.n_vars)
        )
        for idx, pid in enumerate(
            range(self.pid_offset, self.pid_offset + self.n_insts)
        ):
            inst = get_instance_data(self.size, self.split, self.seed, pid)
            self.coords[idx] = torch.from_numpy(inst["coords"])
            self.dists[idx] = torch.from_numpy(inst["dists"])

        self.dists = self.dists.float().to(self.device) / self.MAX_DIST_ON_GRID
        self.coords = self.coords.float().to(self.device) / self.GRID_DIM
        self.coords = torch.cat((self.coords, compute_stat_features(self.dists)), dim=-1)

    def get_instance_data(self, pids: torch.Tensor):
        idxs = pids - self.pid_offset
        return self.coords[idxs], self.dists[idxs]

    def _set_epoch_node_ids(self) -> None:
        try:
            pos = next(self.pos_ids_iter)
        except StopIteration:
            self.pos_ids_iter = iter(self.pos_ids_loader)
            pos = next(self.pos_ids_iter)

        try:
            neg = next(self.neg_ids_iter)
        except StopIteration:
            self.neg_ids_iter = iter(self.neg_ids_loader)
            neg = next(self.neg_ids_iter)

        epoch_ids = torch.cat((pos[0], neg[0]))
        perm = torch.randperm(epoch_ids.shape[0], generator=self.generator)
        self.epoch_ids = epoch_ids[perm.to(epoch_ids.device)]

    def get_epoch_node_dataset(self) -> Subset:
        if self.resample:
            self._set_epoch_node_ids()
        return Subset(self.node_dataset, self.epoch_ids)

    def __len__(self) -> int:
        return self.n_samples_pos + self.n_samples_neg


def get_dataloader(
    dataset: TSPNodeDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    epoch_node_dataset = dataset.get_epoch_node_dataset()
    return DataLoader(
        epoch_node_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )


__all__ = ["TSPNodeDataset", "get_dataloader"]
