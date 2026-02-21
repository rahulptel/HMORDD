"""Trainer for the TSP Pareto node predictor."""

from __future__ import annotations

import math
import pickle as pkl
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix

from hmordd import Paths
from hmordd.tsp.dataset import TSPNodeDataset, get_dataloader
from hmordd.tsp.model import ParetoNodePredictor
from hmordd.tsp.utils import get_exp_str, get_model_str, get_optimizer_str


def _save_model(save_path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )


def _save_result(
    save_path: Path,
    ep: int,
    global_step: int,
    train_result: Optional[dict] = None,
    val_result: Optional[dict] = None,
) -> None:
    pkl.dump(
        {
            "epoch": ep,
            "global_step": global_step,
            "train_result": train_result,
            "val_result": val_result,
        },
        open(str(save_path), "wb"),
    )


def _flatten_batch(batch, dataset: TSPNodeDataset):
    node_feat, label = batch

    pid = node_feat[:, 0].long()
    lid = node_feat[:, 1].long()
    nid = node_feat[:, 2].long()
    ns = node_feat[:, 3]

    flat_idx = dataset.nodes_in_layer_prefix[lid] + nid
    dd_node_data = dataset.dd_flat[flat_idx]

    coords, dists = dataset.get_instance_data(pid)
    lw = dataset.get_layer_weights_exponential(lid)
    label = label.long()

    return coords, dists, lid, dd_node_data, lw, ns, label


def _is_better(prev_best: float, new_result: float, metric: str) -> bool:
    if metric in {"f1", "accuracy", "precision", "recall"}:
        return new_result > prev_best
    if metric == "loss":
        return new_result < prev_best
    return False


def _initialize_eval_metric(metric: str) -> float:
    if metric in {"f1", "accuracy", "precision", "recall"}:
        return 0.0
    if metric == "loss":
        return float("inf")
    return 0.0


def _adjust_learning_rate(
    cfg,
    step: int,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Linearly warm up then cosine decay learning rate."""
    lr = cfg.lr
    if cfg.warmup > 0 and step < warmup_steps:
        lr = cfg.lr * (step + 1) / warmup_steps
    elif cfg.decay_lr and step > decay_steps:
        lr = cfg.min_lr
    elif cfg.decay_lr and step <= decay_steps:
        decay_ratio = (step - warmup_steps) / (decay_steps - warmup_steps)
        if not 0 <= decay_ratio <= 1:
            raise ValueError("Decay ratio must be between 0 and 1.")
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def _print_eval_result(
    split: str,
    ep: int,
    max_epochs: int,
    global_step: int,
    max_steps: int,
    result: Dict[str, float],
) -> None:
    print(
        "Epoch {}/{}, Step {}/{}, Split: {}".format(
            ep, max_epochs, global_step, max_steps, split
        )
    )
    print(
        "\tF1: {}, Recall: {}, Precision: {}, Acc: {}, Loss: {}".format(
            result["f1"],
            result["recall"],
            result["precision"],
            result["accuracy"],
            result["loss"],
        )
    )


@torch.no_grad()
def _evaluate(cfg, model, dataset, dataloader, loss_fn) -> Dict[str, float]:
    model.eval()
    tn, fp, fn, tp = 0, 0, 0, 0
    running_loss = 0.0
    n_items = 0.0
    for batch in dataloader:
        coords, dists, lids, states, lw, sw, labels = _flatten_batch(batch, dataset)

        logits = model(coords, dists, lids, states)
        loss = loss_fn(logits, labels, reduction="none")
        if cfg.weighted_loss:
            loss *= lw + sw
        loss = loss.mean()
        running_loss += loss.cpu().item() * coords.shape[0]
        n_items += coords.shape[0]

        pred_probs = F.softmax(logits.cpu(), dim=-1)
        pred_classes = pred_probs.argmax(dim=-1)
        tn_, fp_, fn_, tp_ = confusion_matrix(
            labels.cpu().numpy(),
            pred_classes.cpu().numpy(),
            labels=[0, 1],
        ).ravel()
        tn += tn_
        fp += fp_
        fn += fn_
        tp += tp_

    result = {
        "loss": running_loss / max(1.0, n_items),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / max(1, tp + fn + fp + tn),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    if tp + fp > 0:
        result["precision"] = tp / (tp + fp)
    if tp + fn > 0:
        result["recall"] = tp / (tp + fn)
    if result["precision"] > 0 and result["recall"] > 0:
        result["f1"] = (2 * tp) / ((2 * tp) + fp + fn)

    return result


class TSPTrainer:
    """Trainer wrapper for the TSP Pareto node predictor."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if getattr(cfg, "device", None) else "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.optimizer = None
        self.loss_fn = F.cross_entropy
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.exp_path: Optional[Path] = None

    def setup(self) -> None:
        if self.cfg.optimizer.decay_lr:
            if self.cfg.optimizer.min_lr is None:
                self.cfg.optimizer.min_lr = self.cfg.optimizer.lr / 10

        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)

        dd_path = getattr(self.cfg, "dd_path", None)
        dd_path = Path(dd_path) if dd_path else None
        dataset_root = (
            Path(self.cfg.dataset_root)
            if getattr(self.cfg, "dataset_root", None)
            else None
        )

        self.train_dataset = TSPNodeDataset(
            self.cfg.prob.n_objs,
            self.cfg.prob.n_vars,
            "train",
            self.device,
            n_insts=self.cfg.n_insts.train,
            resample=self.cfg.resample,
            subsample=self.cfg.subsample.train,
            pos_ratio=self.cfg.pos_ratio,
            neg_to_pos_ratio=self.cfg.neg_to_pos_ratio,
            generator=generator,
            seed=self.cfg.seed,
            dataset_root=dataset_root,
            dd_path=dd_path,
        )
        self.train_loader = get_dataloader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

        val_generator = torch.Generator()
        val_generator.manual_seed(self.cfg.seed + 1)
        self.val_dataset = TSPNodeDataset(
            self.cfg.prob.n_objs,
            self.cfg.prob.n_vars,
            "val",
            self.device,
            n_insts=self.cfg.n_insts.val,
            resample=False,
            subsample=self.cfg.subsample.val,
            pos_ratio=1,
            neg_to_pos_ratio=self.cfg.neg_to_pos_ratio,
            generator=val_generator,
            seed=self.cfg.seed,
            dataset_root=dataset_root,
            dd_path=dd_path,
        )
        self.val_loader = get_dataloader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.model = ParetoNodePredictor(self.cfg.model).to(self.device)
        self.optimizer = self.model.configure_optimizer(self.cfg.optimizer)

        exp_str = get_model_str(self.cfg.model)
        exp_str += "-" + get_optimizer_str(self.cfg.optimizer)
        exp_str += "-" + get_exp_str(self.cfg)
        run_name = getattr(self.cfg, "run_name", None)
        if run_name:
            exp_str += f"-{run_name}"

        self.exp_path = Paths.checkpoints / "tsp" / self.cfg.prob.size / exp_str
        self.exp_path.mkdir(exist_ok=True, parents=True)
        OmegaConf.save(self.cfg, self.exp_path / "config.yaml")

    def train(self) -> None:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Trainer not initialized; call setup() first.")

        max_steps = (len(self.train_dataset) // self.cfg.batch_size) * self.cfg.epochs
        warmup_steps = int((self.cfg.optimizer.warmup / 100) * max_steps)

        global_step = 0
        best_metric = _initialize_eval_metric(self.cfg.metric)
        best_epoch = -1
        best_step = -1

        tick = time.time()
        for ep in range(self.cfg.epochs):
            for batch in self.train_loader:
                self.model.train()
                lr = _adjust_learning_rate(
                    self.cfg.optimizer, global_step, self.optimizer, warmup_steps, max_steps
                )

                coords, dists, lids, states, lw, sw, labels = _flatten_batch(
                    batch, self.train_dataset
                )
                logits = self.model(coords, dists, lids, states)
                loss = self.loss_fn(logits, labels, reduction="none")
                if self.cfg.weighted_loss:
                    loss *= lw + sw
                loss = loss.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

                global_step += 1
                if global_step % self.cfg.eval_every == 0:
                    train_result = _evaluate(
                        self.cfg, self.model, self.train_dataset, self.train_loader, self.loss_fn
                    )
                    train_result.update({"epoch": ep, "global_step": global_step, "lr": lr})
                    _print_eval_result(
                        "Train", ep, self.cfg.epochs, global_step, max_steps, train_result
                    )

                    val_result = _evaluate(
                        self.cfg, self.model, self.val_dataset, self.val_loader, self.loss_fn
                    )
                    val_result.update({"epoch": ep, "global_step": global_step, "lr": lr})
                    _print_eval_result(
                        "Val", ep, self.cfg.epochs, global_step, max_steps, val_result
                    )

                    _save_model(
                        self.exp_path / f"ckpt_{ep}_{global_step}.pt",
                        self.model,
                        self.optimizer,
                    )
                    _save_result(
                        self.exp_path / f"result_{ep}_{global_step}.pkl",
                        ep,
                        global_step,
                        train_result,
                        val_result,
                    )
                    _save_result(self.exp_path / "last.pkl", ep, global_step)

                    prefix = ""
                    if _is_better(best_metric, val_result[self.cfg.metric], self.cfg.metric):
                        prefix = "***"
                        best_metric = val_result[self.cfg.metric]
                        best_epoch = ep
                        best_step = global_step

                        _save_model(
                            self.exp_path / "best_model.pt",
                            self.model,
                            self.optimizer,
                        )
                        _save_result(
                            self.exp_path / "best_result.pkl",
                            ep,
                            global_step,
                            train_result,
                            val_result,
                        )

                    print(
                        "\t{}Best epoch:step={}:{}, Best {}: {}".format(
                            prefix, best_epoch, best_step, self.cfg.metric, best_metric
                        )
                    )
                    print()

            if self.cfg.resample and self.cfg.subsample.train > 0:
                print("Resampling train dataset...")
                self.train_loader = get_dataloader(
                    self.train_dataset,
                    self.cfg.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                print("N dataloader: train: {}".format(len(self.train_loader)))

        wallclock = (time.time() - tick) / 3600
        pkl.dump({"train": wallclock}, open(str(self.exp_path / "log.pkl"), "wb"))


__all__ = ["TSPTrainer"]
