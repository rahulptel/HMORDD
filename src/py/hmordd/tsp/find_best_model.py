"""Select the best-performing TSP model per size."""

from __future__ import annotations

import json
import pickle as pkl
import shutil
from pathlib import Path
from typing import Dict, Optional

import hydra
from omegaconf import OmegaConf

from hmordd import Paths


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


def _load_best_metric(result_path: Path, metric: str) -> Optional[float]:
    try:
        with open(result_path, "rb") as fh:
            data = pkl.load(fh)
    except Exception as exc:
        print(f"Failed to read {result_path}: {exc}")
        return None

    val_result = data.get("val_result") if isinstance(data, dict) else None
    if not isinstance(val_result, dict):
        return None

    return val_result.get(metric)


def _load_model_type(exp_path: Path) -> Optional[str]:
    cfg_path = exp_path / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        cfg = OmegaConf.load(cfg_path)
    except Exception as exc:
        print(f"Failed to read {cfg_path}: {exc}")
        return None
    return getattr(getattr(cfg, "model", None), "type", None)


def _iter_sizes(cfg) -> Dict[str, Path]:
    base_path = Paths.checkpoints / "tsp"
    if cfg.sizes:
        return {size: base_path / size for size in cfg.sizes}
    if not base_path.exists():
        print(f"No checkpoints found at {base_path}")
        return {}
    return {p.name: p for p in base_path.iterdir() if p.is_dir()}


@hydra.main(config_path="./configs", config_name="find_best_model.yaml", version_base="1.2")
def main(cfg):
    size_dirs = _iter_sizes(cfg)
    if not size_dirs:
        return

    for size, size_dir in size_dirs.items():
        if not size_dir.exists():
            print(f"Size directory not found: {size_dir}")
            continue

        best_metric = _initialize_eval_metric(cfg.metric)
        best_exp = None
        best_model = None
        best_model_type = None

        for exp_path in sorted([p for p in size_dir.iterdir() if p.is_dir()]):
            result_path = exp_path / "best_result.pkl"
            model_path = exp_path / "best_model.pt"
            if not result_path.exists() or not model_path.exists():
                continue

            model_type = _load_model_type(exp_path)
            if cfg.model_type and model_type != cfg.model_type:
                continue

            metric_value = _load_best_metric(result_path, cfg.metric)
            if metric_value is None:
                continue

            if best_exp is None or _is_better(best_metric, metric_value, cfg.metric):
                best_metric = metric_value
                best_exp = exp_path
                best_model = model_path
                best_model_type = model_type

        if best_exp is None or best_model is None:
            print(f"No valid best models found for size {size}")
            continue

        summary = {
            "size": size,
            "metric": cfg.metric,
            "best_metric": best_metric,
            "exp_path": str(best_exp),
            "model_type": best_model_type,
            "checkpoint_path": str(best_model),
        }
        summary_path = size_dir / "best_model.json"
        try:
            with open(summary_path, "w") as fh:
                json.dump(summary, fh, indent=2)
        except Exception as exc:
            print(f"Failed to write summary for {size}: {exc}")

        if cfg.copy_to_resources:
            if best_model_type is None:
                print(f"Skipping copy for {size}; model type missing.")
                continue
            dest_dir = Paths.resources / "checkpoints" / "tsp" / size
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{best_model_type}_best_model.pt"
            try:
                shutil.copy2(best_model, dest_path)
                print(f"Copied best model for {size} to {dest_path}")
            except Exception as exc:
                print(f"Failed to copy best model for {size}: {exc}")
        else:
            print(f"Best model for {size}: {best_model}")


if __name__ == "__main__":
    main()
