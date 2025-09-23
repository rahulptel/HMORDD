"""Utility functions for knapsack instances."""

from __future__ import annotations

import importlib
from pathlib import Path
import zipfile

from hmordd import Paths
from hmordd.knapsack import PROB_NAME, PROB_PREFIX


def get_env(n_objs: int):
    module_name = f"libknapsackenvo{n_objs}"
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        try:
            module = importlib.import_module("libknapsackenv")
        except ImportError as exc:
            raise ImportError(
                f"Could not import knapsack environment for {n_objs} objectives."
            ) from exc
    return module.KnapsackEnv()


def get_instance_path(size: str, split: str, seed: int, pid: int) -> Path:
    filename = f"{PROB_PREFIX}_{seed}_{size}_{pid}.dat"
    return Paths.instances / PROB_NAME / size / split / filename


def get_instance_data(size: str, split: str, seed: int, pid: int) -> dict:
    file_path = get_instance_path(size, split, seed, pid)
    if file_path.exists():
        return _read_dat(file_path.read_text())

    archive_path = Paths.instances / PROB_NAME / f"{size}.zip"
    if archive_path.exists():
        member = f"{size}/{split}/{file_path.name}"
        with zipfile.ZipFile(archive_path, "r") as zf:
            with zf.open(member) as fh:
                content = fh.read().decode("utf-8")
        return _read_dat(content)

    raise FileNotFoundError(f"Instance {pid} for size {size} and split {split} not found.")


def _read_dat(content: str) -> dict:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    n_vars = int(lines[0])
    n_cons = int(lines[1])
    n_objs = int(lines[2])

    offset = 3
    values = []
    for _ in range(n_objs):
        values.append([int(v) for v in lines[offset].split()])
        offset += 1

    weight = [int(v) for v in lines[offset].split()]
    capacity = int(lines[offset + 1])
    cons_coeffs = [weight]
    rhs = [capacity]

    return {
        "n_vars": n_vars,
        "n_cons": n_cons,
        "n_objs": n_objs,
        "value": values,
        "weight": weight,
        "capacity": capacity,
        "cons_coeffs": cons_coeffs,
        "rhs": rhs,
    }


__all__ = ["get_env", "get_instance_data", "get_instance_path"]
