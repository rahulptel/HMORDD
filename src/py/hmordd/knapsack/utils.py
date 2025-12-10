"""Utility functions for knapsack instances."""
import importlib
import zipfile
from operator import itemgetter
from pathlib import Path

import numpy as np
from hmordd import Paths
from hmordd.knapsack import PROB_NAME, PROB_PREFIX


def get_env(n_objs):
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


def get_instance_path(size, split, seed, pid) -> Path:
    filename = f"{PROB_PREFIX}_{seed}_{size}_{pid}.dat"
    return Paths.instances / PROB_NAME / size / split / filename


def get_instance_data(size, split, seed, pid) -> dict:
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
    n_cons = 1
    n_objs = int(lines[1])

    offset = 2
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


def get_static_order(order_type, data):
    if order_type == 'MinWt':
        idx_weight = [(i, w) for i, w in enumerate(data['weight'])]
        idx_weight.sort(key=itemgetter(1))

        return np.array([i[0] for i in idx_weight])
    elif order_type == 'MaxRatio':
        min_profit = np.min(data['value'], 0)
        profit_by_weight = [v / w for v, w in zip(min_profit, data['weight'])]
        idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
        idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)

        return np.array([i[0] for i in idx_profit_by_weight])
    elif order_type == 'Lex':
        return np.arange(data['n_vars'])
    

def get_dataset_prefix(with_parent=False, layer_weight=None, neg_to_pos_ratio=1.0):
    prefix = []
    if with_parent:
        prefix.append("wp")
    if layer_weight is not None:
        prefix.append(f"{layer_weight}")
    if neg_to_pos_ratio != 1.0:
        prefix.append(f"{neg_to_pos_ratio}")

    if len(prefix):
        prefix = "-".join(prefix)
    else:
        prefix = "default"

    return prefix