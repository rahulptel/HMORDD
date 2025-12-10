
"""Utility helpers for TSP instances."""

import io
import zipfile

import numpy as np
import torch
from hmordd import Paths
from hmordd.tsp import PROB_NAME, PROB_PREFIX


def get_env(n_objs):
    try:
        lib = __import__("libtspenvo" + str(n_objs))
        return lib.TSPEnv()
    except:
        raise ImportError(f"Could not import library for {n_objs} objectives.")


def get_instance_path(size, split, seed, pid):
    filename = f"{PROB_PREFIX}_{seed}_{size}_{pid}.npz"
    return Paths.instances / PROB_NAME / size / split / filename


def get_instance_data(size, split, seed, pid):
    file_path = get_instance_path(size, split, seed, pid)
    if file_path.exists():
        return _load_npz(file_path)

    archive_path = Paths.instances / PROB_NAME / f"{size}.zip"
    if archive_path.exists():
        member = f"{size}/{split}/{file_path.name}"
        with zipfile.ZipFile(archive_path, "r") as zf:
            with zf.open(member) as fh:
                buffer = io.BytesIO(fh.read())
        return _load_npz(buffer)

    raise FileNotFoundError(f"Instance {pid} for size {size} and split {split} not found.")


def _load_npz(source):
    with np.load(source) as data:
        coords = data["coords"]
        dists = data["dists"]
    return {
        "coords": coords,
        "dists": dists,
        "n_objs": int(coords.shape[0]),
        "n_vars": int(coords.shape[1]),
    }


def compute_stat_features(dists):
    """Compute per-node distance statistics."""

    return torch.cat(
        (
            dists.max(dim=-1, keepdim=True)[0],
            dists.min(dim=-1, keepdim=True)[0],
            dists.std(dim=-1, keepdim=True),
            dists.median(dim=-1, keepdim=True)[0],
            dists.quantile(0.75, dim=-1, keepdim=True)
            - dists.quantile(0.25, dim=-1, keepdim=True),
        ),
        dim=-1,
    )


__all__ = ["get_env", "get_instance_data", "get_instance_path", "compute_stat_features"]
