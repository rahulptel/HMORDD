"""Utility helpers for TSP instances."""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

import numpy as np

from hmordd import Paths
from hmordd.tsp import PROB_NAME, PROB_PREFIX


def get_instance_path(size: str, split: str, seed: int, pid: int) -> Path:
    filename = f"{PROB_PREFIX}_{seed}_{size}_{pid}.npz"
    return Paths.instances / PROB_NAME / size / split / filename


def get_instance_data(size: str, split: str, seed: int, pid: int) -> dict:
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


def _load_npz(source) -> dict:
    with np.load(source) as data:
        coords = data["coords"]
        dists = data["dists"]
    return {"coords": coords, "dists": dists}


__all__ = ["get_instance_data", "get_instance_path"]
