"""Shared utilities for generating problem instances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import zipfile
from typing import Callable, Iterable, Optional

import numpy as np

from hmordd import Paths


@dataclass(frozen=True)
class SplitSpec:
    """Metadata describing one split of the dataset."""

    name: str
    count: int


@dataclass(frozen=True)
class InstanceSize:
    """Normalises size identifiers between file/directory conventions."""

    value: str
    delimiter: str

    def as_dirname(self) -> str:
        tokens = self._split_tokens()
        return self.delimiter.join(tokens)

    def as_filename(self, delimiter: Optional[str] = None) -> str:
        delim = delimiter if delimiter is not None else self.delimiter
        tokens = self._split_tokens()
        return delim.join(tokens)

    def _split_tokens(self) -> Iterable[str]:
        if "-" in self.value:
            return self.value.split("-")
        if "_" in self.value:
            return self.value.split("_")
        # Already atomic, fallback to two numbers? keep as string
        return (self.value,)


def generate_instances_for_problem(
    cfg,
    prob_name: str,
    prob_prefix: str,
    generate_fn: Callable[[np.random.RandomState, object], dict],
    write_fn: Callable[[Path, dict], None],
    *,
    size_delimiter: str,
    file_delimiter: Optional[str] = None,
    zip_output: bool = False,
    split_specs: Optional[Iterable[SplitSpec]] = None,
) -> None:
    """Common entry point for Hydra-powered instance generation scripts."""

    rng = np.random.RandomState(cfg.seed)
    size_str = getattr(cfg, "size", None)
    if size_str is None:
        size_str = f"{cfg.n_objs}{size_delimiter}{cfg.n_vars}"
        cfg.size = size_str

    size = InstanceSize(size_str, size_delimiter)
    size_dir = Paths.instances / prob_name / size.as_dirname()

    specs = list(split_specs) if split_specs is not None else _default_splits(cfg)
    next_pid = 0
    for spec in specs:
        split_dir = size_dir / spec.name
        split_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(spec.count):
            inst_path = split_dir / f"{prob_prefix}_{cfg.seed}_{size.as_filename(file_delimiter)}_{next_pid}"
            data = generate_fn(rng, cfg)
            write_fn(inst_path, data)
            next_pid += 1

    if zip_output:
        archive_path = size_dir.parent / f"{size_dir.name}.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in size_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(size_dir.parent))
        shutil.rmtree(size_dir)


def _default_splits(cfg) -> Iterable[SplitSpec]:
    return (
        SplitSpec("train", cfg.n_train),
        SplitSpec("val", cfg.n_val),
        SplitSpec("test", cfg.n_test),
    )


__all__ = [
    "InstanceSize",
    "SplitSpec",
    "generate_instances_for_problem",
]
