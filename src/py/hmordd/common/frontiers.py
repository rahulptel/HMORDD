"""Utilities for locating and loading saved Pareto frontiers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from hmordd import Paths
from hmordd.common.utils import append_pf_dom_path


def _dd_type(cfg, dd_type: Optional[str] = None) -> str:
    if dd_type is not None:
        return dd_type
    return getattr(getattr(cfg, "dd", None), "type", getattr(cfg, "dd_type", None))


def _setpacking_nosh_rule(cfg):
    nosh = getattr(getattr(cfg, "dd", None), "nosh", None)
    return getattr(nosh, "rule", nosh)


def _variant_get(variant, key, default=None):
    if variant is None:
        return default
    if isinstance(variant, dict):
        return variant.get(key, default)
    return getattr(variant, key, default)


def dd_sols_dir(cfg, dd_type: Optional[str] = None) -> Path:
    """Return the solution directory used by DD runners for a config."""

    problem = cfg.prob.name
    resolved_dd_type = _dd_type(cfg, dd_type)
    save_path = Paths.sols / cfg.prob.name / cfg.prob.size / cfg.split / resolved_dd_type

    if problem == "knapsack":
        save_path = append_pf_dom_path(
            save_path,
            cfg,
            include_dominance=True,
            include_track_x=True,
            include_order_type=True,
        )
        if resolved_dd_type == "restricted":
            save_path = save_path / f"{cfg.dd.nosh}-{cfg.dd.width}"
        return save_path

    if problem == "setpacking":
        save_path = append_pf_dom_path(save_path, cfg, include_dominance=True)
        if resolved_dd_type == "restricted":
            save_path = save_path / f"width-{cfg.dd.width}-nosh-{_setpacking_nosh_rule(cfg)}"
        return save_path

    if problem == "tsp":
        save_path = append_pf_dom_path(
            save_path,
            cfg,
            include_dominance=False,
            include_track_x=True,
        )
        nosh = getattr(getattr(cfg, "dd", None), "nosh", None)
        if resolved_dd_type == "restricted" and nosh:
            save_path = save_path / str(nosh)
        return save_path

    raise ValueError(f"Unsupported problem '{problem}'")


def restricted_dd_sols_dir_for_variant(cfg, variant) -> Path:
    """Return the restricted DD solution directory for a concrete variant."""

    problem = cfg.prob.name
    save_path = Paths.sols / cfg.prob.name / cfg.prob.size / cfg.split / "restricted"

    if problem == "knapsack":
        save_path = append_pf_dom_path(
            save_path,
            cfg,
            include_dominance=True,
            include_track_x=True,
            include_order_type=True,
        )
        return save_path / f"{_variant_get(variant, 'nosh')}-{_variant_get(variant, 'width')}"

    if problem == "setpacking":
        save_path = append_pf_dom_path(save_path, cfg, include_dominance=True)
        return save_path / (
            f"width-{_variant_get(variant, 'width')}-nosh-{_variant_get(variant, 'nosh_rule')}"
        )

    if problem == "tsp":
        save_path = append_pf_dom_path(
            save_path,
            cfg,
            include_dominance=False,
            include_track_x=True,
        )
        return save_path / str(_variant_get(variant, "nosh"))

    raise ValueError(f"Unsupported problem '{problem}'")


def dd_frontier_candidates(cfg, pid: int, dd_type: Optional[str] = None) -> list[Path]:
    """Return possible files for a saved DD frontier."""

    base = dd_sols_dir(cfg, dd_type)
    candidates = [base / f"{pid}.npz", base / f"{pid}.npy"]

    # Older setpacking restricted runs wrote one extra nested directory in
    # save(), while _get_save_path() had already included the restricted params.
    if cfg.prob.name == "setpacking" and _dd_type(cfg, dd_type) == "restricted":
        legacy_base = base / f"{cfg.dd.nosh}-{cfg.dd.width}"
        candidates.extend([legacy_base / f"{pid}.npz", legacy_base / f"{pid}.npy"])

    return candidates


def restricted_dd_frontier_candidates_for_variant(cfg, pid: int, variant) -> list[Path]:
    """Return possible files for a saved restricted DD frontier variant."""

    base = restricted_dd_sols_dir_for_variant(cfg, variant)
    return [base / f"{pid}.npz", base / f"{pid}.npy"]


def load_frontier(path: Path):
    """Load objective vectors from a saved frontier file."""

    if path.suffix == ".npz":
        with np.load(path) as data:
            if "z" in data:
                return np.asarray(data["z"])
            if not data.files:
                return None
            return np.asarray(data[data.files[0]])
    if path.suffix == ".npy":
        return np.asarray(np.load(path))
    raise ValueError(f"Unsupported frontier file extension: {path}")


def load_first_existing_frontier(paths: list[Path]):
    """Load the first existing frontier from a list of candidate paths."""

    for path in paths:
        if not path.exists():
            continue
        return load_frontier(path), path
    return None, None


def nsga2_defaults(problem: str, n_vars: int, n_objs: int, cutoff: str):
    """Return the default NSGA-II pop_size/run_time used by run_nsga2.py."""

    if cutoff not in ["restrict", "5xrestrict"]:
        raise ValueError(f"Unsupported cutoff: {cutoff}")

    pop_size, run_time = None, None
    if problem == "knapsack":
        if n_vars == 40 and n_objs == 7:
            pop_size, run_time = 25000, 54
        elif n_vars == 50 and n_objs == 4:
            pop_size, run_time = 3600, 12
        elif n_vars == 80 and n_objs == 3:
            pop_size, run_time = 2500, 52
    elif problem == "setpacking":
        if n_vars == 100 and n_objs <= 3:
            pop_size, run_time = 250, 1
        elif n_vars == 100 and n_objs == 4:
            pop_size, run_time = 1100, 1
        elif n_vars == 100 and n_objs == 5:
            pop_size, run_time = 4600, 1
        elif n_vars == 100 and n_objs == 6:
            pop_size, run_time = 9000, 2
        elif n_vars == 100 and n_objs >= 7:
            pop_size, run_time = 25000, 13
        elif n_vars == 150 and n_objs <= 3:
            pop_size, run_time = 800, 1
        elif n_vars == 150 and n_objs == 4:
            pop_size, run_time = 6100, 7
        elif n_vars == 150 and n_objs == 5:
            pop_size, run_time = 30000, 77
        elif n_vars == 150 and n_objs == 6:
            pop_size, run_time = 60000, 182
        elif n_vars == 150 and n_objs >= 7:
            pop_size, run_time = 10400, 313
    elif problem == "tsp":
        if n_vars == 15 and n_objs <= 3:
            pop_size, run_time = 900, 3
        elif n_vars == 15 and n_objs == 4:
            pop_size, run_time = 9300, 25
    else:
        raise ValueError(f"Unsupported problem '{problem}'")

    if pop_size is None or run_time is None:
        raise ValueError(
            f"Unsupported NSGA-II defaults for {problem}: {n_vars} vars, {n_objs} objs"
        )

    if "5x" in cutoff:
        run_time = min(run_time * 5, 600)
    return pop_size, run_time


def nsga2_run_params(cfg):
    """Return the NSGA-II pop_size/run_time represented by a config."""

    nsga2_cfg = getattr(cfg, "nsga2", None)
    cutoff = getattr(nsga2_cfg, "cutoff", "restrict")
    default_pop, default_time = nsga2_defaults(
        cfg.prob.name,
        int(cfg.prob.n_vars),
        int(cfg.prob.n_objs),
        cutoff,
    )
    pop_size = getattr(nsga2_cfg, "pop_size", None) or default_pop
    run_time = getattr(nsga2_cfg, "run_time", None) or default_time
    return pop_size, run_time


def nsga2_postprocess_run_params(cfg):
    """Return NSGA-II pop_size/run_time pairs to post-process."""

    nsga2_cfg = getattr(cfg, "nsga2", None)
    cutoff = getattr(nsga2_cfg, "cutoff", "restrict")
    default_pop, default_time = nsga2_defaults(
        cfg.prob.name,
        int(cfg.prob.n_vars),
        int(cfg.prob.n_objs),
        cutoff,
    )
    params = []
    for pop_size in (100, 500, default_pop):
        pair = (pop_size, default_time)
        if pair not in params:
            params.append(pair)
    return params


def nsga2_sols_dir_for_params(cfg, pop_size: int, run_time: int) -> Path:
    """Return the NSGA-II solution directory for explicit run parameters."""

    return (
        Paths.sols
        / cfg.prob.name
        / cfg.prob.size
        / cfg.split
        / "nsga2"
        / f"pop{pop_size}_time{run_time}"
    )


def nsga2_sols_dir(cfg) -> Path:
    """Return the solution directory used by NSGA-II runners for a config."""

    pop_size, run_time = nsga2_run_params(cfg)
    return nsga2_sols_dir_for_params(cfg, pop_size, run_time)


def nsga2_frontier_candidates_for_params(
    cfg,
    pid: int,
    run_seed: int,
    pop_size: int,
    run_time: int,
) -> list[Path]:
    """Return possible files for a saved NSGA-II frontier with explicit params."""

    return [nsga2_sols_dir_for_params(cfg, pop_size, run_time) / f"{pid}-{run_seed}.npy"]


def nsga2_frontier_candidates(cfg, pid: int, run_seed: int) -> list[Path]:
    """Return possible files for a saved NSGA-II frontier."""

    pop_size, run_time = nsga2_run_params(cfg)
    return nsga2_frontier_candidates_for_params(
        cfg,
        pid,
        run_seed,
        pop_size,
        run_time,
    )
