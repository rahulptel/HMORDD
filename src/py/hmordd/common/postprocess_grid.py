"""Helpers for running problem-specific post-processing experiment grids."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import Iterable

from omegaconf import OmegaConf

from hmordd.common.postprocess import PostProcessor


@dataclass(frozen=True)
class ExperimentSpec:
    n_objs: int
    n_vars: int
    restricted_variants: tuple[dict, ...]
    nsga2_variants: tuple[dict, ...]

    @property
    def size(self):
        return f"{self.n_objs}_{self.n_vars}"


def restricted_variant(key, **values):
    return {"key": key, **values}


def nsga2_variant(key, pop_size: int, run_time: int):
    return {"key": key, "pop_size": pop_size, "run_time": run_time}


def _prob_size_overrides(argv: Iterable[str]) -> set[str]:
    keys = {"prob.size", "prob.n_objs", "prob.n_vars"}
    return {
        arg.split("=", 1)[0]
        for arg in argv
        if "=" in arg and arg.split("=", 1)[0] in keys
    }


def _matches_requested_size(cfg, spec: ExperimentSpec, overrides: set[str]) -> bool:
    if "prob.size" in overrides:
        requested_size = str(getattr(cfg.prob, "size", spec.size))
        if requested_size not in {spec.size, spec.size.replace("_", "")}:
            return False
    if "prob.n_objs" in overrides and int(getattr(cfg.prob, "n_objs", spec.n_objs)) != spec.n_objs:
        return False
    if "prob.n_vars" in overrides and int(getattr(cfg.prob, "n_vars", spec.n_vars)) != spec.n_vars:
        return False
    return True


def _configured_specs(cfg, specs: tuple[ExperimentSpec, ...]) -> tuple[ExperimentSpec, ...]:
    overrides = _prob_size_overrides(sys.argv[1:])
    if overrides:
        matched = tuple(spec for spec in specs if _matches_requested_size(cfg, spec, overrides))
        if matched:
            return matched
        return (
            ExperimentSpec(
                n_objs=int(cfg.prob.n_objs),
                n_vars=int(cfg.prob.n_vars),
                restricted_variants=(),
                nsga2_variants=(),
            ),
        )
    return specs


def _concrete_cfg(cfg, spec: ExperimentSpec):
    concrete = copy.deepcopy(cfg)
    OmegaConf.set_struct(concrete, False)
    OmegaConf.set_struct(concrete.prob, False)
    OmegaConf.set_struct(concrete.postprocess, False)
    concrete.prob.n_objs = spec.n_objs
    concrete.prob.n_vars = spec.n_vars
    concrete.prob.size = spec.size
    concrete.postprocess.restricted_dd_variants = list(spec.restricted_variants)
    concrete.postprocess.nsga2_variants = list(spec.nsga2_variants)
    return concrete


def run_postprocess_grid(cfg, specs: tuple[ExperimentSpec, ...]):
    for spec in _configured_specs(cfg, specs):
        print(f"Post-processing {cfg.prob.name} size {spec.size}")
        runner = PostProcessor(_concrete_cfg(cfg, spec))
        runner.run()
