"""Post-process saved frontiers into metric CSVs."""

from __future__ import annotations

import pandas as pd
from hmordd import Paths
from hmordd.common.base_runner import BaseRunner
from hmordd.common.frontiers import (
    dd_frontier_candidates,
    dd_sols_dir,
    load_first_existing_frontier,
    nsga2_frontier_candidates_for_params,
    nsga2_postprocess_run_params,
    nsga2_sols_dir_for_params,
    restricted_dd_frontier_candidates_for_variant,
    restricted_dd_sols_dir_for_variant,
)
from hmordd.common.utils import MetricCalculator


class PostProcessor(BaseRunner):
    """Compute metrics from saved exact, restricted DD, and NSGA-II frontiers."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.metric_calculator = MetricCalculator(cfg.prob.n_objs)

    def _metrics_dir_from_sols_dir(self, sols_dir):
        try:
            relative = sols_dir.relative_to(Paths.sols)
        except ValueError:
            relative = sols_dir.name
        metrics_dir = Paths.outputs / "metrics" / relative
        metrics_dir.mkdir(parents=True, exist_ok=True)
        return metrics_dir

    def _save_row(self, row, metrics_dir, filename):
        path = metrics_dir / filename
        if path.exists() and not getattr(self.cfg, "overwrite", True):
            print(f"Skipping existing metrics file: {path}")
            return
        pd.DataFrame([row]).to_csv(path, index=False)

    def _load_exact_frontier(self, pid):
        return load_first_existing_frontier(
            dd_frontier_candidates(self.cfg, pid, dd_type="exact")
        )

    def _base_row(self, pid, exact_pf, exact_path):
        return {
            "pid": pid,
            "problem": self.cfg.prob.name,
            "size": self.cfg.prob.size,
            "split": self.cfg.split,
            "n_objectives": self.cfg.prob.n_objs,
            "n_variables": getattr(self.cfg.prob, "n_vars", None),
            "inst_seed": getattr(self.cfg, "seed", None),
            "exact_found": exact_pf is not None,
            "exact_path": str(exact_path) if exact_path else None,
        }

    def _metric_row(self, pid, exact_pf, exact_path, approx_pf, approx_path, method, **extra):
        status = "ok"
        if approx_pf is None:
            status = "missing_approx"
        elif exact_pf is None:
            status = "missing_exact"

        metrics = self.metric_calculator.compute(true_pf=exact_pf, approx_pf=approx_pf)
        return {
            **self._base_row(pid, exact_pf, exact_path),
            "method": method,
            "status": status,
            "approx_found": approx_pf is not None,
            "approx_path": str(approx_path) if approx_path else None,
            **extra,
            **metrics,
        }

    def _process_restricted_dd(self, pid, exact_pf, exact_path):
        variants = getattr(self.cfg.postprocess, "restricted_dd_variants", None)
        if variants:
            for variant in variants:
                approx_pf, approx_path = load_first_existing_frontier(
                    restricted_dd_frontier_candidates_for_variant(self.cfg, pid, variant)
                )
                row = self._metric_row(
                    pid,
                    exact_pf,
                    exact_path,
                    approx_pf,
                    approx_path,
                    method="dd",
                    dd_type="restricted",
                    run_seed=None,
                    restricted_key=getattr(variant, "key", None),
                    nosh=getattr(variant, "nosh", None),
                    width=getattr(variant, "width", None),
                    nosh_rule=getattr(variant, "nosh_rule", None),
                )
                metrics_dir = self._metrics_dir_from_sols_dir(
                    restricted_dd_sols_dir_for_variant(self.cfg, variant)
                )
                self._save_row(row, metrics_dir, f"{pid}.csv")
            return

        approx_pf, approx_path = load_first_existing_frontier(
            dd_frontier_candidates(self.cfg, pid, dd_type="restricted")
        )
        row = self._metric_row(
            pid,
            exact_pf,
            exact_path,
            approx_pf,
            approx_path,
            method="dd",
            dd_type="restricted",
            run_seed=None,
        )
        metrics_dir = self._metrics_dir_from_sols_dir(dd_sols_dir(self.cfg, "restricted"))
        self._save_row(row, metrics_dir, f"{pid}.csv")

    def _process_nsga2(self, pid, exact_pf, exact_path):
        variants = getattr(self.cfg.postprocess, "nsga2_variants", None)
        if variants:
            run_params = [
                (
                    getattr(variant, "key", None),
                    getattr(variant, "pop_size"),
                    getattr(variant, "run_time"),
                )
                for variant in variants
            ]
        else:
            run_params = [
                (None, pop_size, run_time)
                for pop_size, run_time in nsga2_postprocess_run_params(self.cfg)
            ]

        for nsga2_key, pop_size, run_time in run_params:
            metrics_dir = self._metrics_dir_from_sols_dir(
                nsga2_sols_dir_for_params(self.cfg, pop_size, run_time)
            )
            for run_seed in getattr(self.cfg, "trial_seeds", []):
                approx_pf, approx_path = load_first_existing_frontier(
                    nsga2_frontier_candidates_for_params(
                        self.cfg,
                        pid,
                        run_seed,
                        pop_size,
                        run_time,
                    )
                )
                row = self._metric_row(
                    pid,
                    exact_pf,
                    exact_path,
                    approx_pf,
                    approx_path,
                    method="nsga2",
                    dd_type=None,
                    run_seed=run_seed,
                    nsga2_key=nsga2_key,
                    pop_size=pop_size,
                    run_time=run_time,
                )
                self._save_row(row, metrics_dir, f"{pid}-{run_seed}.csv")

    def worker(self, rank):
        include_restricted_dd = getattr(self.cfg.postprocess, "restricted_dd", True)
        include_nsga2 = getattr(self.cfg.postprocess, "nsga2", True)

        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"Post-processing PID: {pid} on rank {rank}")
            try:
                exact_pf, exact_path = self._load_exact_frontier(pid)
                if include_restricted_dd:
                    self._process_restricted_dd(pid, exact_pf, exact_path)
                if include_nsga2:
                    self._process_nsga2(pid, exact_pf, exact_path)
            except Exception as exc:
                print(f"Error post-processing PID {pid}: {exc}")
