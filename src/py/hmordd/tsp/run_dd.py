import json
import multiprocessing as mp
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hmordd import Paths
from hmordd.common.base_runner import BaseRunner
from hmordd.common.utils import MetricCalculator
from hmordd.tsp.dd import DDManagerFactory
from hmordd.tsp.utils import get_instance_data


class Runner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.metric_calculator = MetricCalculator(cfg.prob.n_objs)
        self._set_memory_limit()

    def _get_save_path(self, save_type: str) -> Path:
        if save_type == "sols":
            base_path = Paths.sols
        elif save_type == "dds":
            base_path = Paths.dds
        else:
            raise ValueError(f"Unknown save_type '{save_type}'")

        save_path = base_path / self.cfg.prob.name / self.cfg.prob.size 
        save_path = save_path / self.cfg.split / self.cfg.dd.type
        nosh = getattr(self.cfg.dd, "nosh", None)
        if nosh:
            save_path = save_path / nosh
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def _load_exact_pf(self, pid: int):
        exact_sol_path = Paths.sols / self.cfg.prob.name / self.cfg.prob.size
        exact_sol_path = exact_sol_path / self.cfg.split / "exact" / f"{pid}.npz"
        if not exact_sol_path.exists():
            print(f"Exact Pareto front not found for PID {pid} at {exact_sol_path}")
            return None
        try:
            return np.load(exact_sol_path)["z"]
        except Exception as exc:
            print(f"Error loading exact Pareto front for PID {pid}: {exc}")
            return None

    def _extract_frontier_array(self, frontier):
        if frontier is None:
            return None
        if isinstance(frontier, dict):
            return frontier.get("z")
        return frontier

    def _stats_dict(
        self,
        pid: int,
        dd_manager,
        cardinality_result: dict,
        frontier_size: int,
        instance_data: dict,
        exact_pf_size: int,
    ) -> dict:
        return {
            "pid": [pid],
            "cardinality": [cardinality_result.get("cardinality")],
            "precision": [cardinality_result.get("precision")],
            "cardinality_raw": [cardinality_result.get("cardinality_raw")],
            "n_objectives": [instance_data.get("n_objs") if instance_data else None],
            "n_variables": [instance_data.get("n_vars") if instance_data else None],
            "inst_seed": [self.cfg.seed],
            "exact_pf_size": [exact_pf_size],
            "build_time": [dd_manager.time_build],
            "frontier_time": [dd_manager.time_frontier],
            "total_time": [self._sum_times(dd_manager.time_build, dd_manager.time_frontier)],
            "pareto_points": [frontier_size],
        }

    def _sum_times(self, build_time, frontier_time):
        if build_time is None or frontier_time is None:
            return None
        return build_time + frontier_time

    def _frontier_size(self, frontier):
        if frontier is None:
            return None
        if isinstance(frontier, dict) and "z" in frontier:
            return len(frontier["z"])
        return len(frontier)

    def _save_stats(
        self,
        pid: int,
        dd_manager,
        dds_path: Path,
        sols_path: Path,
        cardinality_result: dict,
        frontier_size: int,
        instance_data: dict,
        exact_pf_size: int,
    ) -> None:
        stats = pd.DataFrame(
            self._stats_dict(
                pid,
                dd_manager,
                cardinality_result,
                frontier_size,
                instance_data,
                exact_pf_size,
            )
        )
        try:
            stats.to_csv(dds_path / f"{pid}.csv", index=False)
        except Exception as exc:
            print(f"Error saving DD stats for PID {pid}: {exc}")
        try:
            stats.to_csv(sols_path / f"{pid}.csv", index=False)
        except Exception as exc:
            print(f"Error saving frontier stats for PID {pid}: {exc}")

    def _save_frontier(self, pid: int, dd_manager, sols_path: Path) -> None:
        frontier = dd_manager.frontier
        if frontier is None:
            return
        try:
            np.savez(
                sols_path / f"{pid}.npz",
                **{k: np.asarray(v) for k, v in frontier.items() if v is not None},
            )
        except Exception as exc:
            print(f"Error saving frontier for PID {pid}: {exc}")

    def _maybe_save_dd(self, pid: int, dd_manager, dds_path: Path) -> None:
        if not getattr(self.cfg, "save_dd", False):
            return
        try:
            dd_structure = dd_manager.get_decision_diagram()
        except Exception as exc:
            print(f"Failed to fetch DD for PID {pid}: {exc}")
            return
        if dd_structure is None:
            return
        try:
            with open(dds_path / f"{pid}.json", "w") as fp:
                json.dump(dd_structure, fp)
        except Exception as exc:
            print(f"Error saving DD for PID {pid}: {exc}")

    def save(
        self,
        pid: int,
        dd_manager,
        cardinality_result: dict,
        frontier_size: int,
        instance_data: dict,
        exact_pf_size: int,
    ) -> None:
        dds_path = self._get_save_path("dds")
        sols_path = self._get_save_path("sols")
        self._save_stats(
            pid,
            dd_manager,
            dds_path,
            sols_path,
            cardinality_result,
            frontier_size,
            instance_data,
            exact_pf_size,
        )
        self._save_frontier(pid, dd_manager, sols_path)
        self._maybe_save_dd(pid, dd_manager, dds_path)

    def worker(self, rank: int) -> None:
        size = self.cfg.prob.size
        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"Processing PID: {pid} on rank {rank}")
            inst = get_instance_data(size, self.cfg.split, self.cfg.seed, pid)

            dd_manager = DDManagerFactory.create_dd_manager(self.cfg)
            dd_manager.reset(inst)
            dd_manager.build_dd()
            dd_manager.compute_frontier(self.cfg.prob.pf_enum_method, time_limit=self.cfg.time_limit)
            approx_pf = self._extract_frontier_array(dd_manager.frontier)
            frontier_size = approx_pf.shape[0] if approx_pf is not None else 0
            exact_pf = self._load_exact_pf(pid)
            exact_pf_size = len(exact_pf) if exact_pf is not None else None
            cardinality_result = self.metric_calculator.compute_cardinality(
                true_pf=exact_pf,
                approx_pf=approx_pf,
            )
            print(cardinality_result)
            self.save(
                pid,
                dd_manager,
                cardinality_result,
                frontier_size,
                inst,
                exact_pf_size,
            )

@hydra.main(config_path="./configs", config_name="run_dd.yaml", version_base="1.2")
def main(cfg):
    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
