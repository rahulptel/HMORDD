import json
import multiprocessing as mp
from pathlib import Path

import hydra
import numpy as np
import pandas as pd

from hmordd import Paths
from hmordd.common.utils import MetricCalculator
from hmordd.setpacking.dd import DDManagerFactory
from hmordd.setpacking.utils import get_instance_data


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.metric_calculator = MetricCalculator(cfg.prob.n_objs)

    def _get_save_path(self, save_type: str) -> Path:
        if save_type == "sols":
            base_path = Paths.sols
        elif save_type == "dds":
            base_path = Paths.dds
        else:
            raise ValueError(f"Unknown save_type '{save_type}'")

        save_path = base_path / self.cfg.prob.name / self.cfg.prob.size / self.cfg.split / self.cfg.dd_type
        if self.cfg.dd_type == "restricted":
            save_path = save_path / f"width-{self.cfg.dd.width}-nosh-{self.cfg.dd.nosh.rule}"
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def _save_dd_stats(self, pid: int, dd_manager, dds_save_path: Path) -> None:
        stats = {
            "pid": [pid],
            "total_nodes": [self._safe_get(dd_manager.env, "initial_node_count")],
            "total_incoming_arcs": [self._safe_get(dd_manager.env, "initial_arcs_count")],
            "width": [self._safe_get(dd_manager.env, "initial_width")],
            "build_time": [dd_manager.time_build],
            "frontier_time": [dd_manager.time_frontier],
            "total_time": [self._sum_times(dd_manager.time_build, dd_manager.time_frontier)],
        }
        df_stats = pd.DataFrame(stats)
        try:
            df_stats.to_csv(dds_save_path / f"{pid}.csv", index=False)
        except Exception as exc:
            print(f"Error saving DD statistics for PID {pid}: {exc}")

    def _sum_times(self, build_time, frontier_time):
        if build_time is None or frontier_time is None:
            return None
        return build_time + frontier_time

    def _safe_get(self, obj, attr):
        try:
            return getattr(obj, attr)
        except Exception:
            return None

    def _save_frontier_stats(self, pid: int, dd_manager, sols_save_path: Path) -> None:
        stats = {
            "pid": [pid],
            "build_time": [dd_manager.time_build],
            "frontier_time": [dd_manager.time_frontier],
            "total_time": [self._sum_times(dd_manager.time_build, dd_manager.time_frontier)],
        }
        df_stats = pd.DataFrame(stats)
        try:
            df_stats.to_csv(sols_save_path / f"{pid}.csv", index=False)
        except Exception as exc:
            print(f"Error saving frontier statistics for PID {pid}: {exc}")

    def _save_frontier(self, pid: int, dd_manager, sols_save_path: Path) -> None:
        frontier = dd_manager.frontier
        if frontier is None:
            return
        try:
            if isinstance(frontier, dict):
                np.savez(sols_save_path / f"{pid}.npz", **{k: np.asarray(v) for k, v in frontier.items()})
            else:
                np.save(sols_save_path / f"{pid}.npy", np.asarray(frontier))
        except Exception as exc:
            print(f"Error saving frontier for PID {pid}: {exc}")

    def _maybe_save_dd(self, pid: int, dd_manager, dds_save_path: Path) -> None:
        if not getattr(self.cfg, "save_dd", False):
            return
        if not hasattr(dd_manager, "get_decision_diagram"):
            return
        try:
            dd_structure = dd_manager.get_decision_diagram()
        except Exception as exc:
            print(f"Failed to fetch DD for PID {pid}: {exc}")
            return
        if dd_structure is None:
            return
        try:
            with open(dds_save_path / f"{pid}.json", "w") as fp:
                json.dump(dd_structure, fp)
        except Exception as exc:
            print(f"Error saving DD for PID {pid}: {exc}")

    def save(self, pid: int, dd_manager) -> None:
        dds_save_path = self._get_save_path("dds")
        sols_save_path = self._get_save_path("sols")

        self._save_dd_stats(pid, dd_manager, dds_save_path)
        self._save_frontier_stats(pid, dd_manager, sols_save_path)
        self._save_frontier(pid, dd_manager, sols_save_path)
        self._maybe_save_dd(pid, dd_manager, dds_save_path)

    def worker(self, rank: int) -> None:
        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"Processing PID: {pid} on rank {rank}")
            data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.split, self.cfg.seed, pid)

            dd_manager = DDManagerFactory.create_dd_manager(self.cfg)
            dd_manager.reset(data)
            dd_manager.build_dd()
            dd_manager.compute_frontier(self.cfg.prob.pf_enum_method, time_limit=self.cfg.time_limit)
            self.save(pid, dd_manager)

    def run(self) -> None:
        if self.cfg.n_processes == 1:
            self.worker(0)
            return

        pool = mp.Pool(processes=self.cfg.n_processes)
        results = [pool.apply_async(self.worker, args=(rank,)) for rank in range(self.cfg.n_processes)]
        for result in results:
            result.get()
        pool.close()
        pool.join()


@hydra.main(config_path="./configs", config_name="run_dd.yaml", version_base="1.2")
def main(cfg):
    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
