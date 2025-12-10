import json
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
from hmordd import Paths
from hmordd.common.base_runner import BaseRunner
from hmordd.common.utils import MetricCalculator
from hmordd.setpacking.dd import DDManagerFactory
from hmordd.setpacking.utils import get_instance_data


class Runner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.metric_calculator = MetricCalculator(cfg.prob.n_objs)

        self._set_memory_limit()

    def _get_save_path(self, save_type):
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

    def _save_dd_stats(self, pid, dd_manager, dds_save_path):
        stats = {
            "pid": [pid],
            "total_nodes": [self._get_env_stat(dd_manager.env, "get_total_nodes_count")],
            "total_incoming_arcs": [self._get_env_stat(dd_manager.env, "get_total_incoming_arcs_count")],
            "width": [self._get_env_stat(dd_manager.env, "get_width")],
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

    def _get_env_stat(self, env, method_name):
        try:
            getter = getattr(env, method_name)
        except AttributeError:
            return -1
        try:
            return getter()
        except Exception:
            return -2

    def _save_frontier_stats(self, pid, dd_manager, sols_save_path, exact_pf):
        cardinality_result = self.metric_calculator.compute_cardinality(
            true_pf=exact_pf,
            approx_pf=dd_manager.frontier
        )
        pprint(cardinality_result)
        
        stats = {
            "pid": [pid],
            "n_exact_pf": [cardinality_result['n_exact_pf']],
            "n_approx_pf": [cardinality_result['n_approx_pf']],
            "cardinality": [cardinality_result['cardinality']],
            "precision": [cardinality_result['precision']],
            "cardinality_raw": [cardinality_result['cardinality_raw']],
            "build_time": [dd_manager.time_build],
            "frontier_time": [dd_manager.time_frontier],
            "total_time": [self._sum_times(dd_manager.time_build, dd_manager.time_frontier)],
        }
        df_stats = pd.DataFrame(stats)
        try:
            df_stats.to_csv(sols_save_path / f"{pid}.csv", index=False)
        except Exception as exc:
            print(f"Error saving frontier statistics for PID {pid}: {exc}")

    def _save_frontier(self, pid, dd_manager, sols_save_path):
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

    def _maybe_save_dd(self, pid, dd_manager, dds_save_path):
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

    def save(self, pid, dd_manager, exact_pf):
        dds_save_path = self._get_save_path("dds")
        sols_save_path = self._get_save_path("sols")

        self._save_dd_stats(pid, dd_manager, dds_save_path)
        self._save_frontier(pid, dd_manager, sols_save_path)
        self._save_frontier_stats(pid, dd_manager, sols_save_path, exact_pf)
        self._maybe_save_dd(pid, dd_manager, dds_save_path)

    def worker(self, rank):
        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"Processing PID: {pid} on rank {rank}")
            if self.cfg.dd.type == "restricted":
                # Only process if exact Pareto front is available
                exact_sol_path = Paths.sols / self.cfg.prob.name / self.cfg.prob.size 
                exact_sol_path = exact_sol_path / self.cfg.split / "exact" / f"{pid}.npy"
                exact_pf = None
                try:
                    exact_pf = np.load(exact_sol_path)
                except Exception as e:
                    print(f"Error loading exact Pareto front for PID {pid}: {e}")
                if exact_pf is None:
                    continue
                
            inst = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.split, self.cfg.seed, pid)

            dd_manager = DDManagerFactory.create_dd_manager(self.cfg)
            dd_manager.reset(inst)
            dd_manager.build_dd()
            dd_manager.compute_frontier(self.cfg.prob.pf_enum_method, time_limit=self.cfg.time_limit)
            self.save(pid, dd_manager, exact_pf)

@hydra.main(config_path="./configs", config_name="run_dd.yaml", version_base="1.2")
def main(cfg):
    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
