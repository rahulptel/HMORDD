import json
from pprint import pprint
import signal

import hydra
import numpy as np
import pandas as pd
from hmordd import Paths
from hmordd.common.base_runner import BaseRunner
from hmordd.common.utils import MetricCalculator, append_pf_dom_path
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
        save_path = append_pf_dom_path(save_path, self.cfg, include_dominance=True)
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

    def _save_frontier_stats(self, pid, dd_manager, sols_save_path, cardinality_result, status):
        # Allow cardinality_result to be None (e.g., when build failed)
        if cardinality_result is None:
            cardinality_result = {
                "n_exact_pf": -1,
                "n_approx_pf": -1,
                "cardinality": -1,
                "precision": -1,
                "cardinality_raw": -1,
                "igd": None,
                "igd_raw": None,
            }

        stats = {
            "pid": [pid],
            "n_exact_pf": [cardinality_result.get("n_exact_pf")],
            "n_approx_pf": [cardinality_result.get("n_approx_pf")],
            "cardinality": [cardinality_result.get("cardinality")],
            "precision": [cardinality_result.get("precision")],
            "cardinality_raw": [cardinality_result.get("cardinality_raw")],
            "igd": [cardinality_result.get("igd")],
            "igd_raw": [cardinality_result.get("igd_raw")],
            "build_time": [dd_manager.time_build],
            "frontier_time": [dd_manager.time_frontier],
            "total_time": [self._sum_times(dd_manager.time_build, dd_manager.time_frontier)],
            "status": [status],
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

    def save(self, pid, dd_manager, cardinality_result, status="SUCCESS"):
        dds_save_path = self._get_save_path("dds")
        sols_save_path = self._get_save_path("sols")
                
        self._save_dd_stats(pid, dd_manager, dds_save_path)
        self._save_frontier(pid, dd_manager, sols_save_path)
        self._save_frontier_stats(pid, dd_manager, sols_save_path, cardinality_result, status)
        self._maybe_save_dd(pid, dd_manager, dds_save_path)

    def worker(self, rank):
        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"Processing PID: {pid} on rank {rank}")
            if self.cfg.dd.type == "restricted":
                # Only process if exact Pareto front is available
                exact_sol_path = Paths.sols / self.cfg.prob.name / self.cfg.prob.size
                exact_sol_path = exact_sol_path / self.cfg.split / "exact"
                exact_sol_path = append_pf_dom_path(exact_sol_path, self.cfg, include_dominance=True)
                exact_sol_path = exact_sol_path / f"{pid}.npy"
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

            # Enforce build time limit using the same configured time_limit
            build_timeout = self.cfg.time_limit
            status = "SUCCESS"
            try:
                signal.alarm(build_timeout)
                dd_manager.build_dd()
                signal.alarm(0)
            except MemoryError:
                # Build failed due to memory
                status = "BUILD_FAILED:MEMLIMIT"
                dd_manager.time_build = getattr(dd_manager, "time_build", build_timeout)
                dd_manager.frontier = None
                dd_manager.time_frontier = None
                # Compute cardinality with no approx frontier
                cardinality_result = self.metric_calculator.compute(true_pf=None, approx_pf=None)
                print(status)
                self.save(pid, dd_manager, cardinality_result, status=status)
                continue
            except Exception:
                # Timeout or other error during build
                status = "BUILD_FAILED:TIMELIMIT"
                dd_manager.time_build = getattr(dd_manager, "time_build", build_timeout)
                dd_manager.frontier = None
                dd_manager.time_frontier = None
                cardinality_result = self.metric_calculator.compute(true_pf=None, approx_pf=None)
                print(status)
                self.save(pid, dd_manager, cardinality_result, status=status)
                continue

            # Build succeeded; enumerate frontier with configured time_limit
            dd_manager.compute_frontier(self.cfg.prob.pf_enum_method, time_limit=self.cfg.time_limit)

            approx_pf = dd_manager.frontier
            if self.cfg.dd.type == "exact":
                exact_pf = approx_pf

            # Determine enumeration status: check dd_manager.frontier and possible error flag
            if approx_pf is None:
                fe = getattr(dd_manager, "frontier_error", None)
                if fe == "MEMLIMIT":
                    status = "ENUM_FAILED:MEMLIMIT"
                else:
                    status = "ENUM_FAILED:TIMELIMIT"

            cardinality_result = self.metric_calculator.compute(
                true_pf=exact_pf,
                approx_pf=approx_pf,
            )
            print(cardinality_result)
            print(status)
            self.save(pid, dd_manager, cardinality_result, status=status)

@hydra.main(config_path="./configs", config_name="run_dd.yaml", version_base="1.2")
def main(cfg):
    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
