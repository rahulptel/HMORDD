import json

import hydra
import numpy as np
import pandas as pd
from hmordd import Paths
from hmordd.common.base_runner import BaseRunner
from hmordd.common.utils import append_pf_dom_path
from hmordd.tsp.dd import DDManagerFactory
from hmordd.tsp.utils import get_instance_data


class Runner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._set_memory_limit()

    def _get_save_path(self, save_type):
        if save_type == "sols":
            base_path = Paths.sols
        elif save_type == "dds":
            base_path = Paths.dds
        else:
            raise ValueError(f"Unknown save_type '{save_type}'")

        save_path = base_path / self.cfg.prob.name / self.cfg.prob.size
        save_path = save_path / self.cfg.split / self.cfg.dd.type
        save_path = append_pf_dom_path(
            save_path, self.cfg, include_dominance=False, include_track_x=True
        )
        nosh = getattr(self.cfg.dd, "nosh", None)
        if nosh:
            save_path = save_path / nosh
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def _stats_dict(
        self,
        pid,
        dd_manager,
        instance_data,
    ):
        return {
            "pid": [pid],
            "n_objectives": [instance_data.get("n_objs") if instance_data else None],
            "n_variables": [instance_data.get("n_vars") if instance_data else None],
            "inst_seed": [self.cfg.seed],            
            "build_time": [dd_manager.time_build],
            "frontier_time": [dd_manager.time_frontier],
            "total_time": [self._sum_times(dd_manager.time_build, dd_manager.time_frontier)]            
        }

    def _sum_times(self, build_time, frontier_time):
        if build_time is None or frontier_time is None:
            return None
        return build_time + frontier_time

    def _save_stats(
        self,
        pid,
        dd_manager,
        dds_path,
        sols_path,
        instance_data,
    ):
        stats = pd.DataFrame(
            self._stats_dict(
                pid,
                dd_manager,
                instance_data,
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

    def _save_frontier(self, pid, dd_manager, sols_path):
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

    def _maybe_save_dd(self, pid, dd_manager, dds_path):
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
        pid,
        dd_manager,
        instance_data,
    ):
        dds_path = self._get_save_path("dds")
        sols_path = self._get_save_path("sols")
        self._save_stats(
            pid,
            dd_manager,
            dds_path,
            sols_path,
            instance_data,
        )
        self._save_frontier(pid, dd_manager, sols_path)
        self._maybe_save_dd(pid, dd_manager, dds_path)

    def worker(self, rank):
        size = self.cfg.prob.size
        dd_manager = DDManagerFactory.create_dd_manager(self.cfg)
        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"Processing PID: {pid} on rank {rank}")
            inst = get_instance_data(size, self.cfg.split, self.cfg.seed, pid)
            dd_manager.reset(inst)
            dd_manager.build_dd()
            dd_manager.compute_frontier(self.cfg.prob.pf_enum_method, time_limit=self.cfg.time_limit)
            self.save(
                pid,
                dd_manager,
                inst,
            )

@hydra.main(config_path="./configs", config_name="run_dd.yaml", version_base="1.2")
def main(cfg):
    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
