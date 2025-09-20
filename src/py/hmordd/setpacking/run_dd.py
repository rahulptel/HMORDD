import json
import multiprocessing as mp

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
        self.metric_calculator = MetricCalculator(cfg.prob.n_objs) # Use n_objs for MetricCalculator
        self.pred_pf = None

    def _get_save_path(self, save_type="sols"):
        if save_type == "sols":
            base_path = Paths.sols
        elif save_type == "dds":
            base_path = Paths.dds
        else:
            raise ValueError("Invalid save_type. Must be 'sols' or 'dds'.")

        save_path = base_path / self.cfg.prob.name / self.cfg.prob.size / self.cfg.split / self.cfg.dd_type
        if self.cfg.dd_type == "restricted":
            save_path = save_path / f"width-{self.cfg.dd.width}-nosh-{self.cfg.dd.nosh.rule}"
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path
    
    def _save_dd_stats(self, pid, dd_manager, dds_save_path):
        total_nodes = -1
        total_incoming_arcs = -1
        width = -1
        try:
            total_nodes = dd_manager.env.get_total_nodes_count()
        except Exception as e:
            print(f"Error getting total nodes for PID {pid}: {e}")
        try:
            total_incoming_arcs = dd_manager.env.get_total_incoming_arcs_count()
        except Exception as e:
            print(f"Error getting total incoming arcs for PID {pid}: {e}")
        try:
            width = dd_manager.env.get_width()
        except Exception as e:
            print(f"Error getting width for PID {pid}: {e}")

        total_time = dd_manager.time_build + dd_manager.time_frontier

        stats_data = {
            "pid": [pid],
            "total_nodes": [total_nodes],
            "total_incoming_arcs": [total_incoming_arcs],
            "width": [width],
            "build_time": [dd_manager.time_build],
            "frontier_time": [dd_manager.time_frontier],
            "total_time": [total_time],
        }
        df_stats = pd.DataFrame(stats_data)

        try:
            df_stats.to_csv(dds_save_path / f"{pid}.csv", index=False)
        except Exception as e:
            print(f"Error saving BDD statistics for PID {pid}: {e}")

    def _save_frontier_stats(self, pid, dd_manager, sols_save_path):
        build_time, pareto_time, total_time = -1, -1, -1
        try:
            build_time = dd_manager.time_build
        except Exception as e:
            print(f"Error getting build time for PID {pid}: {e}")
        try:
            pareto_time = dd_manager.time_frontier
        except Exception as e:
            print(f"Error getting pareto time for PID {pid}: {e}")
        if build_time != -1 and pareto_time != -1:
            total_time = build_time + pareto_time
            
        stats_data = {
            "pid": [pid],
            "build_time": [dd_manager.time_build],
            "frontier_time": [dd_manager.time_frontier],
            "total_time": [total_time],
        }
        df_stats = pd.DataFrame(stats_data)
        
        try:
            df_stats.to_csv(sols_save_path / f"{pid}.csv", index=False)
        except Exception as e:
            print(f"Error saving frontier statistics for PID {pid}: {e}")

        
    def _save_frontier(self, pid, dd_manager, sols_save_path):
        try:
            if dd_manager.frontier is not None:
                np.save(sols_save_path / f"{pid}.npy", dd_manager.frontier)
        except Exception as e:
            print(f"Error saving frontier for PID {pid}: {e}")

    
    def save(self, pid, dd_manager):
        dds_save_path = self._get_save_path("dds")
        sols_save_path = self._get_save_path("sols")
        
        self._save_dd_stats(pid, dd_manager, dds_save_path)
        self._save_frontier_stats(pid, dd_manager, sols_save_path)
        self._save_frontier(pid, dd_manager, sols_save_path)

    def worker(self, rank):
        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"Processing PID: {pid} on rank {rank}")

            # Load instance data
            data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.split, self.cfg.seed, pid)

            # Instantiate DDManager based on config
            dd_manager = DDManagerFactory.create_dd_manager(self.cfg)
            assert dd_manager is not None, "DD Manager instantiation failed."
            
            dd_manager.reset(data) # Pass instance data and order
            dd_manager.build_dd()
            dd_manager.compute_frontier(self.cfg.prob.pf_enum_method, time_limit=self.cfg.time_limit)
            if dd_manager.frontier is not None:
                print(f"Pid: {pid}, PF size: {dd_manager.frontier.shape}")
                
            self.save(pid, dd_manager)


    def run(self):
        if self.cfg.n_processes == 1:
            self.worker(0)
        else:
            pool = mp.Pool(processes=self.cfg.n_processes)
            results = []

            for rank in range(self.cfg.n_processes):
                results.append(pool.apply_async(self.worker, args=(rank,)))

            for r in results:
                r.get()

        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"Processing PID: {pid} on rank {rank}")

            # Load instance data
            data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.split, self.cfg.seed, pid)

            # Instantiate DDManager based on config
            dd_manager = DDManagerFactory.create_dd_manager(self.cfg)
            assert dd_manager is not None, "DD Manager instantiation failed."
            
            dd_manager.reset(data) # Pass instance data and order
            dd_manager.build_dd()
            dd_manager.compute_frontier(self.cfg.prob.pf_enum_method, time_limit=self.cfg.time_limit)
            if dd_manager.frontier is not None:
                print(f"Pid: {pid}, PF size: {dd_manager.frontier.shape}")
                
            self.save(pid, dd_manager)


    def run(self):
        if self.cfg.n_processes == 1:
            self.worker(0)
        else:
            pool = mp.Pool(processes=self.cfg.n_processes)
            results = []

            for rank in range(self.cfg.n_processes):
                results.append(pool.apply_async(self.worker, args=(rank,)))

            for r in results:
                r.get()


@hydra.main(config_path="./configs", config_name="run_dd.yaml", version_base="1.2")
def main(cfg):
    runner = Runner(cfg)
    runner.run()
    


if __name__ == '__main__':
    main()