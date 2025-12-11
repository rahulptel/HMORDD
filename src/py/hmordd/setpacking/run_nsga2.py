import time
from pprint import pprint

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmordd import Paths
from hmordd.common.base_runner import BaseRunner
from hmordd.common.utils import MetricCalculator
from hmordd.setpacking.utils import get_instance_data
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class SetPackingProblem(Problem):
    def __init__(self, instance_data):
        self.n_vars = instance_data["n_vars"]
        self.n_objs = instance_data["n_objs"]
        self.n_cons = instance_data["n_cons"]

        self.obj_coeffs = np.array(instance_data["obj_coeffs"], dtype=np.float64)

        self.A = np.zeros((self.n_cons, self.n_vars), dtype=np.float64)
        for idx, cons in enumerate(instance_data["cons_coeffs"]):
            cons_indices = np.asarray(cons, dtype=np.int64)
            # Assume the indices are 0-based
            self.A[idx, cons_indices] = 1

        super().__init__(
            n_var=self.n_vars,
            n_obj=self.n_objs,
            n_ieq_constr=self.n_cons,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = -np.dot(X, self.obj_coeffs.T)
        out["G"] = np.dot(X, self.A.T) - 1


class Runner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.metric_calculator = MetricCalculator(cfg.prob.n_objs)

    def _get_save_path(self, save_type="sols"):
        if save_type == "sols":
            base_path = Paths.sols
        else:
            raise ValueError("Invalid save_type. Must be 'sols'.")

        save_path = (
            base_path
            / self.cfg.prob.name
            / self.cfg.prob.size
            / self.cfg.split
            / "nsga2"
        )
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def _compute_defaults(self, n_vars, n_objs, cutoff):
        if cutoff not in ["restrict", "5xrestrict"]:
            raise ValueError(f"Unsupported cutoff: {cutoff}")
        
        pop_size, run_time = None, None        
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
        else:
            raise ValueError(f"Unsupported n_vars: {n_vars}, n_objs: {n_objs}")

        if "5x" in cutoff:
            run_time = min(5 * run_time, 600)
            
        return pop_size, run_time

    def _run_nsga2(self, instance_data, pid, pop_size, run_time, run_seed):
        problem = SetPackingProblem(instance_data)
        mutation_kwargs = {}
        if getattr(self.cfg.nsga2, "mutation_prob", None) is not None:
            mutation_kwargs["prob"] = self.cfg.nsga2.mutation_prob
        if getattr(self.cfg.nsga2, "mutation_prob_var", None) is not None:
            mutation_kwargs["prob_var"] = self.cfg.nsga2.mutation_prob_var
        mutation = BitflipMutation(**mutation_kwargs)
        
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(prob=self.cfg.nsga2.crossover_prob),
            mutation=mutation,
            eliminate_duplicates=True,   
        )

        termination = get_termination("time", run_time)

        res = minimize(
            problem,
            algorithm,
            termination,
            seed=int(run_seed),
            verbose=False,
            save_history=False,
        )

        if res.F is None or not len(res.F):
            return None

        return -np.array(res.F, dtype=np.int64)
    
    def _save_frontier(self, pid, run_seed, pareto_front, sols_save_path):
        try:
            if pareto_front is not None and pareto_front.size:
                np.save(sols_save_path / f"{pid}-{run_seed}.npy", pareto_front)
        except Exception as e:
            print(f"Error saving frontier for PID {pid}: {e}")

    def _save_stats(
        self,
        pid,
        cardinality_result,
        time_taken,
        pareto_front_size,
        sols_save_path,
        instance_data,
        pop_size_used,
        run_time,
        cutoff,
        run_seed,
        inst_seed,
    ):
        stats_data = {
            "pid": [pid],
            "cardinality": [cardinality_result['cardinality']],
            "precision": [cardinality_result['precision']],
            "cardinality_raw": [cardinality_result['cardinality_raw']],
            "pop_size": [pop_size_used],
            "crossover_prob": [self.cfg.nsga2.crossover_prob],
            "mutation_prob": [self.cfg.nsga2.mutation_prob],
            "mutation_prob_var": [self.cfg.nsga2.mutation_prob_var],
            "n_objectives": [instance_data["n_objs"]],
            "n_variables": [instance_data["n_vars"]],
            "n_constraints": [instance_data["n_cons"]],
            "time_taken": [time_taken],
            "pareto_front_size": [pareto_front_size],
            "run_time": [run_time],
            "cutoff": [cutoff],
            "run_seed": [run_seed],
            "inst_seed": [inst_seed],
        }
        df_stats = pd.DataFrame(stats_data)
        try:
            df_stats.to_csv(sols_save_path / f"{pid}-{run_seed}.csv", index=False)
        except Exception as e:
            print(f"Error saving statistics for PID {pid}: {e}")

    def worker(self, rank):
        self._set_memory_limit()
        sols_save_path = self._get_save_path("sols")

        for pid in range(
            self.cfg.from_pid + rank,
            self.cfg.to_pid,
            self.cfg.n_processes,
        ):
            print(f"Processing PID: {pid} on rank {rank}")

            instance_data = get_instance_data(
                self.cfg.prob.name,
                self.cfg.prob.size,
                self.cfg.split,
                self.cfg.inst_seed,
                pid,
            )

            # Compute defaults based on instance size and config cutoff
            cutoff = getattr(self.cfg.nsga2, "cutoff", "restrict")
            default_pop, default_time = self._compute_defaults(
                instance_data["n_vars"], instance_data["n_objs"], cutoff
            )

            pop_size = (
                default_pop
                if getattr(self.cfg.nsga2, "pop_size", None) is None
                else self.cfg.nsga2.pop_size
            )
            run_time = (
                default_time
                if getattr(self.cfg.nsga2, "run_time", None) is None
                else self.cfg.nsga2.run_time
            )
            
            exact_sol_path = Paths.sols / self.cfg.prob.name / self.cfg.prob.size 
            exact_sol_path = exact_sol_path / self.cfg.split / "exact" / f"{pid}.npy"
            exact_pf = None
            if exact_sol_path.exists():
                try:
                    exact_pf = np.load(exact_sol_path).astype(np.int64)
                except Exception as e:
                    print(f"Error loading exact Pareto front for PID {pid}: {e}")
            else:
                print(f"Exact Pareto front not found for PID {pid} at {exact_sol_path}")                
            if exact_pf is None:
                print(f"Skipping PID {pid} as exact Pareto front is not available.")
                continue
            
            sols_save_path_run = sols_save_path / f"pop{pop_size}_time{run_time}"
            sols_save_path_run.mkdir(parents=True, exist_ok=True)
            for run_seed in getattr(self.cfg, "trial_seeds", [1, 2, 3, 4, 5]):
                start_time = time.time()
                approx_pf = self._run_nsga2(instance_data, pid, pop_size, run_time, run_seed)
                time_taken = time.time() - start_time
                
                cardinality_result = {'cardinality': -10, 'cardinality_raw': -10, 'precision': -10}
                n_approx_pf = 0
                if approx_pf is not None:                    
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(exact_pf[:, 0], exact_pf[:, 1], exact_pf[:, 2], label='True PF', alpha=0.6)
                    # ax.scatter(approx_pf[:, 0], approx_pf[:, 1], approx_pf[:, 2], label='Approx PF', alpha=0.6)
                    # plt.title('Pareto Front Comparison')
                    # plt.legend()
                    # plt.show()                  
                    print(f"Pid: {pid}, Seed: {run_seed}, PF size: {approx_pf.shape}")
                    n_approx_pf = approx_pf.shape[0]
                    cardinality_result = self.metric_calculator.compute_cardinality(
                        true_pf=exact_pf,
                        approx_pf=approx_pf,
                    )
                    pprint(cardinality_result)
                    
                else:
                    print(f"Did not find feasible solution: {pid} (seed={run_seed})")
                    
                
                self._save_frontier(pid, run_seed, approx_pf, sols_save_path_run)
                self._save_stats(
                    pid,
                    cardinality_result,
                    time_taken,
                    n_approx_pf,
                    sols_save_path_run,
                    instance_data,
                    pop_size,
                    run_time,
                    cutoff,
                    run_seed,
                    self.cfg.inst_seed,
                )

@hydra.main(config_path="./configs", config_name="run_nsga2.yaml", version_base="1.2")
def main(cfg):
    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
