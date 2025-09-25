import multiprocessing as mp
import time

import hydra
import numpy as np
import pandas as pd
from hmordd import Paths
from hmordd.setpacking.utils import get_instance_data
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

try:
    import resource
except ImportError:  # pragma: no cover - Windows compatibility
    resource = None


class SetPackingProblem(Problem):
    def __init__(self, instance_data):
        self.n_vars = instance_data["n_vars"]
        self.n_objs = instance_data["n_objs"]
        self.n_cons = instance_data["n_cons"]

        self.obj_coeffs = np.array(instance_data["obj_coeffs"], dtype=np.float64)

        self.A = np.zeros((self.n_cons, self.n_vars), dtype=np.float64)
        for idx, cons in enumerate(instance_data["cons_coeffs"]):
            cons_indices = np.asarray(cons, dtype=np.int64)
            if cons_indices.size and cons_indices.min() >= 1:
                cons_indices = cons_indices - 1
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


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self._memory_limit_gb = getattr(cfg, "memory_limit_gb", 16)

    def _set_memory_limit(self):
        if resource is None:
            return

        if self._memory_limit_gb is None:
            return

        try:
            limit_bytes = int(self._memory_limit_gb * (1024 ** 3))
        except TypeError:
            print(f"Invalid memory limit configuration: {self._memory_limit_gb}")
            return

        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except (ValueError, OSError) as exc:
            print(f"Unable to enforce memory limit of {self._memory_limit_gb} GB: {exc}")

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

    def _compute_defaults(self, n_vars: int, n_objs: int, cutoff: str):
        # Default population size based on n_vars and n_objs
        pop_size = None
        if n_vars == 100:
            if n_objs == 3:
                pop_size = 250
            elif n_objs == 4:
                pop_size = 1000
            elif n_objs == 5:
                pop_size = 4000
            elif n_objs == 6:
                pop_size = 9000
            else:
                pop_size = 22000
        elif n_vars == 150:
            if n_objs == 3:
                pop_size = 850
            elif n_objs == 4:
                pop_size = 7000
            elif n_objs == 5:
                pop_size = 30000
            elif n_objs == 6:
                pop_size = 80000
            else:
                pop_size = 90000

        # Default run time based on n_vars, n_objs, and cutoff
        run_time = None
        if n_vars == 100:
            if n_objs == 3:
                run_time = 1
            elif n_objs == 4:
                run_time = 1
            elif n_objs == 5:
                run_time = 1 if cutoff == "restrict" else 2
            elif n_objs == 6:
                run_time = 5 if cutoff == "restrict" else 10
            else:
                run_time = 23 if cutoff == "restrict" else 27
        elif n_vars == 150:
            if n_objs == 3:
                run_time = 1 if cutoff == "restrict" else 14
            elif n_objs == 4:
                run_time = 10 if cutoff == "restrict" else 53
            elif n_objs == 5:
                run_time = 83 if cutoff == "restrict" else 326
            elif n_objs == 6:
                run_time = 333 if cutoff == "restrict" else 904
            else:
                run_time = 380 if cutoff == "restrict" else 851

        return pop_size, run_time

    def _run_nsga2(self, instance_data, pid, pop_size: int, run_time: int, run_seed: int):
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

        return -np.array(res.F, dtype=np.float64)

    def _save_frontier(self, pid, run_seed, pareto_front, sols_save_path):
        try:
            if pareto_front is not None and pareto_front.size:
                np.save(sols_save_path / f"{pid}-{run_seed}.npy", pareto_front)
        except Exception as e:
            print(f"Error saving frontier for PID {pid}: {e}")

    def _save_stats(
        self,
        pid,
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

            for run_seed in getattr(self.cfg, "run_seeds", [self.cfg.inst_seed]):
                start_time = time.time()
                pareto_front = self._run_nsga2(instance_data, pid, pop_size, run_time, run_seed)
                time_taken = time.time() - start_time

                if pareto_front is not None:
                    print(f"Pid: {pid}, Seed: {run_seed}, PF size: {pareto_front.shape}")
                    pareto_size = pareto_front.shape[0]
                else:
                    print(f"Did not find feasible solution: {pid} (seed={run_seed})")
                    pareto_size = 0

                self._save_frontier(pid, run_seed, pareto_front, sols_save_path)
                self._save_stats(
                    pid,
                    time_taken,
                    pareto_size,
                    sols_save_path,
                    instance_data,
                    pop_size,
                    run_time,
                    cutoff,
                    run_seed,
                    self.cfg.inst_seed,
                )

    def run(self):
        if self.cfg.n_processes == 1:
            self.worker(0)
        else:
            pool = mp.Pool(processes=self.cfg.n_processes)
            results = []

            for rank in range(self.cfg.n_processes):
                results.append(pool.apply_async(self.worker, args=(rank,)))

            for result in results:
                result.get()

            pool.close()
            pool.join()


@hydra.main(config_path="./configs", config_name="run_nsga2.yaml", version_base="1.2")
def main(cfg):
    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
