import multiprocessing as mp
import os
import sys
import time

import hydra
import numpy as np
import pandas as pd

from hmordd import Paths
from hmordd.setpacking.nsga2 import NSGAII
from hmordd.setpacking.utils import get_instance_data


@dataclass
class ProblemConfig:
    n_vars: int
    n_objs: int
    vars_per_con: int = 3
    obj_lb: int = 1
    obj_ub: int = 100
    pf_enum: int = 3
    node_select: int = 1

    @property
    def n_cons(self):
        return int(self.n_vars / 5)


@dataclass
class NSGA2Config:
    n_vars: int
    n_objs: int
    seed: int
    cutoff: str = "restrict"

    @property
    def pop_size(self):
        if self.n_vars == 100:
            if self.n_objs == 3:
                return 250
            elif self.n_objs == 4:
                return 1000
            elif self.n_objs == 5:
                return 4000
            elif self.n_objs == 6:
                return 9000
            else:
                return 22000
        elif self.n_vars == 150:
            if self.n_objs == 3:
                return 850
            elif self.n_objs == 4:
                return 7000
            elif self.n_objs == 5:
                return 30000
            elif self.n_objs == 6:
                return 80000
            else:
                return 90000

    @property
    def run_time(self):
        if self.n_vars == 100:
            if self.n_objs == 3:
                return 1
            elif self.n_objs == 4:
                return 1
            elif self.n_objs == 5:
                return 1 if self.cutoff == "restrict" else 2
            elif self.n_objs == 6:
                return 5 if self.cutoff == "restrict" else 10
            else:
                return 23 if self.cutoff == "restrict" else 27
        elif self.n_vars == 150:
            if self.n_objs == 3:
                return 1 if self.cutoff == "restrict" else 14
            elif self.n_objs == 4:
                return 10 if self.cutoff == "restrict" else 53
            elif self.n_objs == 5:
                return 83 if self.cutoff == "restrict" else 326
            elif self.n_objs == 6:
                return 333 if self.cutoff == "restrict" else 904
            else:
                return 380 if self.cutoff == "restrict" else 851


class SetPackingInstance:
    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.n_vars = self.cfg.n_vars
        self.n_objs = self.cfg.n_objs
        self.n_cons = self.cfg.n_cons
        self.obj_coeffs = []
        self.cons_coeffs = []
        if seed is not None:
            self.rng = np.random.RandomState(seed)

    def set_rng(self, rng):
        self.rng = rng

    def generate_instance(self):
        items = list(range(self.cfg.n_vars))

        # Value
        for _ in range(self.cfg.n_objs):
            self.obj_coeffs.append(
                list(
                    self.rng.randint(
                        self.cfg.obj_lb, self.cfg.obj_ub + 1, self.cfg.n_vars
                    )
                )
            )

        # Constraints
        for _ in range(self.n_cons):
            vars_in_con = self.rng.randint(2, (2 * self.cfg.vars_per_con) + 1)
            self.cons_coeffs.append(
                list(self.rng.choice(items, vars_in_con, replace=False))
            )

        # Ensure no variable is missed
        var_count = []
        for con in self.cons_coeffs:
            var_count.extend(con)
        missing_vars = list(set(range(self.cfg.n_vars)).difference(set(var_count)))
        for v in missing_vars:
            cons_id = self.rng.randint(self.n_cons)
            self.cons_coeffs[cons_id].append(v)

    def __repr__(self):
        return f"MIS Instance(n_objs={self.n_objs}, n_vars={self.n_vars}, n_cons={self.n_cons})"


class SetPackingProblem(Problem):
    def __init__(self, graph):
        self.graph = graph
        self.graph.obj_coeffs = np.array(graph.obj_coeffs)
        self.A = np.zeros((graph.n_cons, graph.n_vars))

        for i in range(graph.n_cons):
            self.A[i][graph.cons_coeffs[i]] = 1

        super().__init__(
            n_var=graph.n_vars,
            n_obj=graph.n_objs,
            n_ieq_constr=graph.n_cons,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # Constraint violations: Ensure no conflicting items are packed
        out["F"] = -np.dot(X, self.graph.obj_coeffs.T)
        out["G"] = np.dot(X, self.A.T) - 1  # Constraint handling


def execute(prob_cfg, nsga_cfg, pid):
    prefix = Path(
        f"{prob_cfg.n_objs}-{prob_cfg.n_vars}-10/nsga_frontier/{nsga_cfg.seed}"
    )
    prefix.mkdir(parents=True, exist_ok=True)

    graph = SetPackingInstance(prob_cfg, pid)
    graph.generate_instance()
    problem = SetPackingProblem(graph=graph)
    algorithm = NSGA2(
        pop_size=nsga_cfg.pop_size,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        get_termination("time", nsga_cfg.run_time),
        seed=nsga_cfg.seed,
    )
    print(res.F)
    if res.F is not None and len(res.F):
        z = -np.array(res.F)
        print("n_nds: ", z.shape)
        np.savez_compressed(f"{str(prefix)}/{pid}-frontier.npz", z=z)
    else:
        with open("infeasible.txt", "a") as fp:
            out = []
            out.append(prob_cfg.n_vars)
            out.append(prob_cfg.n_objs)
            out.append(nsga_cfg.seed)
            out.append(pid)
            out_str = " ".join(list(map(str, out))) + "\n"
            fp.write(out_str)
            print("Did not find feasible solution: ", pid)




class Runner:
    def __init__(self, cfg):
        self.cfg = cfg

    def _get_save_path(self, save_type="sols"):
        if save_type == "sols":
            base_path = Paths.sols
        else:
            raise ValueError("Invalid save_type. Must be 'sols'.")

        save_path = base_path / self.cfg.prob.name / self.cfg.prob.size / self.cfg.split / "nsga2"
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg

    def _get_save_path(self, save_type="sols"):
        if save_type == "sols":
            base_path = Paths.sols
        else:
            raise ValueError("Invalid save_type. Must be 'sols'.")

        save_path = base_path / self.cfg.prob.name / self.cfg.prob.size / self.cfg.split / "nsga2"
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    def _save_frontier(self, pid, pareto_front, sols_save_path):
        try:
            if pareto_front is not None:
                np.save(sols_save_path / f"{pid}.npy", pareto_front)
        except Exception as e:
            print(f"Error saving frontier for PID {pid}: {e}")

    def _save_stats(self, pid, time_taken, pareto_front_size, sols_save_path):
        stats_data = {
            "pid": [pid],
            "n_generations": [self.cfg.nsga2.n_generations],
            "pop_size": [self.cfg.nsga2.pop_size],
            "crossover_prob": [self.cfg.nsga2.crossover_prob],
            "mutation_prob": [self.cfg.nsga2.mutation_prob],
            "n_objectives": [self.cfg.prob.n_objs],
            "n_variables": [self.cfg.prob.n_vars],
            "n_constraints": [self.cfg.prob.n_constraints],
            "time_taken": [time_taken],
            "pareto_front_size": [pareto_front_size],
        }
        df_stats = pd.DataFrame(stats_data)
        try:
            df_stats.to_csv(sols_save_path / f"{pid}.csv", index=False)
        except Exception as e:
            print(f"Error saving statistics for PID {pid}: {e}")

    def worker(self, rank):
        sols_save_path = self._get_save_path("sols")

        for pid in range(self.cfg.from_pid + rank, self.cfg.to_pid, self.cfg.n_processes):
            print(f"Processing PID: {pid} on rank {rank}")

            # Load instance data
            instance_data = get_instance_data(self.cfg.prob.name, self.cfg.prob.size, self.cfg.split, self.cfg.seed, pid)

            nsga2 = NSGAII(
                instance_data=instance_data,
                n_generations=self.cfg.nsga2.n_generations,
                pop_size=self.cfg.nsga2.pop_size,
                crossover_prob=self.cfg.nsga2.crossover_prob,
                mutation_prob=self.cfg.nsga2.mutation_prob,
                n_objectives=self.cfg.prob.n_objs,
                n_variables=self.cfg.prob.n_vars,
                n_constraints=self.cfg.prob.n_constraints,
                seed=self.cfg.seed,
                time_limit=self.cfg.time_limit,
            )

            start_time = time.time()
            final_population = nsga2.run()
            end_time = time.time()
            time_taken = end_time - start_time

            # Extract non-dominated solutions (Pareto front)
            pareto_front = np.array([ind.objectives for ind in final_population if ind.rank == 0])

            if pareto_front is not None:
                print(f"Pid: {pid}, PF size: {pareto_front.shape}")

            self._save_frontier(pid, pareto_front, sols_save_path)
            self._save_stats(pid, time_taken, len(pareto_front), sols_save_path)


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


@hydra.main(config_path="./configs", config_name="run_nsga2.yaml", version_base="1.2")
def main(cfg):
    runner = Runner(cfg)
    runner.run()


if __name__ == '__main__':
    main()