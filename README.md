# Heuristic Multiobjective Discrete Optimization using Restricted Decision Diagrams

> Official Implementation. 
The appendix is available at: https://github.com/rahulptel/HMORDD/blob/main/APPENDIX.pdf

Decision diagrams (DDs) have emerged as a state-of-the-art method for exact multiobjective integer linear programming. When the DD is too large to fit into memory or the decision-maker prefers a fast approximation to the Pareto frontier, the complete DD must be restricted to a subset of its states (or nodes). We introduce new node-selection heuristics for constructing restricted DDs that produce a high-quality approximation of the Pareto frontier. Depending on the structure of the problem, our heuristics are based on either simple rules, machine learning with feature engineering, or end-to-end deep learning. Experiments on multiobjective knapsack, set packing, and traveling salesperson problems show that our approach is highly effective: it recovers more than 85% of the Pareto frontier at a fraction of the running time of enumeration over exact DDs while producing very few non-Pareto solutions. The code and appendix is available at https://github.com/rahulptel/HMORDD.

> Multiobjective Optimization, Decision Diagrams and Machine Learning.


## Layout

```text
HMORDD/
|-- src/
|   |-- cpp/                         # C++ DD environments and pybind11 modules
|   |   |-- common/                  # Shared BDD/DD utilities
|   |   |-- knapsack/                # Multiobjective knapsack environment
|   |   |-- setpacking/              # Multiobjective set packing environment
|   |   `-- tsp/                     # Multiobjective TSP environment
|   `-- py/hmordd/                   # Python package, runners, and analysis tools
|       |-- common/                  # Shared runners, frontier loaders, metrics, utilities
|       |-- knapsack/                # Instance generation, DD/NSGA-II runs, postprocess, summaries
|       |-- setpacking/              # Instance generation, DD/NSGA-II runs, postprocess, summaries
|       `-- tsp/                     # Instance generation, DD/NSGA-II runs, postprocess, summaries
|-- resources/
|   |-- bin/<problem>/               # Built shared libraries: lib<problem>envo<N>.so
|   |-- instances/<problem>/         # Generated datasets by size and split
|   |-- checkpoints/                 # Selected TSP checkpoints used by E2E node selection
|   `-- pretrained/                  # Pretrained node-selection models
|-- outputs/                         # Generated experiment artifacts
|   |-- dds/<problem>/               # DD stats and optional saved diagrams
|   |-- sols/<problem>/              # Pareto fronts and baseline solutions
|   |-- dataset/<problem>/           # Collected DD node datasets for training
|   |-- checkpoints/tsp/             # TSP training checkpoints
|   `-- metrics/<problem>/           # Post-processed metric CSVs
|-- results/                         # Summary CSV/LaTeX tables from summarize_results
|-- scripts/                         # Helper scripts such as build_all_cpp.sh
`-- cc/                              # Cluster/run helper material
```

## Build the C++ extensions

Use the helper script to build everything (or filter by problem):

```bash
bash scripts/build_all_cpp.sh          # build all problems
bash scripts/build_all_cpp.sh setpacking knapsack
```

To build one problem manually:

```bash
cd src/cpp/setpacking   # or knapsack, tsp
bash makelibcmd.sh
```

Artifacts are emitted to `resources/bin/<problem>/` with objective-count-specific names (`libsetpackingenvo<N>.so`, `libknapsackenvo<N>.so`, `libtspenvo<N>.so`). Set `machine=cc` or `machine=desktop` to reuse the dynamic extension suffix detected by `python3-config`; otherwise a static suffix is used.

## Run Python workflows

The Python package is not installed as a wheel in this checkout. Run the module commands below from `src/py` after activating the project environment:

```bash
source ~/envs/morbdd/bin/activate
cd src/py
```

## Set packing workflow

- **Generate instances** (Stidsen generator; outputs `sp_<seed>_<n_objs>_<n_vars>_<pid>.dat` under `resources/instances/setpacking/`):
  ```bash
  python -m hmordd.setpacking.generate_instances n_objs=3 n_vars=100 seed=42 n_train=0 n_val=0 n_test=100
  # Hydra multi-run example
  python -m hmordd.setpacking.generate_instances --multirun n_vars=100,150 n_objs=3,4,5 seed=42 n_train=0 n_val=0 n_test=50
  ```
- **Exact DD** (produces reference Pareto fronts in `outputs/sols/setpacking/<size>/<split>/exact/`):
  ```bash
  python -m hmordd.setpacking.run_dd dd=exact split=test from_pid=0 to_pid=50 prob.n_objs=3 prob.n_vars=100
  ```
- **Restricted DD** (constructs a restricted DD using the configured NOSH rule):
  ```bash
  python -m hmordd.setpacking.run_dd dd=restricted dd.width=50 dd.nosh.rule=1 split=test from_pid=0 to_pid=50 prob.n_objs=3 prob.n_vars=100
  ```
  Outputs live under `outputs/{dds,sols}/setpacking/<size>/<split>/restricted/width-<width>-nosh-<rule>/`.
- **NSGA-II baseline** (implemented with pymoo; defaults for population and runtime depend on `n_vars`, `n_objs`, and `nsga2.cutoff`):
  ```bash
  python -m hmordd.setpacking.run_nsga2 split=test from_pid=0 to_pid=20 inst_seed=42 nsga2.cutoff=restrict
  ```
  Results are stored in `outputs/sols/setpacking/<size>/<split>/nsga2/pop<...>_time<...>/` with per-seed CSV/NPY pairs.
- **Post-process metrics** – Compute cardinality, precision, IGD, and frontier-size metrics from saved exact, restricted, and NSGA-II frontiers:
  ```bash
  python -m hmordd.setpacking.postprocess split=test from_pid=0 to_pid=100 prob.n_objs=3 prob.n_vars=100 prob.pf_enum_method=3 prob.dominance=false
  ```
  Metrics are stored under `outputs/metrics/setpacking/<size>/<split>/...` using the same method subdirectories as `outputs/sols/`. Exact frontiers are not required to run the restricted DD, but they are needed here for comparison metrics.
- **Summary tables** – Build the current post-processed MOSP table:
  ```bash
  python -m hmordd.setpacking.summarize_results --split test --from-pid 0 --to-pid 100 --pf-enum-method 3 --dominance 0
  ```
  Summary artifacts are written to `results/setpacking_summary.csv` and `results/setpacking_summary.tex`.
## Knapsack workflow

- **Generate instances** (uncorrelated items, single-capacity constraint):
  ```bash
  python -m hmordd.knapsack.generate_instances n_objs=4 n_vars=50 seed=7
  ```
- **Exact DD** (writes `.npz` fronts under `outputs/sols/knapsack/<size>/<split>/exact/pf-<m>-dom-<d>-trackx-<t>/`):
  ```bash
  python -m hmordd.knapsack.run_dd dd=exact split=train from_pid=0 to_pid=50 prob.n_objs=4 prob.n_vars=50
  ```
- **Objective-only benchmarking mode** (`prob.track_x=0`) keeps Pareto objectives but skips decision-vector tracking:
  ```bash
  python -m hmordd.knapsack.run_dd dd=exact split=test from_pid=1100 to_pid=1110 prob.track_x=0
  ```
- **Dataset generation mode** requires decision tracking (`prob.track_x=1`, default).
- **Restricted DD** – Choose `dd.nosh=Scal+`/`Scal-` for rule-based pruning or `dd.nosh=FE` to use the XGBoost scorer (expects pretrained models under `resources/pretrained/gbt/knapsack/<size>/`):
  ```bash
  python -m hmordd.knapsack.run_dd dd=restricted dd.width=2500 dd.nosh=Scal+ split=test from_pid=1100 to_pid=1150 prob.n_objs=4 prob.n_vars=50
  ```
- **NSGA-II baseline** – Population/time defaults are defined for sizes (7 obj,40 var), (4 obj,50 var), and (3 obj,80 var):
  ```bash
  python -m hmordd.knapsack.run_nsga2 split=test from_pid=1100 to_pid=1120 inst_seed=7 prob.n_objs=4 prob.n_vars=50
  ```
- **Post-process metrics** – Compute metrics for the configured MOKP experiment grid:
  ```bash
  python -m hmordd.knapsack.postprocess split=test from_pid=1100 to_pid=1200 prob.n_objs=4 prob.n_vars=50 prob.pf_enum_method=3 prob.dominance=1 prob.track_x=0
  ```
  Metrics are stored under `outputs/metrics/knapsack/<size>/<split>/...`. Exact frontiers are not required to run the restricted DD, but they are needed here for comparison metrics.
- **Summary tables** – Build the current post-processed MOKP table:
  ```bash
  python -m hmordd.knapsack.summarize_results --split test --from-pid 1100 --to-pid 1200 --pf-enum-method 3 --dominance 1 --track-x 0
  ```
  Summary artifacts are written to `results/knapsack_summary.csv` and `results/knapsack_summary.tex`.

## Travelling salesperson workflow

- **Generate instances** (integer coordinates on a grid saved as `.npz`):
  ```bash
  python -m hmordd.tsp.generate_instances n_objs=3 n_vars=15 seed=7
  ```
- **Exact DD**:
  ```bash
  python -m hmordd.tsp.run_dd dd=exact split=train from_pid=0 to_pid=50 prob.n_objs=3 prob.n_vars=15
  ```
- **Objective-only mode** (`prob.track_x=0`) disables decision-path tracking and stores only objective vectors (`z`) while keeping `x=[]`:
  ```bash
  python -m hmordd.tsp.run_dd dd=exact split=test from_pid=1100 to_pid=1110 prob.track_x=0
  ```
- **Data collection mode** requires decision tracking (`prob.track_x=1`, default).
- Exact/frontier artifact directories are separated by tracking mode via a `trackx-*` suffix.
- **Restricted DD** – `dd.nosh` supports rule-based scoring (`OrdMeanHigh`, `OrdMaxHigh`, `OrdMinHigh`, `OrdMeanLow`, `OrdMaxLow`, `OrdMinLow`) or `E2E`, which loads a graph transformer checkpoint from `resources/checkpoints/tsp/<size>/<model>_best_model.pt`:
  ```bash
  python -m hmordd.tsp.run_dd dd=restricted dd.width=4804 dd.nosh=OrdMeanHigh split=test from_pid=1100 to_pid=1150 prob.n_objs=3 prob.n_vars=15
  ```
- **NSGA-II baseline** – Defaults cover 15-city, 3–4 objective instances:
  ```bash
  python -m hmordd.tsp.run_nsga2 split=test from_pid=1100 to_pid=1120 inst_seed=7 prob.n_objs=3 prob.n_vars=15
  ```
- **Post-process metrics** – Compute metrics for the configured MOTSP experiment grid:
  ```bash
  python -m hmordd.tsp.postprocess split=test from_pid=1100 to_pid=1200 prob.n_objs=3 prob.n_vars=15 prob.pf_enum_method=3 prob.track_x=0
  ```
  Metrics are stored under `outputs/metrics/tsp/<size>/<split>/...`. Exact frontiers are not required to run the restricted DD, but they are needed here for comparison metrics.
- **Summary tables** – Build the current post-processed MOTSP table:
  ```bash
  python -m hmordd.tsp.summarize_results --split test --from-pid 1100 --to-pid 1200 --pf-enum-method 3 --track-x 0
  ```
  Summary artifacts are written to `results/tsp_summary.csv` and `results/tsp_summary.tex`.

## Outputs and metrics

- DD statistics and Pareto fronts are written to `outputs/dds/<problem>/...` and `outputs/sols/<problem>/...`. Set `save_dd=true` to also dump DD structures as JSON.
- Per-problem `postprocess.py` scripts write metric CSVs to `outputs/metrics/<problem>/...`.
- `summarize_results.py` scripts write final summary CSV/LaTeX tables to `results/`.
- Restricted DD runs do not check for saved exact frontiers before running. Post-processing computes metrics against saved exact frontiers when available; rows are marked with statuses such as `ok`, `missing_exact`, or `missing_approx`.
- The bundled `outputs-nibi/outputs/` directory mirrors the same layout for previously collected runs.

## Dependencies

- **C++:** GCC or Clang with C++17, `pybind11`, and Python development headers (`python3-config --includes/--ldflags`).
- **Python (3.8+) core packages:** `hydra-core`, `numpy`, `pandas`, `scipy`, `torch`, `xgboost`, `pymoo`, `matplotlib` (for plotting hooks), plus standard library modules used throughout. Create a virtualenv and install these before running the pipelines.

## Reference

If you find this work useful, please consider citing:

```
@inproceedings{patel2026hmordd,
  title   = {Heuristic Multiobjective Discrete Optimization using Restricted Decision Diagrams},
  author  = {Patel, Rahul and Khalil, Elias B. and Bergman, David},
  booktitle = {Proceedings of the International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research (CPAIOR)},
  year    = {2026},
  note    = {To appear},
  url     = {https://arxiv.org/abs/2403.02482}
}
```
