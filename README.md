# Heuristic Multiobjective Discrete Optimization using Restricted Decision Diagrams

> Official Implementation. 
The appendix is available at: https://github.com/rahulptel/HMORDD/blob/main/APPENDIX.pdf

Decision diagrams (DDs) have emerged as a state-of-the-art method for exact multiobjective integer linear programming. When the DD is too large to fit into memory or the decision-maker prefers a fast approximation to the Pareto frontier, the complete DD must be restricted to a subset of its states (or nodes). We introduce new node-selection heuristics for constructing restricted DDs that produce a high-quality approximation of the Pareto frontier. Depending on the structure of the problem, our heuristics are based on either simple rules, machine learning with feature engineering, or end-to-end deep learning. Experiments on multiobjective knapsack, set packing, and traveling salesperson problems show that our approach is highly effective: it recovers more than $85\%$ of the Pareto frontier at a fraction of the running time of enumeration over exact DDs while producing very few non-Pareto solutions. The code and appendix is available at https://github.com/rahulptel/HMORDD.

> Multiobjective Optimization, Decision Diagrams and Machine Learning.


## Layout

- `src/cpp/` – C++ DD environments (`common/`, `setpacking/`, `knapsack/`, `tsp/`) compiled with pybind11.
- `src/py/hmordd/` – Python package with shared utilities plus problem-specific pipelines (`setpacking/`, `knapsack/`, `tsp/`).
- `resources/bin/<problem>/` – Built shared libraries (`lib<problem>envo<N>.so`), auto-added to `sys.path` by `hmordd.Paths`.
- `resources/instances/<problem>/<size>/<split>/` – Generated datasets; `<size>` is `<n_objs>_<n_vars>`.
- `resources/checkpoints/`, `resources/pretrained/` – Learned models for ML-based node selectors.
- `outputs/{dds,sols}/` – Experiment artifacts (DD stats, saved diagrams, Pareto fronts). `outputs-nibi/outputs/` holds archived runs from the nibi machine for offline analysis.
- `scripts/` – Helpers such as `build_all_cpp.sh` to compile every `makelibcmd.sh`.

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

## Set packing workflow

- **Generate instances** (Stidsen generator; outputs `sp_<seed>_<n_objs>_<n_vars>_<pid>.dat` under `resources/instances/setpacking/`):
  ```bash
  cd src/py
  python -m hmordd.setpacking.generate_instances n_objs=3 n_vars=100 seed=42
  # Hydra multi-run example
  python -m hmordd.setpacking.generate_instances --multirun n_vars=100,150 n_objs=3,4,5 seed=42 n_train=0 n_val=0 n_test=50
  ```
- **Exact DD** (produces reference Pareto fronts in `outputs/sols/setpacking/<size>/<split>/exact/`):
  ```bash
  python -m hmordd.setpacking.run_dd dd=exact split=test from_pid=0 to_pid=50 prob.n_objs=3 prob.n_vars=100
  ```
- **Restricted DD** (prunes with NOSH rule; skips instances without an exact frontier on disk):
  ```bash
  python -m hmordd.setpacking.run_dd dd=restricted dd.width=50 dd.nosh.rule=1 split=test from_pid=0 to_pid=50 prob.n_objs=3 prob.n_vars=100
  ```
  Outputs live under `outputs/{dds,sols}/setpacking/<size>/<split>/restricted/width-<width>-nosh-<rule>/`.
- **NSGA-II baseline** (implemented with pymoo; defaults for population and runtime depend on `n_vars`, `n_objs`, and `nsga2.cutoff`):
  ```bash
  python -m hmordd.setpacking.run_nsga2 split=test from_pid=0 to_pid=20 inst_seed=42 nsga2.cutoff=restrict
  ```
  Results are stored in `outputs/sols/setpacking/<size>/<split>/nsga2/pop<...>_time<...>/` with per-seed CSV/NPY pairs.
- **Summaries** – Aggregate existing runs (local `outputs/` or the archived `outputs-nibi/outputs/`):
  ```bash
  python -m hmordd.setpacking.summarize_results --outputs-root outputs --save-csv setpacking-summary.csv
  ```

## Knapsack workflow

- **Generate instances** (uncorrelated items, single-capacity constraint):
  ```bash
  cd src/py
  python -m hmordd.knapsack.generate_instances prob.n_objs=4 prob.n_vars=50 seed=42
  ```
- **Exact DD** (writes `.npz` fronts to `outputs/sols/knapsack/<size>/<split>/exact/`):
  ```bash
  python -m hmordd.knapsack.run_dd dd=exact split=train from_pid=0 to_pid=50 prob.n_objs=4 prob.n_vars=50
  ```
- **Restricted DD** – Choose `dd.nosh=Scal+`/`Scal-` for rule-based pruning or `dd.nosh=FE` to use the XGBoost scorer (expects pretrained models under `resources/pretrained/gbt/knapsack/<size>/`):
  ```bash
  python -m hmordd.knapsack.run_dd dd=restricted dd.width=2500 dd.nosh=Scal+ split=test from_pid=0 to_pid=50 prob.n_objs=4 prob.n_vars=50
  ```
- **NSGA-II baseline** – Population/time defaults are defined for sizes (7 obj,40 var), (4 obj,50 var), and (3 obj,80 var):
  ```bash
  python -m hmordd.knapsack.run_nsga2 split=test from_pid=0 to_pid=20 inst_seed=42 prob.n_objs=4 prob.n_vars=50
  ```

## Travelling salesperson workflow

- **Generate instances** (integer coordinates on a grid saved as `.npz`):
  ```bash
  cd src/py
  python -m hmordd.tsp.generate_instances prob.n_objs=3 prob.n_vars=15 seed=7
  ```
- **Exact DD**:
  ```bash
  python -m hmordd.tsp.run_dd dd=exact split=train from_pid=0 to_pid=50 prob.n_objs=3 prob.n_vars=15
  ```
- **Restricted DD** – `dd.nosh` supports rule-based scoring (`OrdMeanHigh`, `OrdMaxHigh`, `OrdMinHigh`, `OrdMeanLow`, `OrdMaxLow`, `OrdMinLow`) or `E2E`, which loads a graph transformer checkpoint from `resources/checkpoints/tsp/<size>/<model>_best_model.pt`:
  ```bash
  python -m hmordd.tsp.run_dd dd=restricted dd.width=4804 dd.nosh=OrdMeanHigh split=test from_pid=0 to_pid=50 prob.n_objs=3 prob.n_vars=15
  ```
- **NSGA-II baseline** – Defaults cover 15-city, 3–4 objective instances:
  ```bash
  python -m hmordd.tsp.run_nsga2 split=test from_pid=0 to_pid=20 inst_seed=42 prob.n_objs=3 prob.n_vars=15
  ```

## Outputs and metrics

- DD statistics and Pareto fronts are written to `outputs/dds/<problem>/...` and `outputs/sols/<problem>/...`. Set `save_dd=true` to also dump DD structures as JSON.
- Restricted runs report cardinality and precision against the saved exact frontier when available; otherwise metrics are set to sentinel values.
- The bundled `outputs-nibi/outputs/` directory mirrors the same layout for previously collected runs.

## Dependencies

- **C++:** GCC or Clang with C++17, `pybind11`, and Python development headers (`python3-config --includes/--ldflags`).
- **Python (3.8+) core packages:** `hydra-core`, `numpy`, `pandas`, `scipy`, `torch`, `xgboost`, `pymoo`, `matplotlib` (for plotting hooks), plus standard library modules used throughout. Create a virtualenv and install these before running the pipelines.
