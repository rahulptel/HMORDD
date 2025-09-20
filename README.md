# HMORDD

Heuristic Multiobjective Optimization using Restricted Decision Diagrams (HMORDD) is a research codebase for exploring decision-diagram-based methods for multiobjective discrete optimization.  The project studies how restricted decision diagrams built with handcrafted or machine-learned heuristics can approximate the Pareto frontier with far less computation than enumerating an exact DD.  In particular, we focus on heuristics that identify the small fraction of *Pareto nodes*—nodes that lie on paths leading to non-dominated solutions—so that the remaining nodes can be pruned without sacrificing frontier quality.

## Repository layout

```
HMORDD/
├── src/
│   ├── cpp/
│   │   ├── common/              # Generic BDD utilities shared by all problems
│   │   └── setpacking/          # Set packing specific decision diagram implementation
│   └── py/
│       └── hmordd/
│           ├── __init__.py      # Path management and shared package initialisation
│           ├── common/          # Python-side DD interfaces and utilities
│           └── setpacking/      # Set packing specific pipelines and configs
├── resources/                   # Expected location for compiled libs and generated instances
├── outputs/                     # Generated decision diagrams and solution frontiers
└── results/                     # Experiment summaries and analysis artifacts
```

Problem-specific code lives under `src/cpp/<problem>` and `src/py/hmordd/<problem>`.  Each problem package is free to tailor its decision diagram builders, instance generators, heuristics, and ML workflows without impacting other problem classes.  During the refactor we will add `knapsack/` and `tsp/` siblings to the existing `setpacking/` modules.

## C++ decision diagram libraries

The C++ layer constructs, reduces, and enumerates multiobjective decision diagrams.  Common building blocks such as BDD nodes, algorithms, and utility routines sit under `src/cpp/common`.  Problem-specific environments wrap these primitives.  For example, `src/cpp/setpacking/setpackingenv.hpp` exposes the `SetpackingEnv` class that owns the set packing instance data, builds exact or restricted BDDs, and enumerates the Pareto frontier.

Shared libraries are built with `pybind11` so they can be loaded from Python.  Each problem package provides a helper script for compilation; for set packing the script is `src/cpp/setpacking/makelibcmd.sh`.  It recompiles `libsetpackingenv.cpp` for each supported objective count and emits modules named `libsetpackingenvo<N>.so`.  Run the script from the problem directory after installing a C++17 compiler and `pybind11`:

```bash
cd src/cpp/setpacking
bash makelibcmd.sh
```

Place the generated `.so` files in `resources/bin/setpacking/` (the Python package automatically adds this directory to `sys.path`), or adjust `PYTHONPATH` so the modules are discoverable.

## Python workflows

The Python package orchestrates experiments and provides integration points for heuristics and learning-based models.  The top-level initialiser (`src/py/hmordd/__init__.py`) defines a `Paths` helper that standardises where resources, outputs, and results live.  The `common/` package contributes abstract interfaces such as `DDManager` for managing the lifecycle of a decision diagram run, generic node-selection heuristics (NOSH), and utility helpers for metrics and multiprocessing support.

The set packing package demonstrates the expected structure for a problem class:

- **Configurations** – Hydra configuration files in `src/py/hmordd/setpacking/configs/` capture experiment parameters such as instance sizes (`generate_instances.yaml`), Pareto enumeration settings (`prob/setpacking.yaml`), and DD types (`dd/exact.yaml` and `dd/restricted.yaml`).
- **Instance generation** – `generate_instances.py` produces synthetic instances following the Stidsen generator.  It stores training/validation/test splits under `resources/instances/setpacking/<n_objs>-<n_vars>/`.
- **Decision diagram runners** – `dd.py` implements `SetPackingDDManager` plus exact and restricted subclasses that call into the compiled C++ environment.  `run_dd.py` loads Hydra configs, fetches instances, builds the requested DD, enumerates the Pareto frontier, and saves frontier statistics under `outputs/`.
- **Baselines and heuristics** – `common/nosh.py` contains reusable node-selection heuristics.  The `setpacking/nsga2.py` module currently stubs out an NSGA-II baseline that will be extended with a full evolutionary algorithm during future development.

### Typical workflow for set packing

1. **Compile the problem-specific libraries** (if not already built)
   ```bash
   cd src/cpp/setpacking
   bash makelibcmd.sh
   mkdir -p ../../resources/bin/setpacking
   mv libsetpackingenvo*.so ../../resources/bin/setpacking/
   ```

2. **Generate instances**
   ```bash
   cd src/py
   python -m hmordd.setpacking.generate_instances
   ```
   Edit `configs/generate_instances.yaml` to control the number of variables, objectives, and dataset sizes.

3. **Run decision diagram experiments**
   ```bash
   cd src/py
   python -m hmordd.setpacking.run_dd dd=restricted dd.width=50 dd.nosh.rule=1 split=train from_pid=0 to_pid=10
   ```
   Adjust the Hydra overrides to toggle between exact and restricted diagrams, select dataset splits, and control batch sizes.

4. **Analyse results** – Frontier `.npy` files and DD statistics are written to `outputs/`.  Scripts for ML-based heuristics and evolutionary baselines will read from the same directory structure.

## Adding new problem classes

The upcoming `knapsack` and `tsp` problem classes will mirror the set packing layout:

- Implement the core decision diagram environment in `src/cpp/<problem>/`, reusing shared code from `src/cpp/common/`.
- Provide a `makelibcmd.sh` script (or similar) that compiles pybind11 modules named `lib<problem>envo<N>.so`.
- Create a Python package at `src/py/hmordd/<problem>/` with Hydra configs, instance generators, DD managers, and integration hooks for heuristics or learning components.
- Update `hmordd/__init__.py` to add the new binary directory to `sys.path` so Python can import the compiled modules.

Following this structure keeps each problem self-contained while sharing utilities across the repository, enabling rapid experimentation without cross-problem coupling.  It also matches the experimental scope of the HMORDD paper, which compares handcrafted and learning-guided restricted DD heuristics on multiobjective knapsack, set packing, and travelling salesperson benchmarks.

## Dependencies

- **C++:** GCC or Clang with C++17 support, `pybind11`, and the Python development headers used by `python3-config`.
- **Python:** Python 3.8+, `hydra-core`, `numpy`, `pandas`, `torch`, and other packages imported throughout the modules (install via `pip install -r requirements.txt` once defined).  Some scripts also rely on `multiprocessing` and `signal` from the standard library.

Set up a virtual environment, install the Python requirements, build the C++ extensions, and populate `resources/instances/` with generated or benchmark instances before launching experiments.
