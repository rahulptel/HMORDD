#!/usr/bin/env python3
"""Builds META-Farm case definitions for exact TSP DD runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Configuration for the sweep
SIZES: Iterable[Tuple[int, int]] = ((3, 15), (4, 15))
INSTANCES_PER_CASE = 10
SPLIT = "test"
TRACK_X = 0

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PY = PROJECT_ROOT / "src" / "py"
INSTANCES_ROOT = PROJECT_ROOT / "resources" / "instances" / "tsp"
TABLE_PATH = Path(__file__).resolve().parent / "table.dat"

# E2E-specific Hydra model configs per problem size
E2E_MODEL_CONFIG: Dict[Tuple[int, int], str] = {
    (3, 15): "gtf_best_3_15",
    (4, 15): "gtf_best_4_15",
}


def build_command(n_objs, n_vars, pid_start, pid_end, rule):
    """Create the command line executed by META-Farm for one case."""
    pythonpath = SRC_PY.as_posix()
    parts = [
        f"PYTHONPATH={pythonpath}:$PYTHONPATH",
        "python -m hmordd.tsp.run_dd",
        "dd=restricted",
        f"dd.nosh={rule}",
        f"prob.n_objs={n_objs}",
        f"prob.n_vars={n_vars}",
        f"prob.track_x={TRACK_X}",
        f"split={SPLIT}",
        f"from_pid={pid_start}",
        f"to_pid={pid_end}",
        "n_processes=1",
    ]
    if rule == "E2E":
        model_cfg = E2E_MODEL_CONFIG.get((n_objs, n_vars))
        if model_cfg is None:
            raise ValueError(f"Missing E2E model config for size {(n_objs, n_vars)}")
        parts.append(f"model={model_cfg}")
    return " ".join(parts)


def count_instances(n_objs, n_vars):
    """Count how many instances exist for a given size/split."""
    split_dir = INSTANCES_ROOT / f"{n_objs}_{n_vars}" / SPLIT
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing instances directory: {split_dir}")
    print(split_dir)
    count = len(list(split_dir.glob("*.npz")))
    if count == 0:
        raise ValueError(f"No instances found in {split_dir}")
    return count


def generate_table_lines():
    """Enumerate all cases for the table.dat file."""
    lines: list[str] = []
    case_id = 1
    for n_objs, n_vars in SIZES:
        if SPLIT == "val":
            offset = 1000
        elif SPLIT == "test":
            offset = 1100    
        else:
            offset = 0
            
        total_instances = count_instances(n_objs, n_vars)
        cases = math.ceil(total_instances / INSTANCES_PER_CASE)
        print(total_instances, cases)
        for case_idx in range(cases):
            pid_start = offset + case_idx * INSTANCES_PER_CASE
            pid_end = min(pid_start + INSTANCES_PER_CASE, offset + total_instances)
            for rule in ["E2E", "OrdMeanHigh", "OrdMaxHigh", "OrdMinHigh", 
                         "OrdMeanLow", "OrdMaxLow", "OrdMinLow"]:
                command = build_command(n_objs, n_vars, pid_start, pid_end, rule)
                lines.append(f"{case_id} {command}")
                case_id += 1
    return lines


def main():
    lines = generate_table_lines()
    TABLE_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} cases to {TABLE_PATH}")


if __name__ == "__main__":
    main()
