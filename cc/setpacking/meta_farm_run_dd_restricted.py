#!/usr/bin/env python3
"""Builds META-Farm case definitions for restricted set packing DD runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple

# Configuration for the sweep
SIZES = tuple((n_objs, n_vars) for n_objs in range(3, 8) for n_vars in (100, 150))
RULE = 1
INSTANCES_PER_CASE = 10
SPLIT = "test"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PY = PROJECT_ROOT / "src" / "py"
INSTANCES_ROOT = PROJECT_ROOT / "resources" / "instances" / "setpacking"
TABLE_PATH = Path(__file__).resolve().parent / "table_dd_restricted.dat"


def build_command(n_objs: int, n_vars: int, pid_start: int, pid_end: int, rule: int, width) -> str:
    """Create the command line executed by META-Farm for one case."""
    pythonpath = SRC_PY.as_posix()
    return (
        f"PYTHONPATH={pythonpath}:$PYTHONPATH "
        f"python -m hmordd.setpacking.run_dd "
        f"dd=restricted dd.width={width} dd.nosh.rule={rule} "
        f"prob.n_objs={n_objs} prob.n_vars={n_vars} "
        f"split={SPLIT} from_pid={pid_start} to_pid={pid_end} "
        f"n_processes=1"
    )


def count_instances(n_objs: int, n_vars: int) -> int:
    """Count how many instances exist for a given size/split."""
    split_dir = INSTANCES_ROOT / f"{n_objs}_{n_vars}" / SPLIT
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing instances directory: {split_dir}")
    count = len(list(split_dir.glob("*.dat")))
    if count == 0:
        raise ValueError(f"No instances found in {split_dir}")
    return count


def generate_table_lines() -> list[str]:
    """Enumerate all cases for the table file."""
    lines: list[str] = []
    case_id = 1
    for n_objs, n_vars in SIZES:
        width = 50 if n_vars == 100 else 5000
        total_instances = count_instances(n_objs, n_vars)
        cases = math.ceil(total_instances / INSTANCES_PER_CASE)
        for case_idx in range(cases):
            pid_start = case_idx * INSTANCES_PER_CASE
            pid_end = min(pid_start + INSTANCES_PER_CASE, total_instances)
            command = build_command(n_objs, n_vars, pid_start, pid_end, RULE, width)
            lines.append(f"{case_id} {command}")
            case_id += 1
    return lines


def main() -> None:
    lines = generate_table_lines()
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} cases to {TABLE_PATH}")


if __name__ == "__main__":
    main()
