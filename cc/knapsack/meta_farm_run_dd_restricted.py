#!/usr/bin/env python3
"""Builds META-Farm case definitions for restricted knapsack DD runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Configuration for the sweep
SIZES_AND_WIDTHS: Dict[Tuple[int, int], Tuple[int, int]] = {
    (3, 80): (4000, 6000),
    (4, 50): (2500, 3500),
    (7, 40): (2000, 3000),
}
NOSH_OPTIONS: Iterable[str] = ("Scal+", "FE")
INSTANCES_PER_CASE = 10
SPLIT = "test"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PY = PROJECT_ROOT / "src" / "py"
INSTANCES_ROOT = PROJECT_ROOT / "resources" / "instances" / "knapsack"
TABLE_PATH = Path(__file__).resolve().parent / "table_dd_restricted.dat"

# FE-specific model overrides per problem size
FE_OVERRIDES: Dict[Tuple[int, int], Dict[str, int]] = {
    (3, 80): {"model.min_child_weight": 1000},
    (4, 50): {"model.max_depth": 7, "model.min_child_weight": 10000},
    (7, 40): {"model.max_depth": 9, "model.min_child_weight": 10000},
}


def build_command(n_objs: int, n_vars: int, pid_start: int, pid_end: int, width: int, nosh: str) -> str:
    """Create the command line executed by META-Farm for one case."""
    pythonpath = SRC_PY.as_posix()
    parts = [
        f"PYTHONPATH={pythonpath}:$PYTHONPATH",
        "python -m hmordd.knapsack.run_dd",
        "dd=restricted",
        f"dd.width={width}",
        f"dd.nosh={nosh}",
        f"prob.n_objs={n_objs}",
        f"prob.n_vars={n_vars}",
        f"split={SPLIT}",
        f"from_pid={pid_start}",
        f"to_pid={pid_end}",
        "n_processes=1",
    ]
    if nosh == "FE":
        overrides = FE_OVERRIDES.get((n_objs, n_vars), {})
        parts.extend(f"{key}={value}" for key, value in overrides.items())
    return " ".join(parts)


def parse_pid(instance_path: Path) -> int:
    """Extract the PID encoded in an instance filename."""
    try:
        return int(instance_path.stem.split("_")[-1])
    except (IndexError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unrecognized instance filename: {instance_path.name}") from exc


def pid_offset_and_count(n_objs: int, n_vars: int) -> tuple[int, int]:
    """Return the smallest PID and number of instances for a size/split."""
    split_dir = INSTANCES_ROOT / f"{n_objs}_{n_vars}" / SPLIT
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing instances directory: {split_dir}")

    pids = sorted(parse_pid(path) for path in split_dir.glob("*.dat"))
    if not pids:
        raise ValueError(f"No instances found in {split_dir}")

    return pids[0], len(pids)


def generate_table_lines() -> list[str]:
    """Enumerate all cases for the table file."""
    lines: list[str] = []
    case_id = 1
    for (n_objs, n_vars), widths in SIZES_AND_WIDTHS.items():
        pid_offset, total_instances = pid_offset_and_count(n_objs, n_vars)
        cases = math.ceil(total_instances / INSTANCES_PER_CASE)
        for width in widths:
            for nosh in NOSH_OPTIONS:
                for case_idx in range(cases):
                    pid_start = pid_offset + case_idx * INSTANCES_PER_CASE
                    pid_end = min(pid_start + INSTANCES_PER_CASE, pid_offset + total_instances)
                    command = build_command(n_objs, n_vars, pid_start, pid_end, width, nosh)
                    lines.append(f"{case_id} {command}")
                    case_id += 1
    return lines


def main() -> None:
    lines = generate_table_lines()
    TABLE_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} cases to {TABLE_PATH}")


if __name__ == "__main__":
    main()
