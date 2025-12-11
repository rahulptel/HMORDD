#!/usr/bin/env python3
"""Builds META-Farm case definitions for exact knapsack DD runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple

# Configuration for the sweep
SIZES: Iterable[Tuple[int, int]] = ((3, 80), (4, 50), (7, 40))
INSTANCES_PER_CASE = 10
SPLIT = "test"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PY = PROJECT_ROOT / "src" / "py"
INSTANCES_ROOT = PROJECT_ROOT / "resources" / "instances" / "knapsack"
TABLE_PATH = Path(__file__).resolve().parent / "table.dat"


def build_command(n_objs: int, n_vars: int, pid_start: int, pid_end: int) -> str:
    """Create the command line executed by META-Farm for one case."""
    pythonpath = SRC_PY.as_posix()
    return (
        f"PYTHONPATH={pythonpath}:$PYTHONPATH "
        f"python -m hmordd.knapsack.run_dd "
        f"dd=exact prob.n_objs={n_objs} prob.n_vars={n_vars} "
        f"split={SPLIT} "
        f"from_pid={pid_start} to_pid={pid_end} n_processes=1"
    )


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
    """Enumerate all cases for the table.dat file."""
    lines: list[str] = []
    case_id = 1
    for n_objs, n_vars in SIZES:
        pid_offset, total_instances = pid_offset_and_count(n_objs, n_vars)
        cases = math.ceil(total_instances / INSTANCES_PER_CASE)
        for case_idx in range(cases):
            pid_start = pid_offset + case_idx * INSTANCES_PER_CASE
            pid_end = min(pid_start + INSTANCES_PER_CASE, pid_offset + total_instances)
            command = build_command(n_objs, n_vars, pid_start, pid_end)
            lines.append(f"{case_id} {command}")
            case_id += 1
    return lines


def main() -> None:
    lines = generate_table_lines()
    TABLE_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} cases to {TABLE_PATH}")


if __name__ == "__main__":
    main()
