#!/usr/bin/env python3
"""Builds META-Farm case definitions for exact set packing DD runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Configuration for the sweep
N_OBJS_VALUES: Iterable[int] = range(3, 8)
N_VARS_VALUES: Iterable[int] = (100, 150)
CASES_PER_COMBINATION = 10
INSTANCES_PER_CASE = 10
SPLIT = "test"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PY = PROJECT_ROOT / "src" / "py"
TABLE_PATH = Path(__file__).resolve().parent / "table.dat"


def build_command(n_objs: int, n_vars: int, pid_start: int, pid_end: int) -> str:
    """Create the command line executed by META-Farm for one case."""
    size_token = f"{n_objs}_{n_vars}"
    pythonpath = SRC_PY.as_posix()
    return (
        f"PYTHONPATH={pythonpath}:$PYTHONPATH "
        f"python -m hmordd.setpacking.run_dd "
        f"dd=exact prob.n_objs={n_objs} prob.n_vars={n_vars} "
        f"split={SPLIT} "
        f"from_pid={pid_start} to_pid={pid_end} n_processes=1"
    )


def generate_table_lines() -> list[str]:
    """Enumerate all cases for the table.dat file."""
    lines: list[str] = []
    case_id = 1
    for n_objs in N_OBJS_VALUES:
        for n_vars in N_VARS_VALUES:
            for case_idx in range(CASES_PER_COMBINATION):
                pid_start = case_idx * INSTANCES_PER_CASE
                pid_end = pid_start + INSTANCES_PER_CASE
                command = build_command(n_objs, n_vars, pid_start, pid_end)
                lines.append(f"{case_id} {command}")
                case_id += 1
    return lines


def main() -> None:
    lines = generate_table_lines()
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} cases to {TABLE_PATH}")


if __name__ == "__main__":
    main()
