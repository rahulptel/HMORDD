#!/usr/bin/env python3
"""Builds missing-only META-Farm definitions for exact set packing DD runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Configuration for missing-only generation
TARGET_SIZES: Iterable[tuple[int, int]] = ((5, 150), (6, 150), (7, 150))
EXPECTED_PIDS = range(100)
SPLIT = "test"
PF_ENUM_METHOD = 3
DOMINANCE = 0

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PY = PROJECT_ROOT / "src" / "py"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs-nibi" / "outputs" / "sols" / "setpacking"
TABLE_PATH = Path(__file__).resolve().parent / "table.dat"


def build_command(n_objs: int, n_vars: int, pid: int) -> str:
    """Create the command line executed by META-Farm for one missing instance."""
    pythonpath = "/scratch/rahulpat/HMORDD/src/py"
    return (
        f"PYTHONPATH={pythonpath}:$PYTHONPATH "
        f"python -m hmordd.setpacking.run_dd "
        f"dd=exact prob.n_objs={n_objs} prob.n_vars={n_vars} "
        f"prob.pf_enum_method={PF_ENUM_METHOD} prob.dominance={DOMINANCE} "
        f"split={SPLIT} "
        f"from_pid={pid} to_pid={pid + 1} n_processes=1"
    )


def get_missing_pids(n_objs: int, n_vars: int) -> list[int]:
    """Return missing instance ids for one size by scanning expected CSV outputs."""
    out_dir = OUTPUTS_ROOT / f"{n_objs}_{n_vars}" / SPLIT / "exact" / "pf-3-dom-0"
    if not out_dir.exists():
        raise FileNotFoundError(f"Missing output directory: {out_dir}")
    missing = [pid for pid in EXPECTED_PIDS if not (out_dir / f"{pid}.csv").exists()]
    return missing


def generate_table_lines() -> tuple[list[str], list[str]]:
    """Enumerate missing-only cases for the table file."""
    lines: list[str] = []
    summary: list[str] = []
    case_id = 1
    for n_objs, n_vars in TARGET_SIZES:
        missing = get_missing_pids(n_objs, n_vars)
        summary.append(f"{n_objs}_{n_vars}: {len(missing)} missing")
        for pid in missing:
            command = build_command(n_objs, n_vars, pid)
            lines.append(f"{case_id} {command}")
            case_id += 1
    return lines, summary


def main() -> None:
    lines, summary = generate_table_lines()
    TABLE_PATH.write_text("\n".join(lines) + "\n")
    for entry in summary:
        print(entry)
    print(f"Total missing cases: {len(lines)}")
    print(f"Wrote {len(lines)} cases to {TABLE_PATH}")


if __name__ == "__main__":
    main()
