"""Create summary tables from post-processed setpacking metrics."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
from hmordd import Paths


NSGA2_DIR_RE = re.compile(r"pop(?P<pop>\d+)_time(?P<time>\d+)$")


@dataclass(frozen=True)
class SizeSpec:
    n_objs: int
    n_vars: int

    @property
    def size(self):
        return f"{self.n_objs}_{self.n_vars}"


@dataclass(frozen=True)
class Nsga2Spec:
    pop_size: int
    run_time: int

    @property
    def dirname(self):
        return f"pop{self.pop_size}_time{self.run_time}"


def _read_csvs(paths):
    frames = []
    for path in paths:
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except EmptyDataError:
                pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _mean_or_none(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _fmt_int(value):
    if value is None or pd.isna(value):
        return "--"
    return f"{int(round(value)):,}"


def _fmt_time(value):
    if value is None or pd.isna(value):
        return "--"
    return f"{int(math.ceil(value)):,}"


def _fmt_percent(value):
    if value is None or pd.isna(value):
        return "--"
    return str(int(round(100 * value)))


def _fmt_igd(value):
    if value is None or pd.isna(value):
        return "--"
    return f"{value:.3f}"


def _latex_textgray(value):
    return rf"\textcolor{{gray}}{{{value}}}"


def _discover_sizes(split, pf_enum_method, dominance):
    exact_root = Paths.sols / "setpacking"
    specs = []
    for path in sorted(exact_root.glob(f"*/{split}/exact/pf-{pf_enum_method}-dom-{dominance}")):
        try:
            n_objs, n_vars = [int(part) for part in path.parts[-4].split("_")]
        except ValueError:
            continue
        specs.append(SizeSpec(n_objs=n_objs, n_vars=n_vars))
    return tuple(sorted(specs, key=lambda spec: (spec.n_vars, spec.n_objs)))


def _exact_dir(spec, split, pf_enum_method, dominance):
    return (
        Paths.sols
        / "setpacking"
        / spec.size
        / split
        / "exact"
        / f"pf-{pf_enum_method}-dom-{dominance}"
    )


def _metrics_nsga2_dir(spec, split, nsga2):
    return (
        Paths.outputs
        / "metrics"
        / "setpacking"
        / spec.size
        / split
        / "nsga2"
        / nsga2.dirname
    )


def _sols_nsga2_dir(spec, split, nsga2):
    return (
        Paths.sols
        / "setpacking"
        / spec.size
        / split
        / "nsga2"
        / nsga2.dirname
    )


def _discover_nsga2_specs(spec, split):
    nsga2_root = Paths.outputs / "metrics" / "setpacking" / spec.size / split / "nsga2"
    specs = []
    for path in sorted(nsga2_root.glob("pop*_time*")):
        match = NSGA2_DIR_RE.match(path.name)
        if match is None:
            continue
        specs.append(
            Nsga2Spec(
                pop_size=int(match.group("pop")),
                run_time=int(match.group("time")),
            )
        )
    return tuple(sorted(specs, key=lambda item: item.pop_size))


def _exact_success_pids(spec, args):
    exact_dir = _exact_dir(spec, args.split, args.pf_enum_method, int(args.dominance))
    pids = []
    for path in exact_dir.glob("*.csv"):
        if path.stat().st_size == 0:
            continue
        try:
            pids.append(int(path.stem))
        except ValueError:
            continue
    return tuple(sorted(pid for pid in pids if args.from_pid <= pid < args.to_pid))


def _filter_pids(df, pids):
    if df.empty or "pid" not in df.columns:
        return df
    pid_values = pd.to_numeric(df["pid"], errors="coerce")
    return df[pid_values.isin(pids)].copy()


def _metrics_for_exact_pids(df, pids):
    metrics = _filter_pids(df, pids)
    if metrics.empty or "status" not in metrics.columns:
        return metrics

    metrics = metrics[metrics["status"] != "missing_exact"].copy()
    missing_approx = metrics["status"] == "missing_approx"
    for column in ("cardinality", "precision", "cardinality_raw", "n_approx_pf"):
        if column in metrics.columns:
            metrics.loc[missing_approx, column] = 0
    return metrics


def _exact_summary(spec, args, exact_pids):
    exact_dir = _exact_dir(spec, args.split, args.pf_enum_method, int(args.dominance))
    df = _read_csvs([exact_dir / f"{pid}.csv" for pid in exact_pids])
    return {
        "method": "Exact",
        "pop_size": None,
        "run_time": None,
        "time": _mean_or_none(df.get("total_time", pd.Series(dtype=float))),
        "cardinality": 1.0,
        "precision": 1.0,
        "igd": 0.0,
        "frontier_size": _mean_or_none(df.get("n_exact_pf", pd.Series(dtype=float))),
        "inst.": len(exact_pids),
    }


def _nsga2_summary(spec, args, nsga2, exact_pids):
    metrics_dir = _metrics_nsga2_dir(spec, args.split, nsga2)
    metrics = _metrics_for_exact_pids(_read_csvs(metrics_dir.glob("*.csv")), exact_pids)

    sols_dir = _sols_nsga2_dir(spec, args.split, nsga2)
    stats = _filter_pids(_read_csvs(sols_dir.glob("*.csv")), exact_pids)

    return {
        "method": f"NSGA-II-{nsga2.pop_size}",
        "pop_size": nsga2.pop_size,
        "run_time": nsga2.run_time,
        "time": _mean_or_none(stats.get("time_taken", pd.Series(dtype=float))),
        "cardinality": _mean_or_none(metrics.get("cardinality", pd.Series(dtype=float))),
        "precision": _mean_or_none(metrics.get("precision", pd.Series(dtype=float))),
        "igd": _mean_or_none(metrics.get("igd", pd.Series(dtype=float))),
        "frontier_size": _mean_or_none(metrics.get("n_approx_pf", pd.Series(dtype=float))),
        "inst.": len(exact_pids),
    }


def build_summary(args):
    rows = []
    for spec in _discover_sizes(args.split, args.pf_enum_method, int(args.dominance)):
        exact_pids = _exact_success_pids(spec, args)
        rows.append(
            {
                "n_vars": spec.n_vars,
                "n_objs": spec.n_objs,
                **_exact_summary(spec, args, exact_pids),
            }
        )
        for nsga2 in _discover_nsga2_specs(spec, args.split):
            rows.append(
                {
                    "n_vars": spec.n_vars,
                    "n_objs": spec.n_objs,
                    **_nsga2_summary(spec, args, nsga2, exact_pids),
                }
            )
    return pd.DataFrame(rows)


def render_latex(summary):
    lines = [
        r"\begin{table}[htbp!]",
        r"    \caption{MOSP results averaged across exact-success test instances.}",
        r"    \centering",
        r"    \footnotesize",
        r"    \resizebox{\linewidth}{!}{",
        r"    \begin{tabular}{rrlrrrrrrr}",
        r"    \toprule",
        r"    $N$ & $K$ & ~~Method & ~~Pop. & ~~Time $\downarrow$ & ~~Cardinality $\uparrow$ & ~~Precision $\uparrow$ & ~~IGD $\downarrow$ & ~~$|\hat{\mathcal{Z}}^\star|$ & ~~Inst. \\",
        r"    \midrule",
    ]

    groups = list(summary.groupby(["n_vars", "n_objs"], sort=False))
    for size_idx, ((n_vars, n_objs), group) in enumerate(groups):
        rows = group.to_dict("records")
        row_count = len(rows)
        for idx, row in enumerate(rows):
            prefix = "        & & "
            if idx == 0:
                prefix = rf"    \multirow{{{row_count}}}{{*}}{{{n_vars}}} & \multirow{{{row_count}}}{{*}}{{{n_objs}}} & "

            method = row["method"]
            pop_size = _fmt_int(row["pop_size"])
            time = _fmt_time(row["time"])
            cardinality = _fmt_percent(row["cardinality"])
            precision = _fmt_percent(row["precision"])
            igd = _fmt_igd(row["igd"])
            frontier_size = _fmt_int(row["frontier_size"])
            inst = _fmt_int(row["inst."])

            if method == "Exact":
                method = _latex_textgray(method)
                pop_size = _latex_textgray("--")
                time = _latex_textgray(time)
                cardinality = _latex_textgray(cardinality)
                precision = _latex_textgray(precision)
                igd = _latex_textgray(igd)
                frontier_size = _latex_textgray(frontier_size)
                inst = _latex_textgray(inst)

            lines.append(
                rf"{prefix}{method} & {pop_size} & {time} & {cardinality} & {precision} & {igd} & {frontier_size} & {inst} \\"
            )
            if idx == 0 and row_count > 1:
                lines.append(r"    \cmidrule{3-10}")

        if size_idx != len(groups) - 1:
            lines.extend(["", r"    \midrule", ""])

    lines.extend(
        [
            r"    \bottomrule",
            r"    \end{tabular}}",
            r"    \label{tab:sp_result_complete}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX MOSP summary table from postprocessed metrics."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--from-pid", type=int, default=0)
    parser.add_argument("--to-pid", type=int, default=100)
    parser.add_argument("--pf-enum-method", type=int, default=3)
    parser.add_argument("--dominance", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Paths.results / "setpacking_summary.tex",
        help="Path to write the LaTeX table.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Paths.results / "setpacking_summary.csv",
        help="Path to write the numeric summary CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_summary(args)
    latex = render_latex(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
