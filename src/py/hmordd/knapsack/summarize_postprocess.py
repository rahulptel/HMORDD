"""Create a LaTeX summary table from post-processed knapsack metrics."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from hmordd import Paths


@dataclass(frozen=True)
class SizeSpec:
    n_vars: int
    n_objs: int
    widths: tuple[int, int]
    nsga2_time: int


SIZE_SPECS = (
    SizeSpec(n_vars=40, n_objs=7, widths=(2000, 3000), nsga2_time=60),
    SizeSpec(n_vars=50, n_objs=4, widths=(2500, 3500), nsga2_time=12),
    SizeSpec(n_vars=80, n_objs=3, widths=(4000, 6000), nsga2_time=58),
)

TRIAL_SEEDS = (7, 8, 9, 10, 11)
NSGA2_POPS = (100, 500)
NOSHES = (("Scal+", r"\noshRule{}"), ("FE", r"\noshFE{}"))


def _read_csvs(paths):
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _pid_paths(directory: Path, suffix: str, from_pid: int, to_pid: int):
    return [directory / f"{pid}{suffix}" for pid in range(from_pid, to_pid)]


def _pid_frame_paths(directory: Path, suffix: str, pids):
    return [directory / f"{pid}{suffix}" for pid in pids]


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


def _latex_bold(value, flag):
    return rf"\textbf{{{value}}}" if flag and value != "--" else value


def _exact_dir(spec: SizeSpec, split: str, pf: int, dominance: int, track_x, order_type):
    size = f"{spec.n_objs}_{spec.n_vars}"
    base = Paths.sols / "knapsack" / size / split / "exact"
    suffix = f"pf-{pf}-dom-{dominance}"
    if track_x is not None:
        suffix += f"-trackx-{int(track_x)}"
    if order_type is not None:
        suffix += f"-order-{order_type}"
    return base / suffix


def _restricted_dir(
    spec: SizeSpec,
    split: str,
    pf: int,
    dominance: int,
    track_x,
    order_type,
    nosh: str,
    width: int,
):
    size = f"{spec.n_objs}_{spec.n_vars}"
    base = Paths.sols / "knapsack" / size / split / "restricted"
    suffix = f"pf-{pf}-dom-{dominance}"
    if track_x is not None:
        suffix += f"-trackx-{int(track_x)}"
    if order_type is not None:
        suffix += f"-order-{order_type}"
    return base / suffix / f"{nosh}-{width}"


def _metrics_restricted_dir(
    spec: SizeSpec,
    split: str,
    pf: int,
    dominance: int,
    track_x,
    order_type,
    nosh: str,
    width: int,
):
    size = f"{spec.n_objs}_{spec.n_vars}"
    base = Paths.outputs / "metrics" / "knapsack" / size / split / "restricted"
    suffix = f"pf-{pf}-dom-{dominance}"
    if track_x is not None:
        suffix += f"-trackx-{int(track_x)}"
    if order_type is not None:
        suffix += f"-order-{order_type}"
    return base / suffix / f"{nosh}-{width}"


def _metrics_nsga2_dir(spec: SizeSpec, split: str, pop_size: int):
    size = f"{spec.n_objs}_{spec.n_vars}"
    return (
        Paths.outputs
        / "metrics"
        / "knapsack"
        / size
        / split
        / "nsga2"
        / f"pop{pop_size}_time{spec.nsga2_time}"
    )


def _nsga2_sols_dir(spec: SizeSpec, split: str, pop_size: int):
    size = f"{spec.n_objs}_{spec.n_vars}"
    return (
        Paths.sols
        / "knapsack"
        / size
        / split
        / "nsga2"
        / f"pop{pop_size}_time{spec.nsga2_time}"
    )


def _exact_success_pids(spec, args):
    exact_dir = _exact_dir(
        spec,
        args.split,
        args.pf_enum_method,
        args.dominance,
        args.track_x,
        args.order_type,
    )
    pids = []
    for pid in range(args.from_pid, args.to_pid):
        csv_path = exact_dir / f"{pid}.csv"
        if csv_path.exists():
            pids.append(pid)
    return tuple(pids)


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
    exact_dir = _exact_dir(
        spec,
        args.split,
        args.pf_enum_method,
        args.dominance,
        args.track_x,
        args.order_type,
    )
    df = _read_csvs(_pid_frame_paths(exact_dir, ".csv", exact_pids))
    return {
        "method": "Exact",
        "width": _mean_or_none(df.get("initial_width", pd.Series(dtype=float))),
        "time": _mean_or_none(df.get("total_time", pd.Series(dtype=float))),
        "cardinality": 1.0,
        "precision": 1.0,
        "igd": 0.0,
        "frontier_size": _mean_or_none(df.get("n_exact_pf", pd.Series(dtype=float))),
        "inst.": len(exact_pids),
    }


def _nsga2_summary(spec, args, pop_size, exact_pids):
    metrics_dir = _metrics_nsga2_dir(spec, args.split, pop_size)
    metrics_paths = [
        metrics_dir / f"{pid}-{seed}.csv"
        for pid in exact_pids
        for seed in TRIAL_SEEDS
    ]
    metrics = _metrics_for_exact_pids(_read_csvs(metrics_paths), exact_pids)

    sols_dir = _nsga2_sols_dir(spec, args.split, pop_size)
    sols_paths = [
        sols_dir / f"{pid}-{seed}.csv"
        for pid in exact_pids
        for seed in TRIAL_SEEDS
    ]
    stats = _filter_pids(_read_csvs(sols_paths), exact_pids)

    return {
        "method": f"NSGA-II-{pop_size}",
        "width": None,
        "time": _mean_or_none(stats.get("time_taken", pd.Series(dtype=float))),
        "cardinality": _mean_or_none(metrics.get("cardinality", pd.Series(dtype=float))),
        "precision": _mean_or_none(metrics.get("precision", pd.Series(dtype=float))),
        "igd": _mean_or_none(metrics.get("igd", pd.Series(dtype=float))),
        "frontier_size": _mean_or_none(metrics.get("n_approx_pf", pd.Series(dtype=float))),
        "inst.": len(exact_pids),
    }


def _restricted_summary(spec, args, nosh, method_label, width, exact_pids):
    metrics_dir = _metrics_restricted_dir(
        spec,
        args.split,
        args.pf_enum_method,
        args.dominance,
        args.track_x,
        args.order_type,
        nosh,
        width,
    )
    metrics = _metrics_for_exact_pids(
        _read_csvs(_pid_frame_paths(metrics_dir, ".csv", exact_pids)), exact_pids
    )

    sols_dir = _restricted_dir(
        spec,
        args.split,
        args.pf_enum_method,
        args.dominance,
        args.track_x,
        args.order_type,
        nosh,
        width,
    )
    stats = _filter_pids(_read_csvs(_pid_frame_paths(sols_dir, ".csv", exact_pids)), exact_pids)

    return {
        "method": method_label,
        "width": width,
        "time": _mean_or_none(stats.get("total_time", pd.Series(dtype=float))),
        "cardinality": _mean_or_none(metrics.get("cardinality", pd.Series(dtype=float))),
        "precision": _mean_or_none(metrics.get("precision", pd.Series(dtype=float))),
        "igd": _mean_or_none(metrics.get("igd", pd.Series(dtype=float))),
        "frontier_size": _mean_or_none(metrics.get("n_approx_pf", pd.Series(dtype=float))),
        "inst.": len(exact_pids),
    }


def build_summary(args):
    rows = []
    for spec in SIZE_SPECS:
        exact_pids = _exact_success_pids(spec, args)
        rows.append(
            {"n_vars": spec.n_vars, "n_objs": spec.n_objs, **_exact_summary(spec, args, exact_pids)}
        )
        for pop_size in NSGA2_POPS:
            rows.append(
                {
                    "n_vars": spec.n_vars,
                    "n_objs": spec.n_objs,
                    **_nsga2_summary(spec, args, pop_size, exact_pids),
                }
            )
        for nosh, method_label in NOSHES:
            for width in spec.widths:
                rows.append(
                    {
                        "n_vars": spec.n_vars,
                        "n_objs": spec.n_objs,
                        **_restricted_summary(spec, args, nosh, method_label, width, exact_pids),
                    }
                )
    return pd.DataFrame(rows)


def _best_flags(rows):
    restricted = [row for row in rows if row["method"] in {r"\noshRule{}", r"\noshFE{}"}]

    def best_key(metric, reverse):
        values = [row[metric] for row in restricted if row[metric] is not None and not pd.isna(row[metric])]
        if not values:
            return None
        return max(values) if reverse else min(values)

    best = {
        "time": best_key("time", reverse=False),
        "cardinality": best_key("cardinality", reverse=True),
        "precision": best_key("precision", reverse=True),
        "igd": best_key("igd", reverse=False),
    }
    flags = []
    for row in rows:
        row_flags = {}
        for metric, value in best.items():
            row_flags[metric] = (
                value is not None
                and row["method"] in {r"\noshRule{}", r"\noshFE{}"}
                and row[metric] is not None
                and not pd.isna(row[metric])
                and abs(row[metric] - value) <= 1e-12
            )
        flags.append(row_flags)
    return flags


def render_latex(summary):
    lines = [
        r"\begin{table}[htbp!]",
        r"    \caption{MOKP results averaged across test instances.",
        r"    NSGA-II-$p$ denotes NSGA-II with population size $p$.",
        r"    Refer to \Cref{sec:setup} for column description.}",
        r"    \centering",
        r"    \footnotesize",
        r"    \resizebox{0.7\linewidth}{!}{",
        r"    \begin{tabular}{rrlrrrrrrr}",
        r"    \toprule",
        r"    $N$ & $K$ & ~~Method & ~~Width & ~~Time $\downarrow$ & ~~Cardinality $\uparrow$ & ~~Precision $\uparrow$ & ~~IGD $\downarrow$ & ~~$|\hat{\mathcal{Z}}^\star|$ & ~~Inst. \\",
        r"    \midrule",
    ]

    groups = list(summary.groupby(["n_vars", "n_objs"], sort=False))
    for size_idx, (spec, group) in enumerate(groups):
        n_vars, n_objs = spec
        rows = group.to_dict("records")
        flags = _best_flags(rows)
        for idx, (row, row_flags) in enumerate(zip(rows, flags)):
            prefix = "        & & "
            if idx == 0:
                prefix = rf"    \multirow{{8}}{{*}}{{{n_vars}}} & \multirow{{8}}{{*}}{{{n_objs}}} & "

            method = row["method"]
            width = _fmt_int(row["width"])
            time = _fmt_time(row["time"])
            cardinality = _fmt_percent(row["cardinality"])
            precision = _fmt_percent(row["precision"])
            igd = _fmt_igd(row["igd"])
            frontier_size = _fmt_int(row["frontier_size"])
            inst = _fmt_int(row["inst."])

            if method == "Exact":
                method = _latex_textgray("Exact")
                width = _latex_textgray(width)
                time = _latex_textgray(time)
                cardinality = _latex_textgray(cardinality)
                precision = _latex_textgray(precision)
                igd = _latex_textgray(igd)
                frontier_size = _latex_textgray(frontier_size)
                inst = _latex_textgray(inst)
            else:
                time = _latex_bold(time, row_flags["time"])
                cardinality = _latex_bold(cardinality, row_flags["cardinality"])
                precision = _latex_bold(precision, row_flags["precision"])
                igd = _latex_bold(igd, row_flags["igd"])

            if method in {r"\noshRule{}", r"\noshFE{}"} and idx in {3, 5}:
                method = rf"\multirow{{2}}{{*}}{{{method}}}"
            elif method in {r"\noshRule{}", r"\noshFE{}"}:
                method = ""

            lines.append(
                rf"{prefix}{method} & {width} & {time} & {cardinality} & {precision} & {igd} & {frontier_size} & {inst} \\"
            )
            if idx in {0, 2, 4}:
                lines.append(r"    \cmidrule{3-10}")

        if size_idx != len(groups) - 1:
            lines.extend(["", r"    \midrule", ""])

    lines.extend(
        [
            r"    \bottomrule",
            r"    \end{tabular}}",
            r"    \label{tab:kp_result_complete}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX MOKP summary table from postprocessed metrics."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--from-pid", type=int, default=1100)
    parser.add_argument("--to-pid", type=int, default=1200)
    parser.add_argument("--pf-enum-method", type=int, default=3)
    parser.add_argument("--dominance", type=int, default=1)
    parser.add_argument("--track-x", type=int, default=0)
    parser.add_argument("--order-type", default="MinWt")
    parser.add_argument(
        "--output",
        type=Path,
        default=Paths.results / "knapsack_summary.tex",
        help="Path to write the LaTeX table.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Paths.results / "knapsack_summary.csv",
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
