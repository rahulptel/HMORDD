"""Create a LaTeX summary table from post-processed knapsack metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from hmordd import Paths
from hmordd.knapsack.postprocess import EXPERIMENT_SPECS

TRIAL_SEEDS = (7, 8, 9, 10, 11)
NOSH_LABELS = {"Scal+": r"\noshRule{}", "FE": r"\noshFE{}"}


def _read_csvs(paths):
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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


def _exact_dir(spec, args):
    base = Paths.sols / "knapsack" / spec.size / args.split / "exact"
    suffix = f"pf-{args.pf_enum_method}-dom-{args.dominance}"
    if args.track_x is not None:
        suffix += f"-trackx-{int(args.track_x)}"
    if args.order_type is not None:
        suffix += f"-order-{args.order_type}"
    return base / suffix


def _restricted_dir(spec, args, variant):
    base = Paths.sols / "knapsack" / spec.size / args.split / "restricted"
    suffix = f"pf-{args.pf_enum_method}-dom-{args.dominance}"
    if args.track_x is not None:
        suffix += f"-trackx-{int(args.track_x)}"
    if args.order_type is not None:
        suffix += f"-order-{args.order_type}"
    return base / suffix / variant["key"]


def _metrics_restricted_dir(spec, args, variant):
    base = Paths.outputs / "metrics" / "knapsack" / spec.size / args.split / "restricted"
    suffix = f"pf-{args.pf_enum_method}-dom-{args.dominance}"
    if args.track_x is not None:
        suffix += f"-trackx-{int(args.track_x)}"
    if args.order_type is not None:
        suffix += f"-order-{args.order_type}"
    return base / suffix / variant["key"]


def _metrics_nsga2_dir(spec, args, nsga2):
    return (
        Paths.outputs
        / "metrics"
        / "knapsack"
        / spec.size
        / args.split
        / "nsga2"
        / nsga2["key"]
    )


def _nsga2_sols_dir(spec, args, nsga2):
    return (
        Paths.sols
        / "knapsack"
        / spec.size
        / args.split
        / "nsga2"
        / nsga2["key"]
    )


def _exact_available_pids(spec, args):
    exact_dir = _exact_dir(spec, args)
    pids = []
    for pid in range(args.from_pid, args.to_pid):
        frontier_path = exact_dir / f"{pid}.npz"
        if frontier_path.exists() and frontier_path.stat().st_size > 0:
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
    exact_dir = _exact_dir(spec, args)
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


def _nsga2_summary(spec, args, nsga2, exact_pids):
    metrics_dir = _metrics_nsga2_dir(spec, args, nsga2)
    metrics_paths = [
        metrics_dir / f"{pid}-{seed}.csv"
        for pid in exact_pids
        for seed in TRIAL_SEEDS
    ]
    metrics = _metrics_for_exact_pids(_read_csvs(metrics_paths), exact_pids)

    sols_dir = _nsga2_sols_dir(spec, args, nsga2)
    sols_paths = [
        sols_dir / f"{pid}-{seed}.csv"
        for pid in exact_pids
        for seed in TRIAL_SEEDS
    ]
    stats = _filter_pids(_read_csvs(sols_paths), exact_pids)

    return {
        "method": f"NSGA-II-{nsga2['pop_size']}",
        "width": None,
        "time": _mean_or_none(stats.get("time_taken", pd.Series(dtype=float))),
        "cardinality": _mean_or_none(metrics.get("cardinality", pd.Series(dtype=float))),
        "precision": _mean_or_none(metrics.get("precision", pd.Series(dtype=float))),
        "igd": _mean_or_none(metrics.get("igd", pd.Series(dtype=float))),
        "frontier_size": _mean_or_none(metrics.get("n_approx_pf", pd.Series(dtype=float))),
        "inst.": len(exact_pids),
    }


def _restricted_summary(spec, args, variant, exact_pids):
    metrics_dir = _metrics_restricted_dir(spec, args, variant)
    metrics = _metrics_for_exact_pids(
        _read_csvs(_pid_frame_paths(metrics_dir, ".csv", exact_pids)), exact_pids
    )

    sols_dir = _restricted_dir(spec, args, variant)
    stats = _filter_pids(_read_csvs(_pid_frame_paths(sols_dir, ".csv", exact_pids)), exact_pids)

    return {
        "method": NOSH_LABELS.get(variant["nosh"], variant["key"]),
        "width": variant["width"],
        "time": _mean_or_none(stats.get("total_time", pd.Series(dtype=float))),
        "cardinality": _mean_or_none(metrics.get("cardinality", pd.Series(dtype=float))),
        "precision": _mean_or_none(metrics.get("precision", pd.Series(dtype=float))),
        "igd": _mean_or_none(metrics.get("igd", pd.Series(dtype=float))),
        "frontier_size": _mean_or_none(metrics.get("n_approx_pf", pd.Series(dtype=float))),
        "inst.": len(exact_pids),
    }


def _selected_specs(args):
    if args.size is None:
        return EXPERIMENT_SPECS
    return tuple(spec for spec in EXPERIMENT_SPECS if spec.size == args.size)


def build_summary(args):
    rows = []
    for spec in _selected_specs(args):
        exact_pids = _exact_available_pids(spec, args)
        rows.append(
            {"n_vars": spec.n_vars, "n_objs": spec.n_objs, **_exact_summary(spec, args, exact_pids)}
        )
        for nsga2 in spec.nsga2_variants:
            rows.append(
                {
                    "n_vars": spec.n_vars,
                    "n_objs": spec.n_objs,
                    **_nsga2_summary(spec, args, nsga2, exact_pids),
                }
            )
        for variant in spec.restricted_variants:
            rows.append(
                {
                    "n_vars": spec.n_vars,
                    "n_objs": spec.n_objs,
                    **_restricted_summary(spec, args, variant, exact_pids),
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


def _is_restricted_method(method):
    return method in {r"\noshRule{}", r"\noshFE{}"}


def _method_block_key(method):
    if method.startswith("NSGA-II-"):
        return "NSGA-II"
    return method


def _method_blocks(rows):
    blocks = []
    start = 0
    while start < len(rows):
        block_key = _method_block_key(rows[start]["method"])
        end = start + 1
        if block_key != "Exact":
            while end < len(rows) and _method_block_key(rows[end]["method"]) == block_key:
                end += 1
        blocks.append((start, end))
        start = end
    return blocks


def render_latex(summary, requested_methods, label_suffix):
    lines = [
        r"\begin{table}[htbp!]",
        r"    \caption{MOKP results averaged across test instances.",
    ]
    if any(method.startswith("NSGA-II-") for method in requested_methods):
        lines.append(r"    NSGA-II-$p$ denotes NSGA-II with population size $p$.")
    lines.extend([
        r"    Refer to \Cref{sec:setup} for column description.}",
        r"    \centering",
        r"    \footnotesize",
        r"    \resizebox{0.7\linewidth}{!}{",
        r"    \begin{tabular}{rrlrrrrrrr}",
        r"    \toprule",
        r"    $N$ & $K$ & ~~Method & ~~Width & ~~Time $\downarrow$ & ~~Cardinality $\uparrow$ & ~~Precision $\uparrow$ & ~~IGD $\downarrow$ & ~~$|\hat{\mathcal{Z}}^\star|$ \\",
        r"    \midrule",
    ])

    groups = list(summary.groupby(["n_vars", "n_objs"], sort=False))
    for size_idx, (spec, group) in enumerate(groups):
        n_vars, n_objs = spec
        rows = [
            row
            for row in group.to_dict("records")
            if row["method"] in requested_methods
        ]
        flags = _best_flags(rows)
        block_ends = {end - 1 for _, end in _method_blocks(rows)}
        for idx, (row, row_flags) in enumerate(zip(rows, flags)):
            prefix = "        & & "
            if idx == 0:
                prefix = rf"    \multirow{{{len(rows)}}}{{*}}{{{n_vars}}} & \multirow{{{len(rows)}}}{{*}}{{{n_objs}}} & "

            method = row["method"]
            width = _fmt_int(row["width"])
            time = _fmt_time(row["time"])
            cardinality = _fmt_percent(row["cardinality"])
            precision = _fmt_percent(row["precision"])
            igd = _fmt_igd(row["igd"])
            frontier_size = _fmt_int(row["frontier_size"])

            if method == "Exact":
                method = _latex_textgray("Exact")
                width = _latex_textgray(width)
                time = _latex_textgray(time)
                cardinality = _latex_textgray(cardinality)
                precision = _latex_textgray(precision)
                igd = _latex_textgray(igd)
                frontier_size = _latex_textgray(frontier_size)
            else:
                time = _latex_bold(time, row_flags["time"])
                cardinality = _latex_bold(cardinality, row_flags["cardinality"])
                precision = _latex_bold(precision, row_flags["precision"])
                igd = _latex_bold(igd, row_flags["igd"])

            if _is_restricted_method(method):
                block_size = next(
                    end - start
                    for start, end in _method_blocks(rows)
                    if start <= idx < end
                )
                if idx == next(start for start, end in _method_blocks(rows) if start <= idx < end):
                    method = rf"\multirow{{{block_size}}}{{*}}{{{method}}}"
                else:
                    method = ""

            lines.append(
                rf"{prefix}{method} & {width} & {time} & {cardinality} & {precision} & {igd} & {frontier_size} \\"
            )
            if idx in block_ends and idx != len(rows) - 1:
                lines.append(r"    \cmidrule{3-9}")

        if size_idx != len(groups) - 1:
            lines.extend(["", r"    \midrule", ""])

    lines.extend(
        [
            r"    \bottomrule",
            r"    \end{tabular}}",
            rf"    \label{{tab:kp_result_complete_{label_suffix}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _with_suffix(path, suffix):
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


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
    parser.add_argument("--size", choices=[spec.size for spec in EXPERIMENT_SPECS])
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
    short_output = _with_suffix(args.output, "short")
    long_output = _with_suffix(args.output, "long")

    short_latex = render_latex(
        summary,
        requested_methods=("Exact", r"\noshRule{}", r"\noshFE{}"),
        label_suffix="short",
    )
    long_latex = render_latex(
        summary,
        requested_methods=("Exact", "NSGA-II-100", "NSGA-II-500", r"\noshRule{}", r"\noshFE{}"),
        label_suffix="long",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    short_output.write_text(short_latex)
    long_output.write_text(long_latex)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)

    print(f"Wrote {short_output}")
    print(f"Wrote {long_output}")
    print(f"Wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
