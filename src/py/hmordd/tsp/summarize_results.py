"""Create summary tables from post-processed TSP metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from hmordd import Paths
from hmordd.tsp.postprocess import EXPERIMENT_SPECS

TRIAL_SEEDS = (7, 8, 9, 10, 11)
RESTRICTED_WIDTH = 4804
TSP_METHOD_ORDER = (
    "Exact",
    "NSGA-II-100",
    "NSGA-II-500",
    r"\texttt{OrdMeanHigh}",
    r"\texttt{OrdMeanLow}",
    r"\texttt{OrdMaxHigh}",
    r"\texttt{OrdMaxLow}",
    r"\texttt{OrdMinHigh}",
    r"\texttt{OrdMinLow}",
    r"\noshEE{}",
)


def _read_csvs(paths):
    frames = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
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


def _first_valid_mean(df, columns):
    for column in columns:
        if column not in df.columns:
            continue
        value = _mean_or_none(df[column])
        if value is not None and value > 0:
            return value
    return None


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


def _method_label(method):
    if method == "E2E":
        return r"\noshEE{}"
    if method.startswith("Ord"):
        return rf"\texttt{{{method}}}"
    return method


def _exact_dir(spec, args):
    return (
        Paths.sols
        / "tsp"
        / spec.size
        / args.split
        / "exact"
        / f"pf-{args.pf_enum_method}-trackx-{args.track_x}"
    )


def _exact_dd_dir(spec, args):
    return Paths.dds / "tsp" / spec.size / args.split / "exact"


def _restricted_dir(spec, args, variant):
    return (
        Paths.sols
        / "tsp"
        / spec.size
        / args.split
        / "restricted"
        / f"pf-{args.pf_enum_method}-trackx-{args.track_x}"
        / variant["nosh"]
    )


def _metrics_restricted_dir(spec, args, variant):
    return (
        Paths.outputs
        / "metrics"
        / "tsp"
        / spec.size
        / args.split
        / "restricted"
        / f"pf-{args.pf_enum_method}-trackx-{args.track_x}"
        / variant["nosh"]
    )


def _metrics_nsga2_dir(spec, args, nsga2):
    return (
        Paths.outputs
        / "metrics"
        / "tsp"
        / spec.size
        / args.split
        / "nsga2"
        / nsga2["key"]
    )


def _sols_nsga2_dir(spec, args, nsga2):
    return (
        Paths.sols
        / "tsp"
        / spec.size
        / args.split
        / "nsga2"
        / nsga2["key"]
    )


def _dd_width(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open() as handle:
            layers = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    if not layers:
        return None
    return max(len(layer) for layer in layers)


def _exact_width(spec, args, exact_pids):
    for pid in exact_pids:
        width = _dd_width(_exact_dd_dir(spec, args) / f"{pid}.json")
        if width is not None:
            return width
    return None


def _exact_available_pids(spec, args):
    exact_dir = _exact_dir(spec, args)
    pids = []
    for path in exact_dir.glob("*.npz"):
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
    exact_dir = _exact_dir(spec, args)
    df = _read_csvs([exact_dir / f"{pid}.csv" for pid in exact_pids])
    return {
        "method": "Exact",
        "setting": None,
        "width": _exact_width(spec, args, exact_pids),
        "time": _mean_or_none(df.get("total_time", pd.Series(dtype=float))),
        "cardinality": 1.0,
        "precision": 1.0,
        "igd": 0.0,
        "frontier_size": _first_valid_mean(df, ("n_exact_pf", "n_approx_pf")),
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

    sols_dir = _sols_nsga2_dir(spec, args, nsga2)
    sols_paths = [
        sols_dir / f"{pid}-{seed}.csv"
        for pid in exact_pids
        for seed in TRIAL_SEEDS
    ]
    stats = _filter_pids(_read_csvs(sols_paths), exact_pids)

    return {
        "method": f"NSGA-II-{nsga2['pop_size']}",
        "setting": nsga2["key"],
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
        _read_csvs([metrics_dir / f"{pid}.csv" for pid in exact_pids]), exact_pids
    )

    sols_dir = _restricted_dir(spec, args, variant)
    stats = _filter_pids(_read_csvs([sols_dir / f"{pid}.csv" for pid in exact_pids]), exact_pids)

    return {
        "method": _method_label(variant["key"]),
        "setting": variant["key"],
        "width": variant.get("width", RESTRICTED_WIDTH),
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
            {
                "n_vars": spec.n_vars,
                "n_objs": spec.n_objs,
                **_exact_summary(spec, args, exact_pids),
            }
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
    restricted = [
        row
        for row in rows
        if row["method"].startswith(r"\texttt{Ord") or row["method"] == r"\noshEE{}"
    ]

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
        is_restricted = row in restricted
        for metric, value in best.items():
            row_flags[metric] = (
                value is not None
                and is_restricted
                and row[metric] is not None
                and not pd.isna(row[metric])
                and abs(row[metric] - value) <= 1e-12
            )
        flags.append(row_flags)
    return flags


def _ordered_rows(group, requested_methods):
    rows_by_method = {row["method"]: row for row in group.to_dict("records")}
    return [rows_by_method[method] for method in requested_methods if method in rows_by_method]


def render_latex(summary, requested_methods, label_suffix):
    lines = [
        r"\begin{table}[htbp!]",
        r"    \caption{\gls*{motsp} results averaged across test instances.",
    ]
    if any(method.startswith(r"\texttt{Ord") for method in requested_methods):
        lines.append(r"    Methods prefixed with \texttt{Ord} correspond to different rule-based NOSHs.")
    if any(method.startswith("NSGA-II-") for method in requested_methods):
        lines.append(r"    NSGA-II-$p$ denotes NSGA-II with population size $p$.")
    lines.extend([
        r"    Refer to \Cref{sec:setup} for column description.}",
        r"    \centering",
        r"    \footnotesize",
        r"    \resizebox{0.7\linewidth}{!}{",
        r"    \begin{tabular}{rrlrrrrrr}",
        r"    \toprule",
        r"    $N$ & $K$ & ~~Method & ~~Width & ~~Time $\downarrow$ & ~~Cardinality $\uparrow$& ~~Precision $\uparrow$& ~~IGD $\downarrow$& ~~$|\hat{\mathcal{Z}}^\star|$ \\",
        r"    \midrule",
    ])

    n_groups = list(summary.groupby("n_vars", sort=False))
    for n_idx, (n_vars, n_group) in enumerate(n_groups):
        k_groups = list(n_group.groupby("n_objs", sort=False))
        k_rows = [(n_objs, _ordered_rows(group, requested_methods)) for n_objs, group in k_groups]
        n_row_count = sum(len(rows) for _, rows in k_rows)
        first_n_row = True

        for k_idx, (n_objs, rows) in enumerate(k_rows):
            flags = _best_flags(rows)
            for row_idx, (row, row_flags) in enumerate(zip(rows, flags)):
                if row_idx == 0:
                    n_cell = rf"\multirow{{{n_row_count}}}{{*}}{{{n_vars}}}" if first_n_row else ""
                    prefix = rf"    {n_cell} & \multirow{{{len(rows)}}}{{*}}{{{n_objs}}} & "
                    first_n_row = False
                else:
                    prefix = "      &  & "

                method = row["method"]
                width = _fmt_int(row["width"])
                time = _fmt_time(row["time"])
                cardinality = _fmt_percent(row["cardinality"])
                precision = _fmt_percent(row["precision"])
                igd = _fmt_igd(row["igd"])
                frontier_size = _fmt_int(row["frontier_size"])

                if method == "Exact":
                    method = _latex_textgray(method)
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

                lines.append(
                    rf"{prefix}{method} & {width} & {time} & {cardinality} & {precision} & {igd} & {frontier_size} \\"
                )

            if k_idx != len(k_rows) - 1:
                lines.append(r"    \cmidrule{2-9}")

        if n_idx != len(n_groups) - 1:
            lines.extend(["", r"    \midrule", ""])

    lines.extend(
        [
            r"    \bottomrule",
            r"    \end{tabular}}  ",
            rf"    \label{{tab:tsp_result_complete_{label_suffix}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _with_suffix(path, suffix):
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX MOTSP summary table from postprocessed metrics."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--from-pid", type=int, default=1100)
    parser.add_argument("--to-pid", type=int, default=1200)
    parser.add_argument("--pf-enum-method", type=int, default=3)
    parser.add_argument("--track-x", type=int, default=0)
    parser.add_argument("--size", choices=[spec.size for spec in EXPERIMENT_SPECS])
    parser.add_argument(
        "--output",
        type=Path,
        default=Paths.results / "tsp_summary.tex",
        help="Path to write the LaTeX table.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Paths.results / "tsp_summary.csv",
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
        requested_methods=("Exact", r"\noshEE{}"),
        label_suffix="short",
    )
    long_latex = render_latex(
        summary,
        requested_methods=TSP_METHOD_ORDER,
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
