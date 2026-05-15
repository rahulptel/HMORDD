"""Create summary tables from post-processed setpacking metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from hmordd import Paths
from hmordd.setpacking.postprocess import EXPERIMENT_SPECS

NOSH_LABEL = r"\noshRule{}"
TRIAL_SEEDS = (7, 8, 9, 10, 11)
ALL_RESULT_COLUMNS = [
    "problem",
    "size",
    "n_vars",
    "n_objs",
    "split",
    "pid",
    "method",
    "setting",
    "run_seed",
    "pop_size",
    "run_time",
    "width",
    "time",
    "cardinality",
    "precision",
    "igd",
    "frontier_size",
    "status",
    "n_exact_pf",
    "n_approx_pf",
    "cardinality_raw",
    "igd_raw",
]


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
    return (
        Paths.sols
        / "setpacking"
        / spec.size
        / args.split
        / "exact"
        / f"pf-{args.pf_enum_method}-dom-{args.dominance}"
    )


def _exact_dd_dir(spec, args):
    return (
        Paths.dds
        / "setpacking"
        / spec.size
        / args.split
        / "exact"
        / f"pf-{args.pf_enum_method}-dom-{args.dominance}"
    )


def _restricted_dir(spec, args, variant):
    return (
        Paths.sols
        / "setpacking"
        / spec.size
        / args.split
        / "restricted"
        / f"pf-{args.pf_enum_method}-dom-{args.dominance}"
        / variant["key"]
    )


def _metrics_restricted_dir(spec, args, variant):
    return (
        Paths.outputs
        / "metrics"
        / "setpacking"
        / spec.size
        / args.split
        / "restricted"
        / f"pf-{args.pf_enum_method}-dom-{args.dominance}"
        / variant["key"]
    )


def _metrics_nsga2_dir(spec, args, nsga2):
    return (
        Paths.outputs
        / "metrics"
        / "setpacking"
        / spec.size
        / args.split
        / "nsga2"
        / nsga2["key"]
    )


def _sols_nsga2_dir(spec, args, nsga2):
    return (
        Paths.sols
        / "setpacking"
        / spec.size
        / args.split
        / "nsga2"
        / nsga2["key"]
    )


def _exact_available_pids(spec, args):
    exact_dir = _exact_dir(spec, args)
    pids = []
    for path in exact_dir.glob("*.npy"):
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
    dd_df = _read_csvs([_exact_dd_dir(spec, args) / f"{pid}.csv" for pid in exact_pids])
    return {
        "method": "Exact",
        "pop_size": None,
        "run_time": None,
        "width": _mean_or_none(dd_df.get("width", pd.Series(dtype=float))),
        "time": _mean_or_none(df.get("total_time", pd.Series(dtype=float))),
        "cardinality": 1.0,
        "precision": 1.0,
        "igd": 0.0,
        "frontier_size": _mean_or_none(df.get("n_exact_pf", pd.Series(dtype=float))),
        "inst.": len(exact_pids),
    }


def _nsga2_summary(spec, args, nsga2, exact_pids):
    metrics_dir = _metrics_nsga2_dir(spec, args, nsga2)
    metrics = _metrics_for_exact_pids(_read_csvs(metrics_dir.glob("*.csv")), exact_pids)

    sols_dir = _sols_nsga2_dir(spec, args, nsga2)
    stats = _filter_pids(_read_csvs(sols_dir.glob("*.csv")), exact_pids)

    return {
        "method": f"NSGA-II-{nsga2['pop_size']}",
        "pop_size": nsga2["pop_size"],
        "run_time": nsga2["run_time"],
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
        "method": NOSH_LABEL,
        "pop_size": None,
        "run_time": None,
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


def _value(row, column, default=None):
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return value


def _all_result_frame(rows):
    return pd.DataFrame(rows, columns=ALL_RESULT_COLUMNS)


def _all_exact_rows(spec, args, exact_pids):
    base = pd.DataFrame({"pid": list(exact_pids)})
    if base.empty:
        return []

    exact_dir = _exact_dir(spec, args)
    stats = _filter_pids(_read_csvs([exact_dir / f"{pid}.csv" for pid in exact_pids]), exact_pids)
    dd_df = _filter_pids(
        _read_csvs([_exact_dd_dir(spec, args) / f"{pid}.csv" for pid in exact_pids]),
        exact_pids,
    )
    if not stats.empty:
        base = base.merge(stats, on="pid", how="left")
    if not dd_df.empty and "width" in dd_df.columns:
        base = base.merge(dd_df[["pid", "width"]], on="pid", how="left")

    rows = []
    for stat in base.to_dict("records"):
        rows.append(
            {
                "problem": "setpacking",
                "size": spec.size,
                "n_vars": spec.n_vars,
                "n_objs": spec.n_objs,
                "split": args.split,
                "pid": _value(stat, "pid"),
                "method": "Exact",
                "setting": "exact",
                "width": _value(stat, "width"),
                "time": _value(stat, "total_time"),
                "cardinality": 1.0,
                "precision": 1.0,
                "igd": 0.0,
                "frontier_size": _value(stat, "n_exact_pf"),
                "status": "ok",
                "n_exact_pf": _value(stat, "n_exact_pf"),
                "n_approx_pf": _value(stat, "n_exact_pf"),
                "cardinality_raw": _value(stat, "n_exact_pf"),
            }
        )
    return rows


def _all_nsga2_rows(spec, args, nsga2, exact_pids):
    base = pd.DataFrame(
        [{"pid": pid, "run_seed": seed} for pid in exact_pids for seed in TRIAL_SEEDS]
    )
    if base.empty:
        return []

    metrics_dir = _metrics_nsga2_dir(spec, args, nsga2)
    metrics = _metrics_for_exact_pids(_read_csvs(metrics_dir.glob("*.csv")), exact_pids)

    sols_dir = _sols_nsga2_dir(spec, args, nsga2)
    stats = _filter_pids(_read_csvs(sols_dir.glob("*.csv")), exact_pids)

    if not metrics.empty:
        metrics = metrics.drop(columns=["time"], errors="ignore")
        base = base.merge(metrics, on=["pid", "run_seed"], how="left")
    if not stats.empty:
        stat_cols = [column for column in ("pid", "run_seed", "time_taken") if column in stats.columns]
        base = base.merge(stats[stat_cols], on=["pid", "run_seed"], how="left")

    rows = []
    for row in base.to_dict("records"):
        rows.append(
            {
                "problem": "setpacking",
                "size": spec.size,
                "n_vars": spec.n_vars,
                "n_objs": spec.n_objs,
                "split": args.split,
                "pid": _value(row, "pid"),
                "method": f"NSGA-II-{nsga2['pop_size']}",
                "setting": nsga2["key"],
                "run_seed": _value(row, "run_seed"),
                "pop_size": nsga2["pop_size"],
                "run_time": nsga2["run_time"],
                "time": _value(row, "time_taken"),
                "cardinality": _value(row, "cardinality"),
                "precision": _value(row, "precision"),
                "igd": _value(row, "igd"),
                "frontier_size": _value(row, "n_approx_pf"),
                "status": _value(row, "status"),
                "n_exact_pf": _value(row, "n_exact_pf"),
                "n_approx_pf": _value(row, "n_approx_pf"),
                "cardinality_raw": _value(row, "cardinality_raw"),
                "igd_raw": _value(row, "igd_raw"),
            }
        )
    return rows


def _all_restricted_rows(spec, args, variant, exact_pids):
    base = pd.DataFrame({"pid": list(exact_pids)})
    if base.empty:
        return []

    metrics_dir = _metrics_restricted_dir(spec, args, variant)
    metrics = _metrics_for_exact_pids(
        _read_csvs([metrics_dir / f"{pid}.csv" for pid in exact_pids]), exact_pids
    )

    sols_dir = _restricted_dir(spec, args, variant)
    stats = _filter_pids(_read_csvs([sols_dir / f"{pid}.csv" for pid in exact_pids]), exact_pids)

    if not metrics.empty:
        metrics = metrics.drop(columns=["time"], errors="ignore")
        base = base.merge(metrics, on="pid", how="left")
    if not stats.empty:
        stat_cols = [column for column in ("pid", "total_time") if column in stats.columns]
        base = base.merge(stats[stat_cols], on="pid", how="left")

    rows = []
    for row in base.to_dict("records"):
        rows.append(
            {
                "problem": "setpacking",
                "size": spec.size,
                "n_vars": spec.n_vars,
                "n_objs": spec.n_objs,
                "split": args.split,
                "pid": _value(row, "pid"),
                "method": NOSH_LABEL,
                "setting": variant["key"],
                "width": variant["width"],
                "time": _value(row, "total_time"),
                "cardinality": _value(row, "cardinality"),
                "precision": _value(row, "precision"),
                "igd": _value(row, "igd"),
                "frontier_size": _value(row, "n_approx_pf"),
                "status": _value(row, "status"),
                "n_exact_pf": _value(row, "n_exact_pf"),
                "n_approx_pf": _value(row, "n_approx_pf"),
                "cardinality_raw": _value(row, "cardinality_raw"),
                "igd_raw": _value(row, "igd_raw"),
            }
        )
    return rows


def build_all_results(args):
    rows = []
    for spec in _selected_specs(args):
        exact_pids = _exact_available_pids(spec, args)
        rows.extend(_all_exact_rows(spec, args, exact_pids))
        for nsga2 in spec.nsga2_variants:
            rows.extend(_all_nsga2_rows(spec, args, nsga2, exact_pids))
        for variant in spec.restricted_variants:
            rows.extend(_all_restricted_rows(spec, args, variant, exact_pids))
    return _all_result_frame(rows)


def _table_columns(summary):
    specs = summary[["n_vars", "n_objs"]].drop_duplicates()
    return [(int(row.n_vars), int(row.n_objs)) for row in specs.itertuples(index=False)]


def _summary_lookup(summary):
    lookup = {}
    for row in summary.to_dict("records"):
        lookup[(int(row["n_vars"]), int(row["n_objs"]), row["method"])] = row
    return lookup


def _methods_for_table(summary, requested_methods):
    methods = []
    for method in summary["method"]:
        if method not in methods:
            methods.append(method)
    return [method for method in requested_methods if method in methods]


def _metric_value(row, metric):
    if row is None:
        return None
    return row.get(metric)


def _format_metric(value, metric):
    if metric in {"width", "frontier_size", "inst."}:
        return _fmt_int(value)
    if metric == "time":
        return _fmt_time(value)
    if metric in {"cardinality", "precision"}:
        return _fmt_percent(value)
    if metric == "igd":
        return _fmt_igd(value)
    raise ValueError(f"Unknown metric '{metric}'")


def _best_by_column(lookup, columns, methods, metric):
    if metric not in {"cardinality", "precision", "igd"}:
        return {}

    reverse = metric in {"cardinality", "precision"}
    best = {}
    for n_vars, n_objs in columns:
        values = []
        for method in methods:
            if method == "Exact":
                continue
            value = _metric_value(lookup.get((n_vars, n_objs, method)), metric)
            if value is not None and not pd.isna(value):
                values.append(float(value))
        if values:
            best[(n_vars, n_objs)] = max(values) if reverse else min(values)
    return best


def _format_row_values(lookup, columns, method, metric, best, *, bold_best=True):
    values = []
    for n_vars, n_objs in columns:
        row = lookup.get((n_vars, n_objs, method))
        value = _metric_value(row, metric)
        rendered = _format_metric(value, metric)
        if method == "Exact":
            rendered = _latex_textgray(rendered)
        else:
            best_value = best.get((n_vars, n_objs))
            is_best = (
                best_value is not None
                and value is not None
                and not pd.isna(value)
                and abs(float(value) - best_value) <= 1e-12
            )
            rendered = _latex_bold(rendered, is_best and bold_best)
        values.append(rendered)
    return values


def render_latex(summary, requested_methods, label_suffix, *, bold_best=True):
    columns = _table_columns(summary)
    lookup = _summary_lookup(summary)
    methods = _methods_for_table(summary, requested_methods)
    n_groups = []
    for n_vars, n_objs in columns:
        if not n_groups or n_groups[-1][0] != n_vars:
            n_groups.append([n_vars, []])
        n_groups[-1][1].append(n_objs)

    col_spec = "ll" + ("r" * len(columns))
    group_headers = ["", ""]
    cmidrules = []
    next_col = 3
    for n_vars, n_objs_values in n_groups:
        group_headers.append(rf"\multicolumn{{{len(n_objs_values)}}}{{c}}{{${'N'} = {n_vars}$}}")
        start_col = next_col
        end_col = next_col + len(n_objs_values) - 1
        cmidrules.append(rf"\cmidrule(lr){{{start_col}-{end_col}}}")
        next_col = end_col + 1
    header_values = [rf"$K={n_objs}$" for _, n_objs in columns]

    lines = [
        r"\begin{table}[htbp!]",
        r"    \caption{\gls*{mospp} results averaged over test instances.",
        r"    Each column corresponds to a specific instance size $(N,K)$.",
    ]
    if any(method.startswith("NSGA-II-") for method in methods):
        lines.append(r"    NSGA-II-$p$ denotes NSGA-II with population size $p$.")
    lines.extend(
        [
            r"    Refer to \Cref{sec:setup} for column description.}",
            r"    \centering",
            r"    \footnotesize",
            r"    \resizebox{\linewidth}{!}{",
            rf"    \begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join(group_headers) + r"\\",
            "".join(cmidrules),
            "Metric & Method & " + " & ".join(header_values) + r" \\",
            r"\midrule",
        ]
    )

    metric_blocks = [
        ("width", "Width"),
        ("time", r"Time $\downarrow$"),
        ("cardinality", r"Cardinality $\uparrow$"),
        ("precision", r"Precision $\uparrow$"),
        ("igd", r"IGD $\downarrow$"),
        ("frontier_size", r"$|\hat{\mathcal{Z}}^\star|$"),
    ]
    for metric_idx, (metric, label) in enumerate(metric_blocks):
        best = _best_by_column(lookup, columns, methods, metric)
        lines.append("")
        for method_idx, method in enumerate(methods):
            metric_label = rf"\multirow{{{len(methods)}}}{{*}}{{{label}}} " if method_idx == 0 else ""
            method_label = _latex_textgray("Exact") if method == "Exact" else method
            values = _format_row_values(
                lookup, columns, method, metric, best, bold_best=bold_best
            )
            lines.append(rf"{metric_label}  & {method_label} & " + " & ".join(values) + r" \\")
        if metric_idx != len(metric_blocks) - 1:
            lines.append(r"\midrule")

    inst_values = []
    for n_vars, n_objs in columns:
        exact_row = lookup.get((n_vars, n_objs, "Exact"))
        inst_values.append(_fmt_int(_metric_value(exact_row, "inst.")))
    lines.extend(["", "Inst. &  & " + " & ".join(inst_values) + r" \\  "])

    lines.extend(
        [
            r"\bottomrule",
            r"    \end{tabular}}",
            rf"    \label{{tab:mis_result_complete_{label_suffix}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _with_suffix(path, suffix):
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX MOSP summary table from postprocessed metrics."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--from-pid", type=int, default=0)
    parser.add_argument("--to-pid", type=int, default=100)
    parser.add_argument("--pf-enum-method", type=int, default=3)
    parser.add_argument("--dominance", type=int, default=0)
    parser.add_argument("--size", choices=[spec.size for spec in EXPERIMENT_SPECS])
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
    parser.add_argument(
        "--all-results-csv",
        type=Path,
        default=Paths.results / "setpacking_all_results.csv",
        help="Path to write the per-instance results CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_summary(args)
    all_results = build_all_results(args)
    short_output = _with_suffix(args.output, "short")
    long_output = _with_suffix(args.output, "long")

    short_latex = render_latex(
        summary,
        requested_methods=("Exact", NOSH_LABEL),
        label_suffix="short",
        bold_best=False,
    )
    long_latex = render_latex(
        summary,
        requested_methods=("Exact", "NSGA-II-100", "NSGA-II-500", NOSH_LABEL),
        label_suffix="long",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    short_output.write_text(short_latex)
    long_output.write_text(long_latex)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    args.all_results_csv.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(args.all_results_csv, index=False)

    print(f"Wrote {short_output}")
    print(f"Wrote {long_output}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.all_results_csv}")


if __name__ == "__main__":
    main()
