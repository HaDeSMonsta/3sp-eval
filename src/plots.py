from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatter

from . import aggregate as agg
from .parse import BATCH_QUERY, BATCH_VARIANT, Run

BUILD_LABEL = "Index build time (ms)"
SIZE_LABEL = "Index size (MiB)"
LATENCY_LABEL = "Query latency (ms)"
TIME_LABEL = "Elapsed time (s)"

PHASE_COLORS = {
    "data read": "#1f77b4",
    "data parse": "#ff7f0e",
    "index build": "#2ca02c",
    "knn": "#d62728",
    "range": "#9467bd",
    "pip (contained)": "#8c564b",
    "pip (not contained)": "#e377c2",
    "knn batch": "#7f7f7f",
}

LOADING_PHASES = ("data read", "data parse")
NON_QUERY_PHASES = LOADING_PHASES + ("index build",)
SYSTEM_ORDER = ("sedona", "fastquadtree", "quadtree-quad", "quadtree-oct")
PHASE_LABELS = {
    "data read": "Data read",
    "data parse": "Data parse",
    "index build": "Index build",
    "knn": "KNN",
    "range": "Range",
    "pip (contained)": "PiP (contained)",
    "pip (not contained)": "PiP (not contained)",
    "knn batch": "KNN batch",
}


def _register_external_fonts() -> None:
    """Register fonts from the shell (``QUADTREE_FONT_DIR``), if provided."""
    font_dir = os.environ.get("QUADTREE_FONT_DIR")
    if not font_dir:
        return

    for path in sorted(Path(font_dir).glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))


def setup_style() -> None:
    _register_external_fonts()
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Liberation Serif",
                "Nimbus Roman",
                "FreeSerif",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def palette_for(benchmarks: list[agg.Benchmark]) -> dict[str, tuple[float, float, float]]:
    slugs = [bench.slug for bench in benchmarks]
    colors = sns.color_palette("tab10", max(len(slugs), 1))
    return dict(zip(slugs, colors, strict=False))


def _save(fig: plt.Figure, out_dir: Path, name: str, extensions: tuple[str, ...] | str) -> None:
    if isinstance(extensions, str):
        extensions = (extensions,)

    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in extensions:
        path = out_dir / f"{name}.{extension}"
        fig.savefig(path)
        print(f"  saved {path}")


def _slug(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace(".", "")


def _decimal(value: float) -> str:
    """Plain decimal string, never scientific notation (e.g. ``0.0003``)."""
    if value == 0 or not np.isfinite(value):
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    return f"{value:,.{max(0, -magnitude)}f}"


class _DecimalLogFormatter(LogFormatter):
    """A log formatter that prints its ticks as plain decimals."""

    def _num_to_string(self, x, vmin, vmax):  # noqa: ANN001 - matplotlib hook
        return _decimal(x)


def _log_formatter(axis, *, x: bool = False, y: bool = False) -> None:
    if x:
        axis.xaxis.set_major_formatter(_DecimalLogFormatter())
        axis.xaxis.set_minor_formatter(_DecimalLogFormatter())
    if y:
        axis.yaxis.set_major_formatter(_DecimalLogFormatter())
        axis.yaxis.set_minor_formatter(_DecimalLogFormatter())


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    if value == 0:
        return "0"

    digits = max(0, 2 - math.floor(math.log10(abs(value))))
    return f"{value:,.{digits}f}"


def _case_titles(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """One (dataset, title) per facet, annotated with the point count."""
    titles: list[tuple[str, str]] = []
    for dataset, group in frame.groupby("dataset", sort=False):
        points = group["points"].dropna()
        suffix = f" - {agg.points_label(int(points.max()))} points" if not points.empty else ""
        titles.append((str(dataset), f"{dataset}{suffix}"))

    return titles


def _memory_figure(
    summary: pd.DataFrame,
    palette: dict,
    *,
    xticks: dict[str, str],
    title: str,
    show_headings: bool,
) -> plt.Figure:
    sub = summary.sort_values("slug", key=lambda slugs: slugs.map(SYSTEM_ORDER.index)).reset_index(drop=True)
    colors = [palette.get(slug, "grey") for slug in sub["slug"]]
    positions = np.arange(len(sub))
    values = sub["size_mib"].to_numpy(dtype=float)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(positions, values, color=colors)
    for position, value in zip(positions, values):
        if np.isfinite(value):
            axis.text(position, value, _fmt(value), ha="center", va="bottom", fontsize=10)

    axis.set_xticks(positions)
    axis.set_xticklabels([xticks.get(slug, slug) for slug in sub["slug"]], rotation=15, ha="right")
    axis.set_ylabel(SIZE_LABEL)
    axis.set_yscale("log")
    _log_formatter(axis, y=True)
    if show_headings:
        axis.set_title(title)

    fig.tight_layout()

    return fig


def _render_memory(
    benchmarks: list[agg.Benchmark],
    out_dir: Path,
    name: str,
    title: str,
    palette: dict,
    xticks: dict[str, str],
    *,
    show_headings: bool,
    extensions: tuple[str, ...],
) -> None:
    summary = agg.reference_rows(agg.build_summary(benchmarks))
    if summary.empty:
        return

    fig = _memory_figure(
        summary,
        palette,
        xticks=xticks,
        title=title,
        show_headings=show_headings,
    )
    _save(fig, out_dir, name, extensions)
    plt.close(fig)


def _timestamp(value: float, total: float) -> str:
    if value == 0:
        return "0"
    digits = 3 if total < 10 else 2 if total < 100 else 1
    return f"{value:,.{digits}f}"


def _duration(value: float) -> str:
    if value == 0:
        return "0 s"
    if value >= 1:
        return f"{value:.3g} s"
    if value >= 1e-3:
        return f"{value * 1e3:.3g} ms"
    return f"{value * 1e6:.3g} µs"


def _phase_figure(
    phases: list[tuple[str, float]],
    *,
    title: str,
    show_headings: bool,
    show_durations: bool,
) -> plt.Figure:
    labels = [label for label, _ in phases]
    durations = np.array([duration for _, duration in phases]) / 1000.0
    starts = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
    total = float(durations.sum())
    positions = np.arange(len(phases))
    colors = [PHASE_COLORS.get(label, "grey") for label in labels]
    tick_labels = [
        f"{PHASE_LABELS[label]}\n{_duration(duration)}" if show_durations else PHASE_LABELS[label]
        for label, duration in zip(labels, durations)
    ]

    fig, axis = plt.subplots(figsize=(11, 0.55 * len(phases) + 1.9))
    axis.barh(positions, durations, left=starts, height=0.5, color=colors)
    for start, position in zip(starts, positions):
        axis.axvline(start, color="0.35", linestyle=":", linewidth=1)
        axis.text(
            start,
            position + 0.31,
            _timestamp(start, total),
            ha="left",
            va="bottom",
            fontsize=8,
            color="0.25",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85},
        )

    axis.set_yticks(positions)
    axis.set_yticklabels(tick_labels)
    axis.set_xlim(0, total * 1.03)
    axis.set_xlabel(TIME_LABEL)
    axis.set_ylabel("Phase")
    if show_headings:
        axis.set_title(title)

    fig.tight_layout()

    return fig


def _query_figure(
    rows: list[tuple[str, str, float, float]],
    palette: dict,
    *,
    title: str,
    show_headings: bool,
) -> plt.Figure:
    positions = np.arange(len(rows))
    values = np.array([value for _, _, value, _ in rows])
    medians = np.array([median for _, _, _, median in rows])
    colors = [palette.get(slug, "grey") for slug, *_ in rows]
    tick_labels = [label for _, label, *_ in rows]

    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(positions, values, color=colors)
    axis.hlines(medians, positions - 0.3, positions + 0.3, color="0.1", linewidth=1.6, zorder=3)
    for position, value in zip(positions, values):
        axis.text(
            position,
            value,
            _fmt(value),
            ha="center",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85},
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(tick_labels, rotation=15, ha="right")
    axis.set_ylabel(LATENCY_LABEL)
    axis.set_yscale("log")
    _log_formatter(axis, y=True)
    axis.legend(
        handles=[
            Patch(facecolor="0.75", edgecolor="0.4", label="Mean"),
            Line2D([], [], color="0.1", linewidth=1.6, label="Median"),
        ],
        loc="best",
        frameon=False,
    )
    if show_headings:
        axis.set_title(title)

    fig.tight_layout()

    return fig


def _render_query(
    rows: list[tuple[str, str, float, float]],
    out_dir: Path,
    name: str,
    title: str,
    palette: dict,
    *,
    show_headings: bool,
    extensions: tuple[str, ...],
) -> None:
    fig = _query_figure(
        rows,
        palette,
        title=title,
        show_headings=show_headings,
    )
    _save(fig, out_dir, name, extensions)
    plt.close(fig)


def _render_phases(
    phases: list[tuple[str, float]],
    out_dir: Path,
    name: str,
    title: str,
    *,
    show_headings: bool,
    show_durations: bool,
    extensions: tuple[str, ...],
) -> None:
    if not phases:
        return

    fig = _phase_figure(
        phases,
        title=title,
        show_headings=show_headings,
        show_durations=show_durations,
    )
    _save(fig, out_dir, name, extensions)
    plt.close(fig)


def _overview_figure(benches: list[agg.Benchmark], reports: int) -> plt.Figure:
    datasets: dict[str, list[agg.Benchmark]] = {}
    for bench in benches:
        datasets.setdefault(bench.dataset, []).append(bench)

    lines = ["quadtree-eval overview", ""]
    lines.append(f"reports: {reports}")
    lines.append(f"benchmarks: {len(benches)}")
    lines.append(f"datasets: {len(datasets)}")

    for dataset, group in datasets.items():
        dims = sorted({bench.dim for bench in group})
        points = sorted(
            {
                agg.benchmark_points(bench)
                for bench in group
                if agg.benchmark_points(bench) is not None
            }
        )
        lines += ["", f"{dataset} ({', '.join(f'{dim}D' for dim in dims)})"]
        if points:
            lines.append("  points: " + ", ".join(f"{count:,}" for count in points))
        for bench in sorted(group, key=lambda bench: SYSTEM_ORDER.index(bench.slug)):
            queries = ", ".join(agg.queries_of(bench))
            lines.append(f"  {bench.engine:<18s} {len(bench.runs):>2d} runs  {queries}")

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(
        0.06,
        0.96,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        linespacing=1.4,
    )

    return fig


def render(
    runs: list[Run],
    out_dir: Path,
    *,
    show_headings: bool,
    show_phase_durations: bool,
    extensions: tuple[str, ...],
) -> None:
    setup_style()
    benches = agg.benchmarks(runs)
    palette = palette_for(benches)

    xticks = {bench.slug: f"{bench.label}\n{bench.dataset}" for bench in benches}

    fig = _overview_figure(benches, len(runs))
    _save(fig, out_dir, "overview", extensions)
    plt.close(fig)

    _render_memory(
        benches,
        out_dir,
        "memory",
        "Index Memory",
        palette,
        xticks,
        show_headings=show_headings,
        extensions=extensions,
    )

    simra = [bench for bench in benches if bench.dataset == "SimRa"]
    _render_memory(
        simra,
        out_dir,
        "memory_simra",
        "Index Memory - SimRa",
        palette,
        xticks,
        show_headings=show_headings,
        extensions=extensions,
    )

    ordered = sorted(benches, key=lambda bench: SYSTEM_ORDER.index(bench.slug))
    for query, variant in agg.query_keys():
        key = (query, variant)
        rows = []
        for bench in ordered:
            summary = agg.query_summary(bench, key)
            if summary is not None:
                rows.append((bench.slug, xticks[bench.slug], *summary))
                if key == ("knn", ""):
                    batch = agg.query_summary(bench, (BATCH_QUERY, BATCH_VARIANT))
                    if batch is not None:
                        rows.append(
                            (
                                bench.slug,
                                f"{bench.label} (batch total)\n{bench.dataset}",
                                *batch,
                            )
                        )
        if len(rows) < 2:
            continue

        if variant and variant != BATCH_VARIANT:
            label = f"{query} ({variant})"
            parts = [query, variant]
        else:
            label = query
            parts = [query]
        name = "latency_" + "_".join(part.replace(" ", "_") for part in parts)
        _render_query(
            rows,
            out_dir,
            name,
            f"Query Latency - {PHASE_LABELS[label]}",
            palette,
            show_headings=show_headings,
            extensions=extensions,
        )

    pip_keys = (("pip", "contained"), ("pip", "not contained"))
    rows = []
    for bench in ordered:
        summary = agg.query_summary(bench, *pip_keys)
        if summary is not None:
            rows.append((bench.slug, xticks[bench.slug], *summary))
    if len(rows) >= 2:
        _render_query(
            rows,
            out_dir,
            "latency_pip",
            "Query Latency - PiP",
            palette,
            show_headings=show_headings,
            extensions=extensions,
        )

    for bench in benches:
        base = f"{bench.label} - {bench.dataset}"
        phases = agg.phase_durations(bench)
        bench_dir = out_dir / bench.slug

        _render_phases(
            [phase for phase in phases if phase[0] not in LOADING_PHASES],
            bench_dir,
            "phases",
            f"{base} - Phase Timings",
            show_headings=show_headings,
            show_durations=show_phase_durations,
            extensions=extensions,
        )
        _render_phases(
            phases,
            bench_dir,
            "phases_data",
            f"{base} - Phase Timings with Data Loading",
            show_headings=show_headings,
            show_durations=show_phase_durations,
            extensions=extensions,
        )
        _render_phases(
            [phase for phase in phases if phase[0] not in NON_QUERY_PHASES],
            bench_dir,
            "phases-query",
            f"{base} - Query Timings",
            show_headings=show_headings,
            show_durations=show_phase_durations,
            extensions=extensions,
        )
