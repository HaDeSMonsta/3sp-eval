"""Group parsed reports into benchmarks and turn them into tidy data frames.

A *benchmark* is one engine/type pair (e.g. the Rust ``quadtree`` running
``BuildQuad``); every report of that pair is one run.  A *comparison* is a
group of benchmarks drawn on the same figures: all 2D benchmarks against each
other, and the quad/octree pair of an engine measured in both dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

import pandas as pd

from .parse import BATCH_QUERY, BATCH_VARIANT, QUERY_KEYS, Run

NANOS_PER_MS = 1_000_000.0
BYTES_PER_MIB = 1024.0 * 1024.0

# Engines whose report name is not a good directory name.
ENGINE_SLUGS = {"sedonadb": "sedona"}
ENGINE_DIM_LABELS = {("quadtree", 2): "QuadTree", ("quadtree", 3): "OcTree"}
DIM_SLUG = {2: "quad", 3: "oct"}
DIM_LABEL = {2: "quad", 3: "octree"}

QUERY_ORDER = ("knn", "range", "pip", "knn batch")


def bench_dim(bench_type: str) -> int:
    return 3 if "oct" in bench_type.lower() else 2


def engine_slug(engine: str) -> str:
    head = engine.strip().split(" ")[0].lower()
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in head).strip("-")
    return ENGINE_SLUGS.get(cleaned, cleaned or "unknown")


@dataclass(frozen=True)
class Benchmark:
    """One engine/type pair and every run measured for it."""

    key: str
    engine: str
    bench_type: str
    dim: int
    dataset: str
    slug: str
    label: str
    runs: tuple[Run, ...]


@dataclass(frozen=True)
class Comparison:
    """A set of benchmarks drawn together on shared figures."""

    name: str
    label: str
    benchmarks: tuple[Benchmark, ...]


def _dominant_dataset(runs: tuple[Run, ...]) -> str:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.dataset] = counts.get(run.dataset, 0) + 1
    return max(counts, key=lambda dataset: counts[dataset])


def benchmarks(runs: list[Run]) -> list[Benchmark]:
    """Collapse the runs into one :class:`Benchmark` per engine/type pair."""
    grouped: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        grouped.setdefault((run.engine, run.bench_type), []).append(run)

    types_per_engine: dict[str, set[str]] = {}
    for engine, bench_type in grouped:
        types_per_engine.setdefault(engine, set()).add(bench_type)

    result: list[Benchmark] = []
    for (engine, bench_type), group in grouped.items():
        dim = bench_dim(bench_type)
        shared = len(types_per_engine[engine]) > 1
        slug = engine_slug(engine)
        label = engine.strip().split(" ")[0]
        if shared:
            slug = f"{slug}-{DIM_SLUG[dim]}"
            label = ENGINE_DIM_LABELS.get((engine, dim), f"{label} ({DIM_LABEL[dim]})")

        result.append(
            Benchmark(
                key=f"{engine}\x00{bench_type}",
                engine=engine,
                bench_type=bench_type,
                dim=dim,
                dataset=_dominant_dataset(tuple(group)),
                slug=slug,
                label=label,
                runs=tuple(group),
            )
        )

    return sorted(result, key=lambda bench: bench.slug)


def comparisons(benchmarks: list[Benchmark]) -> list[Comparison]:
    """The comparison groups: the 2D field, and any engine spanning both dims."""
    result: list[Comparison] = []

    two_d = [bench for bench in benchmarks if bench.dim == 2]
    if len(two_d) >= 2:
        result.append(Comparison("2d", "2D benchmarks", tuple(two_d)))

    by_engine: dict[str, list[Benchmark]] = {}
    for bench in benchmarks:
        by_engine.setdefault(bench.engine, []).append(bench)
    for group in by_engine.values():
        dims = {bench.dim for bench in group}
        if len(dims) >= 2:
            name = "_".join(DIM_SLUG[dim] for dim in sorted(dims))
            label = " vs ".join(DIM_LABEL[dim] for dim in sorted(dims))
            result.append(Comparison(name, label, tuple(sorted(group, key=lambda b: b.dim))))

    return result


def benchmark_points(benchmark: Benchmark) -> int | None:
    counts = [run.num_points for run in benchmark.runs if run.num_points]
    return max(counts) if counts else None


def queries_of(benchmark: Benchmark) -> list[str]:
    present = {query for run in benchmark.runs for query, _ in run.series}
    return _order_queries(present)


def phase_durations(benchmark: Benchmark) -> list[tuple[str, float]]:
    """Mean milliseconds spent in each measured phase, in execution order."""
    phases: list[tuple[str, list[float]]] = []

    for attribute, label in (
        ("read_ns", "data read"),
        ("parse_ns", "data parse"),
        ("build_ns", "index build"),
    ):
        values = [getattr(run, attribute) for run in benchmark.runs]
        values = [value for value in values if value is not None]
        if values:
            phases.append((label, values))

    for query, variant in query_keys():
        values = [
            sum(run.series[(query, variant)])
            for run in benchmark.runs
            if (query, variant) in run.series
        ]
        if not values:
            continue
        label = query if variant in ("", BATCH_VARIANT) else f"{query} ({variant})"
        phases.append((label, values))

    return [(label, sum(values) / len(values) / NANOS_PER_MS) for label, values in phases]


def query_keys() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for query, variants in QUERY_KEYS.values():
        for variant in variants.values():
            keys.append((query, variant))
            if query == "knn":
                keys.append((BATCH_QUERY, BATCH_VARIANT))

    return keys


def query_summary(
    benchmark: Benchmark, *keys: tuple[str, str]
) -> tuple[float, float] | None:
    """Mean and median of the per-run samples for the given query keys, in ms."""
    samples = [
        [ns for key in keys if key in run.series for ns in run.series[key]]
        for run in benchmark.runs
        if any(key in run.series for key in keys)
    ]
    samples = [run_samples for run_samples in samples if run_samples]
    if not samples:
        return None

    means = [mean(run_samples) / NANOS_PER_MS for run_samples in samples]
    medians = [median(run_samples) / NANOS_PER_MS for run_samples in samples]

    return mean(means), median(medians)


def queries(benchmarks: list[Benchmark]) -> list[str]:
    present = {query for bench in benchmarks for query in queries_of(bench)}
    return _order_queries(present)


def _order_queries(present: set[str]) -> list[str]:
    ordered = [query for query in QUERY_ORDER if query in present]
    return ordered + sorted(present - set(ordered))


def points_label(points: int | None) -> str:
    """Compact approximate point count, e.g. ``~370k`` or ``~230M``."""
    if not points:
        return "?"
    if points >= 1_000_000_000:
        return f"~{points / 1_000_000_000:.0f}B"
    if points >= 1_000_000:
        return f"~{points / 1_000_000:.0f}M"
    if points >= 1_000:
        return f"~{points / 1_000:.0f}k"
    return f"~{points}"


def build_frame(benchmarks: list[Benchmark]) -> pd.DataFrame:
    """One row per run with its index metrics."""
    rows = []
    for bench in benchmarks:
        for run in bench.runs:
            rows.append(
                {
                    "dataset": run.dataset,
                    "dim": bench.dim,
                    "slug": bench.slug,
                    "system": bench.label,
                    "points": run.num_points,
                    "build_ns": run.build_ns,
                    "size_bytes": run.size_bytes,
                    "parse_ns": run.parse_ns,
                    "read_ns": run.read_ns,
                    "full_ns": run.full_ns,
                    "path": str(run.path),
                }
            )

    return pd.DataFrame(rows)


def latency_frame(benchmarks: list[Benchmark]) -> pd.DataFrame:
    """One row per executed query."""
    rows = []
    for bench in benchmarks:
        for run in bench.runs:
            for (query, variant), samples in run.series.items():
                for ns in samples:
                    rows.append(
                        {
                            "dataset": run.dataset,
                            "dim": bench.dim,
                            "slug": bench.slug,
                            "system": bench.label,
                            "points": run.num_points,
                            "query": query,
                            "variant": variant,
                            "ns": ns,
                            "ms": ns / NANOS_PER_MS,
                        }
                    )

    return pd.DataFrame(rows)


def build_summary(benchmarks: list[Benchmark]) -> pd.DataFrame:
    """Aggregate build time and index size per run size of each benchmark."""
    frame = build_frame(benchmarks)
    if frame.empty:
        return frame

    summary = (
        frame.groupby(["dataset", "dim", "slug", "system", "points"], dropna=False)
        .agg(
            build_mean=("build_ns", "mean"),
            build_min=("build_ns", "min"),
            build_max=("build_ns", "max"),
            size_mean=("size_bytes", "mean"),
            size_min=("size_bytes", "min"),
            size_max=("size_bytes", "max"),
            runs=("path", "count"),
        )
        .reset_index()
    )

    summary["build_ms"] = summary["build_mean"] / NANOS_PER_MS
    summary["build_min_ms"] = summary["build_min"] / NANOS_PER_MS
    summary["build_max_ms"] = summary["build_max"] / NANOS_PER_MS
    summary["size_mib"] = summary["size_mean"] / BYTES_PER_MIB
    summary["size_min_mib"] = summary["size_min"] / BYTES_PER_MIB
    summary["size_max_mib"] = summary["size_max"] / BYTES_PER_MIB

    return summary


def reference_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """The largest measured size of each benchmark, for single-bar figures."""
    if summary.empty:
        return summary

    ordered = summary.sort_values("points", na_position="first")
    return ordered.groupby("slug", as_index=False).last()
