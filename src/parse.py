"""Parse the quadtree benchmark reports.

The report schema is fixed and the example report of every bench type defines
it.  A report is ``{"type", "engine", "data"}``; ``data`` carries the metric
keys that apply to that engine plus the per-query result lists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Exactly one of these is set per report; it holds the index build cost.
BUILD_TIME_KEYS = (
    "quadtree construction time",
    "octree construction time",
    "sedona build time",
)

# Exactly one of these is set per report; it holds the index footprint.
SIZE_KEYS = (
    "quadtree size",
    "octree size",
    "table bytes",
)

FULL_KEY = "full_runtime"
PARSE_KEY = "data parse time"
READ_KEY = "data read time"
POINTS_KEY = "num points"

# Per-query result lists: report key -> (query name, {duration key: variant}).
QUERY_KEYS: dict[str, tuple[str, dict[str, str]]] = {
    "knn results": ("knn", {"dur": ""}),
    "range results": ("range", {"dur": ""}),
    "pip results": (
        "pip",
        {"containedDur": "contained", "notContainedDur": "not contained"},
    ),
}

# A whole-batch query reports a single duration instead of a per-query list.
BATCH_KEY = "knn batch runtime"
BATCH_QUERY = "knn batch"
BATCH_VARIANT = "batch"


def to_ns(value: Any) -> float | None:
    """Return the nanoseconds of a ``{secs, nanos}`` duration, else ``None``."""
    if not isinstance(value, dict):
        return None

    secs = value.get("secs")
    nanos = value.get("nanos")
    if isinstance(secs, (int, float)) and isinstance(nanos, (int, float)):
        return float(secs) * 1_000_000_000.0 + float(nanos)

    return None


@dataclass
class Run:
    """A single benchmark report."""

    path: Path
    engine: str
    bench_type: str
    dataset: str
    build_ns: float | None
    build_key: str | None
    size_bytes: int | None
    size_key: str | None
    full_ns: float | None
    parse_ns: float | None
    read_ns: float | None
    num_points: int | None
    series: dict[tuple[str, str], list[float]] = field(default_factory=dict)


def _first_key(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in data:
            return key

    return None


def detect_dataset(doc: dict[str, Any], data: dict[str, Any]) -> str:
    """3D (octree) reports are DWD; 2D (quadtree) reports are SimRa."""
    source = data.get("source points")
    if isinstance(source, list):
        for record in source:
            if isinstance(record, dict) and isinstance(record.get("point"), dict):
                return "DWD" if "z" in record["point"] else "SimRa"

    return "DWD" if "oct" in str(doc.get("type", "")).lower() else "SimRa"


def parse_run(doc: dict[str, Any], path: Path) -> Run:
    data = doc["data"]
    engine = str(doc.get("engine", "") or "")
    bench_type = str(doc.get("type", "") or "")
    dataset = detect_dataset(doc, data)

    build_key = _first_key(data, BUILD_TIME_KEYS)
    size_key = _first_key(data, SIZE_KEYS)

    series: dict[tuple[str, str], list[float]] = {}

    for report_key, (query, variants) in QUERY_KEYS.items():
        records = data.get(report_key)
        if not isinstance(records, list):
            continue

        for variant_key, variant in variants.items():
            samples = [
                ns
                for record in records
                if isinstance(record, dict)
                for ns in [to_ns(record.get(variant_key))]
                if ns is not None
            ]
            if samples:
                series[(query, variant)] = samples

    batch_ns = to_ns(data.get(BATCH_KEY))
    if batch_ns is not None:
        series[(BATCH_QUERY, BATCH_VARIANT)] = [batch_ns]

    num_points = data.get(POINTS_KEY)
    if not isinstance(num_points, int) or isinstance(num_points, bool):
        raise ValueError(f"{path}: missing or invalid {POINTS_KEY!r}")

    return Run(
        path=path,
        engine=engine,
        bench_type=bench_type,
        dataset=dataset,
        build_ns=to_ns(data.get(build_key)) if build_key else None,
        build_key=build_key,
        size_bytes=data.get(size_key) if size_key else None,
        size_key=size_key,
        full_ns=to_ns(data.get(FULL_KEY)),
        parse_ns=to_ns(data.get(PARSE_KEY)),
        read_ns=to_ns(data.get(READ_KEY)),
        num_points=num_points,
        series=series,
    )


def load_runs(root: Path) -> tuple[list[Run], list[Path]]:
    """Parse every ``*.json`` below *root*; return the runs and failed paths."""
    runs: list[Run] = []
    failed: list[Path] = []

    for path in sorted(root.rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"  skip {path}: {error}")
            failed.append(path)
            continue

        if not isinstance(doc, dict) or not isinstance(doc.get("data"), dict):
            print(f"  skip {path}: not a benchmark report")
            failed.append(path)
            continue

        runs.append(parse_run(doc, path))

    return runs, failed
