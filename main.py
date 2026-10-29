from pathlib import Path

from src import aggregate as agg
from src import plots
from src.parse import load_runs

IN_DIR = Path("data-in")
OUT_DIR = Path("data-out")
FILE_EXTENSIONS = "pdf"
# FILE_EXTENSIONS = ("pdf", "svg")
SHOW_HEADINGS = True
SHOW_PHASE_DURATIONS = True


def describe(runs) -> None:
    print(f"Loaded {len(runs)} report(s)")

    benches = agg.benchmarks(runs)
    for bench in benches:
        points = agg.benchmark_points(bench)
        size = f"~{points:,}" if points else "unknown"
        plural = "run" if len(bench.runs) == 1 else "runs"
        queries = agg.queries_of(bench)
        print(
            f"\n{bench.slug} ({bench.label}, {bench.dataset}, {size} points, "
            f"{len(bench.runs)} {plural})"
        )
        print(f"  queries: {', '.join(queries) or 'none'}")

    comparisons = agg.comparisons(benches)
    if comparisons:
        print("\ncomparisons:")
        for comparison in comparisons:
            members = ", ".join(bench.slug for bench in comparison.benchmarks)
            print(f"  {comparison.name}: {members}")


def main() -> None:
    if not IN_DIR.exists():
        raise SystemExit(f"input directory not found: {IN_DIR}")

    runs, failed = load_runs(IN_DIR)
    if failed:
        print(f"{len(failed)} file(s) could not be parsed")

    if not runs:
        raise SystemExit(f"no benchmark reports found under {IN_DIR}")

    describe(runs)
    print(f"\nWriting to {OUT_DIR}")
    plots.render(
        runs,
        OUT_DIR,
        show_headings=SHOW_HEADINGS,
        show_phase_durations=SHOW_PHASE_DURATIONS,
        extensions=FILE_EXTENSIONS,
    )


if __name__ == "__main__":
    main()
