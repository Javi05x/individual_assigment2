"""Controller for isolated dense matrix memory measurements."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


BASE_SEED = 20261028
MEASURED_RUNS = 5

ALGORITHMS = (
    "basic_ijk",
    "row_oriented_ikj",
    "tiled_ikj_32",
)


def run_isolated_measurement(
    algorithm: str,
    size: int,
    seed: int,
) -> dict[str, object]:
    """Execute one memory measurement in a fresh Python process."""
    command = [
        sys.executable,
        "-m",
        "python.src.matrix.benchmark.dense_memory_worker",
        "--algorithm",
        algorithm,
        "--size",
        str(size),
        "--seed",
        str(seed),
    ]

    completed_process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    standard_output = completed_process.stdout.strip()

    if not standard_output:
        raise RuntimeError(
            f"Memory worker returned no output for "
            f"{algorithm}, size {size}"
        )

    try:
        measurement = json.loads(standard_output)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Memory worker did not return valid JSON:\n"
            f"{standard_output}"
        ) from error

    return measurement


def validate_measurement(
    measurement: dict[str, object],
    expected_algorithm: str,
    expected_size: int,
    expected_seed: int,
) -> None:
    """Validate metadata and numerical relationships in one measurement."""
    if measurement["algorithm"] != expected_algorithm:
        raise RuntimeError("Worker returned an unexpected algorithm")

    if int(measurement["size"]) != expected_size:
        raise RuntimeError("Worker returned an unexpected matrix size")

    if int(measurement["seed"]) != expected_seed:
        raise RuntimeError("Worker returned an unexpected seed")

    baseline_rss = float(measurement["baseline_rss_mb"])
    peak_rss = float(measurement["peak_rss_mb"])
    final_rss = float(measurement["final_rss_mb"])
    peak_delta = float(measurement["peak_delta_mb"])

    if peak_rss < baseline_rss:
        raise RuntimeError("Peak RSS is lower than baseline RSS")

    if peak_rss < final_rss:
        raise RuntimeError("Peak RSS is lower than final RSS")

    if peak_delta < 0:
        raise RuntimeError("Peak RSS delta is negative")


def validate_checksums(
    measurements: list[dict[str, object]],
    size: int,
) -> None:
    """Verify that all algorithms produced the same checksum."""
    checksums = {
        float(measurement["checksum"])
        for measurement in measurements
        if int(measurement["size"]) == size
    }

    if len(checksums) != 1:
        raise RuntimeError(
            f"Checksum disagreement for size {size}: {checksums}"
        )


def run_benchmark(
    sizes: list[int],
    output_path: Path,
) -> None:
    """Run isolated memory measurements and save every result."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for size in sizes:
        configuration_seed = BASE_SEED + size

        print(f"\nSize: {size} x {size}")
        size_measurements: list[dict[str, object]] = []

        for run_number in range(1, MEASURED_RUNS + 1):
            for algorithm in ALGORITHMS:
                print(
                    f"  Run {run_number}/{MEASURED_RUNS}: "
                    f"{algorithm}"
                )

                measurement = run_isolated_measurement(
                    algorithm=algorithm,
                    size=size,
                    seed=configuration_seed,
                )

                validate_measurement(
                    measurement=measurement,
                    expected_algorithm=algorithm,
                    expected_size=size,
                    expected_seed=configuration_seed,
                )

                measurement["run"] = run_number
                rows.append(measurement)
                size_measurements.append(measurement)

        validate_checksums(size_measurements, size)

        checksum = float(size_measurements[0]["checksum"])
        print(f"  Correctness validated. Checksum: {checksum:.0f}")

    write_csv(rows, output_path)
    print_summary(rows)

    print(f"\nRaw memory measurements saved to: {output_path}")


def write_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write all raw memory measurements to CSV."""
    fieldnames = [
        "algorithm",
        "size",
        "run",
        "seed",
        "baseline_rss_mb",
        "peak_rss_mb",
        "final_rss_mb",
        "peak_delta_mb",
        "checksum",
        "sampling_interval_seconds",
        "memory_metric",
        "memory_tool",
        "tile_size",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    rows: list[dict[str, object]],
) -> None:
    """Print median and IQR for peak RSS and peak RSS delta."""
    print("\nDENSE MEMORY BENCHMARK SUMMARY")
    print("Reported values: median and interquartile range (IQR)")

    sizes = sorted({int(row["size"]) for row in rows})

    for size in sizes:
        print(f"\nSize: {size} x {size}")

        for algorithm in ALGORITHMS:
            algorithm_rows = [
                row
                for row in rows
                if int(row["size"]) == size
                and row["algorithm"] == algorithm
            ]

            peak_values = [
                float(row["peak_rss_mb"])
                for row in algorithm_rows
            ]
            delta_values = [
                float(row["peak_delta_mb"])
                for row in algorithm_rows
            ]

            peak_quartiles = statistics.quantiles(
                peak_values,
                n=4,
                method="inclusive",
            )
            delta_quartiles = statistics.quantiles(
                delta_values,
                n=4,
                method="inclusive",
            )

            peak_median = statistics.median(peak_values)
            peak_iqr = peak_quartiles[2] - peak_quartiles[0]

            delta_median = statistics.median(delta_values)
            delta_iqr = (
                delta_quartiles[2] - delta_quartiles[0]
            )

            print(
                f"  {algorithm:<20} "
                f"peak median={peak_median:.3f} MB, "
                f"peak IQR={peak_iqr:.3f} MB, "
                f"delta median={delta_median:.3f} MB, "
                f"delta IQR={delta_iqr:.3f} MB"
            )


def parse_arguments() -> argparse.Namespace:
    """Parse benchmark profile and output path."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure dense matrix memory usage using isolated processes."
        )
    )

    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help=(
            "quick uses sizes 64 and 128; "
            "full uses sizes 64, 128, 256 and 512"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/dense_memory.csv"),
        help="Path of the raw CSV output file",
    )

    return parser.parse_args()


def main() -> None:
    """Execute the requested memory benchmark profile."""
    arguments = parse_arguments()

    if arguments.profile == "quick":
        sizes = [64, 128]
    else:
        sizes = [64, 128, 256, 512]

    print("DENSE MATRIX MEMORY BENCHMARK")
    print(f"Profile: {arguments.profile}")
    print(f"Sizes: {sizes}")
    print(f"Algorithms: {list(ALGORITHMS)}")
    print(f"Measured runs per configuration: {MEASURED_RUNS}")
    print("Execution model: one fresh process per measurement")
    print("Memory metric: process RSS sampled with psutil")

    run_benchmark(
        sizes=sizes,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()
