"""Controller for isolated dense-versus-sparse memory measurements."""

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
    "dense_row_oriented",
    "custom_csr",
    "scipy_csr",
)


def configuration_seed(
    size: int,
    sparsity: float,
) -> int:
    """Reproduce the seed formula used by the time benchmark."""
    sparsity_code = round(sparsity * 10_000)

    return BASE_SEED + size * 100_000 + sparsity_code


def run_isolated_measurement(
    algorithm: str,
    size: int,
    sparsity: float,
    seed: int,
) -> dict[str, object]:
    """Run one memory measurement in a fresh Python process."""
    command = [
        sys.executable,
        "-m",
        "python.src.matrix.benchmark.sparse_memory_worker",
        "--algorithm",
        algorithm,
        "--size",
        str(size),
        "--sparsity",
        str(sparsity),
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
            f"{algorithm}, size={size}, sparsity={sparsity}"
        )

    try:
        return json.loads(standard_output)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Memory worker did not return valid JSON:\n"
            f"{standard_output}"
        ) from error


def validate_measurement(
    measurement: dict[str, object],
    expected_algorithm: str,
    expected_size: int,
    expected_sparsity: float,
    expected_seed: int,
) -> None:
    """Validate the metadata and memory relationships."""
    if measurement["algorithm"] != expected_algorithm:
        raise RuntimeError("Worker returned an unexpected algorithm")

    if int(measurement["size"]) != expected_size:
        raise RuntimeError("Worker returned an unexpected size")

    if int(measurement["seed"]) != expected_seed:
        raise RuntimeError("Worker returned an unexpected seed")

    measured_sparsity = float(
        measurement["requested_sparsity"]
    )

    if abs(measured_sparsity - expected_sparsity) > 1e-12:
        raise RuntimeError(
            "Worker returned an unexpected sparsity"
        )

    representation_rss = float(
        measurement["representation_rss_mb"]
    )
    pre_multiplication_rss = float(
        measurement["pre_multiplication_rss_mb"]
    )
    peak_rss = float(measurement["peak_rss_mb"])
    final_rss = float(measurement["final_rss_mb"])

    if peak_rss < pre_multiplication_rss:
        raise RuntimeError(
            "Peak RSS is lower than pre-multiplication RSS"
        )

    if peak_rss < final_rss:
        raise RuntimeError(
            "Peak RSS is lower than final RSS"
        )

    if float(measurement["total_peak_delta_mb"]) < 0:
        raise RuntimeError(
            "Total peak RSS delta is negative"
        )

    if (
        float(measurement["multiplication_peak_delta_mb"])
        < 0
    ):
        raise RuntimeError(
            "Multiplication peak RSS delta is negative"
        )

    if representation_rss < 0:
        raise RuntimeError(
            "Representation RSS cannot be negative"
        )


def validate_configuration_results(
    measurements: list[dict[str, object]],
) -> None:
    """Verify logical equivalence across all algorithms and runs."""
    checksums = {
        float(measurement["checksum"])
        for measurement in measurements
    }

    output_nonzero_counts = {
        int(measurement["output_nonzero_count"])
        for measurement in measurements
    }

    input_a_nonzero_counts = {
        int(measurement["input_a_nonzero_count"])
        for measurement in measurements
    }

    input_b_nonzero_counts = {
        int(measurement["input_b_nonzero_count"])
        for measurement in measurements
    }

    if len(checksums) != 1:
        raise RuntimeError(
            f"Checksum disagreement: {checksums}"
        )

    if len(output_nonzero_counts) != 1:
        raise RuntimeError(
            "Output nonzero-count disagreement: "
            f"{output_nonzero_counts}"
        )

    if len(input_a_nonzero_counts) != 1:
        raise RuntimeError(
            "Input A nonzero-count disagreement"
        )

    if len(input_b_nonzero_counts) != 1:
        raise RuntimeError(
            "Input B nonzero-count disagreement"
        )


def run_benchmark(
    sizes: list[int],
    sparsities: list[float],
    output_path: Path,
) -> None:
    """Run all memory configurations in isolated processes."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []

    for size in sizes:
        for sparsity in sparsities:
            seed = configuration_seed(size, sparsity)

            print(
                f"\nConfiguration: {size} x {size}, "
                f"{sparsity * 100:.0f}% sparsity"
            )
            print(f"Seed: {seed}")

            configuration_measurements: list[
                dict[str, object]
            ] = []

            for run_number in range(
                1,
                MEASURED_RUNS + 1,
            ):
                for algorithm in ALGORITHMS:
                    print(
                        f"  Run {run_number}/"
                        f"{MEASURED_RUNS}: {algorithm}"
                    )

                    measurement = (
                        run_isolated_measurement(
                            algorithm=algorithm,
                            size=size,
                            sparsity=sparsity,
                            seed=seed,
                        )
                    )

                    validate_measurement(
                        measurement=measurement,
                        expected_algorithm=algorithm,
                        expected_size=size,
                        expected_sparsity=sparsity,
                        expected_seed=seed,
                    )

                    measurement["run"] = run_number

                    rows.append(measurement)
                    configuration_measurements.append(
                        measurement
                    )

            validate_configuration_results(
                configuration_measurements
            )

            first = configuration_measurements[0]

            print(
                "  Correctness validated. "
                f"Input NNZ: "
                f"A={first['input_a_nonzero_count']}, "
                f"B={first['input_b_nonzero_count']}; "
                f"output NNZ="
                f"{first['output_nonzero_count']}; "
                f"checksum={float(first['checksum']):.0f}"
            )

    write_csv(rows, output_path)
    print_summary(rows)

    print(
        f"\nRaw memory measurements saved to: "
        f"{output_path}"
    )


def write_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write every raw memory measurement to CSV."""
    fieldnames = [
        "algorithm",
        "size",
        "requested_sparsity",
        "run",
        "seed",
        "input_a_nonzero_count",
        "input_b_nonzero_count",
        "output_nonzero_count",
        "initial_rss_mb",
        "representation_rss_mb",
        "representation_delta_mb",
        "pre_multiplication_rss_mb",
        "peak_rss_mb",
        "final_rss_mb",
        "multiplication_peak_delta_mb",
        "total_peak_delta_mb",
        "checksum",
        "sampling_interval_seconds",
        "memory_metric",
        "memory_tool",
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


def median_and_iqr(
    values: list[float],
) -> tuple[float, float]:
    """Return the median and inclusive interquartile range."""
    quartiles = statistics.quantiles(
        values,
        n=4,
        method="inclusive",
    )

    return (
        statistics.median(values),
        quartiles[2] - quartiles[0],
    )


def print_summary(
    rows: list[dict[str, object]],
) -> None:
    """Print memory summaries for every configuration."""
    print("\nSPARSE MEMORY BENCHMARK SUMMARY")
    print("Reported values: median and IQR")

    configurations = sorted(
        {
            (
                int(row["size"]),
                float(row["requested_sparsity"]),
            )
            for row in rows
        }
    )

    for size, sparsity in configurations:
        print(
            f"\nSize {size} x {size}, "
            f"sparsity {sparsity * 100:.0f}%"
        )

        for algorithm in ALGORITHMS:
            algorithm_rows = [
                row
                for row in rows
                if int(row["size"]) == size
                and float(row["requested_sparsity"])
                == sparsity
                and row["algorithm"] == algorithm
            ]

            representation_values = [
                float(row["representation_delta_mb"])
                for row in algorithm_rows
            ]

            multiplication_values = [
                float(
                    row[
                        "multiplication_peak_delta_mb"
                    ]
                )
                for row in algorithm_rows
            ]

            total_values = [
                float(row["total_peak_delta_mb"])
                for row in algorithm_rows
            ]

            representation_median, representation_iqr = (
                median_and_iqr(representation_values)
            )

            multiplication_median, multiplication_iqr = (
                median_and_iqr(multiplication_values)
            )

            total_median, total_iqr = median_and_iqr(
                total_values
            )

            print(
                f"  {algorithm:<20} "
                f"representation="
                f"{representation_median:.3f} MB "
                f"(IQR {representation_iqr:.3f}), "
                f"multiplication="
                f"{multiplication_median:.3f} MB "
                f"(IQR {multiplication_iqr:.3f}), "
                f"total peak="
                f"{total_median:.3f} MB "
                f"(IQR {total_iqr:.3f})"
            )


def parse_arguments() -> argparse.Namespace:
    """Parse profile and CSV destination."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure dense and sparse multiplication memory "
            "using isolated processes."
        )
    )

    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help=(
            "quick uses sizes 64 and 128 with 90% and 99%; "
            "full uses sizes 64, 128, 256 and 512 with "
            "50%, 70%, 90%, 95% and 99%"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/raw/sparse_memory.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Execute the selected memory profile."""
    arguments = parse_arguments()

    if arguments.profile == "quick":
        sizes = [64, 128]
        sparsities = [0.90, 0.99]
    else:
        sizes = [64, 128, 256, 512]
        sparsities = [
            0.50,
            0.70,
            0.90,
            0.95,
            0.99,
        ]

    print("DENSE VERSUS SPARSE MEMORY BENCHMARK")
    print(f"Profile: {arguments.profile}")
    print(f"Sizes: {sizes}")
    print(
        "Sparsity levels: "
        f"{[f'{value * 100:.0f}%' for value in sparsities]}"
    )
    print(
        f"Measured runs per configuration: "
        f"{MEASURED_RUNS}"
    )
    print(
        "Execution model: one fresh process "
        "per measurement"
    )
    print(
        "Memory metric: process RSS sampled "
        "with psutil"
    )

    run_benchmark(
        sizes=sizes,
        sparsities=sparsities,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()
