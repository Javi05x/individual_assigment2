"""Reproducible execution-time benchmark for dense multiplication."""

import argparse
import csv
import gc
import random
import statistics
import time
from pathlib import Path
from typing import Callable

from python.src.matrix.dense.generator import generate_dense_matrix_pair
from python.src.matrix.dense.matrix_basic import Matrix, multiply_basic
from python.src.matrix.dense.matrix_row_oriented import (
    multiply_row_oriented,
)
from python.src.matrix.dense.matrix_tiled import multiply_tiled


MultiplicationFunction = Callable[[Matrix, Matrix], Matrix]

BASE_SEED = 20261028
VALUE_MINIMUM = -10
VALUE_MAXIMUM = 10
WARMUP_RUNS = 1
MEASURED_RUNS = 5
TILE_SIZE = 32

ALGORITHMS: dict[str, MultiplicationFunction] = {
    "basic_ijk": multiply_basic,
    "row_oriented_ikj": multiply_row_oriented,
    "tiled_ikj_32": lambda matrix_a, matrix_b: multiply_tiled(
        matrix_a,
        matrix_b,
        tile_size=TILE_SIZE,
    ),
}


def calculate_checksum(matrix: Matrix) -> float:
    """Consume the multiplication result and return a simple checksum."""
    return float(sum(sum(row) for row in matrix))


def execute_once(
    multiplication_function: MultiplicationFunction,
    matrix_a: Matrix,
    matrix_b: Matrix,
) -> tuple[float, float]:
    """Measure one multiplication and return elapsed time and checksum."""
    gc.collect()

    start_time = time.perf_counter_ns()
    result = multiplication_function(matrix_a, matrix_b)
    end_time = time.perf_counter_ns()

    elapsed_seconds = (end_time - start_time) / 1_000_000_000
    checksum = calculate_checksum(result)

    return elapsed_seconds, checksum


def validate_algorithms(
    matrix_a: Matrix,
    matrix_b: Matrix,
) -> float:
    """Verify that all algorithms produce the same result."""
    reference_result = multiply_basic(matrix_a, matrix_b)
    reference_checksum = calculate_checksum(reference_result)

    for algorithm_name, multiplication_function in ALGORITHMS.items():
        result = multiplication_function(matrix_a, matrix_b)

        if result != reference_result:
            raise RuntimeError(
                f"Correctness validation failed for {algorithm_name}"
            )

        checksum = calculate_checksum(result)

        if checksum != reference_checksum:
            raise RuntimeError(
                f"Checksum validation failed for {algorithm_name}"
            )

    return reference_checksum


def run_benchmark(
    sizes: list[int],
    output_path: Path,
) -> None:
    """Run all dense configurations and preserve every raw measurement."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algorithm",
        "size",
        "run",
        "elapsed_seconds",
        "checksum",
        "seed",
        "value_minimum",
        "value_maximum",
        "warmup_runs",
        "measured_runs",
        "tile_size",
    ]

    rows: list[dict[str, object]] = []

    for size in sizes:
        configuration_seed = BASE_SEED + size

        print(f"\nGenerating matched {size} x {size} input matrices...")

        matrix_a, matrix_b = generate_dense_matrix_pair(
            size=size,
            seed=configuration_seed,
            minimum_value=VALUE_MINIMUM,
            maximum_value=VALUE_MAXIMUM,
        )

        reference_checksum = validate_algorithms(matrix_a, matrix_b)
        print(f"Correctness validated. Checksum: {reference_checksum:.0f}")

        for algorithm_name, multiplication_function in ALGORITHMS.items():
            for _ in range(WARMUP_RUNS):
                warmup_result = multiplication_function(matrix_a, matrix_b)
                calculate_checksum(warmup_result)

        algorithm_order = list(ALGORITHMS)

        for run_number in range(1, MEASURED_RUNS + 1):
            order_generator = random.Random(
                configuration_seed + run_number
            )
            order_generator.shuffle(algorithm_order)

            print(
                f"Measured run {run_number}/{MEASURED_RUNS}: "
                f"{', '.join(algorithm_order)}"
            )

            for algorithm_name in algorithm_order:
                multiplication_function = ALGORITHMS[algorithm_name]

                elapsed_seconds, checksum = execute_once(
                    multiplication_function,
                    matrix_a,
                    matrix_b,
                )

                if checksum != reference_checksum:
                    raise RuntimeError(
                        f"Invalid result from {algorithm_name} "
                        f"during measured run {run_number}"
                    )

                rows.append(
                    {
                        "algorithm": algorithm_name,
                        "size": size,
                        "run": run_number,
                        "elapsed_seconds": elapsed_seconds,
                        "checksum": checksum,
                        "seed": configuration_seed,
                        "value_minimum": VALUE_MINIMUM,
                        "value_maximum": VALUE_MAXIMUM,
                        "warmup_runs": WARMUP_RUNS,
                        "measured_runs": MEASURED_RUNS,
                        "tile_size": (
                            TILE_SIZE
                            if algorithm_name == "tiled_ikj_32"
                            else ""
                        ),
                    }
                )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows)
    print(f"\nRaw measurements saved to: {output_path}")


def print_summary(rows: list[dict[str, object]]) -> None:
    """Print median and interquartile range for each configuration."""
    print("\nDENSE BENCHMARK SUMMARY")
    print("Reported values: median and interquartile range (IQR)")

    sizes = sorted({int(row["size"]) for row in rows})

    for size in sizes:
        print(f"\nSize: {size} x {size}")

        for algorithm_name in ALGORITHMS:
            times = [
                float(row["elapsed_seconds"])
                for row in rows
                if row["algorithm"] == algorithm_name
                and int(row["size"]) == size
            ]

            quartiles = statistics.quantiles(
                times,
                n=4,
                method="inclusive",
            )
            median = statistics.median(times)
            interquartile_range = quartiles[2] - quartiles[0]

            print(
                f"  {algorithm_name:<20} "
                f"median={median:.6f}s, "
                f"IQR={interquartile_range:.6f}s"
            )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark dense matrix multiplication algorithms."
    )

    parser.add_argument(
        "--profile",
        choices=("quick", "medium", "full"),
        default="quick",
        help=(
            "quick uses sizes 32, 64 and 96; "
            "medium uses sizes 128, 192 and 256; "
            "full uses sizes 64, 128, 256 and 512"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/dense_time.csv"),
        help="Path of the raw CSV output file",
    )

    return parser.parse_args()


def main() -> None:
    """Execute the selected benchmark profile."""
    arguments = parse_arguments()

    if arguments.profile == "quick":
        sizes = [32, 64, 96]
    elif arguments.profile == "medium":
        sizes = [128,192,256]
    else:
        sizes = [64, 128, 256, 512]

    print("DENSE MATRIX MULTIPLICATION BENCHMARK")
    print(f"Profile: {arguments.profile}")
    print(f"Sizes: {sizes}")
    print(f"Measured runs per configuration: {MEASURED_RUNS}")
    print(f"Warm-up runs per algorithm and size: {WARMUP_RUNS}")
    print(f"Base seed: {BASE_SEED}")
    print(f"Integer value range: [{VALUE_MINIMUM}, {VALUE_MAXIMUM}]")
    print(f"Tiled algorithm block size: {TILE_SIZE}")

    run_benchmark(sizes, arguments.output)


if __name__ == "__main__":
    main()
