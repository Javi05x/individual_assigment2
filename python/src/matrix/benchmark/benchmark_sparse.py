"""Controlled dense-versus-sparse multiplication benchmark."""

import argparse
import csv
import gc
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from python.src.matrix.dense.matrix_row_oriented import (
    multiply_row_oriented,
)
from python.src.matrix.sparse.generator import (
    calculate_sparsity,
    count_nonzero,
    generate_sparse_matrix_pair,
)
from python.src.matrix.sparse.matrix_csr import SparseMatrixCSR
from python.src.matrix.sparse.matrix_scipy import SparseMatrixSciPy


BASE_SEED = 20261028
VALUE_MINIMUM = -10
VALUE_MAXIMUM = 10
WARMUP_RUNS = 1
MEASURED_RUNS = 5

ALGORITHM_NAMES = (
    "dense_row_oriented",
    "custom_csr",
    "scipy_csr",
)


def dense_checksum(matrix: list[list[Any]]) -> float:
    """Return the checksum of a dense result matrix."""
    return float(sum(sum(row) for row in matrix))


def custom_csr_checksum(matrix: SparseMatrixCSR) -> float:
    """Return the checksum of a custom CSR result matrix."""
    return float(sum(matrix.values))


def scipy_csr_checksum(matrix: SparseMatrixSciPy) -> float:
    """Return the checksum of a SciPy CSR result matrix."""
    return float(matrix.matrix.sum())


def execute_once(
    multiplication_function: Callable[[], Any],
    checksum_function: Callable[[Any], float],
) -> tuple[float, float, int]:
    """
    Measure one multiplication.

    Input generation and representation conversion occur outside this
    function. Checksum and output-NNZ calculation occur after timing.
    """
    gc.collect()

    start_time = time.perf_counter_ns()
    result = multiplication_function()
    end_time = time.perf_counter_ns()

    elapsed_seconds = (
        end_time - start_time
    ) / 1_000_000_000

    checksum = checksum_function(result)

    if isinstance(result, list):
        output_nonzero_count = count_nonzero(result)
    else:
        output_nonzero_count = result.nonzero_count

    return elapsed_seconds, checksum, output_nonzero_count


def validate_results(
    dense_a: list[list[Any]],
    dense_b: list[list[Any]],
    custom_a: SparseMatrixCSR,
    custom_b: SparseMatrixCSR,
    scipy_a: SparseMatrixSciPy,
    scipy_b: SparseMatrixSciPy,
) -> tuple[float, int]:
    """Verify complete result equality before measured execution."""
    dense_result = multiply_row_oriented(dense_a, dense_b)
    custom_result = custom_a.multiply(custom_b).to_dense()
    scipy_result = scipy_a.multiply(scipy_b).to_dense()

    if custom_result != dense_result:
        raise RuntimeError(
            "Custom CSR result differs from the dense reference"
        )

    if scipy_result != dense_result:
        raise RuntimeError(
            "SciPy CSR result differs from the dense reference"
        )

    checksum = dense_checksum(dense_result)
    output_nonzero_count = count_nonzero(dense_result)

    return checksum, output_nonzero_count


def run_benchmark(
    sizes: list[int],
    sparsities: list[float],
    output_path: Path,
) -> None:
    """Run every sparse benchmark configuration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algorithm",
        "size",
        "requested_sparsity",
        "actual_sparsity_a",
        "actual_sparsity_b",
        "run",
        "elapsed_seconds",
        "checksum",
        "input_a_nonzero_count",
        "input_b_nonzero_count",
        "output_nonzero_count",
        "output_sparsity",
        "seed",
        "value_minimum",
        "value_maximum",
        "warmup_runs",
        "measured_runs",
    ]

    rows: list[dict[str, object]] = []

    for size in sizes:
        for sparsity in sparsities:
            sparsity_code = round(sparsity * 10_000)
            configuration_seed = (
                BASE_SEED
                + size * 100_000
                + sparsity_code
            )

            print(
                f"\nConfiguration: {size} x {size}, "
                f"{sparsity * 100:.0f}% requested sparsity"
            )

            dense_a, dense_b = generate_sparse_matrix_pair(
                size=size,
                sparsity=sparsity,
                seed=configuration_seed,
                minimum_value=VALUE_MINIMUM,
                maximum_value=VALUE_MAXIMUM,
            )

            custom_a = SparseMatrixCSR.from_dense(dense_a)
            custom_b = SparseMatrixCSR.from_dense(dense_b)

            scipy_a = SparseMatrixSciPy.from_dense(dense_a)
            scipy_b = SparseMatrixSciPy.from_dense(dense_b)

            input_a_nonzero_count = count_nonzero(dense_a)
            input_b_nonzero_count = count_nonzero(dense_b)

            actual_sparsity_a = calculate_sparsity(dense_a)
            actual_sparsity_b = calculate_sparsity(dense_b)

            reference_checksum, reference_output_nnz = (
                validate_results(
                    dense_a,
                    dense_b,
                    custom_a,
                    custom_b,
                    scipy_a,
                    scipy_b,
                )
            )

            total_output_entries = size * size
            reference_output_sparsity = 1.0 - (
                reference_output_nnz / total_output_entries
            )

            print(
                "Correctness validated. "
                f"Input NNZ: A={input_a_nonzero_count}, "
                f"B={input_b_nonzero_count}; "
                f"output NNZ={reference_output_nnz}"
            )

            algorithms: dict[
                str,
                tuple[Callable[[], Any], Callable[[Any], float]],
            ] = {
                "dense_row_oriented": (
                    lambda: multiply_row_oriented(
                        dense_a,
                        dense_b,
                    ),
                    dense_checksum,
                ),
                "custom_csr": (
                    lambda: custom_a.multiply(custom_b),
                    custom_csr_checksum,
                ),
                "scipy_csr": (
                    lambda: scipy_a.multiply(scipy_b),
                    scipy_csr_checksum,
                ),
            }

            for multiplication_function, checksum_function in (
                algorithms.values()
            ):
                for _ in range(WARMUP_RUNS):
                    warmup_result = multiplication_function()
                    checksum_function(warmup_result)

            for run_number in range(1, MEASURED_RUNS + 1):
                algorithm_order = list(ALGORITHM_NAMES)

                order_generator = random.Random(
                    configuration_seed + run_number
                )
                order_generator.shuffle(algorithm_order)

                print(
                    f"Run {run_number}/{MEASURED_RUNS}: "
                    f"{', '.join(algorithm_order)}"
                )

                for algorithm_name in algorithm_order:
                    (
                        multiplication_function,
                        checksum_function,
                    ) = algorithms[algorithm_name]

                    (
                        elapsed_seconds,
                        checksum,
                        output_nonzero_count,
                    ) = execute_once(
                        multiplication_function,
                        checksum_function,
                    )

                    if checksum != reference_checksum:
                        raise RuntimeError(
                            f"Checksum disagreement for "
                            f"{algorithm_name}"
                        )

                    if (
                        output_nonzero_count
                        != reference_output_nnz
                    ):
                        raise RuntimeError(
                            f"Output NNZ disagreement for "
                            f"{algorithm_name}"
                        )

                    rows.append(
                        {
                            "algorithm": algorithm_name,
                            "size": size,
                            "requested_sparsity": sparsity,
                            "actual_sparsity_a": actual_sparsity_a,
                            "actual_sparsity_b": actual_sparsity_b,
                            "run": run_number,
                            "elapsed_seconds": elapsed_seconds,
                            "checksum": checksum,
                            "input_a_nonzero_count": (
                                input_a_nonzero_count
                            ),
                            "input_b_nonzero_count": (
                                input_b_nonzero_count
                            ),
                            "output_nonzero_count": (
                                output_nonzero_count
                            ),
                            "output_sparsity": (
                                reference_output_sparsity
                            ),
                            "seed": configuration_seed,
                            "value_minimum": VALUE_MINIMUM,
                            "value_maximum": VALUE_MAXIMUM,
                            "warmup_runs": WARMUP_RUNS,
                            "measured_runs": MEASURED_RUNS,
                        }
                    )

    write_csv(rows, fieldnames, output_path)
    print_summary(rows)

    print(f"\nRaw measurements saved to: {output_path}")


def write_csv(
    rows: list[dict[str, object]],
    fieldnames: list[str],
    output_path: Path,
) -> None:
    """Write every raw measurement to CSV."""
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
    """Print median and IQR for every configuration."""
    print("\nSPARSE BENCHMARK SUMMARY")
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

        for algorithm_name in ALGORITHM_NAMES:
            times = [
                float(row["elapsed_seconds"])
                for row in rows
                if int(row["size"]) == size
                and float(row["requested_sparsity"])
                == sparsity
                and row["algorithm"] == algorithm_name
            ]

            quartiles = statistics.quantiles(
                times,
                n=4,
                method="inclusive",
            )

            median = statistics.median(times)
            interquartile_range = (
                quartiles[2] - quartiles[0]
            )

            print(
                f"  {algorithm_name:<20} "
                f"median={median:.6f}s, "
                f"IQR={interquartile_range:.6f}s"
            )


def parse_arguments() -> argparse.Namespace:
    """Parse benchmark profile and output path."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched dense, custom CSR and SciPy CSR "
            "matrix multiplication."
        )
    )

    parser.add_argument(
        "--profile",
        choices=("quick", "medium", "full"),
        default="quick",
        help=(
            "quick uses sizes 64 and 128 with 70%, 90% and 99%; "
            "medium uses size 256 with 50%, 70%, 90%, 95% and 99%; "
            "full uses sizes 64, 128, 256 and 512 with "
            "50%, 70%, 90%, 95% and 99%"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/sparse_time.csv"),
    )

    return parser.parse_args()


def main() -> None:
    """Execute the selected sparse benchmark profile."""
    arguments = parse_arguments()

    if arguments.profile == "quick":
        sizes = [64, 128]
        sparsities = [0.70, 0.90, 0.99]
    elif arguments.profile == "medium":
        sizes = [256]
        sparsities = [0.50, 0.70, 0.90, 0.95, 0.99]
    else:
        sizes = [64, 128, 256, 512]
        sparsities = [0.50, 0.70, 0.90, 0.95, 0.99]

    print("DENSE VERSUS SPARSE MATRIX BENCHMARK")
    print(f"Profile: {arguments.profile}")
    print(f"Sizes: {sizes}")
    print(
        "Sparsity levels: "
        f"{[f'{value * 100:.0f}%' for value in sparsities]}"
    )
    print(f"Measured runs: {MEASURED_RUNS}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Value range: [{VALUE_MINIMUM}, {VALUE_MAXIMUM}]")

    run_benchmark(
        sizes=sizes,
        sparsities=sparsities,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()
