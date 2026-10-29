"""Isolated memory worker for dense-versus-sparse multiplication."""

import argparse
import gc
import json
import threading
import time
from typing import Any, Callable

import psutil

from python.src.matrix.dense.matrix_row_oriented import (
    multiply_row_oriented,
)
from python.src.matrix.sparse.generator import (
    count_nonzero,
    generate_sparse_matrix_pair,
)
from python.src.matrix.sparse.matrix_csr import SparseMatrixCSR
from python.src.matrix.sparse.matrix_scipy import SparseMatrixSciPy


VALUE_MINIMUM = -10
VALUE_MAXIMUM = 10
SAMPLE_INTERVAL_SECONDS = 0.001

ALGORITHM_NAMES = (
    "dense_row_oriented",
    "custom_csr",
    "scipy_csr",
)


def dense_checksum(matrix: list[list[Any]]) -> float:
    """Calculate the checksum of a dense result."""
    return float(sum(sum(row) for row in matrix))


def custom_csr_checksum(matrix: SparseMatrixCSR) -> float:
    """Calculate the checksum of a custom CSR result."""
    return float(sum(matrix.values))


def scipy_csr_checksum(matrix: SparseMatrixSciPy) -> float:
    """Calculate the checksum of a SciPy CSR result."""
    return float(matrix.matrix.sum())


def measure_peak_rss(
    multiplication_function: Callable[[], Any],
    checksum_function: Callable[[Any], float],
) -> dict[str, float | int]:
    """Measure RSS immediately before, during and after multiplication."""
    process = psutil.Process()
    stop_event = threading.Event()

    gc.collect()

    pre_multiplication_rss_bytes = process.memory_info().rss
    peak_rss_bytes = pre_multiplication_rss_bytes

    def sample_memory() -> None:
        nonlocal peak_rss_bytes

        while not stop_event.is_set():
            current_rss_bytes = process.memory_info().rss
            peak_rss_bytes = max(
                peak_rss_bytes,
                current_rss_bytes,
            )
            time.sleep(SAMPLE_INTERVAL_SECONDS)

        final_sample_bytes = process.memory_info().rss
        peak_rss_bytes = max(
            peak_rss_bytes,
            final_sample_bytes,
        )

    monitor = threading.Thread(
        target=sample_memory,
        name="rss-monitor",
        daemon=True,
    )

    monitor.start()

    try:
        result = multiplication_function()
    finally:
        stop_event.set()
        monitor.join()

    final_rss_bytes = process.memory_info().rss
    peak_rss_bytes = max(
        peak_rss_bytes,
        final_rss_bytes,
    )

    if isinstance(result, list):
        output_nonzero_count = count_nonzero(result)
    else:
        output_nonzero_count = result.nonzero_count

    bytes_per_megabyte = 1024 * 1024

    return {
        "pre_multiplication_rss_mb": (
            pre_multiplication_rss_bytes / bytes_per_megabyte
        ),
        "peak_rss_mb": peak_rss_bytes / bytes_per_megabyte,
        "final_rss_mb": final_rss_bytes / bytes_per_megabyte,
        "multiplication_peak_delta_mb": (
            peak_rss_bytes - pre_multiplication_rss_bytes
        ) / bytes_per_megabyte,
        "checksum": checksum_function(result),
        "output_nonzero_count": output_nonzero_count,
    }


def prepare_algorithm(
    algorithm: str,
    size: int,
    sparsity: float,
    seed: int,
) -> tuple[
    Callable[[], Any],
    Callable[[Any], float],
    int,
    int,
]:
    """
    Generate matched logical inputs and retain only the tested representation.
    """
    dense_a, dense_b = generate_sparse_matrix_pair(
        size=size,
        sparsity=sparsity,
        seed=seed,
        minimum_value=VALUE_MINIMUM,
        maximum_value=VALUE_MAXIMUM,
    )

    input_a_nonzero_count = count_nonzero(dense_a)
    input_b_nonzero_count = count_nonzero(dense_b)

    if algorithm == "dense_row_oriented":
        multiplication_function = lambda: multiply_row_oriented(
            dense_a,
            dense_b,
        )
        checksum_function = dense_checksum

    elif algorithm == "custom_csr":
        custom_a = SparseMatrixCSR.from_dense(dense_a)
        custom_b = SparseMatrixCSR.from_dense(dense_b)

        del dense_a
        del dense_b
        gc.collect()

        multiplication_function = lambda: custom_a.multiply(custom_b)
        checksum_function = custom_csr_checksum

    elif algorithm == "scipy_csr":
        scipy_a = SparseMatrixSciPy.from_dense(dense_a)
        scipy_b = SparseMatrixSciPy.from_dense(dense_b)

        del dense_a
        del dense_b
        gc.collect()

        multiplication_function = lambda: scipy_a.multiply(scipy_b)
        checksum_function = scipy_csr_checksum

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return (
        multiplication_function,
        checksum_function,
        input_a_nonzero_count,
        input_b_nonzero_count,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse one isolated memory configuration."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure one dense or sparse multiplication "
            "in an isolated process."
        )
    )

    parser.add_argument(
        "--algorithm",
        required=True,
        choices=ALGORITHM_NAMES,
    )
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument(
        "--sparsity",
        required=True,
        type=float,
    )
    parser.add_argument("--seed", required=True, type=int)

    return parser.parse_args()


def main() -> None:
    """Prepare one representation and measure its memory usage."""
    arguments = parse_arguments()

    if arguments.size <= 0:
        raise ValueError("size must be greater than zero")

    if not 0.0 <= arguments.sparsity <= 1.0:
        raise ValueError(
            "sparsity must be between 0.0 and 1.0"
        )

    process = psutil.Process()

    gc.collect()
    initial_rss_mb = process.memory_info().rss / (1024 * 1024)

    (
        multiplication_function,
        checksum_function,
        input_a_nonzero_count,
        input_b_nonzero_count,
    ) = prepare_algorithm(
        algorithm=arguments.algorithm,
        size=arguments.size,
        sparsity=arguments.sparsity,
        seed=arguments.seed,
    )

    gc.collect()
    representation_rss_mb = (
        process.memory_info().rss / (1024 * 1024)
    )

    measurement = measure_peak_rss(
        multiplication_function=multiplication_function,
        checksum_function=checksum_function,
    )

    output = {
        "algorithm": arguments.algorithm,
        "size": arguments.size,
        "requested_sparsity": arguments.sparsity,
        "seed": arguments.seed,
        "input_a_nonzero_count": input_a_nonzero_count,
        "input_b_nonzero_count": input_b_nonzero_count,
        "output_nonzero_count": measurement[
            "output_nonzero_count"
        ],
        "initial_rss_mb": initial_rss_mb,
        "representation_rss_mb": representation_rss_mb,
        "representation_delta_mb": (
            representation_rss_mb - initial_rss_mb
        ),
        "pre_multiplication_rss_mb": measurement[
            "pre_multiplication_rss_mb"
        ],
        "peak_rss_mb": measurement["peak_rss_mb"],
        "final_rss_mb": measurement["final_rss_mb"],
        "multiplication_peak_delta_mb": measurement[
            "multiplication_peak_delta_mb"
        ],
        "total_peak_delta_mb": (
            float(measurement["peak_rss_mb"])
            - initial_rss_mb
        ),
        "checksum": measurement["checksum"],
        "sampling_interval_seconds": (
            SAMPLE_INTERVAL_SECONDS
        ),
        "memory_metric": "process_rss",
        "memory_tool": "psutil",
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
