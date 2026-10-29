"""Isolated worker for dense matrix memory measurements."""

import argparse
import gc
import json
import threading
import time
from typing import Callable

import psutil

from python.src.matrix.dense.generator import generate_dense_matrix_pair
from python.src.matrix.dense.matrix_basic import Matrix, multiply_basic
from python.src.matrix.dense.matrix_row_oriented import (
    multiply_row_oriented,
)
from python.src.matrix.dense.matrix_tiled import multiply_tiled


MultiplicationFunction = Callable[[Matrix, Matrix], Matrix]

TILE_SIZE = 32
VALUE_MINIMUM = -10
VALUE_MAXIMUM = 10
SAMPLE_INTERVAL_SECONDS = 0.001

ALGORITHMS: dict[str, MultiplicationFunction] = {
    "basic_ijk": multiply_basic,
    "row_oriented_ikj": multiply_row_oriented,
    "tiled_ikj_32": lambda matrix_a, matrix_b: multiply_tiled(
        matrix_a,
        matrix_b,
        tile_size=TILE_SIZE,
    ),
}


def matrix_checksum(matrix: Matrix) -> float:
    """Consume the complete result and return its checksum."""
    return float(sum(sum(row) for row in matrix))


def measure_peak_rss_during_multiplication(
    multiplication_function: MultiplicationFunction,
    matrix_a: Matrix,
    matrix_b: Matrix,
) -> dict[str, float]:
    """
    Measure process RSS before, during and after multiplication.

    RSS is sampled in a monitoring thread while the multiplication runs.
    The reported delta is relative to the RSS measured after input generation
    and immediately before multiplication.
    """
    process = psutil.Process()
    stop_event = threading.Event()

    gc.collect()
    baseline_rss_bytes = process.memory_info().rss
    peak_rss_bytes = baseline_rss_bytes

    def sample_memory() -> None:
        nonlocal peak_rss_bytes

        while not stop_event.is_set():
            current_rss_bytes = process.memory_info().rss
            peak_rss_bytes = max(peak_rss_bytes, current_rss_bytes)
            time.sleep(SAMPLE_INTERVAL_SECONDS)

        final_sample_bytes = process.memory_info().rss
        peak_rss_bytes = max(peak_rss_bytes, final_sample_bytes)

    monitor = threading.Thread(
        target=sample_memory,
        name="rss-monitor",
        daemon=True,
    )

    monitor.start()

    try:
        result = multiplication_function(matrix_a, matrix_b)
    finally:
        stop_event.set()
        monitor.join()

    final_rss_bytes = process.memory_info().rss
    peak_rss_bytes = max(peak_rss_bytes, final_rss_bytes)

    bytes_per_megabyte = 1024 * 1024

    return {
        "baseline_rss_mb": baseline_rss_bytes / bytes_per_megabyte,
        "peak_rss_mb": peak_rss_bytes / bytes_per_megabyte,
        "final_rss_mb": final_rss_bytes / bytes_per_megabyte,
        "peak_delta_mb": (
            peak_rss_bytes - baseline_rss_bytes
        ) / bytes_per_megabyte,
        "checksum": matrix_checksum(result),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse one isolated memory configuration."""
    parser = argparse.ArgumentParser(
        description="Measure one dense multiplication in an isolated process."
    )

    parser.add_argument(
        "--algorithm",
        required=True,
        choices=tuple(ALGORITHMS),
    )
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)

    return parser.parse_args()


def main() -> None:
    """Generate matched inputs and measure one multiplication."""
    arguments = parse_arguments()

    if arguments.size <= 0:
        raise ValueError("size must be greater than zero")

    matrix_a, matrix_b = generate_dense_matrix_pair(
        size=arguments.size,
        seed=arguments.seed,
        minimum_value=VALUE_MINIMUM,
        maximum_value=VALUE_MAXIMUM,
    )

    measurement = measure_peak_rss_during_multiplication(
        ALGORITHMS[arguments.algorithm],
        matrix_a,
        matrix_b,
    )

    output = {
        "algorithm": arguments.algorithm,
        "size": arguments.size,
        "seed": arguments.seed,
        "baseline_rss_mb": measurement["baseline_rss_mb"],
        "peak_rss_mb": measurement["peak_rss_mb"],
        "final_rss_mb": measurement["final_rss_mb"],
        "peak_delta_mb": measurement["peak_delta_mb"],
        "checksum": measurement["checksum"],
        "sampling_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "memory_metric": "process_rss",
        "memory_tool": "psutil",
        "tile_size": (
            TILE_SIZE
            if arguments.algorithm == "tiled_ikj_32"
            else None
        ),
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
