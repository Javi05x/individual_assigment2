"""Reproducible sparse matrix-vector benchmark for mc2depi."""

import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix


VECTOR_SEED = 20261028
WARMUP_RUNS = 1
MEASURED_RUNS = 5
VECTOR_MINIMUM = -10
VECTOR_MAXIMUM = 10


def load_mc2depi(input_path: Path) -> tuple[csr_matrix, float]:
    """Load mc2depi and convert it to canonical CSR format."""
    start_time = time.perf_counter()

    matrix = mmread(input_path).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    end_time = time.perf_counter()

    return matrix, end_time - start_time


def generate_vector(length: int) -> np.ndarray:
    """Generate a reproducible dense integer input vector."""
    generator = np.random.default_rng(VECTOR_SEED)

    return generator.integers(
        low=VECTOR_MINIMUM,
        high=VECTOR_MAXIMUM + 1,
        size=length,
        dtype=np.int64,
    )


def vector_checksum(vector: np.ndarray) -> int:
    """Return a deterministic checksum for correctness verification."""
    return int(vector.sum(dtype=np.int64))


def run_benchmark(
    input_path: Path,
    output_path: Path,
) -> None:
    """Benchmark repeated mc2depi sparse matrix-vector multiplication."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    matrix, load_seconds = load_mc2depi(input_path)

    if matrix.shape != (525_825, 525_825):
        raise RuntimeError(
            f"Unexpected matrix shape: {matrix.shape}"
        )

    if matrix.nnz != 2_100_225:
        raise RuntimeError(
            f"Unexpected nonzero count: {matrix.nnz}"
        )

    vector = generate_vector(matrix.shape[1])

    for _ in range(WARMUP_RUNS):
        warmup_result = matrix @ vector
        vector_checksum(warmup_result)

    rows: list[dict[str, object]] = []
    reference_checksum: int | None = None
    reference_output_nonzero_count: int | None = None

    for run_number in range(1, MEASURED_RUNS + 1):
        gc.collect()

        start_time = time.perf_counter_ns()
        result = matrix @ vector
        end_time = time.perf_counter_ns()

        elapsed_seconds = (
            end_time - start_time
        ) / 1_000_000_000

        checksum = vector_checksum(result)
        output_nonzero_count = int(np.count_nonzero(result))

        if reference_checksum is None:
            reference_checksum = checksum
            reference_output_nonzero_count = (
                output_nonzero_count
            )
        else:
            if checksum != reference_checksum:
                raise RuntimeError(
                    f"Checksum disagreement in run {run_number}"
                )

            if (
                output_nonzero_count
                != reference_output_nonzero_count
            ):
                raise RuntimeError(
                    "Output nonzero-count disagreement in "
                    f"run {run_number}"
                )

        rows.append(
            {
                "dataset": "mc2depi",
                "operation": "sparse_matrix_vector",
                "run": run_number,
                "elapsed_seconds": elapsed_seconds,
                "checksum": checksum,
                "output_nonzero_count": output_nonzero_count,
                "rows": matrix.shape[0],
                "columns": matrix.shape[1],
                "matrix_nonzero_count": matrix.nnz,
                "vector_length": vector.size,
                "vector_seed": VECTOR_SEED,
                "vector_minimum": VECTOR_MINIMUM,
                "vector_maximum": VECTOR_MAXIMUM,
                "warmup_runs": WARMUP_RUNS,
                "measured_runs": MEASURED_RUNS,
                "load_and_conversion_seconds": load_seconds,
                "matrix_data_dtype": str(matrix.data.dtype),
                "vector_dtype": str(vector.dtype),
                "result_dtype": str(result.dtype),
            }
        )

        print(
            f"Run {run_number}/{MEASURED_RUNS}: "
            f"{elapsed_seconds:.9f}s, "
            f"checksum={checksum}, "
            f"output NNZ={output_nonzero_count}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0])

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

    times = [
        float(row["elapsed_seconds"])
        for row in rows
    ]

    quartiles = statistics.quantiles(
        times,
        n=4,
        method="inclusive",
    )

    median = statistics.median(times)
    interquartile_range = quartiles[2] - quartiles[0]

    print("\nMC2DEPI SPARSE MATRIX-VECTOR SUMMARY")
    print(f"Shape: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"Matrix NNZ: {matrix.nnz}")
    print(f"Load and CSR conversion: {load_seconds:.6f}s")
    print(f"Measured runs: {MEASURED_RUNS}")
    print(f"Median multiplication time: {median:.9f}s")
    print(
        f"Multiplication-time IQR: "
        f"{interquartile_range:.9f}s"
    )
    print(f"Checksum: {reference_checksum}")
    print(
        f"Output nonzero count: "
        f"{reference_output_nonzero_count}"
    )
    print(f"Raw results saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse dataset and output paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark sparse matrix-vector multiplication "
            "using the mc2depi dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/mc2depi/mc2depi.mtx"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/raw/mc2depi_spmv_time.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the mc2depi case-study benchmark."""
    arguments = parse_arguments()

    print("MC2DEPI SPARSE MATRIX-VECTOR BENCHMARK")
    print(f"Input: {arguments.input}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Measured runs: {MEASURED_RUNS}")
    print(f"Vector seed: {VECTOR_SEED}")
    print(
        f"Vector value range: "
        f"[{VECTOR_MINIMUM}, {VECTOR_MAXIMUM}]"
    )

    run_benchmark(
        input_path=arguments.input,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()
