"""Final sparse matrix-matrix benchmark for the mc2depi dataset."""

import argparse
import csv
import gc
import hashlib
import statistics
import time
from pathlib import Path

from scipy.io import mmread
from scipy.sparse import csr_matrix


WARMUP_RUNS = 1
MEASURED_RUNS = 5

EXPECTED_ROWS = 525_825
EXPECTED_COLUMNS = 525_825
EXPECTED_INPUT_NONZERO_COUNT = 2_100_225


def load_mc2depi(
    input_path: Path,
) -> tuple[csr_matrix, float]:
    """Load mc2depi and convert it into canonical CSR format."""
    start_time = time.perf_counter()

    matrix = mmread(
        input_path,
        spmatrix=True,
    ).tocsr()

    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    end_time = time.perf_counter()

    if matrix.shape != (
        EXPECTED_ROWS,
        EXPECTED_COLUMNS,
    ):
        raise RuntimeError(
            f"Unexpected mc2depi shape: {matrix.shape}"
        )

    if matrix.nnz != EXPECTED_INPUT_NONZERO_COUNT:
        raise RuntimeError(
            f"Unexpected mc2depi NNZ: {matrix.nnz}"
        )

    return matrix, end_time - start_time


def csr_storage_bytes(matrix: csr_matrix) -> int:
    """Calculate the combined storage of the three CSR arrays."""
    return int(
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )


def csr_digest(matrix: csr_matrix) -> str:
    """
    Calculate a deterministic SHA-256 digest of a canonical CSR matrix.

    The complete data, column-index and row-pointer arrays are included.
    """
    digest = hashlib.sha256()

    digest.update(matrix.data.tobytes())
    digest.update(matrix.indices.tobytes())
    digest.update(matrix.indptr.tobytes())

    return digest.hexdigest()


def canonicalise_result(
    result: csr_matrix,
) -> csr_matrix:
    """Convert a result to canonical CSR form."""
    result = result.tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()

    return result


def run_benchmark(
    input_path: Path,
    output_path: Path,
) -> None:
    """Run the final repeated mc2depi sparse matrix product."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    matrix, load_seconds = load_mc2depi(input_path)

    for _ in range(WARMUP_RUNS):
        warmup_result = canonicalise_result(
            matrix @ matrix
        )
        csr_digest(warmup_result)

    rows: list[dict[str, object]] = []

    reference_digest: str | None = None
    reference_nonzero_count: int | None = None
    reference_storage_bytes: int | None = None
    reference_sum: int | None = None
    reference_absolute_sum: int | None = None

    for run_number in range(1, MEASURED_RUNS + 1):
        gc.collect()

        start_time = time.perf_counter_ns()
        result = matrix @ matrix
        end_time = time.perf_counter_ns()

        elapsed_seconds = (
            end_time - start_time
        ) / 1_000_000_000

        result = canonicalise_result(result)

        result_digest = csr_digest(result)
        output_nonzero_count = int(result.nnz)
        output_storage_bytes = csr_storage_bytes(result)

        output_sum = int(
            sum(int(value) for value in result.data)
        )
        output_absolute_sum = int(
            sum(abs(int(value)) for value in result.data)
        )

        if reference_digest is None:
            reference_digest = result_digest
            reference_nonzero_count = output_nonzero_count
            reference_storage_bytes = output_storage_bytes
            reference_sum = output_sum
            reference_absolute_sum = output_absolute_sum
        else:
            if result_digest != reference_digest:
                raise RuntimeError(
                    f"CSR digest disagreement in run {run_number}"
                )

            if output_nonzero_count != reference_nonzero_count:
                raise RuntimeError(
                    f"Output NNZ disagreement in run {run_number}"
                )

            if output_storage_bytes != reference_storage_bytes:
                raise RuntimeError(
                    f"Storage disagreement in run {run_number}"
                )

            if output_sum != reference_sum:
                raise RuntimeError(
                    f"Output-sum disagreement in run {run_number}"
                )

            if output_absolute_sum != reference_absolute_sum:
                raise RuntimeError(
                    "Absolute-output-sum disagreement in "
                    f"run {run_number}"
                )

        output_sparsity = 1.0 - (
            output_nonzero_count
            / (result.shape[0] * result.shape[1])
        )

        rows.append(
            {
                "dataset": "mc2depi",
                "operation": "sparse_matrix_matrix_A_times_A",
                "run": run_number,
                "elapsed_seconds": elapsed_seconds,
                "rows": matrix.shape[0],
                "columns": matrix.shape[1],
                "input_nonzero_count": matrix.nnz,
                "output_nonzero_count": output_nonzero_count,
                "output_sparsity": output_sparsity,
                "input_csr_storage_bytes": (
                    csr_storage_bytes(matrix)
                ),
                "output_csr_storage_bytes": (
                    output_storage_bytes
                ),
                "output_sum": output_sum,
                "output_absolute_sum": output_absolute_sum,
                "output_sha256": result_digest,
                "warmup_runs": WARMUP_RUNS,
                "measured_runs": MEASURED_RUNS,
                "load_and_conversion_seconds": load_seconds,
                "input_data_dtype": str(matrix.data.dtype),
                "output_data_dtype": str(result.data.dtype),
            }
        )

        print(
            f"Run {run_number}/{MEASURED_RUNS}: "
            f"{elapsed_seconds:.9f}s, "
            f"output NNZ={output_nonzero_count}, "
            f"absolute sum={output_absolute_sum}"
        )

        del result

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0]),
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
    interquartile_range = (
        quartiles[2] - quartiles[0]
    )

    print("\nMC2DEPI SPARSE MATRIX-MATRIX SUMMARY")
    print(f"Shape: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"Input NNZ: {matrix.nnz}")
    print(f"Output NNZ: {reference_nonzero_count}")
    print(
        "Output sparsity: "
        f"{float(rows[0]['output_sparsity']) * 100:.8f}%"
    )
    print(f"Measured runs: {MEASURED_RUNS}")
    print(f"Median multiplication time: {median:.9f}s")
    print(
        f"Multiplication-time IQR: "
        f"{interquartile_range:.9f}s"
    )
    print(f"Output sum: {reference_sum}")
    print(
        f"Output absolute sum: "
        f"{reference_absolute_sum}"
    )
    print(f"Output SHA-256: {reference_digest}")
    print(f"Raw results saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse the input and output file paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark mc2depi sparse matrix-matrix "
            "multiplication."
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
            "results/raw/mc2depi_spgemm_time.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Execute the final mc2depi SpGEMM benchmark."""
    arguments = parse_arguments()

    print("MC2DEPI SPARSE MATRIX-MATRIX BENCHMARK")
    print(f"Input: {arguments.input}")
    print("Operation: A x A")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Measured runs: {MEASURED_RUNS}")

    run_benchmark(
        input_path=arguments.input,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()
