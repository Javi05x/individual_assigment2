"""Load and characterise the mc2depi Matrix Market dataset."""

import argparse
import gc
import json
import time
from pathlib import Path

import psutil
from scipy.io import mmread
from scipy.sparse import csr_matrix


EXPECTED_ROWS = 525_825
EXPECTED_COLUMNS = 525_825
EXPECTED_STORED_ENTRIES = 2_100_225


def read_matrix_market_header(
    input_path: Path,
) -> dict[str, object]:
    """Read the Matrix Market banner and declared dimensions."""
    with input_path.open(
        "r",
        encoding="utf-8",
        errors="strict",
    ) as matrix_file:
        banner = matrix_file.readline().strip()

        banner_parts = banner.split()

        if len(banner_parts) != 5:
            raise ValueError("Invalid Matrix Market banner")

        if banner_parts[0] != "%%MatrixMarket":
            raise ValueError("Missing Matrix Market banner")

        if banner_parts[1:] != [
            "matrix",
            "coordinate",
            "integer",
            "general",
        ]:
            raise ValueError(
                "Expected a general integer matrix in coordinate format"
            )

        for line in matrix_file:
            stripped_line = line.strip()

            if not stripped_line or stripped_line.startswith("%"):
                continue

            dimension_parts = stripped_line.split()

            if len(dimension_parts) != 3:
                raise ValueError(
                    "Invalid Matrix Market dimension line"
                )

            rows, columns, stored_entries = map(
                int,
                dimension_parts,
            )

            return {
                "banner": banner,
                "rows": rows,
                "columns": columns,
                "declared_stored_entries": stored_entries,
            }

    raise ValueError(
        "Matrix Market dimension line was not found"
    )


def validate_declared_metadata(
    metadata: dict[str, object],
) -> None:
    """Validate the known mc2depi dimensions and entry count."""
    if int(metadata["rows"]) != EXPECTED_ROWS:
        raise ValueError(
            "Unexpected mc2depi row count: "
            f"{metadata['rows']}"
        )

    if int(metadata["columns"]) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected mc2depi column count: "
            f"{metadata['columns']}"
        )

    if (
        int(metadata["declared_stored_entries"])
        != EXPECTED_STORED_ENTRIES
    ):
        raise ValueError(
            "Unexpected mc2depi stored-entry count: "
            f"{metadata['declared_stored_entries']}"
        )


def load_as_csr(
    input_path: Path,
) -> tuple[csr_matrix, float]:
    """Load the Matrix Market file and convert it directly to CSR."""
    start_time = time.perf_counter()

    coordinate_matrix = mmread(input_path)
    matrix = coordinate_matrix.tocsr()

    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    end_time = time.perf_counter()

    return matrix, end_time - start_time


def calculate_csr_storage_bytes(
    matrix: csr_matrix,
) -> int:
    """Return the bytes occupied by the three CSR arrays."""
    return (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )


def analyse_dataset(
    input_path: Path,
) -> dict[str, object]:
    """Load, validate and characterise the mc2depi dataset."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    process = psutil.Process()

    gc.collect()
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    metadata = read_matrix_market_header(input_path)
    validate_declared_metadata(metadata)

    matrix, load_seconds = load_as_csr(input_path)

    gc.collect()
    rss_after_mb = process.memory_info().rss / (1024 * 1024)

    rows, columns = matrix.shape
    actual_nonzero_count = int(matrix.nnz)
    total_entries = rows * columns
    density = actual_nonzero_count / total_entries
    sparsity = 1.0 - density

    if rows != int(metadata["rows"]):
        raise RuntimeError(
            "Loaded row count differs from declared row count"
        )

    if columns != int(metadata["columns"]):
        raise RuntimeError(
            "Loaded column count differs from declared column count"
        )

    if (
        actual_nonzero_count
        != int(metadata["declared_stored_entries"])
    ):
        raise RuntimeError(
            "Loaded nonzero count differs from the declared "
            "stored-entry count"
        )

    csr_storage_bytes = calculate_csr_storage_bytes(matrix)

    dense_int64_bytes = total_entries * 8

    return {
        "dataset": "mc2depi",
        "input_path": str(input_path),
        "matrix_market_banner": metadata["banner"],
        "rows": rows,
        "columns": columns,
        "declared_stored_entries": metadata[
            "declared_stored_entries"
        ],
        "actual_nonzero_count": actual_nonzero_count,
        "total_logical_entries": total_entries,
        "density": density,
        "sparsity": sparsity,
        "load_and_conversion_seconds": load_seconds,
        "rss_before_load_mb": rss_before_mb,
        "rss_after_load_mb": rss_after_mb,
        "rss_load_delta_mb": rss_after_mb - rss_before_mb,
        "csr_data_bytes": int(matrix.data.nbytes),
        "csr_indices_bytes": int(matrix.indices.nbytes),
        "csr_indptr_bytes": int(matrix.indptr.nbytes),
        "csr_total_storage_bytes": csr_storage_bytes,
        "csr_total_storage_mib": (
            csr_storage_bytes / (1024 * 1024)
        ),
        "theoretical_dense_int64_bytes": dense_int64_bytes,
        "theoretical_dense_int64_gib": (
            dense_int64_bytes / (1024 ** 3)
        ),
        "data_dtype": str(matrix.data.dtype),
        "indices_dtype": str(matrix.indices.dtype),
        "indptr_dtype": str(matrix.indptr.dtype),
        "explicit_zero_count_after_loading": int(
            (matrix.data == 0).sum()
        ),
        "minimum_stored_value": int(matrix.data.min()),
        "maximum_stored_value": int(matrix.data.max()),
        "sampling_note": (
            "RSS values describe the complete Python process, "
            "not only the matrix arrays."
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse dataset and output paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Load and characterise mc2depi without creating "
            "a dense representation."
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
            "results/raw/mc2depi_characterisation.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Analyse mc2depi and preserve the results as JSON."""
    arguments = parse_arguments()

    results = analyse_dataset(arguments.input)

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with arguments.output.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            results,
            output_file,
            indent=2,
        )

    print("MC2DEPI DATASET CHARACTERISATION")
    print(f"Shape: {results['rows']} x {results['columns']}")
    print(
        f"Nonzero entries: "
        f"{results['actual_nonzero_count']}"
    )
    print(
        f"Sparsity: "
        f"{float(results['sparsity']) * 100:.8f}%"
    )
    print(
        f"Load and CSR conversion: "
        f"{float(results['load_and_conversion_seconds']):.6f}s"
    )
    print(
        f"CSR array storage: "
        f"{float(results['csr_total_storage_mib']):.3f} MiB"
    )
    print(
        f"Theoretical dense int64 storage: "
        f"{float(results['theoretical_dense_int64_gib']):.3f} GiB"
    )
    print(
        f"RSS increase after loading: "
        f"{float(results['rss_load_delta_mb']):.3f} MiB"
    )
    print(
        f"Value range: "
        f"[{results['minimum_stored_value']}, "
        f"{results['maximum_stored_value']}]"
    )
    print(f"Results saved to: {arguments.output}")


if __name__ == "__main__":
    main()
