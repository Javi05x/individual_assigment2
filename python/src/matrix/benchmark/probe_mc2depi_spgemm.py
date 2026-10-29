"""Single exploratory sparse matrix-matrix product for mc2depi."""

import gc
import json
import time
from pathlib import Path

import psutil
from scipy.io import mmread


INPUT_PATH = Path("data/mc2depi/mc2depi.mtx")
OUTPUT_PATH = Path(
    "results/raw/mc2depi_spgemm_probe.json"
)


def csr_storage_bytes(matrix) -> int:
    """Return the combined size of the CSR arrays."""
    return (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )


def main() -> None:
    """Load mc2depi and perform one exploratory A x A product."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    process = psutil.Process()

    print("Loading mc2depi as CSR...")

    load_start = time.perf_counter()

    matrix = mmread(INPUT_PATH).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    load_seconds = time.perf_counter() - load_start

    if matrix.shape != (525_825, 525_825):
        raise RuntimeError(
            f"Unexpected shape: {matrix.shape}"
        )

    if matrix.nnz != 2_100_225:
        raise RuntimeError(
            f"Unexpected input NNZ: {matrix.nnz}"
        )

    gc.collect()
    rss_before_product_mb = (
        process.memory_info().rss / (1024 * 1024)
    )

    print("Computing mc2depi x mc2depi...")

    multiplication_start = time.perf_counter()
    result = matrix @ matrix
    multiplication_seconds = (
        time.perf_counter() - multiplication_start
    )

    result = result.tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()

    gc.collect()
    rss_after_product_mb = (
        process.memory_info().rss / (1024 * 1024)
    )

    input_storage_bytes = csr_storage_bytes(matrix)
    output_storage_bytes = csr_storage_bytes(result)

    output_checksum = int(result.sum())

    measurement = {
        "dataset": "mc2depi",
        "operation": "sparse_matrix_matrix_A_times_A",
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "input_nonzero_count": int(matrix.nnz),
        "output_nonzero_count": int(result.nnz),
        "output_sparsity": (
            1.0
            - result.nnz
            / (result.shape[0] * result.shape[1])
        ),
        "load_and_conversion_seconds": load_seconds,
        "multiplication_seconds": multiplication_seconds,
        "input_csr_storage_bytes": input_storage_bytes,
        "input_csr_storage_mib": (
            input_storage_bytes / (1024 * 1024)
        ),
        "output_csr_storage_bytes": output_storage_bytes,
        "output_csr_storage_mib": (
            output_storage_bytes / (1024 * 1024)
        ),
        "rss_before_product_mb": rss_before_product_mb,
        "rss_after_product_mb": rss_after_product_mb,
        "rss_observed_delta_mb": (
            rss_after_product_mb - rss_before_product_mb
        ),
        "output_checksum": output_checksum,
        "input_data_dtype": str(matrix.data.dtype),
        "output_data_dtype": str(result.data.dtype),
        "memory_note": (
            "RSS was observed before and after the product; "
            "this exploratory probe does not measure peak RSS."
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            measurement,
            output_file,
            indent=2,
        )

    print("\nMC2DEPI SPARSE MATRIX-MATRIX PROBE")
    print(f"Input shape: {matrix.shape}")
    print(f"Input NNZ: {matrix.nnz}")
    print(f"Output NNZ: {result.nnz}")
    print(
        f"Output sparsity: "
        f"{measurement['output_sparsity'] * 100:.8f}%"
    )
    print(
        f"Multiplication time: "
        f"{multiplication_seconds:.6f}s"
    )
    print(
        f"Input CSR storage: "
        f"{measurement['input_csr_storage_mib']:.3f} MiB"
    )
    print(
        f"Output CSR storage: "
        f"{measurement['output_csr_storage_mib']:.3f} MiB"
    )
    print(
        f"Observed RSS change: "
        f"{measurement['rss_observed_delta_mb']:.3f} MiB"
    )
    print(f"Output checksum: {output_checksum}")
    print(f"Probe saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
