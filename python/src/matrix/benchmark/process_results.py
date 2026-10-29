"""Process final benchmark measurements into report-ready summaries."""

from pathlib import Path

import pandas as pd


RAW_DIRECTORY = Path("results/raw")
PROCESSED_DIRECTORY = Path("results/processed")


def summarise_measurements(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
) -> pd.DataFrame:
    """Calculate median, quartiles, IQR, minimum and maximum."""
    summary = (
        dataframe.groupby(group_columns)[value_column]
        .agg(
            measured_runs="count",
            minimum="min",
            first_quartile=lambda values: values.quantile(0.25),
            median="median",
            third_quartile=lambda values: values.quantile(0.75),
            maximum="max",
        )
        .reset_index()
    )

    summary["iqr"] = (
        summary["third_quartile"]
        - summary["first_quartile"]
    )

    return summary


def process_dense_time() -> pd.DataFrame:
    """Process the final dense execution-time benchmark."""
    input_path = RAW_DIRECTORY / "dense_time_full.csv"
    dataframe = pd.read_csv(input_path)

    expected_rows = 60

    if len(dataframe) != expected_rows:
        raise RuntimeError(
            f"{input_path} contains {len(dataframe)} rows; "
            f"expected {expected_rows}"
        )

    summary = summarise_measurements(
        dataframe=dataframe,
        group_columns=["size", "algorithm"],
        value_column="elapsed_seconds",
    )

    baseline = (
        summary[
            summary["algorithm"] == "basic_ijk"
        ][["size", "median"]]
        .rename(
            columns={
                "median": "basic_ijk_median_seconds"
            }
        )
    )

    summary = summary.merge(
        baseline,
        on="size",
        how="left",
    )

    summary["speedup_over_basic"] = (
        summary["basic_ijk_median_seconds"]
        / summary["median"]
    )

    output_path = (
        PROCESSED_DIRECTORY
        / "dense_time_summary.csv"
    )
    summary.to_csv(output_path, index=False)

    return summary


def process_dense_memory() -> pd.DataFrame:
    """Process the final dense RSS benchmark."""
    input_path = (
        RAW_DIRECTORY / "dense_memory_full.csv"
    )
    dataframe = pd.read_csv(input_path)

    expected_rows = 60

    if len(dataframe) != expected_rows:
        raise RuntimeError(
            f"{input_path} contains {len(dataframe)} rows; "
            f"expected {expected_rows}"
        )

    summary = summarise_measurements(
        dataframe=dataframe,
        group_columns=["size", "algorithm"],
        value_column="peak_delta_mb",
    )

    output_path = (
        PROCESSED_DIRECTORY
        / "dense_memory_summary.csv"
    )
    summary.to_csv(output_path, index=False)

    return summary


def process_sparse_time() -> pd.DataFrame:
    """Process the final dense-versus-sparse time benchmark."""
    input_path = RAW_DIRECTORY / "sparse_time_full.csv"
    dataframe = pd.read_csv(input_path)

    expected_rows = 300

    if len(dataframe) != expected_rows:
        raise RuntimeError(
            f"{input_path} contains {len(dataframe)} rows; "
            f"expected {expected_rows}"
        )

    summary = summarise_measurements(
        dataframe=dataframe,
        group_columns=[
            "size",
            "requested_sparsity",
            "algorithm",
        ],
        value_column="elapsed_seconds",
    )

    dense_reference = (
        summary[
            summary["algorithm"]
            == "dense_row_oriented"
        ][
            [
                "size",
                "requested_sparsity",
                "median",
            ]
        ]
        .rename(
            columns={
                "median": (
                    "dense_row_oriented_median_seconds"
                )
            }
        )
    )

    summary = summary.merge(
        dense_reference,
        on=["size", "requested_sparsity"],
        how="left",
    )

    summary["speedup_over_dense"] = (
        summary[
            "dense_row_oriented_median_seconds"
        ]
        / summary["median"]
    )

    metadata_columns = [
        "size",
        "requested_sparsity",
        "actual_sparsity_a",
        "actual_sparsity_b",
        "input_a_nonzero_count",
        "input_b_nonzero_count",
        "output_nonzero_count",
        "output_sparsity",
        "seed",
    ]

    configuration_metadata = (
        dataframe[metadata_columns]
        .drop_duplicates()
    )

    expected_configurations = 20

    if len(configuration_metadata) != expected_configurations:
        raise RuntimeError(
            "Unexpected number of unique sparse "
            f"configurations: {len(configuration_metadata)}"
        )

    summary = summary.merge(
        configuration_metadata,
        on=["size", "requested_sparsity"],
        how="left",
    )

    summary_output_path = (
        PROCESSED_DIRECTORY
        / "sparse_time_summary.csv"
    )
    summary.to_csv(
        summary_output_path,
        index=False,
    )

    metadata_output_path = (
        PROCESSED_DIRECTORY
        / "sparse_configuration_summary.csv"
    )
    configuration_metadata.to_csv(
        metadata_output_path,
        index=False,
    )

    return summary


def process_sparse_memory() -> pd.DataFrame:
    """Process the final dense-versus-sparse RSS benchmark."""
    input_path = (
        RAW_DIRECTORY / "sparse_memory_full.csv"
    )
    dataframe = pd.read_csv(input_path)

    expected_rows = 300

    if len(dataframe) != expected_rows:
        raise RuntimeError(
            f"{input_path} contains {len(dataframe)} rows; "
            f"expected {expected_rows}"
        )

    group_columns = [
        "size",
        "requested_sparsity",
        "algorithm",
    ]

    representation_summary = summarise_measurements(
        dataframe=dataframe,
        group_columns=group_columns,
        value_column="representation_delta_mb",
    ).rename(
        columns={
            column: f"representation_{column}"
            for column in [
                "measured_runs",
                "minimum",
                "first_quartile",
                "median",
                "third_quartile",
                "maximum",
                "iqr",
            ]
        }
    )

    multiplication_summary = summarise_measurements(
        dataframe=dataframe,
        group_columns=group_columns,
        value_column="multiplication_peak_delta_mb",
    ).rename(
        columns={
            column: f"multiplication_{column}"
            for column in [
                "measured_runs",
                "minimum",
                "first_quartile",
                "median",
                "third_quartile",
                "maximum",
                "iqr",
            ]
        }
    )

    total_summary = summarise_measurements(
        dataframe=dataframe,
        group_columns=group_columns,
        value_column="total_peak_delta_mb",
    ).rename(
        columns={
            column: f"total_peak_{column}"
            for column in [
                "measured_runs",
                "minimum",
                "first_quartile",
                "median",
                "third_quartile",
                "maximum",
                "iqr",
            ]
        }
    )

    summary = representation_summary.merge(
        multiplication_summary,
        on=group_columns,
    ).merge(
        total_summary,
        on=group_columns,
    )

    output_path = (
        PROCESSED_DIRECTORY
        / "sparse_memory_summary.csv"
    )
    summary.to_csv(output_path, index=False)

    return summary


def process_mc2depi_spmv() -> pd.DataFrame:
    """Process the final mc2depi matrix-vector benchmark."""
    input_path = (
        RAW_DIRECTORY / "mc2depi_spmv_time.csv"
    )
    dataframe = pd.read_csv(input_path)

    if len(dataframe) != 5:
        raise RuntimeError(
            f"{input_path} contains {len(dataframe)} rows; "
            "expected 5"
        )

    summary = summarise_measurements(
        dataframe=dataframe,
        group_columns=["dataset", "operation"],
        value_column="elapsed_seconds",
    )

    metadata = dataframe.iloc[0][
        [
            "rows",
            "columns",
            "matrix_nonzero_count",
            "vector_length",
            "checksum",
            "output_nonzero_count",
            "vector_seed",
            "load_and_conversion_seconds",
        ]
    ]

    for column, value in metadata.items():
        summary[column] = value

    output_path = (
        PROCESSED_DIRECTORY
        / "mc2depi_spmv_summary.csv"
    )
    summary.to_csv(output_path, index=False)

    return summary


def process_mc2depi_spgemm() -> pd.DataFrame:
    """Process the final mc2depi matrix-matrix benchmark."""
    input_path = (
        RAW_DIRECTORY / "mc2depi_spgemm_time.csv"
    )
    dataframe = pd.read_csv(input_path)

    if len(dataframe) != 5:
        raise RuntimeError(
            f"{input_path} contains {len(dataframe)} rows; "
            "expected 5"
        )

    if dataframe["output_sha256"].nunique() != 1:
        raise RuntimeError(
            "mc2depi SpGEMM outputs do not have one "
            "common SHA-256 digest"
        )

    summary = summarise_measurements(
        dataframe=dataframe,
        group_columns=["dataset", "operation"],
        value_column="elapsed_seconds",
    )

    metadata = dataframe.iloc[0][
        [
            "rows",
            "columns",
            "input_nonzero_count",
            "output_nonzero_count",
            "output_sparsity",
            "input_csr_storage_bytes",
            "output_csr_storage_bytes",
            "output_sum",
            "output_absolute_sum",
            "output_sha256",
            "load_and_conversion_seconds",
        ]
    ]

    for column, value in metadata.items():
        summary[column] = value

    summary["input_csr_storage_mib"] = (
        summary["input_csr_storage_bytes"]
        / (1024 ** 2)
    )
    summary["output_csr_storage_mib"] = (
        summary["output_csr_storage_bytes"]
        / (1024 ** 2)
    )

    output_path = (
        PROCESSED_DIRECTORY
        / "mc2depi_spgemm_summary.csv"
    )
    summary.to_csv(output_path, index=False)

    return summary


def print_dense_key_results(
    summary: pd.DataFrame,
) -> None:
    """Print report-relevant dense results."""
    print("\nDENSE TIME SUMMARY")

    display_columns = [
        "size",
        "algorithm",
        "median",
        "iqr",
        "speedup_over_basic",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
        )
    )


def print_sparse_key_results(
    summary: pd.DataFrame,
) -> None:
    """Print report-relevant sparse results."""
    print("\nSPARSE TIME SUMMARY")

    display_columns = [
        "size",
        "requested_sparsity",
        "algorithm",
        "median",
        "iqr",
        "speedup_over_dense",
        "output_sparsity",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
        )
    )


def main() -> None:
    """Create all processed benchmark summaries."""
    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dense_time = process_dense_time()
    process_dense_memory()

    sparse_time = process_sparse_time()
    process_sparse_memory()

    spmv_summary = process_mc2depi_spmv()
    spgemm_summary = process_mc2depi_spgemm()

    print_dense_key_results(dense_time)
    print_sparse_key_results(sparse_time)

    print("\nMC2DEPI SPMV SUMMARY")
    print(spmv_summary.to_string(index=False))

    print("\nMC2DEPI SPGEMM SUMMARY")
    print(spgemm_summary.to_string(index=False))

    print(
        "\nProcessed summaries saved to: "
        f"{PROCESSED_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
