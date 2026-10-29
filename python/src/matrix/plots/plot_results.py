"""Generate report-ready figures from final processed benchmark results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROCESSED_DIRECTORY = Path("results/processed")
PLOT_DIRECTORY = Path("plots")

ALGORITHM_LABELS = {
    "basic_ijk": "Basic i-j-k",
    "row_oriented_ikj": "Row-oriented i-k-j",
    "tiled_ikj_32": "Tiled i-k-j, block 32",
    "dense_row_oriented": "Dense row-oriented",
    "custom_csr": "Custom CSR",
    "scipy_csr": "SciPy CSR",
}


def save_figure(
    filename: str,
) -> None:
    """Save the current figure and close it."""
    output_path = PLOT_DIRECTORY / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path}")


def plot_dense_execution_time() -> None:
    """Plot dense multiplication median execution time."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "dense_time_summary.csv"
    )

    plt.figure(figsize=(8, 5))

    for algorithm, algorithm_data in dataframe.groupby(
        "algorithm"
    ):
        algorithm_data = algorithm_data.sort_values("size")

        plt.errorbar(
            algorithm_data["size"],
            algorithm_data["median"],
            yerr=algorithm_data["iqr"] / 2,
            marker="o",
            capsize=4,
            label=ALGORITHM_LABELS[algorithm],
        )

    plt.xlabel("Matrix side length")
    plt.ylabel("Median execution time (seconds)")
    plt.title("Dense Matrix Multiplication Time")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    save_figure("dense_execution_time.png")


def plot_dense_speedup() -> None:
    """Plot speedup of dense optimisations over the basic algorithm."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "dense_time_summary.csv"
    )

    optimised = dataframe[
        dataframe["algorithm"] != "basic_ijk"
    ]

    plt.figure(figsize=(8, 5))

    for algorithm, algorithm_data in optimised.groupby(
        "algorithm"
    ):
        algorithm_data = algorithm_data.sort_values("size")

        plt.plot(
            algorithm_data["size"],
            algorithm_data["speedup_over_basic"],
            marker="o",
            label=ALGORITHM_LABELS[algorithm],
        )

    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
        label="Basic i-j-k reference",
    )

    plt.xlabel("Matrix side length")
    plt.ylabel("Speedup over basic i-j-k")
    plt.title("Dense Optimisation Speedup")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure("dense_speedup.png")


def plot_dense_memory() -> None:
    """Plot observed dense multiplication peak RSS increase."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "dense_memory_summary.csv"
    )

    plt.figure(figsize=(8, 5))

    for algorithm, algorithm_data in dataframe.groupby(
        "algorithm"
    ):
        algorithm_data = algorithm_data.sort_values("size")

        plt.errorbar(
            algorithm_data["size"],
            algorithm_data["median"],
            yerr=algorithm_data["iqr"] / 2,
            marker="o",
            capsize=4,
            label=ALGORITHM_LABELS[algorithm],
        )

    plt.xlabel("Matrix side length")
    plt.ylabel("Median observed peak RSS increase (MiB)")
    plt.title("Dense Multiplication Memory Usage")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure("dense_memory.png")


def plot_sparse_execution_time_by_size() -> None:
    """Plot sparse benchmark time in one panel per matrix size."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "sparse_time_summary.csv"
    )

    sizes = sorted(dataframe["size"].unique())

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
    )

    for axis, size in zip(axes.flat, sizes):
        size_data = dataframe[dataframe["size"] == size]

        for algorithm, algorithm_data in size_data.groupby(
            "algorithm"
        ):
            algorithm_data = algorithm_data.sort_values(
                "requested_sparsity"
            )

            axis.errorbar(
                algorithm_data["requested_sparsity"] * 100,
                algorithm_data["median"],
                yerr=algorithm_data["iqr"] / 2,
                marker="o",
                capsize=3,
                label=ALGORITHM_LABELS[algorithm],
            )

        axis.set_title(f"{size} x {size}")
        axis.set_yscale("log")
        axis.set_xlabel("Input sparsity (%)")
        axis.set_ylabel("Median execution time (seconds)")
        axis.grid(True, which="both", alpha=0.3)

    handles, labels = axes.flat[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
    )

    figure.suptitle(
        "Dense and Sparse Multiplication Time",
        y=1.02,
    )

    save_figure("sparse_execution_time.png")


def plot_sparse_speedup() -> None:
    """Plot sparse implementation speedup over the matched dense method."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "sparse_time_summary.csv"
    )

    sparse_only = dataframe[
        dataframe["algorithm"].isin(
            ["custom_csr", "scipy_csr"]
        )
    ]

    sizes = sorted(sparse_only["size"].unique())

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
    )

    for axis, size in zip(axes.flat, sizes):
        size_data = sparse_only[
            sparse_only["size"] == size
        ]

        for algorithm, algorithm_data in size_data.groupby(
            "algorithm"
        ):
            algorithm_data = algorithm_data.sort_values(
                "requested_sparsity"
            )

            axis.plot(
                algorithm_data["requested_sparsity"] * 100,
                algorithm_data["speedup_over_dense"],
                marker="o",
                label=ALGORITHM_LABELS[algorithm],
            )

        axis.axhline(
            y=1.0,
            linestyle="--",
            linewidth=1,
            label="Dense reference",
        )
        axis.set_title(f"{size} x {size}")
        axis.set_yscale("log")
        axis.set_xlabel("Input sparsity (%)")
        axis.set_ylabel("Speedup over dense")
        axis.grid(True, which="both", alpha=0.3)

    handles, labels = axes.flat[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
    )

    figure.suptitle(
        "Sparse Multiplication Speedup over Dense",
        y=1.02,
    )

    save_figure("sparse_speedup.png")


def plot_output_sparsity() -> None:
    """Plot output sparsity relative to input sparsity."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY
        / "sparse_configuration_summary.csv"
    )

    plt.figure(figsize=(8, 5))

    for size, size_data in dataframe.groupby("size"):
        size_data = size_data.sort_values(
            "requested_sparsity"
        )

        plt.plot(
            size_data["requested_sparsity"] * 100,
            size_data["output_sparsity"] * 100,
            marker="o",
            label=f"{size} x {size}",
        )

    plt.plot(
        [50, 99],
        [50, 99],
        linestyle="--",
        linewidth=1,
        label="Input equals output sparsity",
    )

    plt.xlabel("Input sparsity (%)")
    plt.ylabel("Output sparsity (%)")
    plt.title("Product Densification")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure("output_sparsity.png")


def plot_sparse_total_memory() -> None:
    """Plot total observed peak RSS increase at the largest matrix size."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "sparse_memory_summary.csv"
    )

    largest_size = int(dataframe["size"].max())

    dataframe = dataframe[
        dataframe["size"] == largest_size
    ]

    plt.figure(figsize=(8, 5))

    for algorithm, algorithm_data in dataframe.groupby(
        "algorithm"
    ):
        algorithm_data = algorithm_data.sort_values(
            "requested_sparsity"
        )

        plt.errorbar(
            algorithm_data["requested_sparsity"] * 100,
            algorithm_data["total_peak_median"],
            yerr=algorithm_data["total_peak_iqr"] / 2,
            marker="o",
            capsize=4,
            label=ALGORITHM_LABELS[algorithm],
        )

    plt.xlabel("Input sparsity (%)")
    plt.ylabel("Median total observed peak RSS increase (MiB)")
    plt.title(
        f"Memory Usage at {largest_size} x {largest_size}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure("sparse_memory_512.png")


def main() -> None:
    """Generate every final report figure."""
    PLOT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dense_execution_time()
    plot_dense_speedup()
    plot_dense_memory()
    plot_sparse_execution_time_by_size()
    plot_sparse_speedup()
    plot_output_sparsity()
    plot_sparse_total_memory()

    print("\nAll final plots were generated successfully.")


if __name__ == "__main__":
    main()
PYcat > python/src/matrix/plots/plot_results.py <<'PY'
"""Generate report-ready figures from final processed benchmark results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROCESSED_DIRECTORY = Path("results/processed")
PLOT_DIRECTORY = Path("plots")

ALGORITHM_LABELS = {
    "basic_ijk": "Basic i-j-k",
    "row_oriented_ikj": "Row-oriented i-k-j",
    "tiled_ikj_32": "Tiled i-k-j, block 32",
    "dense_row_oriented": "Dense row-oriented",
    "custom_csr": "Custom CSR",
    "scipy_csr": "SciPy CSR",
}


def save_figure(
    filename: str,
) -> None:
    """Save the current figure and close it."""
    output_path = PLOT_DIRECTORY / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path}")


def plot_dense_execution_time() -> None:
    """Plot dense multiplication median execution time."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "dense_time_summary.csv"
    )

    plt.figure(figsize=(8, 5))

    for algorithm, algorithm_data in dataframe.groupby(
        "algorithm"
    ):
        algorithm_data = algorithm_data.sort_values("size")

        plt.errorbar(
            algorithm_data["size"],
            algorithm_data["median"],
            yerr=algorithm_data["iqr"] / 2,
            marker="o",
            capsize=4,
            label=ALGORITHM_LABELS[algorithm],
        )

    plt.xlabel("Matrix side length")
    plt.ylabel("Median execution time (seconds)")
    plt.title("Dense Matrix Multiplication Time")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    save_figure("dense_execution_time.png")


def plot_dense_speedup() -> None:
    """Plot speedup of dense optimisations over the basic algorithm."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "dense_time_summary.csv"
    )

    optimised = dataframe[
        dataframe["algorithm"] != "basic_ijk"
    ]

    plt.figure(figsize=(8, 5))

    for algorithm, algorithm_data in optimised.groupby(
        "algorithm"
    ):
        algorithm_data = algorithm_data.sort_values("size")

        plt.plot(
            algorithm_data["size"],
            algorithm_data["speedup_over_basic"],
            marker="o",
            label=ALGORITHM_LABELS[algorithm],
        )

    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
        label="Basic i-j-k reference",
    )

    plt.xlabel("Matrix side length")
    plt.ylabel("Speedup over basic i-j-k")
    plt.title("Dense Optimisation Speedup")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure("dense_speedup.png")


def plot_dense_memory() -> None:
    """Plot observed dense multiplication peak RSS increase."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "dense_memory_summary.csv"
    )

    plt.figure(figsize=(8, 5))

    for algorithm, algorithm_data in dataframe.groupby(
        "algorithm"
    ):
        algorithm_data = algorithm_data.sort_values("size")

        plt.errorbar(
            algorithm_data["size"],
            algorithm_data["median"],
            yerr=algorithm_data["iqr"] / 2,
            marker="o",
            capsize=4,
            label=ALGORITHM_LABELS[algorithm],
        )

    plt.xlabel("Matrix side length")
    plt.ylabel("Median observed peak RSS increase (MiB)")
    plt.title("Dense Multiplication Memory Usage")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure("dense_memory.png")


def plot_sparse_execution_time_by_size() -> None:
    """Plot sparse benchmark time in one panel per matrix size."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "sparse_time_summary.csv"
    )

    sizes = sorted(dataframe["size"].unique())

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
    )

    for axis, size in zip(axes.flat, sizes):
        size_data = dataframe[dataframe["size"] == size]

        for algorithm, algorithm_data in size_data.groupby(
            "algorithm"
        ):
            algorithm_data = algorithm_data.sort_values(
                "requested_sparsity"
            )

            axis.errorbar(
                algorithm_data["requested_sparsity"] * 100,
                algorithm_data["median"],
                yerr=algorithm_data["iqr"] / 2,
                marker="o",
                capsize=3,
                label=ALGORITHM_LABELS[algorithm],
            )

        axis.set_title(f"{size} x {size}")
        axis.set_yscale("log")
        axis.set_xlabel("Input sparsity (%)")
        axis.set_ylabel("Median execution time (seconds)")
        axis.grid(True, which="both", alpha=0.3)

    handles, labels = axes.flat[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
    )

    figure.suptitle(
        "Dense and Sparse Multiplication Time",
        y=1.02,
    )

    save_figure("sparse_execution_time.png")


def plot_sparse_speedup() -> None:
    """Plot sparse implementation speedup over the matched dense method."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "sparse_time_summary.csv"
    )

    sparse_only = dataframe[
        dataframe["algorithm"].isin(
            ["custom_csr", "scipy_csr"]
        )
    ]

    sizes = sorted(sparse_only["size"].unique())

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
    )

    for axis, size in zip(axes.flat, sizes):
        size_data = sparse_only[
            sparse_only["size"] == size
        ]

        for algorithm, algorithm_data in size_data.groupby(
            "algorithm"
        ):
            algorithm_data = algorithm_data.sort_values(
                "requested_sparsity"
            )

            axis.plot(
                algorithm_data["requested_sparsity"] * 100,
                algorithm_data["speedup_over_dense"],
                marker="o",
                label=ALGORITHM_LABELS[algorithm],
            )

        axis.axhline(
            y=1.0,
            linestyle="--",
            linewidth=1,
            label="Dense reference",
        )
        axis.set_title(f"{size} x {size}")
        axis.set_yscale("log")
        axis.set_xlabel("Input sparsity (%)")
        axis.set_ylabel("Speedup over dense")
        axis.grid(True, which="both", alpha=0.3)

    handles, labels = axes.flat[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
    )

    figure.suptitle(
        "Sparse Multiplication Speedup over Dense",
        y=1.02,
    )

    save_figure("sparse_speedup.png")


def plot_output_sparsity() -> None:
    """Plot output sparsity relative to input sparsity."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY
        / "sparse_configuration_summary.csv"
    )

    plt.figure(figsize=(8, 5))

    for size, size_data in dataframe.groupby("size"):
        size_data = size_data.sort_values(
            "requested_sparsity"
        )

        plt.plot(
            size_data["requested_sparsity"] * 100,
            size_data["output_sparsity"] * 100,
            marker="o",
            label=f"{size} x {size}",
        )

    plt.plot(
        [50, 99],
        [50, 99],
        linestyle="--",
        linewidth=1,
        label="Input equals output sparsity",
    )

    plt.xlabel("Input sparsity (%)")
    plt.ylabel("Output sparsity (%)")
    plt.title("Product Densification")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure("output_sparsity.png")


def plot_sparse_total_memory() -> None:
    """Plot total observed peak RSS increase at the largest matrix size."""
    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "sparse_memory_summary.csv"
    )

    largest_size = int(dataframe["size"].max())

    dataframe = dataframe[
        dataframe["size"] == largest_size
    ]

    plt.figure(figsize=(8, 5))

    for algorithm, algorithm_data in dataframe.groupby(
        "algorithm"
    ):
        algorithm_data = algorithm_data.sort_values(
            "requested_sparsity"
        )

        plt.errorbar(
            algorithm_data["requested_sparsity"] * 100,
            algorithm_data["total_peak_median"],
            yerr=algorithm_data["total_peak_iqr"] / 2,
            marker="o",
            capsize=4,
            label=ALGORITHM_LABELS[algorithm],
        )

    plt.xlabel("Input sparsity (%)")
    plt.ylabel("Median total observed peak RSS increase (MiB)")
    plt.title(
        f"Memory Usage at {largest_size} x {largest_size}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure("sparse_memory_512.png")


def main() -> None:
    """Generate every final report figure."""
    PLOT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dense_execution_time()
    plot_dense_speedup()
    plot_dense_memory()
    plot_sparse_execution_time_by_size()
    plot_sparse_speedup()
    plot_output_sparsity()
    plot_sparse_total_memory()

    print("\nAll final plots were generated successfully.")


if __name__ == "__main__":
    main()
