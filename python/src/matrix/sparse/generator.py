"""Deterministic generators for sparse benchmark matrices."""

import random

from python.src.matrix.dense.matrix_basic import Matrix


def validate_sparsity(sparsity: float) -> None:
    """Validate a sparsity value expressed as a proportion of zero entries."""
    if isinstance(sparsity, bool) or not isinstance(
        sparsity,
        (int, float),
    ):
        raise TypeError("sparsity must be numeric")

    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError("sparsity must be between 0.0 and 1.0")


def generate_sparse_dense_matrix(
    rows: int,
    columns: int,
    sparsity: float,
    seed: int,
    minimum_value: int = 1,
    maximum_value: int = 10,
) -> Matrix:
    """
    Generate a reproducible sparse matrix in dense logical form.

    The exact number of nonzero positions is calculated from the requested
    sparsity. Positions are sampled without replacement and values are drawn
    from a discrete uniform distribution that excludes zero.
    """
    if not isinstance(rows, int) or isinstance(rows, bool):
        raise TypeError("rows must be an integer")

    if not isinstance(columns, int) or isinstance(columns, bool):
        raise TypeError("columns must be an integer")

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix dimensions must be greater than zero")

    validate_sparsity(sparsity)

    if minimum_value > maximum_value:
        raise ValueError(
            "minimum_value must not be greater than maximum_value"
        )

    allowed_values = [
        value
        for value in range(minimum_value, maximum_value + 1)
        if value != 0
    ]

    if not allowed_values:
        raise ValueError(
            "the requested value range must contain a nonzero integer"
        )

    total_entries = rows * columns
    requested_nonzero_count = round(
        total_entries * (1.0 - float(sparsity))
    )

    generator = random.Random(seed)

    selected_positions = generator.sample(
        range(total_entries),
        requested_nonzero_count,
    )

    matrix: Matrix = [
        [0 for _ in range(columns)]
        for _ in range(rows)
    ]

    for flat_position in selected_positions:
        row = flat_position // columns
        column = flat_position % columns
        matrix[row][column] = generator.choice(allowed_values)

    return matrix


def generate_sparse_matrix_pair(
    size: int,
    sparsity: float,
    seed: int,
    minimum_value: int = 1,
    maximum_value: int = 10,
) -> tuple[Matrix, Matrix]:
    """
    Generate two reproducible square sparse matrices.

    The returned matrices use dense logical form so that identical input
    values can later be converted to custom CSR, SciPy CSR or retained as
    dense matrices for controlled comparisons.
    """
    matrix_a = generate_sparse_dense_matrix(
        rows=size,
        columns=size,
        sparsity=sparsity,
        seed=seed,
        minimum_value=minimum_value,
        maximum_value=maximum_value,
    )

    matrix_b = generate_sparse_dense_matrix(
        rows=size,
        columns=size,
        sparsity=sparsity,
        seed=seed + 1,
        minimum_value=minimum_value,
        maximum_value=maximum_value,
    )

    return matrix_a, matrix_b


def count_nonzero(matrix: Matrix) -> int:
    """Count the nonzero values in a logical matrix."""
    return sum(
        value != 0
        for row in matrix
        for value in row
    )


def calculate_sparsity(matrix: Matrix) -> float:
    """Calculate the observed proportion of zero values."""
    rows = len(matrix)
    columns = len(matrix[0])
    total_entries = rows * columns

    return 1.0 - (count_nonzero(matrix) / total_entries)
