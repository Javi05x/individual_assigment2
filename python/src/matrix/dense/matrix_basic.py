"""Basic dense matrix multiplication implementation."""

from numbers import Real


Matrix = list[list[Real]]


def validate_matrix(matrix: Matrix, name: str) -> tuple[int, int]:
    """Validate a dense rectangular matrix and return its dimensions."""
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(f"{name} must be a non-empty list of rows")

    if not all(isinstance(row, list) and row for row in matrix):
        raise ValueError(f"{name} must contain non-empty rows")

    column_count = len(matrix[0])

    if any(len(row) != column_count for row in matrix):
        raise ValueError(f"{name} must be rectangular")

    if any(not isinstance(value, Real) for row in matrix for value in row):
        raise TypeError(f"{name} must contain only numeric values")

    return len(matrix), column_count


def multiply_basic(matrix_a: Matrix, matrix_b: Matrix) -> list[list[Real]]:
    """
    Multiply two dense matrices using the conventional i-j-k algorithm.

    The function computes C = A x B. Matrix generation, file access and
    correctness validation are intentionally kept outside this function so
    that the benchmark can time only the multiplication operation.
    """
    rows_a, columns_a = validate_matrix(matrix_a, "matrix_a")
    rows_b, columns_b = validate_matrix(matrix_b, "matrix_b")

    if columns_a != rows_b:
        raise ValueError(
            "Incompatible dimensions: the number of columns in matrix_a "
            "must equal the number of rows in matrix_b"
        )

    result: list[list[Real]] = [
        [0 for _ in range(columns_b)] for _ in range(rows_a)
    ]

    for i in range(rows_a):
        for j in range(columns_b):
            accumulator: Real = 0
            for k in range(columns_a):
                accumulator += matrix_a[i][k] * matrix_b[k][j]
            result[i][j] = accumulator

    return result
