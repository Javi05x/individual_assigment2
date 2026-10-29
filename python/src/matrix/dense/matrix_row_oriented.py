"""Row-oriented dense matrix multiplication implementation."""

from numbers import Real

from python.src.matrix.dense.matrix_basic import Matrix, validate_matrix


def multiply_row_oriented(
    matrix_a: Matrix,
    matrix_b: Matrix,
) -> list[list[Real]]:
    """
    Multiply two dense matrices using the i-k-j loop order.

    Unlike the basic i-j-k implementation, this version traverses each row
    of matrix B sequentially. The loop reordering is intended to improve
    memory-access locality while preserving the same mathematical result.
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
        result_row = result[i]

        for k in range(columns_a):
            value_a = matrix_a[i][k]
            matrix_b_row = matrix_b[k]

            for j in range(columns_b):
                result_row[j] += value_a * matrix_b_row[j]

    return result
