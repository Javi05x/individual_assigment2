"""Tiled dense matrix multiplication implementation."""

from numbers import Real

from python.src.matrix.dense.matrix_basic import Matrix, validate_matrix


def multiply_tiled(
    matrix_a: Matrix,
    matrix_b: Matrix,
    tile_size: int = 32,
) -> list[list[Real]]:
    """
    Multiply two dense matrices using blocked i-k-j traversal.

    The matrices are processed in tiles to improve data locality. Partial
    tiles are supported, so matrix dimensions do not need to be exact
    multiples of the selected tile size.
    """
    rows_a, columns_a = validate_matrix(matrix_a, "matrix_a")
    rows_b, columns_b = validate_matrix(matrix_b, "matrix_b")

    if columns_a != rows_b:
        raise ValueError(
            "Incompatible dimensions: the number of columns in matrix_a "
            "must equal the number of rows in matrix_b"
        )

    if not isinstance(tile_size, int) or isinstance(tile_size, bool):
        raise TypeError("tile_size must be an integer")

    if tile_size <= 0:
        raise ValueError("tile_size must be greater than zero")

    result: list[list[Real]] = [
        [0 for _ in range(columns_b)] for _ in range(rows_a)
    ]

    for row_start in range(0, rows_a, tile_size):
        row_end = min(row_start + tile_size, rows_a)

        for inner_start in range(0, columns_a, tile_size):
            inner_end = min(inner_start + tile_size, columns_a)

            for column_start in range(0, columns_b, tile_size):
                column_end = min(column_start + tile_size, columns_b)

                for i in range(row_start, row_end):
                    result_row = result[i]
                    matrix_a_row = matrix_a[i]

                    for k in range(inner_start, inner_end):
                        value_a = matrix_a_row[k]
                        matrix_b_row = matrix_b[k]

                        for j in range(column_start, column_end):
                            result_row[j] += value_a * matrix_b_row[j]

    return result
