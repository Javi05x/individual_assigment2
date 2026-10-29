"""Deterministic generators for dense benchmark matrices."""

import random

from python.src.matrix.dense.matrix_basic import Matrix


def generate_dense_matrix(
    rows: int,
    columns: int,
    seed: int,
    minimum_value: int = -10,
    maximum_value: int = 10,
) -> Matrix:
    """
    Generate a reproducible dense matrix of integer values.

    A local random-number generator is used so that matrix generation does
    not modify Python's global random state.
    """
    if not isinstance(rows, int) or isinstance(rows, bool):
        raise TypeError("rows must be an integer")

    if not isinstance(columns, int) or isinstance(columns, bool):
        raise TypeError("columns must be an integer")

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix dimensions must be greater than zero")

    if minimum_value > maximum_value:
        raise ValueError(
            "minimum_value must not be greater than maximum_value"
        )

    generator = random.Random(seed)

    return [
        [
            generator.randint(minimum_value, maximum_value)
            for _ in range(columns)
        ]
        for _ in range(rows)
    ]


def generate_dense_matrix_pair(
    size: int,
    seed: int,
    minimum_value: int = -10,
    maximum_value: int = 10,
) -> tuple[Matrix, Matrix]:
    """
    Generate two reproducible square matrices for one benchmark case.

    Different derived seeds are used for matrices A and B while preserving
    complete reproducibility from the supplied configuration seed.
    """
    matrix_a = generate_dense_matrix(
        rows=size,
        columns=size,
        seed=seed,
        minimum_value=minimum_value,
        maximum_value=maximum_value,
    )

    matrix_b = generate_dense_matrix(
        rows=size,
        columns=size,
        seed=seed + 1,
        minimum_value=minimum_value,
        maximum_value=maximum_value,
    )

    return matrix_a, matrix_b
