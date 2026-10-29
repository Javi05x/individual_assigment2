"""Correctness tests for row-oriented matrix multiplication."""

import random

import numpy as np
import pytest

from python.src.matrix.dense.matrix_basic import multiply_basic
from python.src.matrix.dense.matrix_row_oriented import (
    multiply_row_oriented,
)


def test_known_square_matrices() -> None:
    matrix_a = [
        [1, 2],
        [3, 4],
    ]
    matrix_b = [
        [5, 6],
        [7, 8],
    ]

    assert multiply_row_oriented(matrix_a, matrix_b) == [
        [19, 22],
        [43, 50],
    ]


def test_rectangular_matrices() -> None:
    matrix_a = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    matrix_b = [
        [7, 8],
        [9, 10],
        [11, 12],
    ]

    assert multiply_row_oriented(matrix_a, matrix_b) == [
        [58, 64],
        [139, 154],
    ]


def test_identity_matrix() -> None:
    matrix = [
        [2, -1],
        [0, 3],
    ]
    identity = [
        [1, 0],
        [0, 1],
    ]

    assert multiply_row_oriented(matrix, identity) == matrix
    assert multiply_row_oriented(identity, matrix) == matrix


def test_zero_matrix() -> None:
    matrix = [
        [2, 4],
        [6, 8],
    ]
    zero = [
        [0, 0],
        [0, 0],
    ]

    assert multiply_row_oriented(matrix, zero) == zero


def test_matches_basic_implementation() -> None:
    matrix_a = [
        [2, -1, 3],
        [0, 4, 5],
    ]
    matrix_b = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]

    assert multiply_row_oriented(matrix_a, matrix_b) == multiply_basic(
        matrix_a,
        matrix_b,
    )


def test_matches_numpy_with_deterministic_random_input() -> None:
    generator = random.Random(20261028)

    matrix_a = [
        [generator.randint(-10, 10) for _ in range(7)]
        for _ in range(5)
    ]
    matrix_b = [
        [generator.randint(-10, 10) for _ in range(4)]
        for _ in range(7)
    ]

    actual = multiply_row_oriented(matrix_a, matrix_b)
    expected = np.matmul(
        np.asarray(matrix_a),
        np.asarray(matrix_b),
    )

    np.testing.assert_array_equal(
        np.asarray(actual),
        expected,
    )


def test_incompatible_dimensions() -> None:
    with pytest.raises(ValueError, match="Incompatible dimensions"):
        multiply_row_oriented(
            [[1, 2, 3]],
            [[1, 2], [3, 4]],
        )
