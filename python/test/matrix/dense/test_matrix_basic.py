"""Correctness tests for the basic dense matrix multiplication."""

import pytest

from python.src.matrix.dense.matrix_basic import multiply_basic


def test_known_square_matrices() -> None:
    matrix_a = [
        [1, 2],
        [3, 4],
    ]
    matrix_b = [
        [5, 6],
        [7, 8],
    ]

    assert multiply_basic(matrix_a, matrix_b) == [
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

    assert multiply_basic(matrix_a, matrix_b) == [
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

    assert multiply_basic(matrix, identity) == matrix
    assert multiply_basic(identity, matrix) == matrix


def test_zero_matrix() -> None:
    matrix = [
        [2, 4],
        [6, 8],
    ]
    zero = [
        [0, 0],
        [0, 0],
    ]

    assert multiply_basic(matrix, zero) == zero


def test_single_element_matrices() -> None:
    assert multiply_basic([[4]], [[3]]) == [[12]]


def test_incompatible_dimensions() -> None:
    matrix_a = [[1, 2, 3]]
    matrix_b = [[1, 2], [3, 4]]

    with pytest.raises(ValueError, match="Incompatible dimensions"):
        multiply_basic(matrix_a, matrix_b)


def test_empty_matrix() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        multiply_basic([], [[1]])


def test_non_rectangular_matrix() -> None:
    matrix_a = [
        [1, 2],
        [3],
    ]

    with pytest.raises(ValueError, match="rectangular"):
        multiply_basic(matrix_a, [[1], [2]])


def test_non_numeric_value() -> None:
    matrix_a = [
        [1, "invalid"],
        [3, 4],
    ]

    with pytest.raises(TypeError, match="numeric"):
        multiply_basic(matrix_a, [[1, 0], [0, 1]])
