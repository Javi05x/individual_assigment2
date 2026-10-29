"""Correctness tests for tiled matrix multiplication."""

import random

import numpy as np
import pytest

from python.src.matrix.dense.matrix_basic import multiply_basic
from python.src.matrix.dense.matrix_row_oriented import (
    multiply_row_oriented,
)
from python.src.matrix.dense.matrix_tiled import multiply_tiled


@pytest.mark.parametrize("tile_size", [1, 2, 3, 8, 32])
def test_known_square_matrices(tile_size: int) -> None:
    matrix_a = [
        [1, 2],
        [3, 4],
    ]
    matrix_b = [
        [5, 6],
        [7, 8],
    ]

    assert multiply_tiled(matrix_a, matrix_b, tile_size) == [
        [19, 22],
        [43, 50],
    ]


def test_rectangular_matrices_with_partial_tiles() -> None:
    matrix_a = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    matrix_b = [
        [7, 8],
        [9, 10],
        [11, 12],
    ]

    assert multiply_tiled(matrix_a, matrix_b, tile_size=2) == [
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

    assert multiply_tiled(matrix, identity, tile_size=2) == matrix
    assert multiply_tiled(identity, matrix, tile_size=2) == matrix


def test_zero_matrix() -> None:
    matrix = [
        [2, 4],
        [6, 8],
    ]
    zero = [
        [0, 0],
        [0, 0],
    ]

    assert multiply_tiled(matrix, zero, tile_size=2) == zero


def test_matches_other_implementations() -> None:
    matrix_a = [
        [2, -1, 3],
        [0, 4, 5],
    ]
    matrix_b = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]

    tiled_result = multiply_tiled(matrix_a, matrix_b, tile_size=2)

    assert tiled_result == multiply_basic(matrix_a, matrix_b)
    assert tiled_result == multiply_row_oriented(matrix_a, matrix_b)


def test_matches_numpy_with_deterministic_random_input() -> None:
    generator = random.Random(20261028)

    matrix_a = [
        [generator.randint(-10, 10) for _ in range(11)]
        for _ in range(7)
    ]
    matrix_b = [
        [generator.randint(-10, 10) for _ in range(5)]
        for _ in range(11)
    ]

    actual = multiply_tiled(matrix_a, matrix_b, tile_size=4)
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
        multiply_tiled(
            [[1, 2, 3]],
            [[1, 2], [3, 4]],
        )


@pytest.mark.parametrize("tile_size", [0, -1, -32])
def test_invalid_tile_size(tile_size: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        multiply_tiled([[1]], [[1]], tile_size)


@pytest.mark.parametrize("tile_size", [2.5, "4", True])
def test_non_integer_tile_size(tile_size: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        multiply_tiled([[1]], [[1]], tile_size)  # type: ignore[arg-type]
