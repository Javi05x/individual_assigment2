"""Correctness tests for the pure-Python CSR implementation."""

import random

import numpy as np
import pytest

from python.src.matrix.sparse.matrix_csr import SparseMatrixCSR


def test_dense_to_csr_conversion() -> None:
    dense = [
        [0, 5, 0],
        [2, 0, 3],
        [0, 0, 4],
    ]

    matrix = SparseMatrixCSR.from_dense(dense)

    assert matrix.rows == 3
    assert matrix.columns == 3
    assert matrix.values == (5, 2, 3, 4)
    assert matrix.column_indices == (1, 0, 2, 2)
    assert matrix.row_pointers == (0, 1, 3, 4)
    assert matrix.nonzero_count == 4


def test_csr_to_dense_conversion() -> None:
    dense = [
        [0, 5, 0],
        [2, 0, 3],
        [0, 0, 4],
    ]

    matrix = SparseMatrixCSR.from_dense(dense)

    assert matrix.to_dense() == dense


def test_density_and_sparsity() -> None:
    matrix = SparseMatrixCSR.from_dense(
        [
            [1, 0],
            [0, 0],
        ]
    )

    assert matrix.density == pytest.approx(0.25)
    assert matrix.sparsity == pytest.approx(0.75)


def test_zero_matrix() -> None:
    matrix = SparseMatrixCSR.from_dense(
        [
            [0, 0],
            [0, 0],
        ]
    )

    assert matrix.values == ()
    assert matrix.column_indices == ()
    assert matrix.row_pointers == (0, 0, 0)
    assert matrix.nonzero_count == 0
    assert matrix.to_dense() == [[0, 0], [0, 0]]


def test_known_sparse_multiplication() -> None:
    matrix_a = SparseMatrixCSR.from_dense(
        [
            [1, 0, 2],
            [0, 3, 0],
        ]
    )

    matrix_b = SparseMatrixCSR.from_dense(
        [
            [0, 4],
            [5, 0],
            [0, 6],
        ]
    )

    result = matrix_a.multiply(matrix_b)

    assert result.to_dense() == [
        [0, 16],
        [15, 0],
    ]


def test_rectangular_sparse_multiplication() -> None:
    matrix_a = SparseMatrixCSR.from_dense(
        [
            [1, 0, 2],
            [0, 3, 0],
        ]
    )

    matrix_b = SparseMatrixCSR.from_dense(
        [
            [1, 2, 0, 0],
            [0, 0, 3, 0],
            [4, 0, 0, 5],
        ]
    )

    assert matrix_a.multiply(matrix_b).to_dense() == [
        [9, 2, 0, 10],
        [0, 0, 9, 0],
    ]


def test_cancellation_does_not_store_zero() -> None:
    matrix_a = SparseMatrixCSR.from_dense([[1, 1]])
    matrix_b = SparseMatrixCSR.from_dense([[2], [-2]])

    result = matrix_a.multiply(matrix_b)

    assert result.to_dense() == [[0]]
    assert result.nonzero_count == 0
    assert result.values == ()


def test_identity_matrix() -> None:
    matrix = SparseMatrixCSR.from_dense(
        [
            [2, 0],
            [0, -3],
        ]
    )
    identity = SparseMatrixCSR.from_dense(
        [
            [1, 0],
            [0, 1],
        ]
    )

    assert matrix.multiply(identity).to_dense() == matrix.to_dense()
    assert identity.multiply(matrix).to_dense() == matrix.to_dense()


def test_matches_numpy_with_deterministic_sparse_input() -> None:
    generator = random.Random(20261028)

    dense_a = [
        [
            0 if generator.random() < 0.75
            else generator.randint(1, 9)
            for _ in range(7)
        ]
        for _ in range(5)
    ]

    dense_b = [
        [
            0 if generator.random() < 0.75
            else generator.randint(1, 9)
            for _ in range(4)
        ]
        for _ in range(7)
    ]

    matrix_a = SparseMatrixCSR.from_dense(dense_a)
    matrix_b = SparseMatrixCSR.from_dense(dense_b)

    actual = matrix_a.multiply(matrix_b).to_dense()
    expected = np.matmul(
        np.asarray(dense_a),
        np.asarray(dense_b),
    )

    np.testing.assert_array_equal(
        np.asarray(actual),
        expected,
    )


def test_incompatible_dimensions() -> None:
    matrix_a = SparseMatrixCSR.from_dense([[1, 0, 2]])
    matrix_b = SparseMatrixCSR.from_dense(
        [
            [1, 0],
            [0, 1],
        ]
    )

    with pytest.raises(ValueError, match="Incompatible dimensions"):
        matrix_a.multiply(matrix_b)


def test_invalid_row_pointer_length() -> None:
    with pytest.raises(ValueError, match=r"rows \+ 1"):
        SparseMatrixCSR(
            rows=2,
            columns=2,
            values=(1,),
            column_indices=(0,),
            row_pointers=(0, 1),
        )


def test_explicit_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero"):
        SparseMatrixCSR(
            rows=1,
            columns=1,
            values=(0,),
            column_indices=(0,),
            row_pointers=(0, 1),
        )


def test_unsorted_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="sorted"):
        SparseMatrixCSR(
            rows=1,
            columns=3,
            values=(4, 5),
            column_indices=(2, 0),
            row_pointers=(0, 2),
        )
