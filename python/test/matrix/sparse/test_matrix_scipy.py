"""Correctness tests for the SciPy CSR adapter."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from python.src.matrix.sparse.generator import (
    generate_sparse_matrix_pair,
)
from python.src.matrix.sparse.matrix_csr import SparseMatrixCSR
from python.src.matrix.sparse.matrix_scipy import SparseMatrixSciPy


def test_dense_conversion() -> None:
    dense = [
        [0, 5, 0],
        [2, 0, 3],
        [0, 0, 4],
    ]

    matrix = SparseMatrixSciPy.from_dense(dense)

    assert matrix.rows == 3
    assert matrix.columns == 3
    assert matrix.nonzero_count == 4
    assert matrix.to_dense() == dense


def test_density_and_sparsity() -> None:
    matrix = SparseMatrixSciPy.from_dense(
        [
            [1, 0],
            [0, 0],
        ]
    )

    assert matrix.density == pytest.approx(0.25)
    assert matrix.sparsity == pytest.approx(0.75)


def test_zero_matrix() -> None:
    matrix = SparseMatrixSciPy.from_dense(
        [
            [0, 0],
            [0, 0],
        ]
    )

    assert matrix.nonzero_count == 0
    assert matrix.to_dense() == [
        [0, 0],
        [0, 0],
    ]


def test_known_multiplication() -> None:
    matrix_a = SparseMatrixSciPy.from_dense(
        [
            [1, 0, 2],
            [0, 3, 0],
        ]
    )

    matrix_b = SparseMatrixSciPy.from_dense(
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


def test_rectangular_multiplication() -> None:
    matrix_a = SparseMatrixSciPy.from_dense(
        [
            [1, 0, 2],
            [0, 3, 0],
        ]
    )

    matrix_b = SparseMatrixSciPy.from_dense(
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


def test_zero_cancellation_is_removed() -> None:
    matrix_a = SparseMatrixSciPy.from_dense([[1, 1]])
    matrix_b = SparseMatrixSciPy.from_dense([[2], [-2]])

    result = matrix_a.multiply(matrix_b)

    assert result.to_dense() == [[0]]
    assert result.nonzero_count == 0


def test_matches_custom_csr_and_numpy() -> None:
    dense_a, dense_b = generate_sparse_matrix_pair(
        size=20,
        sparsity=0.85,
        seed=20261028,
        minimum_value=-10,
        maximum_value=10,
    )

    custom_a = SparseMatrixCSR.from_dense(dense_a)
    custom_b = SparseMatrixCSR.from_dense(dense_b)

    scipy_a = SparseMatrixSciPy.from_dense(dense_a)
    scipy_b = SparseMatrixSciPy.from_dense(dense_b)

    custom_result = custom_a.multiply(custom_b).to_dense()
    scipy_result = scipy_a.multiply(scipy_b).to_dense()

    numpy_result = np.matmul(
        np.asarray(dense_a),
        np.asarray(dense_b),
    )

    assert custom_result == scipy_result

    np.testing.assert_array_equal(
        np.asarray(scipy_result),
        numpy_result,
    )


def test_same_logical_input_has_same_nonzero_count() -> None:
    dense_a, _ = generate_sparse_matrix_pair(
        size=30,
        sparsity=0.9,
        seed=300,
    )

    custom_matrix = SparseMatrixCSR.from_dense(dense_a)
    scipy_matrix = SparseMatrixSciPy.from_dense(dense_a)

    assert custom_matrix.nonzero_count == scipy_matrix.nonzero_count
    assert custom_matrix.sparsity == pytest.approx(
        scipy_matrix.sparsity
    )


def test_incompatible_dimensions() -> None:
    matrix_a = SparseMatrixSciPy.from_dense([[1, 0, 2]])
    matrix_b = SparseMatrixSciPy.from_dense(
        [
            [1, 0],
            [0, 1],
        ]
    )

    with pytest.raises(ValueError, match="Incompatible dimensions"):
        matrix_a.multiply(matrix_b)


def test_invalid_operand_type() -> None:
    matrix = SparseMatrixSciPy.from_dense([[1]])

    with pytest.raises(
        TypeError,
        match="SparseMatrixSciPy",
    ):
        matrix.multiply([[1]])  # type: ignore[arg-type]


def test_explicit_zero_is_removed() -> None:
    internal_matrix = csr_matrix(
        (
            np.asarray([4, 0, 5]),
            np.asarray([0, 1, 2]),
            np.asarray([0, 3]),
        ),
        shape=(1, 3),
    )

    matrix = SparseMatrixSciPy(internal_matrix)

    assert matrix.nonzero_count == 2
    assert matrix.to_dense() == [[4, 0, 5]]


def test_invalid_internal_type() -> None:
    with pytest.raises(TypeError, match="csr_matrix"):
        SparseMatrixSciPy(  # type: ignore[arg-type]
            np.asarray([[1, 0], [0, 1]])
        )
