"""Tests for deterministic sparse matrix generation."""

import random

import pytest

from python.src.matrix.sparse.generator import (
    calculate_sparsity,
    count_nonzero,
    generate_sparse_dense_matrix,
    generate_sparse_matrix_pair,
)


def test_generated_matrix_has_requested_dimensions() -> None:
    matrix = generate_sparse_dense_matrix(
        rows=4,
        columns=7,
        sparsity=0.75,
        seed=20261028,
    )

    assert len(matrix) == 4
    assert all(len(row) == 7 for row in matrix)


@pytest.mark.parametrize(
    "sparsity",
    [0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
)
def test_exact_requested_nonzero_count(sparsity: float) -> None:
    rows = 20
    columns = 10

    matrix = generate_sparse_dense_matrix(
        rows=rows,
        columns=columns,
        sparsity=sparsity,
        seed=100,
    )

    expected_nonzero_count = round(
        rows * columns * (1.0 - sparsity)
    )

    assert count_nonzero(matrix) == expected_nonzero_count


def test_same_seed_produces_same_matrix() -> None:
    first = generate_sparse_dense_matrix(
        10,
        10,
        sparsity=0.9,
        seed=200,
    )
    second = generate_sparse_dense_matrix(
        10,
        10,
        sparsity=0.9,
        seed=200,
    )

    assert first == second


def test_different_seed_produces_different_matrix() -> None:
    first = generate_sparse_dense_matrix(
        10,
        10,
        sparsity=0.8,
        seed=300,
    )
    second = generate_sparse_dense_matrix(
        10,
        10,
        sparsity=0.8,
        seed=301,
    )

    assert first != second


def test_generated_values_exclude_zero_at_selected_positions() -> None:
    matrix = generate_sparse_dense_matrix(
        rows=20,
        columns=20,
        sparsity=0.5,
        seed=400,
        minimum_value=-5,
        maximum_value=5,
    )

    nonzero_values = [
        value
        for row in matrix
        for value in row
        if value != 0
    ]

    assert len(nonzero_values) == 200
    assert all(-5 <= value <= 5 for value in nonzero_values)
    assert all(value != 0 for value in nonzero_values)


def test_observed_sparsity_is_correct() -> None:
    matrix = generate_sparse_dense_matrix(
        rows=10,
        columns=10,
        sparsity=0.75,
        seed=500,
    )

    assert calculate_sparsity(matrix) == pytest.approx(0.75)


def test_matrix_pair_is_reproducible() -> None:
    first_a, first_b = generate_sparse_matrix_pair(
        size=20,
        sparsity=0.9,
        seed=600,
    )
    second_a, second_b = generate_sparse_matrix_pair(
        size=20,
        sparsity=0.9,
        seed=600,
    )

    assert first_a == second_a
    assert first_b == second_b


def test_matrix_pair_contains_different_inputs() -> None:
    matrix_a, matrix_b = generate_sparse_matrix_pair(
        size=20,
        sparsity=0.8,
        seed=700,
    )

    assert matrix_a != matrix_b


def test_generator_does_not_modify_global_random_state() -> None:
    random.seed(800)
    expected = random.random()

    random.seed(800)
    generate_sparse_dense_matrix(
        rows=5,
        columns=5,
        sparsity=0.8,
        seed=900,
    )
    actual = random.random()

    assert actual == expected


@pytest.mark.parametrize(
    "sparsity",
    [-0.01, 1.01, -1.0, 2.0],
)
def test_invalid_sparsity_value(sparsity: float) -> None:
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        generate_sparse_dense_matrix(
            rows=5,
            columns=5,
            sparsity=sparsity,
            seed=1000,
        )


@pytest.mark.parametrize(
    "sparsity",
    ["0.9", None, True],
)
def test_non_numeric_sparsity(sparsity: object) -> None:
    with pytest.raises(TypeError, match="numeric"):
        generate_sparse_dense_matrix(  # type: ignore[arg-type]
            rows=5,
            columns=5,
            sparsity=sparsity,
            seed=1100,
        )


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (0, 5),
        (-1, 5),
        (5, 0),
        (5, -1),
    ],
)
def test_invalid_dimensions(rows: int, columns: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        generate_sparse_dense_matrix(
            rows=rows,
            columns=columns,
            sparsity=0.9,
            seed=1200,
        )


def test_range_must_contain_nonzero_integer() -> None:
    with pytest.raises(ValueError, match="nonzero integer"):
        generate_sparse_dense_matrix(
            rows=5,
            columns=5,
            sparsity=0.9,
            seed=1300,
            minimum_value=0,
            maximum_value=0,
        )


def test_invalid_value_range() -> None:
    with pytest.raises(ValueError, match="minimum_value"):
        generate_sparse_dense_matrix(
            rows=5,
            columns=5,
            sparsity=0.9,
            seed=1400,
            minimum_value=5,
            maximum_value=-5,
        )
