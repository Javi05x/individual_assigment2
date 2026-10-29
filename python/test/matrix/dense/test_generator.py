"""Tests for deterministic dense matrix generation."""

import random

import pytest

from python.src.matrix.dense.generator import (
    generate_dense_matrix,
    generate_dense_matrix_pair,
)


def test_matrix_has_requested_dimensions() -> None:
    matrix = generate_dense_matrix(
        rows=3,
        columns=5,
        seed=20261028,
    )

    assert len(matrix) == 3
    assert all(len(row) == 5 for row in matrix)


def test_same_seed_produces_same_matrix() -> None:
    first = generate_dense_matrix(4, 4, seed=100)
    second = generate_dense_matrix(4, 4, seed=100)

    assert first == second


def test_different_seed_produces_different_matrix() -> None:
    first = generate_dense_matrix(4, 4, seed=100)
    second = generate_dense_matrix(4, 4, seed=101)

    assert first != second


def test_values_respect_requested_range() -> None:
    matrix = generate_dense_matrix(
        rows=20,
        columns=20,
        seed=200,
        minimum_value=-3,
        maximum_value=7,
    )

    assert all(
        -3 <= value <= 7
        for row in matrix
        for value in row
    )


def test_matrix_pair_is_reproducible() -> None:
    first_a, first_b = generate_dense_matrix_pair(5, seed=300)
    second_a, second_b = generate_dense_matrix_pair(5, seed=300)

    assert first_a == second_a
    assert first_b == second_b


def test_pair_contains_different_matrices() -> None:
    matrix_a, matrix_b = generate_dense_matrix_pair(5, seed=400)

    assert matrix_a != matrix_b


def test_generator_does_not_change_global_random_state() -> None:
    random.seed(500)
    expected = random.random()

    random.seed(500)
    generate_dense_matrix(3, 3, seed=600)
    actual = random.random()

    assert actual == expected


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (0, 3),
        (-1, 3),
        (3, 0),
        (3, -1),
    ],
)
def test_invalid_dimensions(rows: int, columns: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        generate_dense_matrix(rows, columns, seed=700)


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (2.5, 3),
        (3, "4"),
        (True, 3),
        (3, False),
    ],
)
def test_non_integer_dimensions(
    rows: object,
    columns: object,
) -> None:
    with pytest.raises(TypeError, match="integer"):
        generate_dense_matrix(  # type: ignore[arg-type]
            rows,
            columns,
            seed=800,
        )


def test_invalid_value_range() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_value",
    ):
        generate_dense_matrix(
            rows=3,
            columns=3,
            seed=900,
            minimum_value=10,
            maximum_value=-10,
        )
