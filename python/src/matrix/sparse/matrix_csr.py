"""Pure-Python Compressed Sparse Row matrix implementation."""

from dataclasses import dataclass
from numbers import Real

from python.src.matrix.dense.matrix_basic import Matrix, validate_matrix


@dataclass(frozen=True)
class SparseMatrixCSR:
    """Matrix stored in Compressed Sparse Row format."""

    rows: int
    columns: int
    values: tuple[Real, ...]
    column_indices: tuple[int, ...]
    row_pointers: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the internal CSR representation."""
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError(
                "matrix dimensions must be greater than zero"
            )

        if len(self.values) != len(self.column_indices):
            raise ValueError(
                "values and column_indices must have equal length"
            )

        if len(self.row_pointers) != self.rows + 1:
            raise ValueError(
                "row_pointers must contain rows + 1 entries"
            )

        if self.row_pointers[0] != 0:
            raise ValueError("row_pointers must start at zero")

        if self.row_pointers[-1] != len(self.values):
            raise ValueError(
                "the final row pointer must equal the number "
                "of stored values"
            )

        if any(
            first > second
            for first, second in zip(
                self.row_pointers,
                self.row_pointers[1:],
            )
        ):
            raise ValueError(
                "row_pointers must be non-decreasing"
            )

        if any(
            column < 0 or column >= self.columns
            for column in self.column_indices
        ):
            raise ValueError("column index outside matrix bounds")

        for row in range(self.rows):
            start = self.row_pointers[row]
            end = self.row_pointers[row + 1]
            row_columns = self.column_indices[start:end]

            if tuple(sorted(row_columns)) != row_columns:
                raise ValueError(
                    "column indices must be sorted within each row"
                )

            if len(set(row_columns)) != len(row_columns):
                raise ValueError(
                    "duplicate column indices are not allowed "
                    "within a row"
                )

        if any(value == 0 for value in self.values):
            raise ValueError(
                "explicit zero values must not be stored"
            )

    @classmethod
    def from_dense(cls, matrix: Matrix) -> "SparseMatrixCSR":
        """Convert a validated dense matrix into CSR format."""
        rows, columns = validate_matrix(matrix, "matrix")

        values: list[Real] = []
        column_indices: list[int] = []
        row_pointers = [0]

        for row in matrix:
            for column_index, value in enumerate(row):
                if value != 0:
                    values.append(value)
                    column_indices.append(column_index)

            row_pointers.append(len(values))

        return cls(
            rows=rows,
            columns=columns,
            values=tuple(values),
            column_indices=tuple(column_indices),
            row_pointers=tuple(row_pointers),
        )

    @property
    def nonzero_count(self) -> int:
        """Return the number of explicitly stored nonzero values."""
        return len(self.values)

    @property
    def density(self) -> float:
        """Return the proportion of nonzero matrix entries."""
        return self.nonzero_count / (self.rows * self.columns)

    @property
    def sparsity(self) -> float:
        """Return the proportion of zero matrix entries."""
        return 1.0 - self.density

    def to_dense(self) -> list[list[Real]]:
        """Convert the CSR matrix into a dense list-of-lists matrix."""
        dense: list[list[Real]] = [
            [0 for _ in range(self.columns)]
            for _ in range(self.rows)
        ]

        for row in range(self.rows):
            start = self.row_pointers[row]
            end = self.row_pointers[row + 1]

            for position in range(start, end):
                column = self.column_indices[position]
                dense[row][column] = self.values[position]

        return dense

    def multiply(
        self,
        other: "SparseMatrixCSR",
    ) -> "SparseMatrixCSR":
        """
        Multiply two CSR matrices using sparse row accumulation.

        For every nonzero value A[i, k], the nonzero entries in row k of B
        contribute to row i of the result. Intermediate values are stored
        in a dictionary and zero-valued cancellations are omitted.
        """
        if self.columns != other.rows:
            raise ValueError(
                "Incompatible dimensions: the number of columns "
                "in the left matrix must equal the number of rows "
                "in the right matrix"
            )

        result_values: list[Real] = []
        result_columns: list[int] = []
        result_row_pointers = [0]

        for row in range(self.rows):
            accumulator: dict[int, Real] = {}

            left_start = self.row_pointers[row]
            left_end = self.row_pointers[row + 1]

            for left_position in range(left_start, left_end):
                inner_index = self.column_indices[left_position]
                left_value = self.values[left_position]

                right_start = other.row_pointers[inner_index]
                right_end = other.row_pointers[inner_index + 1]

                for right_position in range(
                    right_start,
                    right_end,
                ):
                    output_column = other.column_indices[
                        right_position
                    ]

                    product = (
                        left_value
                        * other.values[right_position]
                    )

                    accumulator[output_column] = (
                        accumulator.get(output_column, 0)
                        + product
                    )

            for output_column in sorted(accumulator):
                output_value = accumulator[output_column]

                if output_value != 0:
                    result_columns.append(output_column)
                    result_values.append(output_value)

            result_row_pointers.append(len(result_values))

        return SparseMatrixCSR(
            rows=self.rows,
            columns=other.columns,
            values=tuple(result_values),
            column_indices=tuple(result_columns),
            row_pointers=tuple(result_row_pointers),
        )
