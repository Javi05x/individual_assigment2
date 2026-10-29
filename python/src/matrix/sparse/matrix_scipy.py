"""SciPy CSR matrix adapter used for controlled comparisons."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.sparse import csr_matrix

from python.src.matrix.dense.matrix_basic import Matrix, validate_matrix


@dataclass
class SparseMatrixSciPy:
    """Matrix represented internally by SciPy's CSR format."""

    matrix: csr_matrix

    def __post_init__(self) -> None:
        """Validate and normalise the internal sparse matrix."""
        if not isinstance(self.matrix, csr_matrix):
            raise TypeError("matrix must be a SciPy csr_matrix")

        if len(self.matrix.shape) != 2:
            raise ValueError("matrix must be two-dimensional")

        if self.matrix.shape[0] <= 0 or self.matrix.shape[1] <= 0:
            raise ValueError(
                "matrix dimensions must be greater than zero"
            )

        self.matrix.sum_duplicates()
        self.matrix.eliminate_zeros()
        self.matrix.sort_indices()

    @classmethod
    def from_dense(
        cls,
        matrix: Matrix,
    ) -> "SparseMatrixSciPy":
        """Convert the same logical dense input into SciPy CSR format."""
        validate_matrix(matrix, "matrix")

        numpy_matrix = np.asarray(matrix)

        return cls(csr_matrix(numpy_matrix))

    @property
    def rows(self) -> int:
        """Return the number of matrix rows."""
        return int(self.matrix.shape[0])

    @property
    def columns(self) -> int:
        """Return the number of matrix columns."""
        return int(self.matrix.shape[1])

    @property
    def nonzero_count(self) -> int:
        """Return the number of explicitly stored nonzero values."""
        return int(self.matrix.nnz)

    @property
    def density(self) -> float:
        """Return the proportion of nonzero matrix entries."""
        return self.nonzero_count / (self.rows * self.columns)

    @property
    def sparsity(self) -> float:
        """Return the proportion of zero matrix entries."""
        return 1.0 - self.density

    def to_dense(self) -> list[list[Real]]:
        """Convert the SciPy CSR matrix to logical dense form."""
        return self.matrix.toarray().tolist()

    def multiply(
        self,
        other: "SparseMatrixSciPy",
    ) -> "SparseMatrixSciPy":
        """Multiply two SciPy CSR matrices."""
        if not isinstance(other, SparseMatrixSciPy):
            raise TypeError(
                "other must be a SparseMatrixSciPy instance"
            )

        if self.columns != other.rows:
            raise ValueError(
                "Incompatible dimensions: the number of columns "
                "in the left matrix must equal the number of rows "
                "in the right matrix"
            )

        result = self.matrix @ other.matrix

        return SparseMatrixSciPy(result.tocsr())
