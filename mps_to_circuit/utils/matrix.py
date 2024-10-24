# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Matrix utility functions."""

import numpy as np


def _has_orthonormal_columns(matrix: np.ndarray) -> bool:
    """Checks if a given matrix has orthonormal columns, ignoring all-zero columns.

    :param matrix: Matrix whose columns to check.

    :return: True if the matrix has orthonormal columns, False otherwise.
    """
    # Filter out all-zero columns.
    non_zero_columns = matrix[:, np.linalg.norm(matrix, axis=0) > 0]

    assert non_zero_columns.shape[1] > 0, "Requires at least one non-zero column."

    # Check if the remaining columns are mutually orthonormal.
    gram_matrix = non_zero_columns.conj().T @ non_zero_columns
    return np.allclose(gram_matrix, np.eye(non_zero_columns.shape[1]))


def _is_unitary(matrix: np.ndarray) -> bool:
    """Checks if a given matrix is unitary.

    A matrix is considered unitary if its conjugate transpose is equal to its inverse.

    :param matrix: Matrix to check unitarity of.

    :return: True if matrix is unitary, False otherwise.

    """
    return np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]))


def _gram_schmidt(matrix: np.ndarray) -> np.ndarray:
    """Perform the Gram-Schmidt process on a matrix.

    :param matrix: Assumed to be a real or complex matrix with linearly independent columns.

    :return: A unitary matrix.
    """
    num_rows, num_cols = matrix.shape
    unitary = np.zeros((num_rows, num_cols), dtype=matrix.dtype)

    orthonormal_basis = []
    for j in range(num_cols):
        col = matrix[:, j].copy()
        if not np.allclose(col, np.zeros(len(col))):
            orthonormal_basis += [col]

    for j in range(num_cols):
        basis = np.array(orthonormal_basis)
        col = matrix[:, j].copy()
        if np.allclose(col, np.zeros(len(col))):
            if np.iscomplexobj(unitary):
                col = np.random.uniform(-1, 1, len(col)) + 1j * np.random.uniform(
                    -1, 1, len(col)
                )
            else:
                col = np.random.uniform(-1, 1, len(col))

            for vec in basis:
                col -= (vec.conj().T @ col) / (vec.conj().T @ vec) * vec

        unitary[:, j] = col / np.linalg.norm(col)
        orthonormal_basis += [unitary[:, j]]

    return unitary
