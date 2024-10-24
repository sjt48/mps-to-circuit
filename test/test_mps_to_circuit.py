# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test MPS to circuit."""

import numpy as np
import pytest
from qiskit.quantum_info import (
    Statevector,
    state_fidelity,
)
from quimb.tensor import MatrixProductState, MPS_rand_state

from mps_to_circuit import mps_to_circuit


def _convert_to_little_endian(statevector: np.ndarray) -> np.ndarray:
    """Convert a big-endian statevector to little-endian qubit ordering.

    :param statevector: The big-endian statevector to be converted.

    :returns: The converted statevector.
    """
    num_qubits = int(np.log2(len(statevector)))
    assert 2**num_qubits == len(
        statevector
    ), "The input statevector must have a length that is a power of 2."

    statevector = statevector.reshape([2] * num_qubits)
    statevector = np.transpose(statevector, axes=range(num_qubits)[::-1])
    return statevector.flatten()


def _mps_to_statevector(mps: MatrixProductState) -> np.ndarray:
    """
    Convert a Quimb MPS to a full statevector.

    If the MPS is not normalized, the resulting statevector will not be normalized either.

    :param mps: The MPS to be converted.

    :return: A NumPy array representing the full statevector.
    """
    return _convert_to_little_endian(mps.to_dense())


@pytest.mark.parametrize("chi_max", range(1, 17))
def test_mps_to_circuit(chi_max):
    """Test MPS to quantum circuit conversion function."""

    mps = MPS_rand_state(L=8, bond_dim=chi_max)
    mps.compress()
    mps.normalize()

    assert chi_max <= 2 ** (
        mps._L // 2
    ), f"chi_max={chi_max} is too large for L={mps._L}."

    expected = Statevector(_mps_to_statevector(mps))

    arrays = list(mps.arrays)

    qc = mps_to_circuit(arrays)
    result = Statevector(qc)

    fidelity = state_fidelity(expected, result)
    assert np.isclose(fidelity, 1.0)
