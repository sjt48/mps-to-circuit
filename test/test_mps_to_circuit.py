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


# Exact method tests


@pytest.mark.parametrize("chi_max", range(1, 17))
def test_mps_to_circuit_exact_method(chi_max):
    """Test exact MPS to quantum circuit conversion function."""

    mps = MPS_rand_state(L=8, bond_dim=chi_max)
    mps.compress()
    mps.normalize()

    assert chi_max <= 2 ** (
        mps._L // 2
    ), f"chi_max={chi_max} is too large for L={mps._L}."

    expected = Statevector(_mps_to_statevector(mps))

    arrays = list(mps.arrays)

    qc = mps_to_circuit(arrays, method="exact")
    result = Statevector(qc)

    fidelity = state_fidelity(expected, result)
    assert np.isclose(fidelity, 1.0)


# Approximate method tests


@pytest.mark.parametrize("chi_max", range(1, 17))
def test_mps_to_circuit_approx_method(chi_max):
    """
    Test approximate MPS to quantum circuit conversion function.
    - The circuits should have the correct number of gates
    - Fidelity should increase as more layers are added
    """

    mps = MPS_rand_state(L=8, bond_dim=chi_max)
    mps.compress()
    mps.normalize()

    assert chi_max <= 2 ** (
        mps._L // 2
    ), f"chi_max={chi_max} is too large for L={mps._L}."

    expected = Statevector(_mps_to_statevector(mps))

    arrays = list(mps.arrays)

    for num_layers in range(1, 6):
        history = {"circuits": []}

        qc = mps_to_circuit(
            arrays, method="approximate", num_layers=num_layers, history=history
        )

        # The circuit after num_layers layers should have num_sites * num_layers gates.
        assert len(qc.data) == mps._L * num_layers

        # The history should store a number of circuits equal to num_layers.
        assert len(history["circuits"]) == num_layers

        result = Statevector(qc)
        fidelity = state_fidelity(expected, result)
        previous_fidelity = 0.0

        # Fidelity after num_layers layers should be greater than fidelity after num_layers-1
        # layers.
        if num_layers > 0:
            assert fidelity > previous_fidelity

        previous_fidelity = fidelity


def test_mps_to_circuit_approx_method_is_exact_for_chi_2():
    """
    Test approximate MPS to quantum circuit conversion function.
    - The single-layer circuit should be exact for MPS of bond dimension 2
    """

    num_sites = 8
    mps = MPS_rand_state(L=num_sites, bond_dim=2)
    mps.normalize()

    expected = Statevector(_mps_to_statevector(mps))

    arrays = list(mps.arrays)
    qc = mps_to_circuit(arrays, method="approximate", num_layers=1)
    result = Statevector(qc)
    fidelity = state_fidelity(expected, result)

    assert np.isclose(fidelity, 1.0)
