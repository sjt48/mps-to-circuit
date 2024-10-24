# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MPS to circuit exact function."""

import numpy as np
from qiskit import QuantumCircuit

from .mps_to_circuit_exact import _mps_to_circuit_exact


def mps_to_circuit(mps: list[np.ndarray], method: str = "exact") -> QuantumCircuit:
    """Convert a matrix product state to a quantum circuit.

    :param mps: A matrix product state representing a quantum state.
    :param method: The name of the mapping method.

    :return: The matrix product state represented as a quantum circuit.
    """
    match method:
        case "exact":
            return _mps_to_circuit_exact(mps)
        case _:
            raise ValueError(f"Invalid method `{method}`")
