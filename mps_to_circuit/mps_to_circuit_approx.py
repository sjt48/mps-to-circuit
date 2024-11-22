# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exact MPS to circuit mapping function."""

import numpy as np
from qiskit import QuantumCircuit

from .mps_to_circuit_exact import _mps_to_circuit_exact

from .utils import _prepare_mps


def _mps_to_circuit_approx(
    mps: list[np.ndarray],
    shape: str = "lrp",
    num_layers: int = 1,
    compress: bool = True,
    chi_max: int = -1,
) -> QuantumCircuit:
    """Convert a matrix product state to a quantum circuit.

    Approximately convert a generic matrix product state (MPS) with arbitrary bond dimensions to a
    Qiskit quantum circuit. See Section III of "Encoding of matrix product states into quantum
    circuits of one- and two-qubit gates", S J Ran, Phys. Rev. A, 2020.

    :param mps: A matrix product state (MPS) representation of a quantum state.
    :param shape: The ordering of the dimensions of each MPS tensor. 'left', 'right', 'physical' by
    default.
    :param num_layers: The number of layers to add to the circuit.
    :param compress: Set to True to compress the MPS after each layer to a maximum bond dimension of
    chi_max.
    :param chi_max: See description for compress.
    NOTE: If set to the default value of -1, chi_max will be set to the maximum bond dimension
    of the input MPS.
    NOTE: chi_max will be ignored if compress is False.

    :return: A quantum circuit consisting of N layers of two-qubit isometries that approximately
    represents the input MPS.
    """
    # Prepare Quimb MPS in the correct form
    _mps = _prepare_mps(mps, shape)
    compressed_mps = _mps.copy(deep=True)
    disentangled_mps = _mps.copy(deep=True)

    circuits = []

    if compress and chi_max == -1:
        chi_max = _mps.max_bond()
    if compress and chi_max != -1 and (chi_max < 1 or not isinstance(chi_max, int)):
        raise ValueError("chi_max must be an integer greater than 0")
    if not compress and chi_max != -1:
        print("Warning: chi_max will be ignored if compress==False")

    for layer in range(num_layers):
        # Compress the MPS from the previous layer to a maximum bond dimension of 2
        # |ψ_k> -> |ψ'_k>
        compressed_mps = disentangled_mps.copy(deep=True)
        compressed_mps.compress(form="left", max_bond=2)
        compressed_mps.normalize()

        # Find the unitary U_k such that |ψ'_k> = U_k @ |0>
        qc = _mps_to_circuit_exact(list(compressed_mps.arrays), shape="lrp")
        unitaries = [gate.to_matrix() for (gate, _, _) in qc.data]

        # Append inv(U_k) @ ... @ inv(U_1) @ inv(U_0) @ |0> to circuits
        if layer == 0:
            circuits.append(qc)
        else:
            circuits.append(qc.compose(circuits[layer - 1]))

        # Apply the inverse of U_k to disentangle |ψ_k>
        # |ψ_(k+1)> = inv(U_k) @ |ψ_k>
        for i in range(len(unitaries)):
            U_inv = unitaries[-(i + 1)].conj().T
            if U_inv.shape[0] == 4:
                disentangled_mps.gate_split(
                    U_inv, (i - 1, i), inplace=True, cutoff=1e-3
                )
            else:
                disentangled_mps.gate(U_inv, (i), inplace=True, contract=True)

        if compress:
            # Compress |ψ_(k+1)> to have maximum bond dimension chi_max
            disentangled_mps.compress(form="left", max_bond=chi_max)

    return circuits[-1]  # TODO: What do we want to return?
