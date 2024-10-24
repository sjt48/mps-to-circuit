# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tensor utility functions."""

import numpy as np

from .math import _next_power_of_two


def _pad_tensor(tensor: np.ndarray) -> np.ndarray:
    """Pad tensor so that both virtual dimensions have power two.

    :param tensor: Rank-3 tensor to pad.

    :return: Padded tensor where both virtual dimensions have power 2.
    """
    d_left, d, d_right = tensor.shape
    new_d_left = _next_power_of_two(d_left)
    new_d_right = _next_power_of_two(d_right)
    padded_tensor = np.zeros((new_d_left, d, new_d_right), dtype=tensor.dtype)
    padded_tensor[:d_left, :, :d_right] = tensor
    return padded_tensor
