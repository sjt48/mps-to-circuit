# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Math utility functions."""


def _next_power_of_two(number: int) -> int:
    """Return the next power of two greater than or equal to a given number.

    Args:
        number: The natural number.

    Returns:
        The next power of two greater than or equal to the given number.
    """
    n = int(number)
    assert n > 0, "Input must be a natural number."
    return 1 << (n - 1).bit_length()
