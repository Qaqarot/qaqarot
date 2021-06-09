# Copyright 2019 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import cmath
import typing
from typing import List, Tuple, Union

from ..utils import check_unitarity
from ..gate import OneQubitGate, RYGate, RZGate, UGate

if typing.TYPE_CHECKING:
    import numpy as np


def calc_uparams(mat: 'np.ndarray') -> Tuple[float, float, float, float]:
    """Calculate U-gate parameters from a 2x2 unitary matrix."""
    assert mat.shape == (2, 2)
    assert check_unitarity(mat)
    gamma = cmath.phase(mat[0, 0])
    mat = mat * cmath.exp(-1j * gamma)
    theta = math.atan2(abs(mat[1, 0]), mat[0, 0].real) * 2.0
    phi_plus_lambda = cmath.phase(mat[1, 1])
    phi = cmath.phase(mat[1, 0]) % (2.0 * math.pi)
    lam = (phi_plus_lambda - phi) % (2.0 * math.pi)
    return theta, phi, lam, gamma


def u_decomposer(gate: OneQubitGate) -> List[UGate]:
    """Decompose one qubit gate to U gate.

    Args:
        gate (OneQubitGate): The gate which have 2x2 unitary matrix

    Returns:
        List of U gate. length is always 1.
    """
    mat = gate.matrix()
    theta, phi, lam, gamma = calc_uparams(mat)
    return [UGate(gate.targets, theta, phi, lam, gamma)]


def ryrz_decomposer(gate: OneQubitGate) -> List[Union[RYGate, RZGate]]:
    """Decompose one qubit gate to RY and RZ gate.

    Args:
        gate (OneQubitGate): The gate which have 2x2 unitary matrix

    Returns:
        List of RZ and RY gate. The global phase is omitted.
    """
    mat = gate.matrix()
    theta, phi, lam, _ = calc_uparams(mat)
    if theta < 1e-10:
        if phi + lam < 1e-10:
            return []
        return [RZGate(gate.targets, phi + lam)]
    ops = []
    if 1e-10 < lam < 2.0 * math.pi - 1e-10:
        ops.append(RZGate(gate.targets, lam))
    ops.append(RYGate(gate.targets, theta))
    if 1e-10 < phi < 2.0 * math.pi - 1e-10:
        ops.append(RZGate(gate.targets, phi))
    return ops
