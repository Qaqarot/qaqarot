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
from typing import List, Union

from ..utils import check_unitarity
from ..gate import OneQubitGate, RYGate, RZGate, UGate


def u_decomposer(gate: OneQubitGate) -> List[UGate]:
    """Decompose one qubit gate to U gate.

    Args:
        gate (OneQubitGate): The gate which have 2x2 unitary matrix

    Returns:
        List of U gate. length is always 1.
    """
    mat = gate.matrix()
    assert mat.shape == (2, 2)
    assert check_unitarity(mat)
    gamma = cmath.phase(mat[0, 0])
    mat *= cmath.exp(-1j * gamma)
    assert abs(mat[0, 0].imag) < 1e-8
    cos_halftheta = mat[0, 0].real
    sin_halftheta = 1.0 - cos_halftheta ** 2
    try:
        theta = math.acos(cos_halftheta) * 2
    except ValueError: # math domain error
        assert abs(abs(cos_halftheta) - 1.0) < 1e-8
        if cos_halftheta > 0:
            theta = 0.0
        else:
            theta = math.pi
    if abs(sin_halftheta) < 1e-8:
        phi = 0.0
        lam = cmath.phase(mat[1, 1])
    else:
        phi = cmath.phase(mat[1, 0])
        lam = cmath.phase(mat[0, 1])
    return [UGate(gate.targets, theta, phi, lam, gamma)]


def ryrz_decomposer(gate: OneQubitGate) -> List[Union[RYGate, RZGate]]:
    """Decompose one qubit gate to RY and RZ gate.

    Args:
        gate (OneQubitGate): The gate which have 2x2 unitary matrix

    Returns:
        List of RZ and RY gate. The global phase is omitted.
    """
    mat = gate.matrix()
    assert mat.shape == (2, 2)
    assert check_unitarity(mat)
    gamma = cmath.phase(mat[0, 0])
    mat *= cmath.exp(-1j * gamma)
    assert abs(mat[0, 0].imag) < 1e-8
    cos_halftheta = mat[0, 0].real
    sin_halftheta = 1.0 - cos_halftheta ** 2
    try:
        theta = math.acos(cos_halftheta) * 2
    except ValueError: # math domain error
        assert abs(abs(cos_halftheta) - 1.0) < 1e-8
        if cos_halftheta > 0:
            theta = 0.0
        else:
            theta = math.pi
    if abs(sin_halftheta) < 1e-8:
        phi = 0.0
        lam = cmath.phase(mat[1, 1])
    else:
        phi = cmath.phase(mat[1, 0])
        lam = cmath.phase(mat[0, 1])
    if theta < 1e-8:
        return [RZGate(gate.targets, lam + phi)]
    gates = []
    if lam > 1e-8:
        gates.append(RZGate(gate.targets, lam))
    gates.append(RYGate(gate.targets, theta))
    if phi > 1e-8:
        gates.append(RZGate(gate.targets, phi))
    return gates
