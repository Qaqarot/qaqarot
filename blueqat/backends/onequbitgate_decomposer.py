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
from typing import List, Union

from ..utils import calc_u_params
from ..gate import OneQubitGate, RYGate, RZGate, UGate


def u_decomposer(gate: OneQubitGate) -> List[UGate]:
    """Decompose one qubit gate to U gate.

    Args:
        gate (OneQubitGate): The gate which have 2x2 unitary matrix

    Returns:
        List of U gate. length is always 1.
    """
    mat = gate.matrix()
    theta, phi, lam, gamma = calc_u_params(mat)
    return [UGate.create(gate.targets, (theta, phi, lam, gamma))]


def ryrz_decomposer(gate: OneQubitGate) -> List[Union[RYGate, RZGate]]:
    """Decompose one qubit gate to RY and RZ gate.

    Args:
        gate (OneQubitGate): The gate which have 2x2 unitary matrix

    Returns:
        List of RZ and RY gate. The global phase is omitted.
    """
    mat = gate.matrix()
    theta, phi, lam, _ = calc_u_params(mat)
    if theta < 1e-10:
        if phi + lam < 1e-10:
            return []
        return [RZGate.create(gate.targets, (phi + lam,))]
    ops = []
    if 1e-10 < lam < 2.0 * math.pi - 1e-10:
        ops.append(RZGate.create(gate.targets, (lam,)))
    ops.append(RYGate(gate.targets, theta))
    if 1e-10 < phi < 2.0 * math.pi - 1e-10:
        ops.append(RZGate.create(gate.targets, (phi,)))
    return ops
