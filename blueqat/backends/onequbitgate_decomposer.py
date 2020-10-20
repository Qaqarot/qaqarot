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

import cmath
from typing import List

import numpy as np

from ..utils import check_unitarity
from ..gate import *


def u3_decomposer(gate: OneQubitGate) -> List[U3Gate]:
    """Decompose one qubit gate to U3 gate.
    This decomposer ignores global phase.

    Args:
        gate (OneQubitGate): The gate which have 2x2 unitary matrix

    Returns:
        List of U3 gate. length is always 1.
    """
    mat = gate.matrix()
    assert mat.shape == (2, 2)
    assert check_unitarity(mat)
    # Remove global phase. Make SU(2).
    mat /= np.sqrt(np.linalg.det(mat))
    sq_cos_halftheta = (mat[0, 0] * mat[1, 1]).real
    cos_halftheta = np.sqrt(sq_cos_halftheta)
    sin_halftheta = np.sqrt(1. - sq_cos_halftheta)
    theta = np.arccos(cos_halftheta) * 2
    if np.allclose(0.0, cos_halftheta):
        lam = 0.0
        phi = cmath.phase(mat[1, 0]) * 2
    elif np.allclose(0.0, sin_halftheta):
        lam = 0.0
        phi = cmath.phase(mat[1, 1]) * 2
    else:
        angle_phi_plus_lambda_half = cmath.phase(mat[1, 1] / sin_halftheta)
        angle_phi_minus_lambda_half = cmath.phase(mat[1, 0] / cos_halftheta)
        phi = angle_phi_plus_lambda_half + angle_phi_minus_lambda_half
        lam = angle_phi_plus_lambda_half - angle_phi_minus_lambda_half
    return [U3Gate(gate.targets, theta, phi, lam)]
