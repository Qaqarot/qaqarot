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
"""Utilities for convenient."""
import cmath
from collections import Counter
import math
import typing
from typing import Dict, Iterator, Tuple, Union
import warnings

import numpy as np

if typing.TYPE_CHECKING:
    from . import Circuit


def to_inttuple(
    bitstr: Union[str, Counter, Dict[str, int]]
) -> Union[Tuple, Counter, Dict[Tuple, int]]:
    """Convert from bit string likes '01011' to int tuple likes (0, 1, 0, 1, 1)

    Args:
        bitstr (str, Counter, dict): String which is written in "0" or "1".
            If all keys are bitstr, Counter or dict are also can be converted by this function.

    Returns:
        tuple of int, Counter, dict: Converted bits.
            If bitstr is Counter or dict, returns the Counter or dict
            which contains {converted key: original value}.

    Raises:
        ValueError: If bitstr type is unexpected or bitstr contains illegal character.
    """
    if isinstance(bitstr, str):
        return tuple(int(b) for b in bitstr)
    if isinstance(bitstr, Counter):
        return Counter(
            {tuple(int(b) for b in k): v
             for k, v in bitstr.items()})
    if isinstance(bitstr, dict):
        return {tuple(int(b) for b in k): v for k, v in bitstr.items()}
    raise ValueError("bitstr type shall be `str`, `Counter` or `dict`")


def ignore_global_phase(statevec: np.ndarray) -> np.ndarray:
    """Multiple e^-iθ to `statevec` where θ is a phase of first non-zero element.

    Args:
        statevec np.ndarray: Statevector.

    Returns:
        np.ndarray: `statevec` is returned.
    """
    for q in statevec:
        if abs(q) > 0.0000001:
            ang = abs(q) / q
            statevec *= ang
            break
    return statevec


def check_unitarity(mat: np.ndarray) -> bool:
    """Check whether mat is a unitary matrix."""
    shape = mat.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        # Not a square matrix.
        return False
    return np.allclose(mat @ mat.T.conjugate(), np.eye(shape[0]))


def circuit_to_unitary(circ: 'Circuit', *runargs, **runkwargs) -> np.ndarray:
    """Make circuit to unitary. This function is experimental feature and
    may changed or deleted in the future."""
    warnings.warn(
        "blueqat.util.circuit_to_unitary is moved to " +
        "blueqat.circuit_funcs.circuit_to_unitary.circuit_to_unitary.",
        DeprecationWarning)
    from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary as f
    return f(circ, *runargs, **runkwargs)


def calc_u_params(mat: np.ndarray) -> Tuple[float, float, float, float]:
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


def sqrt_2x2_matrix(mat: np.ndarray) -> np.ndarray:
    """Returns square root of 2x2 matrix.

    Reference: https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    """
    assert mat.shape == (2, 2)
    s = np.sqrt(np.linalg.det(mat))
    t = np.sqrt(mat[0, 0] + mat[1, 1] + 2 * s)
    if abs(t) < 1e-8: # Avoid zero division.
        s = -s
        t = np.sqrt(mat[0, 0] + mat[1, 1] + 2 * s)
    return (mat + s * np.eye(2)) / t


def gen_graycode(n: int) -> Iterator[int]:
    """Generate an iterator which returns Gray code."""
    return (v ^ (v >> 1) for v in range(2**n))


def gen_gray_controls(n: int) -> Iterator[Tuple[int, int, int]]:
    """Generate an iterator which returns bit indices for constructing
    Gray code based controlled gate.

    ## Example

    4 controlled Z impletation:

    ```py
    n = 3
    t = 3
    c = Circuit()
    angles = [pi / 2**(n - 1), -pi / 2**(n - 1)]
    for c0, c1, parity in gen_gray_controls(n):
        if c0 >= 0:
            c.cx[c0, c1]
        c.crz(angles[parity])[c1, t]
    """
    def gen_changedbit(n):
        pow2 = [2 ** i for i in range(n)]
        gen = gen_graycode(n)
        try:
            prev = next(gen)
        except StopIteration: # Unreachable.
            raise ValueError() from None
        for g in gen:
            yield pow2.index(g ^ prev)
            prev = g

    def gen_cxtarget():
        k = 0
        while 1:
            for _ in range(2**k):
                yield k
            k += 1

    def gen_parity():
        while 1:
            yield 0
            yield 1

    for c0, c1, p in zip(gen_changedbit(n), gen_cxtarget(), gen_parity()):
        if c0 == c1:
            yield c0 - 1, c1, p
        else:
            yield c0, c1, p
