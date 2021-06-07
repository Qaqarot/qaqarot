from typing import Callable, List

import pytest
import numpy as np

from blueqat import Circuit
from blueqat.gate import OneQubitGate, Mat1Gate
from blueqat.utils import circuit_to_unitary
from blueqat.backends.onequbitgate_decomposer import ryrz_decomposer, u_decomposer

Decomposer = Callable[[OneQubitGate], List[OneQubitGate]]

decomposer_test = pytest.mark.parametrize(
        'decomposer',
        [ryrz_decomposer, u_decomposer])

def check_decomposed(g: OneQubitGate, d: Decomposer, ignore_global: bool):
    u1 = circuit_to_unitary(Circuit(0, [g]))
    u2 = circuit_to_unitary(Circuit(0, d(g)))
    if ignore_global:
        u1 /= np.sqrt(np.linalg.det(u1))
        u2 /= np.sqrt(np.linalg.det(u1))
    assert np.allclose(u1, u2)


@decomposer_test
def test_identity(decomposer):
    g = Mat1Gate(0, np.eye(2, dtype=complex))
    check_decomposed(g, decomposer, False)


@decomposer_test
def test_identity_plus_delta(decomposer):
    g = Mat1Gate(0, np.eye(2, dtype=complex) + np.ones((2, 2)) * 1e-10)
    check_decomposed(g, decomposer, False)
