from math import pi
import random
from typing import Callable, List

import pytest
import numpy as np

from blueqat import Circuit
from blueqat.gate import OneQubitGate, Mat1Gate, HGate, UGate, PhaseGate, RXGate, RYGate, RZGate
from blueqat.utils import circuit_to_unitary
from blueqat.backends.onequbitgate_decomposer import ryrz_decomposer, u_decomposer

Decomposer = Callable[[OneQubitGate], List[OneQubitGate]]

decomposer_test = pytest.mark.parametrize('decomposer',
                                          [ryrz_decomposer, u_decomposer])


def check_decomposed(g: OneQubitGate, d: Decomposer, ignore_global: bool):
    c1 = Circuit(1, [g])
    c2 = Circuit(1, d(g))
    u1 = circuit_to_unitary(c1)
    u2 = circuit_to_unitary(c2)
    try:
        if not ignore_global:
            assert np.allclose(u1, u2)
        else:
            # If u1 and u2 are same, u1 u2† = e^iθ I.
            # But, vice versa? I'm not sure...
            u = u1 @ u2.T.conj()
            gphase = np.sqrt(np.linalg.det(u))
            assert np.allclose(u, gphase * np.eye(2, dtype=complex))
    except AssertionError:
        print("Orig:", c1)
        print(u1)
        print("Conv:", c2)
        print(u2)
        raise


@decomposer_test
def test_identity(decomposer):
    g = Mat1Gate((0, ), np.eye(2, dtype=complex))
    check_decomposed(g, decomposer, False)


@decomposer_test
def test_identity_plus_delta(decomposer):
    g = Mat1Gate((0, ), np.eye(2, dtype=complex) + np.ones((2, 2)) * 1e-10)
    check_decomposed(g, decomposer, False)


@decomposer_test
def test_hadamard(decomposer):
    g = HGate((0, ))
    check_decomposed(g, decomposer, True)


@decomposer_test
def test_random_rx(decomposer):
    for _ in range(20):
        t = random.random() * pi
        g = RXGate((0, ), t)
        check_decomposed(g, decomposer, True)


@decomposer_test
def test_random_ry(decomposer):
    for _ in range(20):
        t = random.random() * pi
        g = RYGate((0, ), t)
        check_decomposed(g, decomposer, False)


@decomposer_test
def test_random_rz(decomposer):
    for _ in range(20):
        t = random.random() * pi
        g = RZGate((0, ), t)
        check_decomposed(g, decomposer, False)


@decomposer_test
def test_random_r(decomposer):
    for _ in range(20):
        t = random.random() * pi
        g = PhaseGate((0, ), t)
        check_decomposed(g, decomposer, False)


@decomposer_test
def test_random_u(decomposer):
    for _ in range(20):
        t1, t2, t3, t4 = [random.random() * pi for _ in range(4)]
        g = UGate((0, ), t1, t2, t3, t4)
        check_decomposed(g, decomposer, True)
