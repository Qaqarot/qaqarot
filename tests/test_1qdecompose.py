import cmath
from math import pi
import random
from typing import Callable, List

import pytest
import numpy as np

from blueqat import Circuit
from blueqat.gate import OneQubitGate, Mat1Gate, HGate, UGate, PhaseGate, RXGate, RYGate, RZGate
from blueqat.circuit_funcs import circuit_to_unitary
from blueqat.backends.onequbitgate_decomposer import ryrz_decomposer, u_decomposer

Decomposer = Callable[[OneQubitGate], List[OneQubitGate]]

decomposer_test = pytest.mark.parametrize('decomposer',
                                          [ryrz_decomposer, u_decomposer])


def check_decomposed(g: OneQubitGate, d: Decomposer, ignore_global: bool):
    c1 = Circuit(1, [g])
    c2 = Circuit(1, d(g))
    u1 = circuit_to_unitary(c1)
    u2 = circuit_to_unitary(c2)
    if ignore_global:
        gphase1 = cmath.phase(np.linalg.det(u1))
        gphase2 = cmath.phase(np.linalg.det(u2))
        su1 = u1 * cmath.exp(-0.5j * gphase1)
        su2 = u2 * cmath.exp(-0.5j * gphase2)
        assert np.isclose(np.linalg.det(su1), 1.0)
        assert np.isclose(np.linalg.det(su2), 1.0)
    else:
        su1 = su2 = np.eye(2) # To avoid static analyzer warning.
    try:
        if ignore_global:
            assert np.allclose(su1, su2) or np.allclose(su1, -su2)
        else:
            assert np.allclose(u1, u2)
    except AssertionError:
        print("Orig:", c1)
        print(u1)
        if ignore_global:
            print("-->")
            print(su1)
        print("Conv:", c2)
        print(u2)
        if ignore_global:
            print("-->")
            print(su2)
        if ignore_global:
            print("abs(Orig - Conv):")
            print(np.abs(su1 - su2))
            print("abs(Orig + Conv):")
            print(np.abs(su1 + su2))
        else:
            print("abs(Orig - Conv):")
            print(np.abs(u1 - u2))
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
        check_decomposed(g, decomposer, True)


@decomposer_test
def test_random_rz(decomposer):
    for _ in range(20):
        t = random.random() * pi
        g = RZGate((0, ), t)
        check_decomposed(g, decomposer, True)


@decomposer_test
def test_random_r(decomposer):
    for _ in range(20):
        t = random.random() * pi
        g = PhaseGate((0, ), t)
        check_decomposed(g, decomposer, True)


@decomposer_test
def test_random_u(decomposer):
    for _ in range(20):
        t1, t2, t3, t4 = [random.random() * pi for _ in range(4)]
        g = UGate((0, ), t1, t2, t3, t4)
        check_decomposed(g, decomposer, True)
