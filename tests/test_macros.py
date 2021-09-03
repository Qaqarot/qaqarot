import math
import cmath
import random
from collections import Counter

import pytest
import numpy as np

from blueqat import Circuit
from blueqat.circuit_funcs import circuit_to_unitary

import blueqat.macros

PAULI_X = np.array([[0, 1], [1, 0]])

def test_c3z():
    assert np.allclose(
            circuit_to_unitary(Circuit().c3z(0, 1, 2, 3)),
            np.diag([1] * 15 + [-1])
    )

def test_c4z():
    assert np.allclose(
            circuit_to_unitary(Circuit().c4z(0, 1, 2, 3, 4)),
            np.diag([1] * 31 + [-1])
    )

def test_mcz_gray_0():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcz_gray([], 0)),
            np.diag([1, -1])
    )

def test_mcz_gray_1():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcz_gray([0], 1)),
            np.diag([1, 1, 1, -1])
    )

def test_mcz_gray_4():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcz_gray([0, 1, 2, 3], 4)),
            np.diag([1] * 31 + [-1])
    )

def test_c3x():
    expected = np.eye(16)
    expected[7, 7] = expected[15, 15] = 0
    expected[7, 15] = expected[15, 7] = 1
    assert np.allclose(
            circuit_to_unitary(Circuit().c3x(0, 1, 2, 3)),
            expected
    )

def test_c4x():
    expected = np.eye(32)
    expected[15, 15] = expected[31, 31] = 0
    expected[15, 31] = expected[31, 15] = 1
    assert np.allclose(
            circuit_to_unitary(Circuit().c4x(0, 1, 2, 3, 4)),
            expected
    )

def test_mcx_gray_0():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcx_gray([], 0)),
            np.array([[0, 1], [1, 0]])
    )

def test_mcx_gray_1():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcx_gray([0], 1)),
            circuit_to_unitary(Circuit().cx[0, 1])
    )

def test_mcx_gray_2():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcx_gray([0, 1], 2)),
            circuit_to_unitary(Circuit().ccx[0, 1, 2])
    )

def test_mcx_gray_4():
    expected = np.eye(32)
    expected[15, 15] = expected[31, 31] = 0
    expected[15, 31] = expected[31, 15] = 1
    assert np.allclose(
            circuit_to_unitary(Circuit().mcx_gray([0, 1, 2, 3], 4)),
            expected
    )
