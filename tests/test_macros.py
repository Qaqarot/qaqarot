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
    np.allclose(
            circuit_to_unitary(Circuit().c3z(0, 1, 2, 3)),
            np.diag([1] * 15 + [-1])
    )

def test_c4z():
    np.allclose(
            circuit_to_unitary(Circuit().c4z(0, 1, 2, 3, 4)),
            np.diag([1] * 31 + [-1])
    )

def test_mcz_gray_0():
    np.allclose(
            circuit_to_unitary(Circuit().mcz_gray([], 0)),
            np.diag([1, -1])
    )

def test_mcz_gray_1():
    np.allclose(
            circuit_to_unitary(Circuit().mcz_gray([0], 1)),
            np.diag([1, 1, 1, -1])
    )

def test_mcz_gray_4():
    np.allclose(
            circuit_to_unitary(Circuit().mcz_gray([0, 1, 2, 3], 4)),
            np.diag([1] * 31 + [-1])
    )
