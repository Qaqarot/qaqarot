import circuit
import numpy as np
import sys

EPS = 1e-16

def vec_distsq(a, b):
    diff = a - b
    return diff.T.conjugate() @ diff

def is_vec_same(a, b, eps=EPS):
    return vec_distsq(a, b) < eps

def test_hgate1():
    assert is_vec_same(circuit.Circuit().h[1].h[0].run(), np.array([0.5, 0.5, 0.5, 0.5]))

def test_hgate2():
    assert is_vec_same(circuit.Circuit().x[0].h[0].run(), np.array([1/np.sqrt(2), -1/np.sqrt(2)]))

def test_cx():
    assert is_vec_same(
            circuit.Circuit().h[0].h[1].cx[1,0].h[0].h[1].run(),
            circuit.Circuit().cx[0,1].run()
    )
