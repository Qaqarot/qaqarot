
"""
`gate` module implements quantum gate operations.
This module is internally used.
"""

import numpy as np
from .baseFunc import *
from .gateOps import *

class Sgate():
    """
    Squeezing gate.
    """
    def __init__(self, obj, idx, r):
        self.obj = obj
        self.N = self.obj.N
        self.idx = idx
        self.r = r

    def run(self, state):
        self.r = _paramCheck(self.r)
        return Xsqueeze(state, self.N, self.idx, self.r)

class PSgate():
    """
    Squeezing gate for p direction.
    """
    def __init__(self, obj, idx, r):
        self.obj = obj
        self.N = self.obj.N
        self.idx = idx
        self.r = r

    def run(self, state):
        self.r = _paramCheck(self.r)
        return Xsqueeze(state, self.N, self.idx, -self.r)

class Rgate():
    """
    Rotation gate.
    """
    def __init__(self, obj, idx, theta):
        self.obj = obj
        self.N = self.obj.N
        self.idx = idx
        self.theta = theta

    def run(self, state):
        self.theta = _paramCheck(self.theta)
        return Rotation(state, self.N, self.idx, self.theta)

class BSgate():
    """
    Beamsplitter gate.
    """
    def __init__(self, obj, idx1, idx2, theta):
        self.obj = obj
        self.N = self.obj.N
        self.idx1 = idx1
        self.idx2 = idx2
        self.theta = theta

    def run(self, state):
        self.theta = _paramCheck(self.theta)
        return Beamsplitter(state, self.N, self.idx1, self.idx2, self.theta)

class Dgate():
    """
    Displacement gate.
    """
    def __init__(self, obj, idx, alpha):
        self.obj = obj
        self.N = self.obj.N
        self.idx = idx
        self.alpha = alpha

    def run(self, state):
        self.alpha = _paramCheck(self.alpha)
        return Displace(state, self.idx, self.alpha)

class Xgate():
    """
    x axis shift in phase space.
    """
    def __init__(self, obj, idx, dx):
        self.obj = obj
        self.N = self.obj.N
        self.idx = idx
        self.dx = dx

    def run(self, state):
        self.dx = _paramCheck(self.dx)
        return Xtrans(state, self.idx, self.dx)

class Zgate():
    """
    p axis shift in phase space.
    """
    def __init__(self, obj, idx, dp):
        self.obj = obj
        self.N = self.obj.N
        self.idx = idx
        self.dp = dp

    def run(self, state):
        self.dp = _paramCheck(self.dp)
        return Ztrans(state, self.idx, self.dp)

class TMSgate():
    """
    Two mode squeezing gate.
    """
    def __init__(self, obj, idx1, idx2, r):
        self.obj = obj
        self.N = self.obj.N
        self.idx1 = idx1
        self.idx2 = idx2
        self.r = r

    def run(self, state):
        self.r = _paramCheck(self.r)
        return twoModeSqueezing(state, self.N, self.idx1, self.idx2, self.r)

class MeasX():
    """
    Homodyne measurement for x quadrature.
    """
    def __init__(self, obj, idx):
        self.obj = obj
        self.N = self.obj.N
        self.idx = idx

    def run(self, state):
        res, state_ = HomodyneX(state, self.idx)
        self.obj.creg[self.idx][0] = res
        return state_

class MeasP():
    """
    Homodyne measurement for p quadrature.
    """
    def __init__(self, obj, idx):
        self.obj = obj
        self.N = self.obj.N
        self.idx = idx

    def run(self, state):
        res, state_ = HomodyneP(state, self.idx)
        self.obj.creg[self.idx][1] = res
        return state_

def _paramCheck(param):
    if isinstance(param, CregReader):
        return param.read()
    else:
         return param

class CregReader():
    """
    Class for reading classical register.
    """
    def __init__(self, reg, idx, var, scale = 1):
        self.reg = reg
        self.idx = idx
        self.var = var
        self.scale = scale

    def read(self):
        if self.var == "x":
            v = 0
        elif self.var == "p":
            v = 1
        else:
            raise ValueError('Creg keeps measurement results of "x" or "p".')
        return self.reg[self.idx][v] * self.scale