"""
`gate` module implements quantum gate operations.
This module is internally used.
"""

import math
import random
from abc import ABC, abstractmethod
import numpy as np

class Gate(ABC):
    """Abstract quantum gate class."""
    lowername = None

    def __init__(self, targets, **kwargs):
        if self.lowername is None:
            raise ValueError(f"{self.__class__.__name__}.lowername is not defined.")
        self.kwargs = kwargs
        self.targets = targets

    def fallback(self):
        """Returns alternative gates to make equivalent circuit."""
        raise NotImplementedError(f"The fallback of {self.__class__.__name__} gate is not defined.")

    @abstractmethod
    def to_qasm(self, helper, targets):
        """Returns OpenQASM. This method is called internally."""
        pass

class OneQubitGate(Gate):
    """Abstract quantum gate class for 1 qubit gate."""
    def target_iter(self, n_qubits):
        return slicing(self.targets, n_qubits)

class TwoQubitGate(Gate):
    """Abstract quantum gate class for 2 qubits gate."""
    def control_target_iter(self, n_qubits):
        return qubit_pairs(self.targets, n_qubits)

class IGate(OneQubitGate):
    """Identity Gate"""
    lowername = "i"
    def to_qasm(self, helper, targets):
        return []

    def fallback(self):
        return []

class XGate(OneQubitGate):
    """Pauli's X Gate"""
    lowername = "x"
    def to_qasm(self, helper, targets):
        n_qubits = helper["n_qubits"]
        qasm = []
        for target in self.target_iter(n_qubits):
            qasm.append("x q[{}];".format(target))
        return qasm

class YGate(OneQubitGate):
    """Pauli's Y Gate"""
    lowername = "y"
    def to_qasm(self, helper, targets):
        n_qubits = helper["n_qubits"]
        qasm = []
        for target in self.target_iter(n_qubits):
            qasm.append("y q[{}];".format(target))
        return qasm

class ZGate(OneQubitGate):
    """Pauli's Z Gate"""
    lowername = "z"
    def to_qasm(self, helper, targets):
        n_qubits = helper["n_qubits"]
        qasm = []
        for target in self.target_iter(n_qubits):
            qasm.append("z q[{}];".format(target))
        return qasm

class HGate(OneQubitGate):
    """Hadamard Gate"""
    lowername = "h"
    def to_qasm(self, helper, targets):
        n_qubits = helper["n_qubits"]
        qasm = []
        for target in self.target_iter(n_qubits):
            qasm.append("h q[{}];".format(target))
        return qasm

class CZGate(TwoQubitGate):
    """Control-Z gate"""
    lowername = "cz"
    def to_qasm(self, helper, args):
        n_qubits = helper["n_qubits"]
        qasm = []
        for control, target in qubit_pairs(args, n_qubits):
            qasm.append("cz q[{}],q[{}];".format(control, target))
        return qasm

class CXGate(TwoQubitGate):
    """Control-X (CNOT) Gate"""
    lowername = "cx"
    def to_qasm(self, helper, args):
        n_qubits = helper["n_qubits"]
        qasm = []
        for control, target in qubit_pairs(args, n_qubits):
            qasm.append("cx q[{}],q[{}];".format(control, target))
        return qasm

class RXGate(OneQubitGate):
    """Rotate-X gate"""
    lowername = "rx"
    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, **kwargs)
        self.theta = theta

    def to_qasm(self, helper, targets):
        n_qubits = helper["n_qubits"]
        qasm = []
        theta = self.theta
        for target in self.target_iter(n_qubits):
            qasm.append("rx({}) q[{}];".format(theta, target))
        return qasm

class RYGate(OneQubitGate):
    """Rotate-Y gate"""
    lowername = "ry"
    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, **kwargs)
        self.theta = theta

    def to_qasm(self, helper, targets):
        n_qubits = helper["n_qubits"]
        qasm = []
        theta = self.theta
        for target in self.target_iter(n_qubits):
            qasm.append("ry({}) q[{}];".format(theta, target))
        return qasm

class RZGate(OneQubitGate):
    """Rotate-Z gate"""
    lowername = "rz"
    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, **kwargs)
        self.theta = theta

    def to_qasm(self, helper, targets):
        n_qubits = helper["n_qubits"]
        qasm = []
        theta = self.theta
        for target in self.target_iter(n_qubits):
            qasm.append("rz({}) q[{}];".format(theta, target))
        return qasm

class TGate(OneQubitGate):
    """T ($\\pi/8$) gate"""
    lowername = "t"
    def to_qasm(self, helper, targets):
        n_qubits = helper["n_qubits"]
        qasm = []
        for target in self.target_iter(n_qubits):
            qasm.append("t q[{}];".format(target))
        return qasm

class SGate(OneQubitGate):
    """S gate"""
    lowername = "s"
    def to_qasm(self, helper, targets):
        n_qubits = helper["n_qubits"]
        qasm = []
        for target in self.target_iter(n_qubits):
            qasm.append("s q[{}];".format(target))
        return qasm

class Measurement(OneQubitGate):
    """Measurement gate"""
    lowername = "measure"
    def to_qasm(self, helper, targets):
        n_qubits = helper["n_qubits"]
        qasm = []
        for target in self.target_iter(n_qubits):
            qasm.append("measure q[{}] -> c[{}];".format(target, target))
        return qasm

def slicing_singlevalue(arg, length):
    """Internally used."""
    if isinstance(arg, slice):
        start, stop, step = arg.indices(length)
        i = start
        if step > 0:
            while i < stop:
                yield i
                i += step
        else:
            while i > stop:
                yield i
                i += step
    else:
        try:
            i = arg.__index__()
        except AttributeError:
            raise TypeError("indices must be integers or slices, not " + arg.__class__.__name__)
        if i < 0:
            i += length
        yield i

def slicing(args, length):
    """Internally used."""
    if isinstance(args, tuple):
        for arg in args:
            yield from slicing_singlevalue(arg, length)
    else:
        yield from slicing_singlevalue(args, length)

def qubit_pairs(args, length):
    """Internally used."""
    if not isinstance(args, tuple):
        raise ValueError("Control and target qubits pair(s) are required.")
    if len(args) != 2:
        raise ValueError("Control and target qubits pair(s) are required.")
    controls = list(slicing(args[0], length))
    targets = list(slicing(args[1], length))
    if len(controls) != len(targets):
        raise ValueError("The number of control qubits and target qubits are must be same.")
    for c, z in zip(controls, targets):
        if c == z:
            raise ValueError("Control qubit and target qubit are must be different.")
    return zip(controls, targets)

def get_maximum_index(indices):
    """Internally used."""
    def _maximum_idx_single(idx):
        if isinstance(idx, slice):
            start = -1
            stop = 0
            if idx.start is not None:
                start = idx.start.__index__()
            if idx.stop is not None:
                stop = idx.stop.__index__()
            return max(start, stop - 1)
        else:
            return idx.__index__()
    if isinstance(indices, tuple):
        return max(_maximum_idx_single(i) for i in indices)
    else:
        return _maximum_idx_single(indices)

def find_n_qubits(gates):
    return max((get_maximum_index(g.targets) for g in gates), default=-1) + 1
