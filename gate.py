import numpy as np
import random
import math

class IGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, targets):
        return qubits

class XGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, targets):
        n_qubits = helper["n_qubits"]
        i = helper["indices"]
        for target in slicing(targets, n_qubits):
            newq = np.zeros_like(qubits)
            newq[(i & (1 << target)) == 0] = qubits[(i & (1 << target)) != 0]
            newq[(i & (1 << target)) != 0] = qubits[(i & (1 << target)) == 0]
            qubits = newq
        return qubits

class YGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, targets):
        n_qubits = helper["n_qubits"]
        i = helper["indices"]
        for target in slicing(targets, n_qubits):
            newq = np.zeros_like(qubits)
            newq[(i & (1 << target)) == 0] = -1.0j * qubits[(i & (1 << target)) != 0]
            newq[(i & (1 << target)) != 0] = 1.0j * qubits[(i & (1 << target)) == 0]
            qubits = newq
        return qubits

class ZGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, targets):
        n_qubits = helper["n_qubits"]
        i = helper["indices"]
        for target in slicing(targets, n_qubits):
            qubits[(i & (1 << target)) != 0] *= -1
        return qubits

class HGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, targets):
        n_qubits = helper["n_qubits"]
        i = helper["indices"]
        for target in slicing(targets, n_qubits):
            newq = np.zeros_like(qubits)
            newq[(i & (1 << target)) == 0] = qubits[(i & (1 << target)) == 0] + qubits[(i & (1 << target)) != 0]
            newq[(i & (1 << target)) != 0] = qubits[(i & (1 << target)) == 0] - qubits[(i & (1 << target)) != 0]
            newq /= np.sqrt(2)
            qubits = newq
        return qubits

class CZGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, args):
        n_qubits = helper["n_qubits"]
        for control, target in qubit_pairs(args, n_qubits):
            i = helper["indices"]
            qubits[((i & (1 << control)) != 0) & ((i & (1 << target)) != 0)] *= -1
        return qubits

class CXGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, args):
        n_qubits = helper["n_qubits"]
        for control, target in qubit_pairs(args, n_qubits):
            h = HGate()
            cz = CZGate()
            qubits = h.apply(helper, qubits, target)
            qubits = cz.apply(helper, qubits, (control, target))
            qubits = h.apply(helper, qubits, target)
        return qubits

class RXGate:
    def __init__(self, theta):
        self.theta = theta

    def apply(self, helper, qubits, targets):
        n_qubits = helper["n_qubits"]
        i = helper["indices"]
        theta = self.theta
        for target in slicing(targets, n_qubits):
            newq = np.zeros_like(qubits)
            newq[(i & (1 << target)) == 0] = (
                np.cos(theta / 2) * qubits[(i & (1 << target)) == 0] +
                -1.0j * np.sin(theta / 2) * qubits[(i & (1 << target)) != 0]
            )
            newq[(i & (1 << target)) != 0] = (
                -1.0j * np.sin(theta / 2) * qubits[(i & (1 << target)) == 0] +
                np.cos(theta / 2) * qubits[(i & (1 << target)) != 0]
            )
            qubits = newq
        return qubits

class RYGate:
    def __init__(self, theta):
        self.theta = theta

    def apply(self, helper, qubits, targets):
        n_qubits = helper["n_qubits"]
        i = helper["indices"]
        theta = self.theta
        for target in slicing(targets, n_qubits):
            newq = np.zeros_like(qubits)
            newq[(i & (1 << target)) == 0] = (
                np.cos(theta / 2) * qubits[(i & (1 << target)) == 0] +
                -np.sin(theta / 2) * qubits[(i & (1 << target)) != 0]
            )
            newq[(i & (1 << target)) != 0] = (
                np.sin(theta / 2) * qubits[(i & (1 << target)) == 0] +
                np.cos(theta / 2) * qubits[(i & (1 << target)) != 0]
            )
            qubits = newq
        return qubits

class RZGate:
    def __init__(self, theta):
        self.theta = theta

    def apply(self, helper, qubits, targets):
        n_qubits = helper["n_qubits"]
        i = helper["indices"]
        theta = self.theta
        for target in slicing(targets, n_qubits):
            qubits[(i & (1 << target)) != 0] *= complex(math.cos(theta), math.sin(theta))
        return qubits

class Measurement:
    def __init__(self):
        pass

    def apply(self, helper, qubits, targets):
        n_qubits = helper["n_qubits"]
        i = helper["indices"]
        for target in slicing(targets, n_qubits):
            p_zero = (qubits[(i & (1 << target)) == 0].T.conjugate() @ qubits[(i & (1 << target)) == 0]).real
            rand = random.random()
            if rand < p_zero:
                qubits[(i & (1 << target)) != 0] = 0.0
                qubits /= p_zero
                helper["cregs"][target] = 0
            else:
                qubits[(i & (1 << target)) == 0] = 0.0
                qubits /= (1.0 - p_zero)
                helper["cregs"][target] = 1
        return qubits

class DebugDisplay:
    def __init__(self, *args, **kwargs):
        self.ctor_args = (args, kwargs)

    def apply(self, helper, qubits, targets):
        n_qubits = helper["n_qubits"]
        for target in slicing(targets, n_qubits):
            print("ctor arguments:", self.ctor_args)
            print("helper:", helper)
            print("target:", target)
            print("qubits:", qubits)
            indices = np.arange(0, 2**n_qubits, dtype=int)
        return qubits

def slicing_singlevalue(arg, length):
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
    if isinstance(args, tuple):
        for arg in args:
            yield from slicing_singlevalue(arg, length)
    else:
        yield from slicing_singlevalue(args, length)

def qubit_pairs(args, length):
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
