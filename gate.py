import numpy as np

class IGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, target):
        return qubits

class ZGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, target):
        n_qubits = helper["n_qubits"]
        i = helper["indices"]
        qubits[(i & (1 << target)) != 0] *= -1
        return qubits

class XGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, target):
        n_qubits = helper["n_qubits"]
        i = helper["indices"]
        newq = np.zeros_like(qubits)
        newq[(i & (1 << target)) == 0] = qubits[(i & (1 << target)) != 0]
        newq[(i & (1 << target)) != 0] = qubits[(i & (1 << target)) == 0]
        return newq

class HGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, target):
        n_qubits = helper["n_qubits"]
        i = helper["indices"]
        newq = np.zeros_like(qubits)
        newq[(i & (1 << target)) == 0] = qubits[(i & (1 << target)) == 0] + qubits[(i & (1 << target)) != 0]
        newq[(i & (1 << target)) != 0] = qubits[(i & (1 << target)) == 0] - qubits[(i & (1 << target)) != 0]
        newq /= np.sqrt(2)
        return newq

class CZGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, control, target):
        i = helper["indices"]
        qubits[((i & (1 << control)) != 0) & ((i & (1 << target)) != 0)] *= -1
        return qubits

class CXGate:
    def __init__(self):
        pass

    def apply(self, helper, qubits, control, target):
        h = HGate()
        cz = CZGate()
        qubits = h.apply(helper, qubits, target)
        qubits = cz.apply(helper, qubits, control, target)
        qubits = h.apply(helper, qubits, target)
        return qubits

class _DebugDisplay:
    def __init__(self, *args, **kwargs):
        self.ctor_args = (args, kwargs)

    def apply(self, helper, qubits, target):
        print("ctor arguments:", self.ctor_args)
        print("helper:", helper)
        print("target:", target)
        print("qubits:", qubits)
        indices = np.arange(0, 2**n_qubits, dtype=int)
        return qubits
