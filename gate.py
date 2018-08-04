import numpy as np

class ZGate:
    def __init__(self):
        pass

    def apply(self, n_qubits, qubits, target):
        qubits[2**(n_qubits-target-1)::2**target] *= -1
        return qubits
