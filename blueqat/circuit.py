import warnings
import numpy as np
from . import gate
from .backends.numpy_backend import NumPyBackend

DEFAULT_GATE_SET = {
    "i": gate.IGate,
    "x": gate.XGate,
    "y": gate.YGate,
    "z": gate.ZGate,
    "h": gate.HGate,
    "t": gate.TGate,
    "s": gate.SGate,
    "cz": gate.CZGate,
    "cx": gate.CXGate,
    "cnot": gate.CXGate,
    "rx": gate.RXGate,
    "ry": gate.RYGate,
    "rz": gate.RZGate,
    "phase": gate.RZGate,
    "u1": gate.RZGate,
    "measure": gate.Measurement,
    "m": gate.Measurement,
}
DEFAULT_DTYPE = np.complex128

class Circuit:
    def __init__(self, n_qubits=0, ops=None, gate_set=None):
        self.gate_set = gate_set or DEFAULT_GATE_SET.copy()
        self.ops = ops or []
        self.cache = None
        self.cache_idx = -1
        self._backend = NumPyBackend()
        if n_qubits > 0:
            self.i[n_qubits - 1]

    def __getattr__(self, name):
        if name in self.gate_set:
            return _GateWrapper(self, name, self.gate_set[name])
        raise AttributeError("'circuit' object has no attribute or gate '" + name + "'")

    def __add__(self, other):
        if not isinstance(other, Circuit):
            return NotImplemented
        c = self.copy()
        c += other
        return c

    def __iadd__(self, other):
        if not isinstance(other, Circuit):
            return NotImplemented
        if self.gate_set != other.gate_set:
            raise ValueError("Cannot connect the circuits between different gate set.")
        self.ops += other.ops
        return self

    def copy(self, copy_cache=True, copy_history=None):
        copied = Circuit(self.n_qubits, self.ops.copy(), self.gate_set.copy())
        if copy_cache and self.cache is not None:
            copied.cache = self.cache.copy()
            copied.cache_idx = self.cache_idx
        if copy_history is not None:
            warnings.warn("copy_history is deprecated", DeprecationWarning)
        return copied

    def run(self, *args, **kwargs):
        return self._backend.run(self.ops, args, kwargs)

    def to_qasm(self, output_prologue=True):
        n_qubits = self.n_qubits
        helper = {
            "n_qubits": n_qubits,
        }
        if output_prologue:
            qasm = [
                "OPENQASM 2.0;",
                'include "qelib1.inc";',
                "qreg q[{}];".format(n_qubits),
                "creg c[{}];".format(n_qubits),
            ]
        else:
            qasm = []
        for op in self.ops:
            qasm += op.to_qasm(helper, op.targets)
        return "\n".join(qasm)

    def last_result(self):
        # Too noisy...
        #warnings.warn("last_result will be deprecated", DeprecationWarning)
        try:
            return self._backend.run_history[-1]
        except IndexError:
            raise ValueError("The Circuit has never been to run.")

    @property
    def n_qubits(self):
        return gate.find_n_qubits(self.ops)

    @property
    def run_history(self):
        warnings.warn("run_history will be deprecated", DeprecationWarning)
        return self._backend.run_history

class _GateWrapper:
    def __init__(self, circuit, name, gate):
        self.circuit = circuit
        self.target = None
        self.name = name
        self.gate = gate
        self.args = ()
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def __getitem__(self, args):
        self.target = args
        self.circuit.ops.append(self.gate(self.target, *self.args, **self.kwargs))
        return self.circuit

    def __str__(self):
        if self.args:
            args_str = str(self.args)
            if self.kwargs:
                args_str = args_str[:-1] + ", kwargs=" + str(self.kwargs) + ")"
        elif self.kwargs:
            args_str = "(kwargs=" + str(self.kwargs) + ")"
        else:
            args_str = ""
        return self.name + args_str + " " + str(self.target)
