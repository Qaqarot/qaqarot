import numpy as np
from . import gate

DEFAULT_GATE_SET = {
    "i": gate.IGate,
    "x": gate.XGate,
    "y": gate.YGate,
    "z": gate.ZGate,
    "h": gate.HGate,
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
    "dbg": gate.DebugDisplay,
}
DEFAULT_DTYPE = np.complex128

class Circuit:
    def __init__(self, n_qubits=0, ops=None, gate_set=None):
        self.gate_set = gate_set or DEFAULT_GATE_SET.copy()
        self.ops = ops or []
        self.n_qubits = n_qubits
        self.run_history = []
        self.cache = None
        self.cache_idx = -1

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
        self.n_qubits = max(self.n_qubits, other.n_qubits)
        return self

    def copy(self, copy_cache=True, copy_history=False):
        copied = Circuit(self.n_qubits, self.ops.copy(), self.gate_set.copy())
        if copy_cache and self.cache is not None:
            copied.cache = self.cache.copy()
            copied.cache_idx = self.cache_idx
        if copy_history:
            copied.run_history = self.run_history.copy()
        return copied

    def run(self):
        n_qubits = self.n_qubits
        if self.cache is not None:
            if len(self.cache) == 2**n_qubits:
                qubits = self.cache.copy()
            else:
                self.cache = None
                self.cache_idx = -1
        if self.cache is None:
            qubits = np.zeros(2**n_qubits, dtype=DEFAULT_DTYPE)
            qubits[0] = 1.0
        helper = {
            "n_qubits": n_qubits,
            "indices": np.arange(2**n_qubits, dtype=np.uint32),
            "cregs": [0] * n_qubits,
        }
        save_cache = True
        for i, op in enumerate(self.ops[self.cache_idx + 1:], start=self.cache_idx + 1):
            gate = op.gate(*op.args, **op.kwargs)
            qubits = gate.apply(helper, qubits, op.target)
            if save_cache:
                if hasattr(gate, "no_cache") and gate.no_cache:
                    save_cache = False
                else:
                    self.cache = qubits.copy()
                    self.cache_idx = i
        self.run_history.append(tuple(helper["cregs"]))
        _ignore_globals(qubits)
        return qubits

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
            gate = op.gate(*op.args, **op.kwargs)
            qasm += gate.to_qasm(helper, op.target)
        return "\n".join(qasm)

    def last_result(self):
        try:
            return self.run_history[-1]
        except IndexError:
            raise ValueError("The Circuit has never been to run.")

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
        self.circuit.n_qubits = max(gate.get_maximum_index(args) + 1, self.circuit.n_qubits)
        self.circuit.ops.append(self)
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

def _ignore_globals(qubits):
    for i, q in enumerate(qubits):
        if abs(q) > 0.0000001:
            ang = abs(q) / q
            qubits *= ang
            return
