import numpy as np
import gate

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
            return _GateWrapper(self, self.gate_set[name])
        raise AttributeError("'circuit' object has no attribute or gate '" + name + "'")

    def copy(self):
        return Circuit(self.n_qubits, self.ops, self.gate_set)

    def run(self):
        n_qubits = self.n_qubits
        if self.cache is None:
            qubits = np.zeros(2**n_qubits, dtype=DEFAULT_DTYPE)
            qubits[0] = 1.0
        else:
            qubits = self.cache.copy()
        helper = {
            "n_qubits": n_qubits,
            "indices": np.arange(2**n_qubits, dtype=np.uint32),
            "cregs": [0] * n_qubits,
        }
        save_cache = True
        for i, op in enumerate(self.ops[self.cache_idx+1:]):
            gate = op.gate(*op.args, **op.kwargs)
            qubits = gate.apply(helper, qubits, op.target)
            if save_cache:
                if hasattr(gate, "no_cache") and gate.no_cache:
                    save_cache = False
                else:
                    self.cache = qubits.copy()
                    self.cache_idx = i
        self.run_history.append(tuple(helper["cregs"]))
        ignore_globals(qubits)
        return qubits

    def last_result(self):
        try:
            return self.run_history[-1]
        except IndexError:
            raise ValueError("The Circuit has never been to run.")

class _GateWrapper:
    def __init__(self, circuit, gate):
        self.circuit = circuit
        self.target = None
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

def ignore_globals(qubits):
    for i,q in enumerate(qubits):
        if abs(q) > 0.0000001:
            ang = abs(q) / q
            qubits *= ang
            return
