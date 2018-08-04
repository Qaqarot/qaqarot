import numpy as np
import gate

DEFAULT_GATE_SET = {
    "i": gate.IGate,
    "z": gate.ZGate,
    "x": gate.XGate,
    "h": gate.HGate,
    "cz": gate.CZGate,
    "_dbg": gate._DebugDisplay,
}
DEFAULT_DTYPE = np.complex64

class Circuit:
    def __init__(self, gate_set=None, ops=None, n_qubits=0):
        self.gate_set = gate_set or DEFAULT_GATE_SET.copy()
        self.ops = ops or []
        self.n_qubits = n_qubits

    def __getattr__(self, name):
        if name in self.gate_set:
            return _GateWrapper(self, self.gate_set[name])
        raise AttributeError("'circuit' object has no attribute or gate '" + name + "'")

    def copy(self):
        return Circuit(self.gate_set, self.ops, n_qubits)

    def run(self):
        n_qubits = self.n_qubits
        qubits = np.zeros(2**n_qubits, dtype=DEFAULT_DTYPE)
        qubits[0] = 1.0
        helper = {
                "n_qubits": n_qubits,
                "indices": np.arange(2**n_qubits, dtype=np.uint32)
        }
        for op in self.ops:
            gate = op.gate(*op.args, **op.kwargs)
            qubits = gate.apply(helper, qubits, *op.target)
        return qubits

class _GateWrapper:
    def __init__(self, circuit, gate):
        self.circuit = circuit
        self.gate = gate
        self.args = ()
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        self.target = args
        self.circuit.n_qubits = max(max(args) + 1, self.circuit.n_qubits)
        self.circuit.ops.append(self)
        return self.circuit
