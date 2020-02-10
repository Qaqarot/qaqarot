from typing import Tuple, NamedTuple
from functools import partial
from .. import gate
from .. import Circuit

GATE_SET = {
    "i": gate.IGate,
    "x": gate.XGate,
    "y": gate.YGate,
    "z": gate.ZGate,
    "h": gate.HGate,
    "t": gate.TGate,
    "tdg": gate.TDagGate,
    "s": gate.SGate,
    "sdg": gate.SDagGate,
    "cz": gate.CZGate,
    "cx": gate.CXGate,
    "cnot": gate.CXGate,
    "cy": gate.CYGate,
    "ch": gate.CHGate,
    "rx": gate.RXGate,
    "ry": gate.RYGate,
    "rz": gate.RZGate,
    "r": gate.PhaseGate,
    "phase": gate.PhaseGate,
    "crx": gate.CRXGate,
    "cry": gate.CRYGate,
    "crz": gate.CRZGate,
    "cr": gate.CPhaseGate,
    "cphase": gate.CPhaseGate,
    "u1": gate.U1Gate,
    "u2": gate.U2Gate,
    "u3": gate.U3Gate,
    "cu1": gate.CU1Gate,
    "cu2": gate.CU2Gate,
    "cu3": gate.CU3Gate,
    "swap": gate.SwapGate,
    "ccx": gate.ToffoliGate,
    "toffoli": gate.ToffoliGate,
    "measure": gate.Measurement,
    "m": gate.Measurement,
}

GLOBAL_MACROS = {}

class Ops(NamedTuple):
    """Immutable type operations"""
    n_qubits: int = 0
    ops: Tuple[gate.Gate] = ()

    def __repr__(self):
        return f'Ops({self.n_qubits}).' + '.'.join(str(op) for op in self.ops)

    def __getattr__(self, name):
        if name in GATE_SET:
            return _GateWrapper(self, name, GATE_SET[name])
        if name in GLOBAL_MACROS:
            return partial(GLOBAL_MACROS[name], self)
        raise AttributeError(f"'circuit' object has no attribute or gate '{name}'")

    def __add__(self, other):
        if not isinstance(other, Ops):
            return NotImplemented
        return Ops(max(self.n_qubits, other.n_qubits), self.ops + other.ops)

    def dagger(self):
        """Get Hamiltonian conjugate of `self`."""
        return Ops(self.n_qubits, tuple(g.dagger() for g in reversed(self.ops)))

    def to_circuit(self):
        return Circuit(self.n_qubits, list(self.ops))


class _GateWrapper:
    def __init__(self, ops, name, gate):
        self.ops = ops
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
        n_qubits = max(self.ops.n_qubits, gate.get_maximum_index(args) + 1)
        ops = self.ops.ops + (self.gate(self.target, *self.args, **self.kwargs),)
        return Ops(n_qubits, ops)

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
