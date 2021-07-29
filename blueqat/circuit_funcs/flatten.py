from .. import Circuit
from .. import gate as g

def flatten(c: Circuit) -> Circuit:
    """expands slice and multiple targets into single target"""
    n_qubits = c.n_qubits
    ops = []
    for op in c.ops:
        if isinstance(op, (g.OneQubitGate, g.Measurement, g.Reset)):
            ops += [op.create(t, op.params) for t in op.target_iter(n_qubits)]
        elif isinstance(op, g.TwoQubitGate):
            ops += [op.create(t, op.params) for t in op.control_target_iter(n_qubits)]
        else:
            raise ValueError(f"Cannot process operation {op.lowername}.")
    return Circuit(n_qubits, ops)
