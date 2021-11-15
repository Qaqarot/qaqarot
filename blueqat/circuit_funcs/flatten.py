from .. import Circuit
from .. import gate as g

def flatten(c: Circuit) -> Circuit:
    """expands slice and multiple targets into single target"""
    n_qubits = c.n_qubits
    ops = []
    for op in c.ops:
        if isinstance(op, (g.OneQubitGate, g.Reset)):
            ops += [op.create(t, op.params) for t in op.target_iter(n_qubits)]
        elif isinstance(op, g.TwoQubitGate):
            ops += [op.create(t, op.params) for t in op.control_target_iter(n_qubits)]
        elif isinstance(op, g.Measurement):
            if op.key is None:
                ops += [op.create(t, op.params) for t in op.target_iter(n_qubits)]
            else:
                options = {'key': op.key}
                if op.duplicated is not None:
                    options['duplicated'] = op.duplicated
                ops += [op.create(tuple(t for t in op.target_iter(n_qubits)), op.params, options)]
        else:
            raise ValueError(f"Cannot process operation {op.lowername}.")
    return Circuit(n_qubits, ops)
