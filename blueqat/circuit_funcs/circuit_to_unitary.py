import numpy as np

from blueqat import Circuit

def circuit_to_unitary(circ: Circuit, *runargs, **runkwargs) -> np.ndarray:
    """Make circuit to unitary. This function is experimental feature and
    may changed or deleted in the future."""
    runkwargs.setdefault('returns', 'statevector')
    runkwargs.setdefault('ignore_global', False)
    n_qubits = circ.n_qubits
    vecs = []
    if n_qubits == 0:
        return np.array([[1]])
    for i in range(1 << n_qubits):
        bitmask = tuple(k for k in range(n_qubits) if (1 << k) & i)
        c = Circuit()
        if bitmask:
            c.x[bitmask]
        c += circ
        vecs.append(c.run(*runargs, **runkwargs))
    return np.array(vecs).T
