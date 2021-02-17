from typing import Callable, Tuple, List, Optional, Sequence, Union

import numpy as np

from ..circuit import Circuit
from ..gate import *
from .backendbase import Backend

_eye = np.eye(2, dtype=complex)

# Dict[str, TwoQubitGate]
BASIS_TABLE = {
    'cx': CXGate,
    'cz': CZGate,
    'zz': ZZGate,
}

# Dict[str, Dict[str, Tuple[np.array, np.array, Tuple[Any], np.array, np.array]]]
DECOMPOSE_TABLE = {
    'cx': {
        'cz': (_eye.copy(), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
               (),
               _eye.copy(), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)),
        'zz': (_eye.copy(), np.array([[1j, -1j], [1j, 1j]]) / np.sqrt(2),
               (),
               np.diag([1 - 1j, 1 + 1j]) / np.sqrt(2),
               np.array([[0.5 - 0.5j, 0.5 - 0.5j], [-0.5 - 0.5j, 0.5 + 0.5j]])),
    },
    'cz': {
        'cx': (_eye.copy(), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
               (),
               _eye.copy(), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)),
        'zz': (_eye.copy(), np.array([[1j, 0], [0, -1j]]),
               (),
               np.diag([1 - 1j, 1 + 1j]) / np.sqrt(2),
               np.array([[1 - 1j, 0], [0, -1 - 1j]]) / np.sqrt(2)),
    },
    'zz': {
        'cx': (_eye.copy(), np.array([[-1j, -1j], [1j, -1j]]) / np.sqrt(2),
               (),
               np.diag([1 + 1j, 1 - 1j]) / np.sqrt(2),
               np.array([[0.5 + 0.5j, -0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]])),
        'cz': (_eye.copy(), np.array([[-1j, 0], [0, 1j]]),
               (),
               np.diag([1 + 1j, 1 - 1j]) / np.sqrt(2),
               np.array([[1 + 1j, 0], [0, -1 + 1j]]) / np.sqrt(2)),
    },
}


class TwoQubitGateDecomposingTranspiler(Backend):
    """Decomposite two qubit gate."""
    def _run_inner(self, gates, operations: List[Operation], singlemats: List[np.array],
            n_qubits: int, basis: Sequence[str],
            mat1_decomposer: Callable[[OneQubitGate], List[Operation]]):
        basisgate = BASIS_TABLE[basis[0]]
        table = DECOMPOSE_TABLE[basis[0]]
        for gate in gates:
            if not isinstance(gate, Gate):
                # Non-gate operations.
                for t in gate.target_iter(n_qubits):
                    if not np.allclose(singlemats[t], _eye):
                        operations += mat1_decomposer(Mat1Gate((t,), singlemats[t]))
                        singlemats[t] = _eye.copy()
                operations.append(gate)
            elif gate.n_qargs == 1:
                for t in gate.target_iter(n_qubits):
                    np.matmul(gate.matrix(), singlemats[t], out=singlemats[t])
            elif gate.lowername in basis:
                for t1, t2 in gate.control_target_iter(n_qubits):
                    if not np.allclose(singlemats[t1], _eye):
                        operations += mat1_decomposer(Mat1Gate((t1,), singlemats[t1]))
                        singlemats[t1] = _eye.copy()
                    if not np.allclose(singlemats[t2], _eye):
                        operations += mat1_decomposer(Mat1Gate((t2,), singlemats[t2]))
                        singlemats[t2] = _eye.copy()
                operations.append(gate)
            elif gate.lowername in table:
                for t1, t2 in gate.control_target_iter(n_qubits):
                    l1, l2, gparams, r1, r2 = table[gate.lowername]
                    np.matmul(l1, singlemats[t1], out=singlemats[t1])
                    np.matmul(l2, singlemats[t2], out=singlemats[t2])
                    if not np.allclose(singlemats[t1], _eye):
                        operations += mat1_decomposer(Mat1Gate((t1,), singlemats[t1]))
                    singlemats[t1] = r1.copy()
                    if not np.allclose(singlemats[t2], _eye):
                        operations += mat1_decomposer(Mat1Gate((t2,), singlemats[t2]))
                    singlemats[t2] = r2.copy()
                    operations.append(basisgate((t1, t2), *gparams))
            else:
                self._run_inner(gate.fallback(n_qubits), operations, singlemats, n_qubits, basis, mat1_decomposer)

    def run(self, gates, n_qubits: int, *,
            basis: Union[str, Sequence[str]],
            mat1_decomposer: Optional[Callable[[OneQubitGate], List[Operation]]] = None,
            **kwargs) -> Circuit:
        """
        Args:
            basis (str or non-empty tuple of str): Name (lowername) of basis 2 qubits gate.
                If multiple basis gates are given, given basis gates are not decomposited and
                non-basis gate is decomposited to first available basis gate.
        """
        if mat1_decomposer is None:
            mat1_decomposer = lambda g: [g]
        if isinstance(basis, str):
            basis = (basis,)
        try:
            primary_basis = [b for b in basis if b in DECOMPOSE_TABLE][0]
        except IndexError:
            raise ValueError(f'Unsupported basis.')
        operations = []
        if not basis:
            raise ValueError('No basis provided.')
        singlemats = [_eye.copy() for _ in range(n_qubits)]
        self._run_inner(gates, operations, singlemats, n_qubits, basis, mat1_decomposer)
        for i, mat in enumerate(singlemats):
            if not np.allclose(mat, _eye):
                operations += mat1_decomposer(Mat1Gate((i,), mat))
        return Circuit(n_qubits, operations)
