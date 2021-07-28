from typing import Callable, List, Optional

import numpy as np

from ..circuit import Circuit
from ..gate import *
from .backendbase import Backend

_eye = np.eye(2, dtype=complex)


class OneQubitGateCompactionTranspiler(Backend):
    """Merge one qubit gate."""
    def _run_inner(self, gates, operations: List[Operation],
                   singlemats: List[np.ndarray], n_qubits: int,
                   mat1_decomposer: Callable[[OneQubitGate], List[Operation]]):
        for gate in gates:
            if isinstance(gate, Gate) and gate.n_qargs == 1:
                for t in gate.target_iter(n_qubits):
                    np.matmul(gate.matrix(), singlemats[t], out=singlemats[t])
            else:
                for t in gate.target_iter(n_qubits):
                    if not np.allclose(singlemats[t], _eye):
                        operations += mat1_decomposer(
                            Mat1Gate.create((t, ), (singlemats[t],)))
                        singlemats[t] = _eye.copy()
                operations.append(gate)

    def run(self,
            gates,
            n_qubits: int,
            *,
            mat1_decomposer: Optional[Callable[[OneQubitGate],
                                               List[Operation]]] = None,
            **kwargs) -> Circuit:
        """
        Args:
            basis (str or non-empty tuple of str): Name (lowername) of basis 2 qubits gate.
                If multiple basis gates are given, given basis gates are not decomposited and
                non-basis gate is decomposited to first available basis gate.
        """
        if mat1_decomposer is None:
            mat1_decomposer = lambda g: [g]
        operations = []
        singlemats = [_eye.copy() for _ in range(n_qubits)]
        self._run_inner(gates, operations, singlemats, n_qubits,
                        mat1_decomposer)
        for i, mat in enumerate(singlemats):
            if not np.allclose(mat, _eye):
                operations += mat1_decomposer(Mat1Gate.create((i, ), (mat,)))
        return Circuit(n_qubits, operations)
