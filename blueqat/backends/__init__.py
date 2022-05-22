from .numpy_backend import NumPyBackend
from .numba_backend import numba_backend_lazy
from .qasm_output_backend import QasmOutputBackend
from .ibmq_backend import ibmq_backend
from .sympy_backend import SympyBackend
from .onequbitgate_transpiler import OneQubitGateCompactionTranspiler
from .twoqubitgate_transpiler import TwoQubitGateDecomposingTranspiler
from .draw_backend import DrawCircuit
from .quimb import Quimb

BACKENDS = {
    "numpy": NumPyBackend,
    "numba": numba_backend_lazy,
    "qasm_output": QasmOutputBackend,
    "ibmq": ibmq_backend,
    "sympy_unitary": SympyBackend,
    "2q_decomposition": TwoQubitGateDecomposingTranspiler,
    "1q_compaction": OneQubitGateCompactionTranspiler,
    "draw": DrawCircuit,
    "quimb": Quimb,
}
DEFAULT_BACKEND_NAME = "quimb"
