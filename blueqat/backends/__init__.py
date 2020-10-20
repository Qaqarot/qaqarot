from .numpy_backend import NumPyBackend
from .numba_backend import numba_backend_lazy
from .qasm_output_backend import QasmOutputBackend
from .ibmq_backend import ibmq_backend
from .sympy_backend import SympyBackend
from .qgate_backend import QgateBackend
from .onequbitgate_transpiler import OneQubitGateCompactionTranspiler
from .twoqubitgate_transpiler import TwoQubitGateDecomposingTranspiler

BACKENDS = {
    "numpy": NumPyBackend,
    "numba": numba_backend_lazy,
    "qasm_output": QasmOutputBackend,
    "ibmq": ibmq_backend,
    "sympy_unitary": SympyBackend,
    "qgate": QgateBackend,
    "2q_decomposition": TwoQubitGateDecomposingTranspiler,
    "1q_compaction": OneQubitGateCompactionTranspiler,
}
DEFAULT_BACKEND_NAME = "numpy"
