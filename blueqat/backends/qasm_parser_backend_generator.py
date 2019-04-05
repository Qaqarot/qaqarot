# Copyright 2019 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..gate import *
from .backendbase import Backend
from .qasm_output_backend import QasmOutputBackend

class QasmParsableBackend(Backend):
    """Backend for third-party library which can read OpenQASM """
    def __init__(self, qasm_runner):
        """Specify qasm_runner which receive OpenQASM text and returns result.

        Args:
            qasm_runner (function (qasm: str, *args, **kwargs) -> Result):
                An function which receives OpenQASM and some arguments and returns a result.
        """
        self.to_qasm = QasmOutputBackend()
        self.qasm_runner = qasm_runner

    def run(self, gates, n_qubits, *args, **kwargs):
        qasm = self.to_qasm.run(gates, n_qubits)
        return self.qasm_runner(qasm, *args, **kwargs)

def generate_backend(qasm_runner):
    """Generate a wrapper of QasmParsableBackend from qasm_runner.

    Due to Blueqat's backend specifications, normally, cannot give arguments to
    QasmParsableBackend's constructor.
    This function wrap the class for specify `qasm_runner` to QasmParsableBackend's constructor.

    Args:
        qasm_runner (function (qasm: str, *args, **kwargs) -> Result):
            An function which receives OpenQASM and some arguments and returns a result.

    Returns:
        function () -> QasmParsableBackend(qasm_runner)
    """
    return lambda: QasmParsableBackend(qasm_runner)
