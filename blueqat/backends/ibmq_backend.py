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

"""This module provides IBM Q Backend via OpenQASM and Qiskit.

To run this backend, call
Circuit.run_with_ibmq or Circuit.run(backend="ibmq", ...).

Args:
    qiskit_backend: Backend of qiskit.
    shots (int, optional): The number of shots.
    returns (str, optional): You can specify "shots", "qiskit_result" or "_exception".
        "shots": Default behavior. Returns the measurement result.
        "qiskit_result": Returns result which is returned by Qiskit.
        "_exception": This is for troubleshooting purpose.
                      Returns the exception which raised during importing Qiskit.
                      If no exception is raised, returns None.
    kwargs: Other arguments which passed to qiskit.execute
Returns:
    The result of run.
"""

import warnings
from collections import Counter
from .qasm_parser_backend_generator import generate_backend


def _qasm_runner_qiskit(qasm, qiskit_backend, shots=None, returns=None, **kwargs):
    if returns is None:
        returns = "shots"
    elif returns not in ("shots", "_exception", "qiskit_result"):
        raise ValueError("`returns` shall be None, 'shots', 'qiskit_result' or '_exception'")

    import_error = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from qiskit import QuantumCircuit, execute
    except Exception as e:
        import_error = e

    if import_error:
        if returns == "_exception":
            return e
        if isinstance(import_error, ImportError):
            raise ImportError("Cannot import qiskit. To use this backend, please install qiskit." +
                              " `pip install qiskit`.")
        else:
            raise ValueError("Unknown error raised when importing qiskit. To get exception, " +
                             'run this backend with arg `returns="_exception"`')
    else:
        if shots is None:
            shots = 1024
        qk_circuit = QuantumCircuit.from_qasm_str(qasm)
        result = execute(qk_circuit, backend=qiskit_backend, shots=shots, **kwargs).result()
        if returns == "qiskit_result":
            return result
        counts = Counter({bits[::-1]: val for bits, val in result.get_counts().items()})
        return counts

ibmq_backend = generate_backend(_qasm_runner_qiskit)
