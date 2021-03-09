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
    qiskit_backend (Backend, optional): Backend of qiskit.
        If this argument is omitted, Blueqat uses Qiskit Aer's qasm_simulator.
    shots (int, optional): The number of shots.
    returns (str, optional): Choose from followings. Default is "shots".
        "shots": Default behavior. Returns the measurement result.
        "draw": Draw the circuit via `qiskit.circuit.QuantumCircuit.draw()`.
        "qiskit_circuit": Returns Qiskit's circuit class. (qiskit.circuit.QuantumCircuit)
        "qiskit_job": Returns Qiskit's job class. (qiskit.providers.BaseJob)
        "qiskit_result": Returns Qiskit's result class, which is returns by BaseJob.result().
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


def _qasm_runner_qiskit(qasm,
                        qiskit_backend=None,
                        shots=None,
                        returns=None,
                        **kwargs):
    if returns is None:
        returns = "shots"
    elif returns not in ("shots", "draw", "_exception", "qiskit_circuit",
                         "qiskit_job", "qiskit_result"):
        raise ValueError(
            "`returns` shall be None, 'shots', 'draw', " +
            "'qiskit_circuit', 'qiskit_job', 'qiskit_result' or '_exception'")

    import_error = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from qiskit import Aer, QuantumCircuit, execute
    except Exception as e:
        import_error = e

    if import_error:
        if returns == "_exception":
            return import_error
        if isinstance(import_error, ImportError):
            raise ImportError(
                "Cannot import qiskit. To use this backend, please install qiskit."
                + " `pip install qiskit`.")
        else:
            raise ValueError(
                "Unknown error raised when importing qiskit. To get exception, "
                + 'run this backend with arg `returns="_exception"`')
    else:
        if returns == "_exception":
            return None
        qk_circuit = QuantumCircuit.from_qasm_str(qasm)
        if returns == "qiskit_circuit":
            return qk_circuit
        if returns == "draw":
            return qk_circuit.draw(**kwargs)
        if shots is None:
            shots = 1024
        if qiskit_backend is None:
            qiskit_backend = Aer.get_backend("qasm_simulator")
        job = execute(qk_circuit,
                      backend=qiskit_backend,
                      shots=shots,
                      **kwargs)
        if returns == "qiskit_job":
            return job
        result = job.result()
        if returns == "qiskit_result":
            return result
        counts = Counter(
            {bits[::-1]: val
             for bits, val in result.get_counts().items()})
        return counts


ibmq_backend = generate_backend(_qasm_runner_qiskit)
