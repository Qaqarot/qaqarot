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

from blueqat import Circuit
from ..gate import *
from .backendbase import Backend


class ManyQubitGateFallbacker(Backend):
    """Decomposite more than 2 qubit gate by fallback system."""
    def _run_inner(self, gates, n_qubits) -> List[Operation]:
        decomposed = []
        for gate in gates:
            if gate.n_qargs > 2:
                decomposed += self._run_inner(gate.fallback(n_qubits),
                                              n_qubits)
            else:
                decomposed.append(gate)
        return decomposed

    def run(self, gates, n_qubits, *args, **kwargs) -> Circuit:
        return Circuit(n_qubits, self._run_inner(gates, n_qubits))
