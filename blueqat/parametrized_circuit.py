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
"""
This module defines ParametrizedCircuit.
"""

from functools import partial, update_wrapper
from typing import cast, Any, List, Mapping, Optional, Type, Union

from . import gate
from .circuit import GLOBAL_MACROS, Circuit
from .gateset import get_op_type
from .typing import GeneralCircuitOperation, T
from .parameter import Parameter, ParamAssign

ParametrizedCircuitOperation = GeneralCircuitOperation['ParametrizedCircuit',
                                                       T]
class ParametrizedCircuit:
    """Store the gate operations and call the backends."""

    def __init__(self,
                 n_qubits: int = 0,
                 ops: Optional[List[gate.Operation]] = None,
                 params: Optional[List[str]] = None):
        self.ops = ops or []
        self.n_qubits = n_qubits
        self.params = params or []

    def add_param(self, param: str) -> int:
        self.params.append(param)
        return len(self.params) - 1

    def __repr__(self):
        return f'ParametrizedCircuit({self.n_qubits}).' + '.'.join(
            str(op) for op in self.ops)

    def __getattr__(self, name: str) -> ParametrizedCircuitOperation[Any]:
        op_type = get_op_type(name)
        if op_type:
            return _GateWrapper(self, op_type)
        if name in GLOBAL_MACROS:
            macro = update_wrapper(partial(GLOBAL_MACROS[name], self),
                                   GLOBAL_MACROS[name])
            return cast(ParametrizedCircuitOperation[Any], macro)
        raise AttributeError(
            f"'ParametrizedCircuit' object has no attribute or gate '{name}'")

    def __add__(self, other):
        if not isinstance(other, (Circuit, ParametrizedCircuit)):
            return NotImplemented
        c = self.copy()
        c += other
        return c

    def __iadd__(self, other):
        if not isinstance(other, (Circuit, ParametrizedCircuit)):
            return NotImplemented
        self.ops += other.ops
        self.n_qubits = max(self.n_qubits, other.n_qubits)
        self.params += get_params(other)
        return self

    def copy(self) -> 'ParametrizedCircuit':
        """Copy the circuit.

        params:
            | copy_backends :bool copy backends if True.
        """
        copied = ParametrizedCircuit(self.n_qubits, self.ops.copy())
        return copied

    def dagger(self,
               ignore_measurement: bool = False) -> 'ParametrizedCircuit':
        """Make Hermitian conjugate of the circuit.

        This feature is beta. Interface may be changed.

        ignore_measurement (bool, optional):
            | If True, ignore the measurement in the circuit.
            | Otherwise, if measurement in the circuit, raises ValueError.
        """
        ops = []
        for g in reversed(self.ops):
            if isinstance(g, gate.Gate):
                ops.append(g.dagger())
            else:
                if not ignore_measurement:
                    raise ValueError(
                        'Cannot make the Hermitian conjugate of this circuit because '
                        'the circuit contains measurement.')

        copied = ParametrizedCircuit(self.n_qubits, ops)
        return copied

    def subs(self, params: ParamAssign) -> Circuit:
        if isinstance(params, Mapping):
            if set(self.params) != set(params.keys()):
                raise ValueError('Parameters are not matching.')
        else:
            if len(self.params) != len(params):
                raise ValueError('The number of parameters is not matching.')
        return Circuit(
                self.n_qubits,
                [op.subs(params) for op in self.ops]
        )


class _GateWrapper(ParametrizedCircuitOperation[ParametrizedCircuit]):

    def __init__(self, circuit: ParametrizedCircuit,
                 op_type: Type[gate.Operation]):
        self.circuit = circuit
        self.op_type = op_type
        self.params = ()
        self.options = None

    def __call__(self, *args, **kwargs) -> '_GateWrapper':

        def make_fparam(param: Union[float, str]) -> Union[float, Parameter]:
            if isinstance(param, str):
                idx = self.circuit.add_param(param)
                return Parameter(param, idx)
            return param

        self.params = tuple(make_fparam(p) for p in args)
        if kwargs:
            self.options = kwargs
        return self

    def __getitem__(self, targets) -> 'ParametrizedCircuit':
        self.circuit.ops.append(
            self.op_type.create(targets, self.params, self.options))
        # ad-hoc
        self.circuit.n_qubits = max(
            gate.get_maximum_index(targets) + 1, self.circuit.n_qubits)
        return self.circuit

    def __str__(self) -> str:
        if self.params:
            args_str = str(self.params)
        else:
            args_str = ""
        if self.options:
            args_str += str(self.options)
        return self.op_type.lowername + args_str


def get_params(c: Union[Circuit, ParametrizedCircuit]) -> List[str]:
    return c.params if isinstance(c, ParametrizedCircuit) else []
