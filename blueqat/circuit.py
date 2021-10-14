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
This module defines Circuit and the setting for circuit.
"""

import warnings
from functools import partial, update_wrapper
import typing
from typing import cast, Any, Callable, Dict, Optional, Tuple, Type

import numpy as np

from . import gate
from .gateset import get_op_type, register_operation, unregister_operation
from .typing import CircuitOperation

if typing.TYPE_CHECKING:
    from .gate import Operation
    from .backends.backendbase import Backend
    BackendUnion = typing.Union[None, str, Backend]

GLOBAL_MACROS = {}


class Circuit:
    """Store the gate operations and call the backends."""
    def __init__(self, n_qubits=0, ops=None):
        self.ops = ops or []
        self._backends: Dict[str, 'Backend'] = {}
        self.n_qubits = n_qubits

    def __repr__(self):
        return f'Circuit({self.n_qubits}).' + '.'.join(
            str(op) for op in self.ops)

    def __get_backend(self, backend_name: str) -> 'Backend':
        try:
            return self._backends[backend_name]
        except KeyError:
            backend = BACKENDS.get(backend_name)
            if backend is None:
                raise ValueError(f"Backend {backend_name} doesn't exist.")
        self._backends[backend_name] = backend()
        return self._backends[backend_name]

    def __backend_runner_wrapper(self, backend_name):
        backend = self.__get_backend(backend_name)

        def runner(*args, **kwargs):
            return backend.run(self.ops, self.n_qubits, *args, **kwargs)

        return runner

    def __getattr__(self, name: str) -> CircuitOperation[Any]:
        op_type = get_op_type(name)
        if op_type:
            return _GateWrapper(self, op_type)
        if name in GLOBAL_MACROS:
            macro = update_wrapper(partial(GLOBAL_MACROS[name], self), GLOBAL_MACROS[name])
            return cast(CircuitOperation[Any], macro)
        if name.startswith("run_with_"):
            backend_name = name[9:]
            if backend_name in BACKENDS:
                return self.__backend_runner_wrapper(backend_name)
            raise AttributeError(f"Backend '{backend_name}' is not exists.")
        raise AttributeError(
            f"'circuit' object has no attribute or gate '{name}'")

    def __add__(self, other):
        if not isinstance(other, Circuit):
            return NotImplemented
        c = self.copy()
        c += other
        return c

    def __iadd__(self, other):
        if not isinstance(other, Circuit):
            return NotImplemented
        self.ops += other.ops
        self.n_qubits = max(self.n_qubits, other.n_qubits)
        return self

    def copy(self,
             copy_backends: bool = True) -> 'Circuit':
        """Copy the circuit.

        params:
            | copy_backends :bool copy backends if True.
        """
        copied = Circuit(self.n_qubits, self.ops.copy())
        if copy_backends:
            copied._backends = {k: v.copy() for k, v in self._backends.items()}
        return copied

    def dagger(self,
               ignore_measurement: bool = False) -> 'Circuit':
        """Make Hermitian conjugate of the circuit.

        This feature is beta. Interface may be changed.

        ignore_measurement (bool, optional):
            | If True, ignore the measurement in the circuit.
            | Otherwise, if measurement in the circuit, raises ValueError.
        """
        ops = []
        for g in reversed(self.ops):
            try:
                ops.append(g.dagger())
            except ValueError:
                if not ignore_measurement:
                    raise ValueError(
                        'Cannot make the Hermitian conjugate of this circuit because '
                        'the circuit contains measurement.')

        copied = Circuit(self.n_qubits, ops)
        return copied

    def run(self, *args, backend=None, **kwargs):
        """Run the circuit.

        `Circuit` have several backends. When `backend` parameter is specified,
        use specified backend, and otherwise, default backend is used.
        Other parameters are passed to the backend.

        The meaning of parameters are depends on the backend specifications.
        However, following parameters are commonly used.

        Commonly used args (Depends on backend):
            | shots (int, optional): The number of sampling the circuit.
            | returns (str, optional):  The category of returns value.
            |   e.g. "statevector" returns the state vector after run the circuit.
            |         "shots" returns the counter of measured value.
            | token, url (str, optional): The token and URL for cloud resource.

        Returns:
            Depends on backend.

        Raises:
            Depends on backend.
        """
        if backend is None:
            backend = self.__get_backend(DEFAULT_BACKEND_NAME)
        elif isinstance(backend, str):
            backend = self.__get_backend(backend)
        return backend.run(self.ops, self.n_qubits, *args, **kwargs)

    def make_cache(self, backend: Optional['BackendUnion'] = None) -> None:
        """Make a cache to reduce the time of run. Some backends may implemented it.

        This is temporary API. It may changed or deprecated."""
        if backend is None:
            backend = DEFAULT_BACKEND_NAME
        if isinstance(backend, str):
            backend = self.__get_backend(backend)
        return backend.make_cache(self.ops, self.n_qubits)

    def to_qasm(self, *args, **kwargs):
        """Returns the OpenQASM output of this circuit."""
        return self.run_with_qasm_output(*args, **kwargs)

    def to_unitary(self, *args, **kwargs):
        """Returns sympy unitary matrix of this circuit."""
        return self.run_with_sympy_unitary(*args, **kwargs)

    def statevector(self,
                    backend: 'BackendUnion' = None,
                    **kwargs) -> np.ndarray:
        """Run the circuit and get a statevector as a result."""
        if kwargs.get('returns'):
            raise ValueError('Circuit.statevector has no argument `returns`.')
        if backend is None:
            backend = self.__get_backend(DEFAULT_BACKEND_NAME)
        elif isinstance(backend, str):
            backend = self.__get_backend(backend)

        if hasattr(backend, 'statevector'):
            return backend.statevector(self.ops, self.n_qubits,
                                       **kwargs)
        return backend.run(self.ops,
                           self.n_qubits,
                           returns='statevector',
                           **kwargs)

    def shots(self,
              shots: int,
              backend: 'BackendUnion' = None,
              **kwargs) -> typing.Counter[str]:
        """Run the circuit and get shots as a result."""
        if kwargs.get('returns'):
            raise ValueError('Circuit.shots has no argument `returns`.')
        if backend is None:
            backend = self.__get_backend(DEFAULT_BACKEND_NAME)
        elif isinstance(backend, str):
            backend = self.__get_backend(backend)

        if hasattr(backend, 'shots'):
            return backend.shots(self.ops,
                                 self.n_qubits,
                                 shots=shots,
                                 **kwargs)
        return backend.run(self.ops,
                           self.n_qubits,
                           shots=shots,
                           returns='shots',
                           **kwargs)

    def oneshot(self,
                backend: 'BackendUnion' = None,
                **kwargs) -> Tuple[np.ndarray, str]:
        """Run the circuit and get shots as a result."""
        if kwargs.get('returns'):
            raise ValueError('Circuit.oneshot has no argument `returns`.')
        if backend is None:
            backend = self.__get_backend(DEFAULT_BACKEND_NAME)
        elif isinstance(backend, str):
            backend = self.__get_backend(backend)

        if hasattr(backend, 'oneshot'):
            return backend.oneshot(self.ops, self.n_qubits, **kwargs)
        v, cnt = backend.run(self.ops,
                             self.n_qubits,
                             shots=1,
                             returns='statevector_and_shots',
                             **kwargs)
        return v, cnt.most_common()[0][0]


class _GateWrapper(CircuitOperation[Circuit]):
    def __init__(self, circuit: Circuit, op_type: Type['Operation']):
        self.circuit = circuit
        self.target = None
        self.op_type = op_type
        self.params = ()

    def __call__(self, *args) -> '_GateWrapper':
        self.params = args
        return self

    def __getitem__(self, args) -> 'Circuit':
        self.target = args
        self.circuit.ops.append(
            self.op_type.create(self.target, self.params))
        # ad-hoc
        self.circuit.n_qubits = max(
            gate.get_maximum_index(args) + 1, self.circuit.n_qubits)
        return self.circuit

    def __str__(self) -> str:
        if self.params:
            args_str = str(self.params)
        else:
            args_str = ""
        return self.op_type.lowername + args_str + " " + str(self.target)


class BlueqatGlobalSetting:
    """Setting for Blueqat."""
    @staticmethod
    def register_macro(name: str,
                       func: Callable,
                       allow_overwrite: bool = False) -> None:
        """Register new macro to Circuit.

        Args:
            | name (str): The name of macro.
            | func (callable): The function to be called.
            | allow_overwrite (bool, optional): If True, allow to overwrite the existing macro.
            |     Otherwise, raise the ValueError.

        Raises:
            ValueError: The name is duplicated with existing macro, gate or method.
                When `allow_overwrite=True`, this error is not raised.
        """
        if hasattr(Circuit, name):
            if allow_overwrite:
                warnings.warn(f"Circuit has attribute `{name}`.")
            else:
                raise ValueError(f"Circuit has attribute `{name}`.")
        if name.startswith("run_with_"):
            if allow_overwrite:
                warnings.warn(
                    f"Gate name `{name}` may conflict with run of backend.")
            else:
                raise ValueError(
                    f"Gate name `{name}` shall not start with 'run_with_'.")
        if not allow_overwrite:
            if get_op_type(name) is not None:
                raise ValueError(
                    f"Gate '{name}' is already exists in gate set.")
            if name in GLOBAL_MACROS:
                raise ValueError(f"Macro '{name}' is already exists.")
        GLOBAL_MACROS[name] = func

    @staticmethod
    def unregister_macro(name):
        """Unregister a macro.

        Args:
            name (str): The name of the macro to be unregistered.

        Raises:
            ValueError: Specified gate is not registered.
        """
        if name not in GLOBAL_MACROS:
            raise ValueError(f"Macro '{name}' is not registered.")
        del GLOBAL_MACROS[name]

    @staticmethod
    def register_gate(name, gateclass, allow_overwrite=False):
        """Register new gate to gate set.

        Args:
            | name (str): The name of gate.
            | gateclass (type): The type object of gate.
            | allow_overwrite (bool, optional): If True, allow to overwrite the existing gate.
            |     Otherwise, raise the ValueError.

        Raises:
            ValueError: The name is duplicated with existing gate.
                When `allow_overwrite=True`, this error is not raised.
        """
        if hasattr(Circuit, name):
            if allow_overwrite:
                warnings.warn(f"Circuit has attribute `{name}`.")
            else:
                raise ValueError(f"Circuit has attribute `{name}`.")
        if name.startswith("run_with_"):
            if allow_overwrite:
                warnings.warn(
                    f"Gate name `{name}` may conflict with run of backend.")
            else:
                raise ValueError(
                    f"Gate name `{name}` shall not start with 'run_with_'.")
        if not allow_overwrite:
            if get_op_type(name) is not None:
                raise ValueError(
                    f"Gate '{name}' is already exists in gate set.")
            if name in GLOBAL_MACROS:
                raise ValueError(f"Macro '{name}' is already exists.")
        register_operation(name, gateclass)

    @staticmethod
    def unregister_gate(name):
        """Unregister a gate from gate set

        Args:
            name (str): The name of the gate to be unregistered.

        Raises:
            ValueError: Specified gate is not registered.
        """
        if get_op_type(name) is None:
            raise ValueError(f"Gate '{name}' is not registered.")
        unregister_operation(name)

    @staticmethod
    def register_backend(name, backend, allow_overwrite=False):
        """Register new backend.

        Args:
            | name (str): The name of backend.
            | gateclass (type): The type object of backend
            | allow_overwrite (bool, optional): If True, allow to overwrite the existing backend.
                Otherwise, raise the ValueError.

        Raises:
            ValueError: The name is duplicated with existing backend.
                When `allow_overwrite=True`, this error is not raised.
        """
        if hasattr(Circuit, "run_with_" + name):
            if allow_overwrite:
                warnings.warn(f"Circuit has attribute `run_with_{name}`.")
            else:
                raise ValueError(f"Circuit has attribute `run_with_{name}`.")
        if not allow_overwrite:
            if name in BACKENDS:
                raise ValueError(
                    f"Backend '{name}' is already registered as backend.")
        BACKENDS[name] = backend

    @staticmethod
    def unregister_backend(name):
        """Unregister a backend.

        Args:
            name (str): The name of the backend to be unregistered.

        Raises:
            ValueError: Specified backend is not registered.
        """
        if name not in BACKENDS:
            raise ValueError(f"Backend '{name}' is not registered.")
        del BACKENDS[name]

    @staticmethod
    def remove_backend(name):
        """This method is deperecated. Use `unregister_backend` method."""
        warnings.warn(
            "remove_backend is deprecated. `unregister_backend` is recommended.",
            DeprecationWarning)
        BlueqatGlobalSetting.unregister_backend(name)

    @staticmethod
    def set_default_backend(name):
        """Set the default backend to be used by `Circuit`.
        Args:
            name (str): The name of the default backend.

        Raises:
            ValueError: Specified backend is not registered.
        """
        if name not in BACKENDS:
            raise ValueError(f"Backend '{name}' is not registered.")
        global DEFAULT_BACKEND_NAME
        DEFAULT_BACKEND_NAME = name

    @staticmethod
    def get_default_backend_name():
        """Get the default backend name.

        Returns:
            str: The name of default backend.
        """
        return DEFAULT_BACKEND_NAME


from .backends import BACKENDS, DEFAULT_BACKEND_NAME
