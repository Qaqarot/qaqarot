import warnings
import numpy as np
from . import gate
from .backends.numpy_backend import NumPyBackend
from .backends.qasm_output_backend import QasmOutputBackend

GATE_SET = {
    "i": gate.IGate,
    "x": gate.XGate,
    "y": gate.YGate,
    "z": gate.ZGate,
    "h": gate.HGate,
    "t": gate.TGate,
    "s": gate.SGate,
    "cz": gate.CZGate,
    "cx": gate.CXGate,
    "cnot": gate.CXGate,
    "rx": gate.RXGate,
    "ry": gate.RYGate,
    "rz": gate.RZGate,
    "phase": gate.RZGate,
    "u1": gate.RZGate,
    "measure": gate.Measurement,
    "m": gate.Measurement,
}

BACKENDS = {
    "run_with_numpy": NumPyBackend,
    "to_qasm": QasmOutputBackend,
}
DEFAULT_BACKEND_NAME = "run_with_numpy"

class Circuit:
    def __init__(self, n_qubits=0, ops=None):
        self.ops = ops or []
        self.cache = None
        self.cache_idx = -1
        self._backends = {
                "_default": NumPyBackend(),
                "to_qasm": QasmOutputBackend(),
        }
        if n_qubits > 0:
            self.i[n_qubits - 1]

    def __get_backend(self, backend_name):
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
            return backend.run(self.ops, *args, **kwargs)
        return runner

    def __getattr__(self, name):
        if name in GATE_SET:
            return _GateWrapper(self, name, GATE_SET[name])
        if name in BACKENDS:
            return self.__backend_runner_wrapper(name)
        raise AttributeError("'circuit' object has no attribute or gate or backend'" + name + "'")

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
        return self

    def copy(self, copy_cache=True, copy_history=None):
        copied = Circuit(self.n_qubits, self.ops.copy())
        if copy_cache and self.cache is not None:
            copied.cache = self.cache.copy()
            copied.cache_idx = self.cache_idx
        if copy_history is not None:
            warnings.warn("copy_history is deprecated", DeprecationWarning)
        return copied

    def run(self, *args, **kwargs):
        return self.__get_backend(DEFAULT_BACKEND_NAME).run(self.ops, *args, **kwargs)

    def run_with_backend(self, backend, *args, **kwargs):
        if isinstance(backend, str):
            self.__get_backend(backend).run(self.ops, *args, **kwargs)
        else:
            backend.run(self.ops, *args, **kwargs)

    def last_result(self):
        # Too noisy...
        #warnings.warn("last_result will be deprecated", DeprecationWarning)
        try:
            return self._backends["run_with_numpy"].run_history[-1]
        except IndexError:
            raise ValueError("The Circuit has never been to run.")

    @property
    def n_qubits(self):
        return gate.find_n_qubits(self.ops)

    @property
    def run_history(self):
        warnings.warn("run_history will be deprecated", DeprecationWarning)
        try:
            return self._backends["run_with_numpy"].run_history
        except KeyError:
            return []

class _GateWrapper:
    def __init__(self, circuit, name, gate):
        self.circuit = circuit
        self.target = None
        self.name = name
        self.gate = gate
        self.args = ()
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def __getitem__(self, args):
        self.target = args
        self.circuit.ops.append(self.gate(self.target, *self.args, **self.kwargs))
        return self.circuit

    def __str__(self):
        if self.args:
            args_str = str(self.args)
            if self.kwargs:
                args_str = args_str[:-1] + ", kwargs=" + str(self.kwargs) + ")"
        elif self.kwargs:
            args_str = "(kwargs=" + str(self.kwargs) + ")"
        else:
            args_str = ""
        return self.name + args_str + " " + str(self.target)

class BlueqatGlobalSetting:
    @staticmethod
    def register_gate(name, gateclass, allow_overwrite=False):
        """Register new gate to gate set."""
        if not allow_overwrite:
            if name in GATE_SET:
                raise ValueError(f"Gate '{name}' is already exists in gate set.")
            if name in BACKENDS:
                raise ValueError(f"Gate '{name}' is not exists but backend '{name}' is exists.")
        GATE_SET[name] = gateclass

    @staticmethod
    def unregister_gate(name):
        """Unregister a gate from gate set"""
        if name not in GATE_SET:
            raise ValueError(f"Gate '{name}' is not registered.")
        del GATE_SET[name]

    @staticmethod
    def register_backend(name, backend, allow_overwrite=False):
        """Register new backend."""
        if not allow_overwrite:
            if name in BACKENDS:
                raise ValueError(f"Backend '{name}' is already registered as backend.")
            if name in GATE_SET:
                raise ValueError(f"Backend '{name}' is not exists but gate '{name}' is exists.")
        BACKENDS[name] = backend

    @staticmethod
    def remove_backend(name):
        """Unregister a backend."""
        if name not in GATE_SET:
            raise ValueError(f"Backend '{name}' is not registered.")
        del BACKENDS[name]

    @staticmethod
    def set_default_backend(name):
        if name not in BACKENDS:
            raise ValueError(f"Backend '{name}' is not registered.")
        DEFAULT_GATE_SET = name
