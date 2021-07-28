"""
`gate` module implements quantum gate operations.
This module is internally used.
"""

import cmath
import math
from typing import Callable, Iterable, Iterator, List, NoReturn, Tuple, Type, TypeVar, Union

import numpy as np

from .typing import Targets

_Op = TypeVar('_Op', bound='Operation')


class Operation:
    """Abstract quantum circuit operation class."""

    lowername: str = ''
    """Lower name of the operation."""
    @property
    def uppername(self) -> str:
        """Upper name of the operation."""
        return self.lowername.upper()

    def __init__(self, targets: Targets, params=(), **kwargs) -> None:
        if self.lowername == '':
            raise ValueError(
                f"{self.__class__.__name__}.lowername is not defined.")
        self.params = params
        self.kwargs = kwargs
        self.targets = targets

    def target_iter(self, n_qubits: int) -> Iterator[int]:
        """The generator which yields the target qubits."""
        return slicing(self.targets, n_qubits)

    @classmethod
    def create(cls: Type[_Op], targets: Targets, params: tuple) -> _Op:
        """Create an operation."""
        raise NotImplementedError(f"{cls.__name__}.create() is not defined.")

    def fallback(self, n_qubits: int) -> List['Operation']:
        """Returns alternative operations to make equivalent circuit."""
        raise NotImplementedError(
            f"The fallback of {self.__class__.__name__} operation is not defined."
        )

    def dagger(self) -> 'Operation':
        """Returns the Hermitian conjugate of `self`."""
        raise NotImplementedError(
            f"Hermitian conjugate of {self.__class__.__name__} operation is not provided."
        )

    def _str_args(self) -> str:
        """Returns printable string of args."""
        if not self.params:
            return ''
        return '(' + ', '.join(str(param) for param in self.params) + ')'

    def _str_targets(self) -> str:
        """Returns printable string of targets."""
        def _slice_to_str(obj):
            if isinstance(obj, slice):
                start = '' if obj.start is None else str(obj.start.__index__())
                stop = '' if obj.stop is None else str(obj.stop.__index__())
                if obj.step is None:
                    return f'{start}:{stop}'
                step = str(obj.step.__index__())
                return f'{start}:{stop}:{step}'
            return str(obj.__index__())

        if isinstance(self.targets, tuple):
            return f"[{', '.join(_slice_to_str(target) for target in self.targets)}]"
        return f"[{_slice_to_str(self.targets)}]"

    def __str__(self) -> str:
        str_args = self._str_args()
        str_targets = self._str_targets()
        return f'{self.lowername}{str_args}{str_targets}'


class Gate(Operation):
    """Abstract quantum gate class."""
    @property
    def n_qargs(self) -> int:
        """Number of qubit arguments of this gate."""
        raise NotImplementedError()

    def fallback(self, n_qubits: int) -> List['Gate']:
        """Returns alternative gates to make equivalent circuit."""
        raise NotImplementedError(
            f"The fallback of {self.__class__.__name__} gate is not defined.")

    def dagger(self) -> 'Gate':
        """Returns the Hermitian conjugate of `self`."""
        raise NotImplementedError(
            "Hermitian conjugate of this gate is not provided.")

    def matrix(self) -> np.ndarray:
        """Returns the matrix of implementations.

        (Non-abstract) subclasses of Gate must implement this method.
        WARNING: qubit order specifications of multi qubit gate is still not defined.
        """
        raise NotImplementedError()


class OneQubitGate(Gate):
    """Abstract quantum gate class for 1 qubit gate."""

    u_params: Tuple[float, float, float, float]
    """Params for U gate."""
    @property
    def n_qargs(self) -> int:
        return 1

    def _make_fallback_for_target_iter(
            self, n_qubits: int,
            fallback: Callable[[int], List['Gate']]) -> List['Gate']:
        gates = []
        for t in self.target_iter(n_qubits):
            gates += fallback(t)
        return gates

    def fallback(self, n_qubits: int) -> List['Gate']:
        if self.u_params:
            return [UGate(self.targets, *self.u_params, **self.kwargs)]
        try:
            mat = self.matrix()
        except NotImplementedError:
            return super().fallback(n_qubits)
        return [Mat1Gate(self.targets, mat, **self.kwargs)]


class TwoQubitGate(Gate):
    """Abstract quantum gate class for 2 qubits gate."""

    cu_params: Tuple[float, float, float, float]

    @property
    def n_qargs(self):
        return 2

    def control_target_iter(self, n_qubits: int) -> Iterator[Tuple[int, int]]:
        """The generator which yields the tuples of (control, target) qubits."""
        return qubit_pairs(self.targets, n_qubits)

    def _make_fallback_for_control_target_iter(
            self, n_qubits: int,
            fallback: Callable[[int, int], List['Gate']]) -> List['Gate']:
        gates = []
        for c, t in self.control_target_iter(n_qubits):
            gates += fallback(c, t)
        return gates

    def fallback(self, n_qubits: int) -> List['Gate']:
        if self.cu_params:
            return [CUGate(self.targets, *self.cu_params, **self.kwargs)]
        return super().fallback(n_qubits)


class HGate(OneQubitGate):
    """Hadamard gate"""
    lowername = "h"
    u_params = (-math.pi / 2.0, math.pi, 0.0, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'HGate':
        return cls(targets, *params)

    def dagger(self):
        return self

    def matrix(self):
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


class IGate(OneQubitGate):
    """Identity gate"""
    lowername = "i"
    u_params = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'IGate':
        return cls(targets, *params)

    def fallback(self, _):
        return []

    def dagger(self):
        return self

    def matrix(self):
        return np.eye(2)


class Mat1Gate(OneQubitGate):
    """Arbitrary 2x2 matrix gate

    `mat` is expected a 2x2 unitary matrix, but not checked.
    (If unexpected matrix is given, backend may raises error or returns weird result)
    """
    lowername = "mat1"
    u_params = None

    def __init__(self, targets, mat: np.ndarray, **kwargs):
        super().__init__(targets, (mat, ), **kwargs)
        self.mat = mat

    @classmethod
    def create(cls, targets, params) -> 'Mat1Gate':
        return cls(targets, params[0])

    def dagger(self):
        return Mat1Gate(self.targets, self.mat.T.conjugate(), **self.kwargs)

    def matrix(self):
        return self.mat

    def fallback(self, n_qubits: int) -> List['Gate']:
        raise NotImplementedError(
            'Fallback implementation for Mat1Gate is not available.')


class PhaseGate(OneQubitGate):
    """Rotate-Z gate but global phase is different.

    Global phase doesn't makes any difference of measured result.
    You may use RZ gate or U1 gate instead, but distinguishing these gates
    may better for debugging or future improvement.

    furthermore, phase gate may efficient for simulating.
    (It depends on backend implementation.
    But matrix of phase gate is simpler than RZ gate or U1 gate.)"""
    lowername = "phase"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta, ), **kwargs)
        self.theta = theta
        self.u_params = (0.0, self.theta, 0.0, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'PhaseGate':
        return cls(targets, params[0])

    def dagger(self):
        return PhaseGate(self.targets, -self.theta, **self.kwargs)

    def matrix(self):
        return np.array([[1, 0], [0, cmath.exp(1j * self.theta)]],
                        dtype=complex)


class RXGate(OneQubitGate):
    """Rotate-X gate"""
    lowername = "rx"

    def __init__(self, targets, theta: float, **kwargs):
        super().__init__(targets, (theta, ), **kwargs)
        self.theta = theta
        self.u_params = (theta, -math.pi / 2.0, math.pi / 2.0, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'RXGate':
        return cls(targets, params[0])

    def dagger(self):
        return RXGate(self.targets, -self.theta, **self.kwargs)

    def matrix(self):
        t = self.theta * 0.5
        a = math.cos(t)
        b = -1j * math.sin(t)
        return np.array([[a, b], [b, a]], dtype=complex)


class RYGate(OneQubitGate):
    """Rotate-Y gate"""
    lowername = "ry"

    def __init__(self, targets, theta: float, **kwargs):
        super().__init__(targets, (theta, ), **kwargs)
        self.theta = theta
        self.u_params = (theta, 0.0, 0.0, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'RYGate':
        return cls(targets, params[0])

    def dagger(self):
        return RYGate(self.targets, -self.theta, **self.kwargs)

    def matrix(self):
        t = self.theta * 0.5
        a = math.cos(t)
        b = math.sin(t)
        return np.array([[a, -b], [b, a]], dtype=complex)


class RZGate(OneQubitGate):
    """Rotate-Z gate"""
    lowername = "rz"

    def __init__(self, targets, theta: float, **kwargs):
        super().__init__(targets, (theta, ), **kwargs)
        self.theta = theta
        self.u_params = (0.0, 0.0, theta, -0.5 * theta)

    @classmethod
    def create(cls, targets, params) -> 'RZGate':
        return cls(targets, params[0])

    def dagger(self):
        return RZGate(self.targets, -self.theta, **self.kwargs)

    def matrix(self):
        a = cmath.exp(0.5j * self.theta)
        return np.array([[a.conjugate(), 0], [0, a]], dtype=complex)


class SGate(OneQubitGate):
    """S gate"""
    lowername = "s"
    u_params = (0.0, 0.0, math.pi / 2, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'SGate':
        return cls(targets, params)

    def dagger(self):
        return SDagGate(self.targets, self.params, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(
            n_qubits, lambda t: [PhaseGate(t, math.pi / 2)])

    def matrix(self):
        return np.array([[1, 0], [0, 1j]])


class SDagGate(OneQubitGate):
    """Dagger of S gate"""
    lowername = "sdg"
    u_params = (0.0, 0.0, -math.pi / 2, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'SDagGate':
        return cls(targets, params)

    def dagger(self):
        return SGate(self.targets, self.params, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(
            n_qubits, lambda t: [PhaseGate(t, -math.pi / 2)])

    def matrix(self):
        return np.array([[1, 0], [0, -1j]])


class SXGate(OneQubitGate):
    """sqrt(X) gate

    This is equivalent as RX(π/2) * (1 + i) / √2."""
    lowername = "sx"
    u_params = (math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0, math.pi / 4.0)

    @classmethod
    def create(cls, targets, params) -> 'SXGate':
        return cls(targets, params)

    def dagger(self):
        return SXDagGate(self.targets, self.params, **self.kwargs)

    def matrix(self):
        return np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])


class SXDagGate(OneQubitGate):
    """sqrt(X)† gate"""
    lowername = "sxdg"
    u_params = (-math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0, -math.pi / 4.0)

    def dagger(self):
        return SXGate(self.targets, self.params, **self.kwargs)

    def matrix(self):
        return np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]])


class TGate(OneQubitGate):
    """T ($\\pi/8$) gate"""
    lowername = "t"
    u_params = (0, 0, math.pi / 4.0, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'TGate':
        return cls(targets, params)

    def dagger(self):
        return TDagGate(self.targets, self.params, **self.kwargs)

    def fallback(self, n_qubits):
        return [PhaseGate(self.targets, math.pi / 4, **self.kwargs)]

    def matrix(self):
        return np.array([[1, 0], [0, math.exp(math.pi * 0.25)]])


class TDagGate(OneQubitGate):
    """Dagger of T ($\\pi/8$) gate"""
    lowername = "tdg"
    u_params = (0, 0, -math.pi / 4.0, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'TDagGate':
        return cls(targets, params)

    def dagger(self):
        return TGate(self.targets, self.params, **self.kwargs)

    def fallback(self, n_qubits):
        return [PhaseGate(self.targets, -math.pi / 4, **self.kwargs)]

    def matrix(self):
        return np.array([[1, 0], [0, math.exp(math.pi * -0.25)]])


class ToffoliGate(Gate):
    """Toffoli (CCX) gate"""
    lowername = "ccx"

    @property
    def n_qargs(self):
        return 3

    @classmethod
    def create(cls, targets, params) -> 'ToffoliGate':
        return cls(targets, params)

    def dagger(self):
        return self

    def fallback(self, n_qubits):
        c1, c2, t = self.targets
        return [
            HGate(t),
            CCZGate((c1, c2, t)),
            HGate(t),
        ]

    def matrix(self):
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
                        dtype=complex)


class UGate(OneQubitGate):
    """Arbitrary 1 qubit unitary gate including global phase.

    U(θ, φ, λ, γ = 0.0) = e^iγ * array([
        [cos(θ/2), -e^iλ sin(θ/2)],
        [e^iφ sin(θ/2), e^i(φ+λ) cos(θ/2)]])

    Note: If SU matrix is required, U(θ, φ, λ, (φ + λ) / 2) works fine.
    """
    lowername = "u"

    def __init__(self,
                 targets,
                 theta: float,
                 phi: float,
                 lam: float,
                 gamma: float = 0.0,
                 **kwargs):
        super().__init__(targets, (theta, phi, lam, gamma), **kwargs)
        self.theta = theta
        self.phi = phi
        self.lam = lam
        self.gamma = gamma
        self.u_params = (theta, phi, lam, gamma)

    @classmethod
    def create(cls, targets, params) -> 'UGate':
        return cls(targets, *params)

    def dagger(self):
        return UGate(self.targets, -self.theta, -self.lam, -self.phi,
                     -self.gamma, **self.kwargs)

    def fallback(self, n_qubits: int) -> List['Gate']:
        raise NotImplementedError(
            'Fallback implementation for UGate is not available.')

    def matrix(self):
        t, p, l, g = self.params
        gphase = cmath.exp(1j * g)
        cos_t = math.cos(0.5 * t)
        sin_t = math.sin(0.5 * t)
        return np.array(
            [[cos_t, -cmath.exp(1j * l) * sin_t],
             [cmath.exp(1j * p) * sin_t,
              cmath.exp(1j * (p + l)) * cos_t]],
            dtype=complex) * gphase


class XGate(OneQubitGate):
    """Pauli's X gate"""
    lowername = "x"
    u_params = (math.pi, 0.0, math.pi, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'XGate':
        return cls(targets, params)

    def dagger(self):
        return self

    def matrix(self):
        return np.array([[0, 1], [1, 0]], dtype=complex)


class YGate(OneQubitGate):
    """Pauli's Y gate"""
    lowername = "y"
    u_params = (math.pi, math.pi / 2, math.pi / 2, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'YGate':
        return cls(targets, params)

    def dagger(self):
        return self

    def matrix(self):
        return np.array([[0, -1j], [1j, 0]])


class ZGate(OneQubitGate):
    """Pauli's Z gate"""
    lowername = "z"
    u_params = (0.0, 0.0, math.pi, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'ZGate':
        return cls(targets, params)

    def dagger(self):
        return self

    def matrix(self):
        return np.array([[1, 0], [0, -1]], dtype=complex)


class CCZGate(Gate):
    """2-Controlled Z gate"""
    lowername = "ccz"

    @property
    def n_qargs(self):
        return 3

    @classmethod
    def create(cls, targets, params) -> 'CCZGate':
        return cls(targets, params)

    def fallback(self, n_qubits):
        c1, c2, t = self.targets
        return [
            CXGate((c2, t)),
            TDagGate(t),
            CXGate((c1, t)),
            TGate(t),
            CXGate((c2, t)),
            TDagGate(t),
            CXGate((c1, t)),
            TGate(c2),
            TGate(t),
            CXGate((c1, c2)),
            TGate(c1),
            TDagGate(c2),
            CXGate((c1, c2)),
        ]

    def dagger(self):
        return self

    def matrix(self):
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, -1]],
                        dtype=complex)


class CHGate(TwoQubitGate):
    """Controlled-H gate"""
    lowername = "ch"
    cu_params = (-math.pi / 2.0, math.pi, 0.0, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'CHGate':
        return cls(targets, params)

    def dagger(self):
        return self

    def matrix(self):
        a = 1.0 / math.sqrt(2)
        return np.array(
            [[1, 0, 0, 0], [0, a, 0, a], [0, 0, 1, 0], [0, a, 0, -a]],
            dtype=complex)


class CPhaseGate(TwoQubitGate):
    """Rotate-Z gate but phase is different."""
    lowername = "cphase"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta, ), **kwargs)
        self.theta = theta
        self.cu_params = (0.0, self.theta, 0.0, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'CPhaseGate':
        return cls(targets, params[0])

    def dagger(self):
        return CPhaseGate(self.targets, -self.theta, **self.kwargs)

    def matrix(self):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                         [0, 0, 0, cmath.exp(1j * self.theta)]],
                        dtype=complex)


class CRXGate(TwoQubitGate):
    """Rotate-X gate"""
    lowername = "crx"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta, ), **kwargs)
        self.theta = theta
        self.cu_params = (theta, -math.pi / 2.0, math.pi / 2.0, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'CRXGate':
        return cls(targets, params[0])

    def dagger(self):
        return CRXGate(self.targets, -self.theta, **self.kwargs)

    def matrix(self):
        t = self.theta * 0.5
        a = math.cos(t)
        b = -1j * math.sin(t)
        return np.array(
            [[1, 0, 0, 0], [0, a, 0, b], [0, 0, 1, 0], [0, b, 0, a]],
            dtype=complex)


class CRYGate(TwoQubitGate):
    """Rotate-Y gate"""
    lowername = "cry"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta, ), **kwargs)
        self.theta = theta
        self.cu_params = (theta, 0.0, 0.0, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'CRYGate':
        return cls(targets, params[0])

    def dagger(self):
        return CRYGate(self.targets, -self.theta, **self.kwargs)

    def matrix(self):
        t = self.theta * 0.5
        a = math.cos(t)
        b = math.sin(t)
        return np.array(
            [[1, 0, 0, 0], [0, a, 0, -b], [0, 0, 1, 0], [0, b, 0, a]],
            dtype=complex)


class CRZGate(TwoQubitGate):
    """Rotate-Z gate"""
    lowername = "crz"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta, ), **kwargs)
        self.theta = theta
        self.cu_params = (0.0, 0.0, theta, -0.5 * theta)

    @classmethod
    def create(cls, targets, params) -> 'CRZGate':
        return cls(targets, params[0])

    def dagger(self):
        return CRZGate(self.targets, -self.theta, **self.kwargs)

    def matrix(self):
        a = cmath.exp(0.5j * self.theta)
        return np.array([[1, 0, 0, 0], [0, a.conjugate(), 0, 0], [0, 0, 1, 0],
                         [0, 0, 0, a]],
                        dtype=complex)


class CSwapGate(Gate):
    """Controlled SWAP gate"""
    lowername = "cswap"

    @property
    def n_qargs(self):
        return 3

    @classmethod
    def create(cls, targets, params) -> 'CSwapGate':
        return cls(targets, params)

    def dagger(self):
        return self

    def fallback(self, n_qubits):
        c, t1, t2 = self.targets
        return [CXGate((t2, t1)), ToffoliGate((c, t1, t2)), CXGate((t2, t1))]

    def matrix(self):
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
                        dtype=complex)


class CUGate(TwoQubitGate):
    """Controlled-U gate."""
    lowername = "cu"

    def __init__(self,
                 targets,
                 theta: float,
                 phi: float,
                 lam: float,
                 gamma: float = 0.0,
                 **kwargs):
        super().__init__(targets, (theta, phi, lam, gamma), **kwargs)
        self.theta = theta
        self.phi = phi
        self.lam = lam
        self.gamma = gamma

    @classmethod
    def create(cls, targets, params) -> 'CUGate':
        return cls(targets, *params)

    def dagger(self):
        return CUGate(self.targets, -self.theta, -self.lam, -self.phi,
                      -self.gamma, **self.kwargs)

    def fallback(self, n_qubits: int) -> List['Gate']:
        theta, phi, lam, gamma = self.params
        return self._make_fallback_for_control_target_iter(
            n_qubits, lambda c, t: [
                PhaseGate((c, ), gamma + 0.5 * (lam + phi)),
                PhaseGate((t, ), 0.5 * (lam - phi)),
                CXGate((c, t)),
                UGate((t, ), -0.5 * theta, 0.0, -0.5 * (phi + lam)),
                CXGate((c, t)),
                UGate((t, ), 0.5 * theta, phi, 0.0)
            ])

    def matrix(self):
        t, p, l, g = self.params
        cos_t = math.cos(0.5 * t)
        sin_t = math.sin(0.5 * t)
        return np.array([[
            1, 0, 0, 0
        ], [0,
            cmath.exp(1j * g) * cos_t, 0, -cmath.exp(1j * (g + l)) * sin_t],
                         [0, 0, 1, 0],
                         [
                             0,
                             cmath.exp(1j * (g + p)) * sin_t, 0,
                             cmath.exp(1j * (g + p + l)) * cos_t
                         ]],
                        dtype=complex)


class CXGate(TwoQubitGate):
    """Controlled-X (CNOT) gate"""
    lowername = "cx"
    cu_params = (math.pi, 0.0, math.pi, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'CXGate':
        return cls(targets, params)

    def dagger(self):
        return self

    def matrix(self):
        return np.array(
            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
            dtype=complex)


class CYGate(TwoQubitGate):
    """Controlled-Y gate"""
    lowername = "cy"
    cu_params = (math.pi, math.pi / 2, math.pi / 2, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'CYGate':
        return cls(targets, params)

    def dagger(self):
        return self

    def matrix(self):
        return np.array(
            [[1, 0, 0, 0], [0, 0, -1j, 0], [0, 0, 1, 0], [0, 1j, 0, 0]],
            dtype=complex)


class CZGate(TwoQubitGate):
    """Controlled-Z gate"""
    lowername = "cz"
    cu_params = (0.0, 0.0, math.pi, 0.0)

    @classmethod
    def create(cls, targets, params) -> 'CZGate':
        return cls(targets, params)

    def dagger(self):
        return self

    def matrix(self):
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dtype=complex)


class RXXGate(TwoQubitGate):
    """Rotate-XX gate"""
    lowername = "rxx"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta, ), **kwargs)
        self.theta = theta

    @classmethod
    def create(cls, targets, params) -> 'RXXGate':
        return cls(targets, *params)

    def dagger(self):
        return RXXGate(self.targets, -self.theta, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits, lambda c, t: [
                HGate(c),
                HGate(t),
                RZZGate((c, t), self.theta),
                HGate(c),
                HGate(t)
            ])

    def matrix(self):
        a = math.cos(self.theta * 0.5)
        b = -1j * math.sin(self.theta * 0.5)
        return np.array(
            [[a, 0, 0, b], [0, a, b, 0], [0, b, a, 0], [a, 0, 0, b]],
            dtype=complex)


class RYYGate(TwoQubitGate):
    """Rotate-YY gate"""
    lowername = "ryy"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta, ), **kwargs)
        self.theta = theta

    @classmethod
    def create(cls, targets, params) -> 'RYYGate':
        return cls(targets, *params)

    def dagger(self):
        return RYYGate(self.targets, -self.theta, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits, lambda c, t: [
                RXGate(c, -math.pi * 0.5),
                RXGate(t, -math.pi * 0.5),
                RZZGate((c, t), self.theta),
                RXGate(c, math.pi * 0.5),
                RXGate(t, math.pi * 0.5)
            ])

    def matrix(self):
        a = math.cos(self.theta * 0.5)
        b = 1j * math.sin(self.theta * 0.5)
        return np.array(
            [[a, 0, 0, b], [0, a, -b, 0], [0, -b, a, 0], [a, 0, 0, b]],
            dtype=complex)


class RZZGate(TwoQubitGate):
    """Rotate-ZZ gate"""
    lowername = "rzz"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta, ), **kwargs)
        self.theta = theta

    @classmethod
    def create(cls, targets, params) -> 'RZZGate':
        return cls(targets, *params)

    def dagger(self):
        return RZZGate(self.targets, -self.theta, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits, lambda c, t:
            [CXGate(
                (c, t)), RZGate(t, self.theta),
             CXGate((c, t))])

    def matrix(self):
        a = cmath.exp(0.5j * self.theta)
        return np.array([[a.conjugate(), 0, 0, 0], [0, a, 0, 0], [0, 0, a, 0],
                         [0, 0, 0, a.conjugate()]],
                        dtype=complex)


class SwapGate(TwoQubitGate):
    """Swap gate"""
    lowername = "swap"

    def dagger(self):
        return self

    @classmethod
    def create(cls, targets, params) -> 'SwapGate':
        return cls(targets, params)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [CXGate(
                (c, t)), CXGate(
                    (t, c)), CXGate((c, t))])

    def matrix(self):
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                         [0, 0, 0, 1]])


class ZZGate(TwoQubitGate):
    """ZZ gate

    This gate is a basis two-qubit gate for some kinds of trapped-ion based machines.
    It is equivalent with RZZ(pi/2) except global phase.
    """
    lowername = "zz"

    def __init__(self, targets, **kwargs):
        super().__init__(targets, (), **kwargs)

    @classmethod
    def create(cls, targets, params) -> 'ZZGate':
        return cls(targets)

    def dagger(self):
        return self

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits, lambda c, t: [
                RYGate(t, math.pi * 0.5),
                CXGate((c, t)),
                RZGate(c, math.pi * 0.5),
                UGate(t, -math.pi * 0.5, math.pi * 0.5, 0.0, math.pi * 0.25),
            ])

    def matrix(self):
        return np.diag([1, 1j, 1j, 1])


class Measurement(Operation):
    """Measurement operation"""
    lowername = "measure"

    @classmethod
    def create(cls, targets, params) -> 'Measurement':
        return cls(targets)

    # Ad-hoc copy and paste programming.
    def target_iter(self, n_qubits):
        """The generator which yields the target qubits."""
        return slicing(self.targets, n_qubits)


class Reset(Operation):
    """Reset operation"""
    lowername = "reset"

    @classmethod
    def create(cls, targets, params) -> 'Reset':
        return cls(targets)

    # Ad-hoc copy and paste programming.
    def target_iter(self, n_qubits):
        """The generator which yields the target qubits."""
        return slicing(self.targets, n_qubits)


class DeprecatedOperation:
    """Inform deprecated operation"""
    def __init__(self, name: str, alternative: str) -> None:
        self.name = name
        self.alt = alternative

    def __call__(self, *_args, **_kwargs) -> NoReturn:
        raise ValueError(
            f'{self.name} operation is deprecated. Use insteads {self.alt}.')


def slicing_singlevalue(arg: Union[slice, int], length: int) -> Iterator[int]:
    """Internally used."""
    if isinstance(arg, slice):
        start, stop, step = arg.indices(length)
        i = start
        if step > 0:
            while i < stop:
                yield i
                i += step
        else:
            while i > stop:
                yield i
                i += step
    else:
        try:
            i = arg.__index__()
        except AttributeError:
            raise TypeError("indices must be integers or slices, not " +
                            arg.__class__.__name__) from None
        if i < 0:
            i += length
        yield i


def slicing(args: Targets, length: int) -> Iterator[int]:
    """Internally used."""
    if isinstance(args, tuple):
        for arg in args:
            yield from slicing_singlevalue(arg, length)
    else:
        yield from slicing_singlevalue(args, length)


def qubit_pairs(args: Tuple[Targets, Targets], length: int) -> Iterator[Tuple[int, int]]:
    """Internally used."""
    if not isinstance(args, tuple):
        raise ValueError("Control and target qubits pair(s) are required.")
    if len(args) != 2:
        raise ValueError("Control and target qubits pair(s) are required.")
    controls = list(slicing(args[0], length))
    targets = list(slicing(args[1], length))
    if len(controls) != len(targets):
        raise ValueError(
            "The number of control qubits and target qubits are must be same.")
    for c, z in zip(controls, targets):
        if c == z:
            raise ValueError(
                "Control qubit and target qubit are must be different.")
    return zip(controls, targets)


def get_maximum_index(indices: Targets) -> int:
    """Internally used."""
    def _maximum_idx_single(idx: int):
        if isinstance(idx, slice):
            start = -1
            stop = 0
            if idx.start is not None:
                start = idx.start.__index__()
            if idx.stop is not None:
                stop = idx.stop.__index__()
            return max(start, stop - 1)
        return idx.__index__()

    if isinstance(indices, tuple):
        return max((_maximum_idx_single(i) for i in indices), default=-1)
    return _maximum_idx_single(indices)


def find_n_qubits(gates: Iterable[Operation]) -> int:
    """Find n_qubits from gates"""
    return max((get_maximum_index(g.targets) for g in gates), default=-1) + 1
