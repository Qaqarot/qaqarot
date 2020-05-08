"""
`gate` module implements quantum gate operations.
This module is internally used.
"""

import math
from abc import ABC
from typing import List


class Gate(ABC):
    """Abstract quantum gate class."""

    """Lower name of the gate."""
    lowername: str = None

    @property
    def uppername(self) -> str:
        """Upper name of the gate."""
        return self.lowername.upper()

    def __init__(self, targets, params=(), **kwargs):
        if self.lowername is None:
            raise ValueError(f"{self.__class__.__name__}.lowername is not defined.")
        self.params = params
        self.kwargs = kwargs
        self.targets = targets

    def fallback(self, n_qubits: int) -> List['Gate']:
        """Returns alternative gates to make equivalent circuit."""
        raise NotImplementedError(f"The fallback of {self.__class__.__name__} gate is not defined.")

    def dagger(self) -> 'Gate':
        """Returns the Hermitian conjugate of `self`."""
        raise NotImplementedError(f"Hermitian conjugate of this gate is not provided.")


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
                else:
                    step = str(obj.step.__index__())
                    return f'{start}:{stop}:{step}'
            else:
                return str(obj.__index__())

        if isinstance(self.targets, tuple):
            return f"[{', '.join(_slice_to_str(target) for target in self.targets)}]"
        else:
            return f"[{_slice_to_str(self.targets)}]"

    def __str__(self) -> str:
        str_args = self._str_args()
        str_targets = self._str_targets()
        return f'{self.lowername}{str_args}{str_targets}'


class OneQubitGate(Gate):
    """Abstract quantum gate class for 1 qubit gate."""

    def target_iter(self, n_qubits):
        """The generator which yields the target qubits."""
        return slicing(self.targets, n_qubits)

    def _make_fallback_for_target_iter(self, n_qubits, fallback):
        gates = []
        for t in self.target_iter(n_qubits):
            gates += fallback(t)
        return gates


class TwoQubitGate(Gate):
    """Abstract quantum gate class for 2 qubits gate."""

    def control_target_iter(self, n_qubits):
        """The generator which yields the tuples of (control, target) qubits."""
        return qubit_pairs(self.targets, n_qubits)

    def _make_fallback_for_control_target_iter(self, n_qubits, fallback):
        gates = []
        for c, t in self.control_target_iter(n_qubits):
            gates += fallback(c, t)
        return gates


class IGate(OneQubitGate):
    """Identity gate"""
    lowername = "i"

    def fallback(self, n_qubits):
        return []

    def dagger(self):
        return self


class XGate(OneQubitGate):
    """Pauli's X gate"""
    lowername = "x"

    def dagger(self):
        return self


class YGate(OneQubitGate):
    """Pauli's Y gate"""
    lowername = "y"

    def dagger(self):
        return self


class ZGate(OneQubitGate):
    """Pauli's Z gate"""
    lowername = "z"

    def dagger(self):
        return self


class HGate(OneQubitGate):
    """Hadamard gate"""
    lowername = "h"

    def dagger(self):
        return self


class RXGate(OneQubitGate):
    """Rotate-X gate"""
    lowername = "rx"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def dagger(self):
        return RXGate(self.targets, -self.theta, **self.kwargs)


class RYGate(OneQubitGate):
    """Rotate-Y gate"""
    lowername = "ry"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def dagger(self):
        return RYGate(self.targets, -self.theta, **self.kwargs)


class RZGate(OneQubitGate):
    """Rotate-Z gate"""
    lowername = "rz"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def dagger(self):
        return RZGate(self.targets, -self.theta, **self.kwargs)


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
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def dagger(self):
        return PhaseGate(self.targets, -self.theta, **self.kwargs)

    def fallback(self, n_qubits):
        # If phase gate is not implemented in the backend, global phase is ignored.
        return self._make_fallback_for_target_iter(n_qubits, lambda t: [RZGate(t, self.theta)])


class TGate(OneQubitGate):
    """T ($\\pi/8$) gate"""
    lowername = "t"

    def dagger(self):
        return TDagGate(self.targets, self.params, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(n_qubits, lambda t: [PhaseGate(t, math.pi / 4)])


class TDagGate(OneQubitGate):
    """Dagger of T ($\\pi/8$) gate"""
    lowername = "tdg"

    def dagger(self):
        return TGate(self.targets, self.params, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(n_qubits, lambda t: [PhaseGate(t, -math.pi / 4)])


class SGate(OneQubitGate):
    """S gate"""
    lowername = "s"

    def dagger(self):
        return SDagGate(self.targets, self.params, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(n_qubits, lambda t: [PhaseGate(t, math.pi / 2)])


class SDagGate(OneQubitGate):
    """Dagger of S gate"""
    lowername = "sdg"

    def dagger(self):
        return SGate(self.targets, self.params, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(n_qubits, lambda t: [PhaseGate(t, -math.pi / 2)])


class U1Gate(OneQubitGate):
    """U1 gate

    U1 gate is as same as RZ gate and CU1 gate is as same as CPhase gate.
    It is because for compatibility with IBM's implementations.

    You should probably use RZ/CRZ gates or Phase/CPhase gates instead of U1/CU1 gates.
    """
    lowername = "u1"

    def __init__(self, targets, lambd, **kwargs):
        super().__init__(targets, (lambd,), **kwargs)
        self.lambd = lambd

    def dagger(self):
        return U1Gate(self.targets, -self.lambd, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(
            n_qubits, lambda t: [U3Gate(t, 0.0, 0.0, self.lambd)])


class U2Gate(OneQubitGate):
    """U2 gate"""
    lowername = "u2"

    def __init__(self, targets, phi, lambd, **kwargs):
        super().__init__(targets, (phi, lambd), **kwargs)
        self.phi = phi
        self.lambd = lambd

    def dagger(self):
        return U3Gate(self.targets, -math.pi / 2, -self.lambd, -self.phi, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(
            n_qubits, lambda t: [U3Gate(t, math.pi / 2, self.phi, self.lambd)])


class U3Gate(OneQubitGate):
    """U3 gate"""
    lowername = "u3"

    def __init__(self, targets, theta, phi, lambd, **kwargs):
        super().__init__(targets, (theta, phi, lambd), **kwargs)
        self.theta = theta
        self.phi = phi
        self.lambd = lambd

    def dagger(self):
        return U3Gate(self.targets, -self.theta, -self.lambd, -self.phi, **self.kwargs)


class CXGate(TwoQubitGate):
    """Controlled-X (CNOT) gate"""
    lowername = "cx"

    def dagger(self):
        return self


class CZGate(TwoQubitGate):
    """Controlled-Z gate"""
    lowername = "cz"

    def dagger(self):
        return self


class CYGate(TwoQubitGate):
    """Controlled-Y gate"""
    lowername = "cy"

    def dagger(self):
        return self

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [SDagGate(t),
                          CXGate((c, t)),
                          SGate(t)])


class CHGate(TwoQubitGate):
    """Controlled-H gate"""
    lowername = "ch"

    def dagger(self):
        return self

    def fallback(self, n_qubits):
        # Ignores global phase
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [RYGate(t, math.pi / 4),
                          CXGate((c, t)),
                          RYGate(t, -math.pi / 4)])


class CRXGate(TwoQubitGate):
    """Rotate-X gate"""
    lowername = "crx"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def dagger(self):
        return CRXGate(self.targets, -self.theta, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [RXGate(t, self.theta / 2),
                          CZGate((c, t)),
                          RXGate(t, -self.theta / 2),
                          CZGate((c, t))])


class CRYGate(TwoQubitGate):
    """Rotate-Y gate"""
    lowername = "cry"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def dagger(self):
        return CRYGate(self.targets, -self.theta, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [RYGate(t, self.theta / 2),
                          CXGate((c, t)),
                          RYGate(t, -self.theta / 2),
                          CXGate((c, t))])


class CRZGate(TwoQubitGate):
    """Rotate-Z gate"""
    lowername = "crz"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def dagger(self):
        return CRZGate(self.targets, -self.theta, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [RZGate(t, self.theta / 2),
                          CXGate((c, t)),
                          RZGate(t, -self.theta / 2),
                          CXGate((c, t))])


class CPhaseGate(TwoQubitGate):
    """Rotate-Z gate but phase is different."""
    lowername = "cphase"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def dagger(self):
        return CPhaseGate(self.targets, -self.theta, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [CRZGate((c, t), self.theta),
                          PhaseGate(c, self.theta / 2)])


class RXXGate(TwoQubitGate):
    """Rotate-XX gate"""
    lowername = "rxx"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def dagger(self):
        return RXXGate(self.targets, -self.theta, **self.kwargs)

    def fallback(self, n_qubits):
        # TODO: test
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [HGate(c),
                          HGate(t),
                          RZZGate((c, t), self.theta),
                          HGate(c),
                          HGate(t)])


class RYYGate(TwoQubitGate):
    """Rotate-YY gate"""
    lowername = "ryy"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def dagger(self):
        return RYYGate(self.targets, -self.theta, **self.kwargs)

    def fallback(self, n_qubits):
        # TODO: test
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [RXGate(c, -math.pi * 0.5),
                          RXGate(t, -math.pi * 0.5),
                          RZZGate((c, t), self.theta),
                          RXGate(c, math.pi * 0.5),
                          RXGate(t, math.pi * 0.5)])


class RZZGate(TwoQubitGate):
    """Rotate-ZZ gate"""
    lowername = "rzz"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def dagger(self):
        return RZZGate(self.targets, -self.theta, **self.kwargs)

    def fallback(self, n_qubits):
        # TODO: test
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [CXGate((c, t)),
                          RZGate(t, self.theta),
                          CXGate((c, t))])


class SwapGate(TwoQubitGate):
    """Swap gate"""
    lowername = "swap"

    def dagger(self):
        return self

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [CXGate((c, t)), CXGate((t, c)), CXGate((c, t))])


class CU1Gate(TwoQubitGate):
    """Controlled U1 gate

    U1 gate is as same as RZ gate and CU1 gate is as same as CPhase gate.
    It is because for compatibility with IBM's implementations.

    You should probably use RZ/CRZ gates or Phase/CPhase gates instead of U1/CU1 gates.
    """
    lowername = "cu1"

    def __init__(self, targets, lambd, **kwargs):
        super().__init__(targets, (lambd,), **kwargs)
        self.lambd = lambd

    def dagger(self):
        return CU1Gate(self.targets, -self.lambd, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits, lambda c, t: [
                U1Gate(c, self.lambd / 2),
                CXGate((c, t)),
                U1Gate(t, -self.lambd / 2),
                CXGate((c, t)),
                U1Gate(t, self.lambd / 2),
            ])


class CU2Gate(TwoQubitGate):
    """Controlled U2 gate"""
    lowername = "cu2"

    def __init__(self, targets, phi, lambd, **kwargs):
        super().__init__(targets, (phi, lambd), **kwargs)
        self.phi = phi
        self.lambd = lambd

    def dagger(self):
        return CU3Gate(self.targets, -math.pi / 2, -self.lambd, -self.phi, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits, lambda c, t: [CU3Gate((c, t), math.pi / 2, self.phi, self.lambd)])


class CU3Gate(TwoQubitGate):
    """Controlled U3 gate"""
    lowername = "cu3"

    def __init__(self, targets, theta, phi, lambd, **kwargs):
        super().__init__(targets, (theta, phi, lambd), **kwargs)
        self.theta = theta
        self.phi = phi
        self.lambd = lambd

    def dagger(self):
        return CU3Gate(self.targets, -self.theta, -self.lambd, -self.phi, **self.kwargs)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits, lambda c, t: [
                U1Gate(t, (self.lambd - self.phi) / 2),
                CXGate((c, t)),
                U3Gate(t, -self.theta / 2, 0, -(self.phi + self.lambd) / 2),
                CXGate((c, t)),
                U3Gate(t, self.theta / 2, self.phi, 0),
            ])


class ToffoliGate(Gate):
    """Toffoli (CCX) gate"""
    lowername = "ccx"

    def dagger(self):
        return self

    def fallback(self, n_qubits):
        c1, c2, t = self.targets
        return [
            HGate(t),
            CXGate((c2, t)),
            TDagGate(t),
            CXGate((c1, t)),
            TGate(t),
            CXGate((c2, t)),
            TDagGate(t),
            CXGate((c1, t)),
            TGate(c2),
            TGate(t),
            HGate(t),
            CXGate((c1, c2)),
            TGate(c1),
            TDagGate(c2),
            CXGate((c1, c2)),
        ]


class CCZGate(Gate):
    """2-Controlled Z gate"""
    lowername = "ccz"

    def dagger(self):
        return self


class CSwapGate(Gate):
    """Controlled SWAP gate"""
    lowername = "cswap"

    def dagger(self):
        return self

    def fallback(self, n_qubits):
        # TODO: test
        c, t1, t2 = self.targets
        return [CXGate((t2, t1)),
                ToffoliGate((c, t1, t2)),
                CXGate((t2, t1))]


class Measurement(OneQubitGate):
    """Measurement gate"""
    lowername = "measure"

    def dagger(self):
        raise ValueError(f"Hermitian conjugate of the measurement is not available.")


def slicing_singlevalue(arg, length):
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
            raise TypeError("indices must be integers or slices, not " + arg.__class__.__name__)
        if i < 0:
            i += length
        yield i


def slicing(args, length):
    """Internally used."""
    if isinstance(args, tuple):
        for arg in args:
            yield from slicing_singlevalue(arg, length)
    else:
        yield from slicing_singlevalue(args, length)


def qubit_pairs(args, length):
    """Internally used."""
    if not isinstance(args, tuple):
        raise ValueError("Control and target qubits pair(s) are required.")
    if len(args) != 2:
        raise ValueError("Control and target qubits pair(s) are required.")
    controls = list(slicing(args[0], length))
    targets = list(slicing(args[1], length))
    if len(controls) != len(targets):
        raise ValueError("The number of control qubits and target qubits are must be same.")
    for c, z in zip(controls, targets):
        if c == z:
            raise ValueError("Control qubit and target qubit are must be different.")
    return zip(controls, targets)


def get_maximum_index(indices):
    """Internally used."""
    def _maximum_idx_single(idx):
        if isinstance(idx, slice):
            start = -1
            stop = 0
            if idx.start is not None:
                start = idx.start.__index__()
            if idx.stop is not None:
                stop = idx.stop.__index__()
            return max(start, stop - 1)
        else:
            return idx.__index__()
    if isinstance(indices, tuple):
        return max((_maximum_idx_single(i) for i in indices), default=-1)
    else:
        return _maximum_idx_single(indices)


def find_n_qubits(gates):
    """Find n_qubits from gates"""
    return max((get_maximum_index(g.targets) for g in gates), default=-1) + 1
