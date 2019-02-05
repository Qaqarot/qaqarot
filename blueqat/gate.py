"""
`gate` module implements quantum gate operations.
This module is internally used.
"""

import math
from abc import ABC


class Gate(ABC):
    """Abstract quantum gate class."""

    """Lower name of the gate."""
    lowername = None

    @property
    def uppername(self):
        """Upper name of the gate."""
        return self.lowername.upper()

    def __init__(self, targets, params=(), **kwargs):
        if self.lowername is None:
            raise ValueError(f"{self.__class__.__name__}.lowername is not defined.")
        self.params = params
        self.kwargs = kwargs
        self.targets = targets

    def fallback(self, n_qubits):
        """Returns alternative gates to make equivalent circuit."""
        raise NotImplementedError(f"The fallback of {self.__class__.__name__} gate is not defined.")

    def _str_args(self):
        """Returns printable string of args."""
        return ""

    def _str_targets(self):
        """Returns printable string of targets."""
        def _slice_to_str(obj):
            if isinstance(obj, slice):
                start = "" if obj.start is None else str(obj.start.__index__())
                stop = "" if obj.stop is None else str(obj.stop.__index__())
                if obj.step is None:
                    return f"{start}:{stop}"
                else:
                    step = str(obj.step.__index__())
                    return f"{start}:{stop}:{step}"
            else:
                return obj.__index__()

        if isinstance(self.targets, tuple):
            return f"[{', '.join(_slice_to_str(idx for idx in self.targets))}]"
        else:
            return f"[{_slice_to_str(self.targets)}]"

    def __str__(self):
        str_args = self._str_args()
        str_targets = self._str_targets()
        return f'{self.uppername}{str_args} {str_targets}'


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
    """Identity Gate"""
    lowername = "i"

    def fallback(self, n_qubits):
        return []


class XGate(OneQubitGate):
    """Pauli's X Gate"""
    lowername = "x"


class YGate(OneQubitGate):
    """Pauli's Y Gate"""
    lowername = "y"


class ZGate(OneQubitGate):
    """Pauli's Z Gate"""
    lowername = "z"


class HGate(OneQubitGate):
    """Hadamard Gate"""
    lowername = "h"


class CZGate(TwoQubitGate):
    """Control-Z gate"""
    lowername = "cz"


class CXGate(TwoQubitGate):
    """Control-X (CNOT) Gate"""
    lowername = "cx"


class RXGate(OneQubitGate):
    """Rotate-X gate"""
    lowername = "rx"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def _str_args(self):
        return f'({self.theta})'


class RYGate(OneQubitGate):
    """Rotate-Y gate"""
    lowername = "ry"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def _str_args(self):
        return f'({self.theta})'


class RZGate(OneQubitGate):
    """Rotate-Z gate"""
    lowername = "rz"

    def __init__(self, targets, theta, **kwargs):
        super().__init__(targets, (theta,), **kwargs)
        self.theta = theta

    def _str_args(self):
        return f'({self.theta})'


class TGate(OneQubitGate):
    """T ($\\pi/8$) gate"""
    lowername = "t"

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(n_qubits, lambda t: [RZGate(t, math.pi / 4)])


class TDagGate(OneQubitGate):
    """Dagger of T ($\\pi/8$) gate"""
    lowername = "tdg"

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(n_qubits, lambda t: [RZGate(t, -math.pi / 4)])


class SGate(OneQubitGate):
    """S gate"""
    lowername = "s"

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(n_qubits, lambda t: [RZGate(t, math.pi / 2)])


class SDagGate(OneQubitGate):
    """Dagger of S gate"""
    lowername = "s"

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(n_qubits, lambda t: [RZGate(t, -math.pi / 2)])


class SwapGate(TwoQubitGate):
    """Swap gate"""
    lowername = "swap"

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [CXGate((c, t)), CXGate((t, c)), CXGate((c, t))])


class ToffoliGate(Gate):
    """Toffoli (CCX) gate"""
    lowername = "ccx"

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


class U1Gate(OneQubitGate):
    """U1 gate"""
    def __init__(self, targets, lambd, **kwargs):
        super().__init__(targets, (lambd,), **kwargs)
        self.lambd = lambd

    lowername = "u1"

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(
            n_qubits, lambda t: [U3Gate(t, 0.0, 0.0, self.lambd)])


class U2Gate(OneQubitGate):
    """U2 gate"""
    def __init__(self, targets, phi, lambd, **kwargs):
        super().__init__(targets, (phi, lambd), **kwargs)
        self.phi = phi
        self.lambd = lambd

    lowername = "u2"

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(
            n_qubits, lambda t: [U3Gate(t, math.pi / 2, self.phi, self.lambd)])


class U3Gate(OneQubitGate):
    """U3 gate"""
    def __init__(self, targets, theta, phi, lambd, **kwargs):
        super().__init__(targets, (theta, phi, lambd), **kwargs)
        self.theta = theta
        self.phi = phi
        self.lambd = lambd

    lowername = "u3"


class CU1Gate(TwoQubitGate):
    """Controlled U1 gate"""
    def __init__(self, targets, lambd, **kwargs):
        super().__init__(targets, (lambd,), **kwargs)
        self.lambd = lambd

    lowername = "cu1"

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
    def __init__(self, targets, phi, lambd, **kwargs):
        super().__init__(targets, (phi, lambd), **kwargs)
        self.phi = phi
        self.lambd = lambd

    lowername = "cu2"

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits, lambda c, t: [CU3Gate((c, t), math.pi / 2, self.phi, self.lambd)])

class CU3Gate(TwoQubitGate):
    """Controlled U3 gate"""
    def __init__(self, targets, theta, phi, lambd, **kwargs):
        super().__init__(targets, (theta, phi, lambd), **kwargs)
        self.theta = theta
        self.phi = phi
        self.lambd = lambd

    lowername = "cu3"

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits, lambda c, t: [
                U1Gate(t, (self.lambd - self.phi) / 2),
                CXGate((c, t)),
                U3Gate(t, -self.theta / 2, 0, -(self.phi + self.lambd) / 2),
                CXGate((c, t)),
                U3Gate(t, self.theta / 2, self.phi, 0),
            ])


class Measurement(OneQubitGate):
    """Measurement gate"""
    lowername = "measure"


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
