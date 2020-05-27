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

import cmath
import math
import random
import warnings
from collections import Counter

import numpy as np
from numba import jit, njit, prange
import numba

from ..gate import *
from .backendbase import Backend

DEFAULT_DTYPE = np.complex128
FASTMATH = True

# Typedef
# Index of Qubit
_QBIdx = numba.uint32
_QBIdx_dtype = np.uint32
# Index of Quantum State
_QSIdx = numba.uint64
_QSIdx_dtype = np.uint64
# Mask of Quantum State
_QSMask = _QSIdx
_QSMask_dtype = _QSIdx_dtype

class _NumbaBackendContext:
    """This class is internally used in NumbaBackend"""

    def __init__(self, n_qubits, save_cache, dtype=DEFAULT_DTYPE):
        self.n_qubits = n_qubits
        self.qubits = np.zeros(2**n_qubits, dtype)
        self.save_cache = save_cache
        self.shots_result = Counter()
        self.cregs = None

    def prepare(self, cache):
        """Prepare to run next shot."""
        if cache is not None:
            np.copyto(self.qubits, cache)
        else:
            self.qubits.fill(0.0)
            self.qubits[0] = 1.0
        self.cregs = [0] * self.n_qubits

    def store_shot(self):
        """Store current cregs to shots_result"""
        def to_str(cregs):
            return ''.join(str(b) for b in cregs)
        key = to_str(self.cregs)
        self.shots_result[key] = self.shots_result.get(key, 0) + 1


@njit(_QSIdx(_QSMask, _QSIdx),
      locals={'lower': _QSMask, 'higher': _QSMask},
      nogil=True, cache=True)
def _shifted(lower_mask, idx):
    lower = idx & lower_mask
    higher = (idx & ~lower_mask) << 1
    return higher + lower


@njit(_QSMask[:](_QBIdx[:]), nogil=True, cache=True)
def _create_masks(indices):
    indices.sort()
    masks = np.empty(len(indices) + 1, dtype=_QSMask_dtype)
    for i, x in enumerate(indices):
        masks[i] = (1 << (x - i)) - 1
    masks[-1] = ~0
    for i in range(len(indices), 0, -1):
        masks[i] &= ~masks[i - 1]
    return masks


@njit(_QSIdx(_QSMask[:], _QSIdx), nogil=True, cache=True)
def _mult_shifted(masks, idx):
    shifted = 0
    for i, x in enumerate(masks):
        shifted |= (idx & x) << i
    return shifted


@njit(locals={'lower_mask': _QSMask},
      nogil=True, parallel=True, fastmath=FASTMATH)
def _zgate(qubits, n_qubits, target):
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        qubits[_shifted(lower_mask, i)] *= -1


@njit(locals={'lower_mask': _QSMask},
      nogil=True, parallel=True, fastmath=FASTMATH)
def _xgate(qubits, n_qubits, target):
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        qubits[i0] = qubits[i0 + (1 << target)]
        qubits[i0 + (1 << target)] = t


@njit(locals={'lower_mask': _QSMask},
      nogil=True, parallel=True, fastmath=FASTMATH)
def _ygate(qubits, n_qubits, target):
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        # Global phase is ignored.
        qubits[i0] = -qubits[i0 + (1 << target)]
        qubits[i0 + (1 << target)] = t


@njit(locals={'lower_mask': _QSMask},
      nogil=True, parallel=True, fastmath=FASTMATH)
def _hgate(qubits, n_qubits, target):
    sqrt2_inv = 0.7071067811865475
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] = (t + u) * sqrt2_inv
        qubits[i0 + (1 << target)] = (t - u) * sqrt2_inv


@njit(locals={'lower_mask': _QSMask},
      nogil=True, parallel=True, fastmath=FASTMATH)
def _diaggate(qubits, n_qubits, target, factor):
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i1 = _shifted(lower_mask, i) + (1 << target)
        # Global phase is ignored.
        qubits[i1] *= factor


@njit(locals={'lower_mask': _QSMask},
      nogil=True, parallel=True, fastmath=FASTMATH)
def _rzgate(qubits, n_qubits, target, ang):
    ang *= 0.5
    eit = cmath.exp(1.j * ang)
    eitstar = eit.conjugate()
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] *= eitstar
        qubits[i0 + (1 << target)] *= eit


@njit(locals={'lower_mask': _QSMask},
      nogil=True, parallel=True, fastmath=FASTMATH)
def _rygate(qubits, n_qubits, target, ang):
    ang *= 0.5
    cos = math.cos(ang)
    sin = math.sin(ang)
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] = cos * t - sin * u
        qubits[i0 + (1 << target)] = sin * t + cos * u


@njit(locals={'lower_mask': _QSMask},
      nogil=True, parallel=True, fastmath=FASTMATH)
def _rxgate(qubits, n_qubits, target, ang):
    ang *= 0.5
    cos = math.cos(ang)
    nisin = math.sin(ang) * -1.j
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] = cos * t + nisin * u
        qubits[i0 + (1 << target)] = nisin * t + cos * u


@njit(locals={'lower_mask': _QSMask},
      nogil=True, parallel=True, fastmath=FASTMATH)
def _u3gate(qubits, n_qubits, target, theta, phi, lambd):
    theta *= 0.5
    cos = math.cos(theta)
    sin = math.sin(theta)
    expadd = cmath.exp((phi + lambd) * 0.5j)
    expsub = cmath.exp((phi - lambd) * 0.5j)
    a = expadd.conjugate() * cos
    b = -expsub.conjugate() * sin
    c = expsub * sin
    d = expadd * cos
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] = a * t + b * u
        qubits[i0 + (1 << target)] = c * t + d * u


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _czgate(qubits, n_qubits, controls_target):
    target = controls_target[-1]
    all1 = _QSMask(0)
    for b in controls_target:
        all1 |= _QSMask(1) << b
    n_loop = 1 << (_QSMask(n_qubits) - _QSMask(len(controls_target)))
    masks = _create_masks(controls_target)
    for i in prange(n_loop):
        i11 = _mult_shifted(masks, i) | all1
        qubits[i11] *= -1


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _cxgate(qubits, n_qubits, controls_target):
    c_mask = _QSMask(0)
    for c in controls_target[:-1]:
        c_mask |= _QSMask(1) << c
    t_mask = 1 << controls_target[-1]
    n_loop = 1 << (_QSMask(n_qubits) - _QSMask(len(controls_target)))
    masks = _create_masks(controls_target)
    for i in prange(n_loop):
        i10 = _mult_shifted(masks, i) | c_mask
        i11 = i10 | t_mask
        t = qubits[i10]
        qubits[i10] = qubits[i11]
        qubits[i11] = t


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _crxgate(qubits, n_qubits, controls_target, ang):
    ang *= 0.5
    cos = math.cos(ang)
    nisin = math.sin(ang) * -1.j
    c_mask = _QSMask(0)
    for c in controls_target[:-1]:
        c_mask |= _QSMask(1) << c
    t_mask = 1 << controls_target[-1]
    n_loop = 1 << (_QSMask(n_qubits) - _QSMask(len(controls_target)))
    masks = _create_masks(controls_target)
    for i in prange(n_loop):
        i10 = _mult_shifted(masks, i) | c_mask
        i11 = i10 | t_mask
        t = qubits[i10]
        u = qubits[i11]
        qubits[i10] = cos * t + nisin * u
        qubits[i11] = nisin * t + cos * u


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _crygate(qubits, n_qubits, controls_target, ang):
    ang *= 0.5
    cos = math.cos(ang)
    sin = math.sin(ang)
    c_mask = _QSMask(0)
    for c in controls_target[:-1]:
        c_mask |= _QSMask(1) << c
    t_mask = 1 << controls_target[-1]
    n_loop = 1 << (_QSMask(n_qubits) - _QSMask(len(controls_target)))
    masks = _create_masks(controls_target)
    for i in prange(n_loop):
        i10 = _mult_shifted(masks, i) | c_mask
        i11 = i10 | t_mask
        t = qubits[i10]
        u = qubits[i11]
        qubits[i10] = cos * t - sin * u
        qubits[i11] = sin * t + cos * u


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _crzgate(qubits, n_qubits, controls_target, ang):
    ang *= 0.5
    eit = cmath.exp(1.j * ang)
    eitstar = eit.conjugate()
    c_mask = _QSMask(0)
    for c in controls_target[:-1]:
        c_mask |= _QSMask(1) << c
    t_mask = 1 << controls_target[-1]
    n_loop = 1 << (_QSMask(n_qubits) - _QSMask(len(controls_target)))
    masks = _create_masks(controls_target)
    for i in prange(n_loop):
        i10 = _mult_shifted(masks, i) | c_mask
        i11 = i10 | t_mask
        qubits[i10] *= eitstar
        qubits[i11] *= eit


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _cphasegate(qubits, n_qubits, controls_target, ang):
    eit = cmath.exp(1.j * ang)
    c_mask = _QSMask(0)
    for c in controls_target[:-1]:
        c_mask |= _QSMask(1) << c
    t_mask = 1 << controls_target[-1]
    n_loop = 1 << (_QSMask(n_qubits) - _QSMask(len(controls_target)))
    masks = _create_masks(controls_target)
    for i in prange(n_loop):
        i11 = _mult_shifted(masks, i) | c_mask | t_mask
        qubits[i11] *= eit


@njit(locals={'lower_mask': _QSMask},
      nogil=True, parallel=True, fastmath=FASTMATH)
def _p0calc(qubits, target, n_qubits):
    p0 = 0.0
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        val = qubits[_shifted(lower_mask, i)]
        p0 += val.real * val.real + val.imag * val.imag
    return p0


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _reduce0(qubits, target, n_qubits, p0):
    sqrtp_inv = 1.0 / math.sqrt(p0)
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        qubits[i0] *= sqrtp_inv
        qubits[i0 + (1 << target)] = 0.0


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _reduce1(qubits, target, n_qubits, p0):
    sqrtp_inv = 1.0 / math.sqrt(1.0 - p0)
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        qubits[i0 + (1 << target)] *= sqrtp_inv
        qubits[i0] = 0.0


class NumbaBackend(Backend):
    """Simulator backend which uses numba."""
    __return_type = {
        "statevector": lambda ctx: ctx.qubits,
        "shots": lambda ctx: ctx.shots_result,
        "statevector_and_shots": lambda ctx: (ctx.qubits, ctx.shots_result),
        "_inner_ctx": lambda ctx: ctx,
    }
    DEFAULT_SHOTS = 1024

    def __init__(self):
        self.cache = None
        self.cache_idx = -1

    def __clear_cache(self):
        self.cache = None
        self.cache_idx = -1

    def __clear_cache_if_invalid(self, n_qubits, dtype):
        if self.cache is None:
            self.__clear_cache()
            return
        if len(self.cache) != 2**n_qubits:
            self.__clear_cache()
            return
        if self.cache.dtype != dtype:
            self.__clear_cache()
            return

    def run(self, gates, n_qubits, *args, **kwargs):
        def __parse_run_args(shots=None, returns=None, enable_cache=True, ignore_global=False,
                             dtype=DEFAULT_DTYPE, **_kwargs):
            if returns is None:
                if shots is None:
                    returns = "statevector"
                else:
                    returns = "shots"
            if returns not in self.__return_type.keys():
                raise ValueError(f"Unknown returns type '{returns}'")
            if shots is None:
                if returns in ("statevector", "_inner_ctx"):
                    shots = 1
                else:
                    shots = self.DEFAULT_SHOTS
            if returns == "statevector" and shots > 1:
                warnings.warn("When `returns` = 'statevector', `shots` = 1 is enough.")
            return shots, returns, dtype, enable_cache, ignore_global

        shots, returns, dtype, enable_cache, ignore_global = __parse_run_args(*args, **kwargs)

        if enable_cache:
            self.__clear_cache_if_invalid(n_qubits, dtype)
        else:
            self.__clear_cache()
        ctx = _NumbaBackendContext(n_qubits, enable_cache, dtype)

        def run_single_gate(gate):
            nonlocal ctx
            action = self._get_action(gate)
            if action is not None:
                ctx = action(gate, ctx)
            else:
                for g in gate.fallback(n_qubits):
                    run_single_gate(g)

        for _ in range(shots):
            ctx.prepare(self.cache)
            cache_idx = self.cache_idx
            for gate in gates[cache_idx + 1:]:
                run_single_gate(gate)
                if ctx.save_cache:
                    cache_idx += 1
            if self.cache_idx != cache_idx:
                self.cache_idx = cache_idx
                if ctx.save_cache:
                    self.cache = ctx.qubits.copy()
            if ctx.cregs:
                ctx.store_shot()

        if ignore_global:
            _ignore_global(ctx.qubits)
        return self.__return_type[returns](ctx)

    def make_cache(self, gates, n_qubits):
        self.run(gates, n_qubits)

    def gate_x(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _xgate(qubits, n_qubits, target)
        return ctx

    def gate_y(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _ygate(qubits, n_qubits, target)
        return ctx

    def gate_z(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _zgate(qubits, n_qubits, target)
        return ctx

    def gate_h(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _hgate(qubits, n_qubits, target)
        return ctx

    def gate_t(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = cmath.exp(0.25j * math.pi)
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    def gate_tdg(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = cmath.exp(-0.25j * math.pi)
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    def gate_s(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = 1.j
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    def gate_sdg(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = -1.j
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    def gate_cz(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for control, target in gate.control_target_iter(n_qubits):
            _czgate(qubits, n_qubits, np.array([control, target], dtype=_QBIdx_dtype))
        return ctx

    def gate_cx(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for control, target in gate.control_target_iter(n_qubits):
            _cxgate(qubits, n_qubits, np.array([control, target], dtype=_QBIdx_dtype))
        return ctx

    def gate_rx(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            _rxgate(qubits, n_qubits, target, theta)
        return ctx

    def gate_ry(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            _rygate(qubits, n_qubits, target, theta)
        return ctx

    def gate_rz(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            _rzgate(qubits, n_qubits, target, theta)
        return ctx

    def gate_phase(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = cmath.exp(1.j * gate.theta)
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    def gate_crx(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for control, target in gate.control_target_iter(n_qubits):
            _crxgate(qubits, n_qubits, np.array([control, target], dtype=_QBIdx_dtype), theta)
        return ctx

    def gate_cry(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for control, target in gate.control_target_iter(n_qubits):
            _crygate(qubits, n_qubits, np.array([control, target], dtype=_QBIdx_dtype), theta)
        return ctx

    def gate_crz(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for control, target in gate.control_target_iter(n_qubits):
            _crzgate(qubits, n_qubits, np.array([control, target], dtype=_QBIdx_dtype), theta)
        return ctx

    def gate_cphase(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for control, target in gate.control_target_iter(n_qubits):
            _cphasegate(qubits, n_qubits, np.array([control, target], dtype=_QBIdx_dtype), theta)
        return ctx

    def gate_ccz(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        _czgate(qubits, n_qubits, np.array(gate.targets, dtype=_QBIdx_dtype))
        return ctx

    def gate_ccx(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        _cxgate(qubits, n_qubits, np.array(gate.targets, dtype=_QBIdx_dtype))
        return ctx

    def gate_u1(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        angle = gate.lambd
        for target in gate.target_iter(n_qubits):
            _rzgate(qubits, n_qubits, target, angle)
        return ctx

    def gate_u3(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        phi = gate.phi
        lambd = gate.lambd
        for target in gate.target_iter(n_qubits):
            _u3gate(qubits, n_qubits, target, theta, phi, lambd)
        return ctx

    def gate_measure(self, gate, ctx):
        if ctx.save_cache:
            self.cache = ctx.qubits.copy()
        ctx.save_cache = False
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            rand = random.random()
            p0 = _p0calc(qubits, target, n_qubits)
            if rand < p0:
                _reduce0(qubits, target, n_qubits, p0)
                ctx.cregs[target] = 0
            else:
                _reduce1(qubits, target, n_qubits, p0)
                ctx.cregs[target] = 1
        return ctx


@njit(nogil=True, cache=True)
def _ignore_global(qubits):
    for q in qubits:
        if abs(q) > 0.0000001:
            ang = abs(q) / q
            qubits *= ang
            break
    return qubits
