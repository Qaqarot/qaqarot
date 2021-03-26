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
"""Blueqat numba backend."""

import cmath
import math
import random
import warnings
from collections import Counter
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numba import njit, prange
import numba

from ..gate import *
from .backendbase import Backend

DEFAULT_DTYPE = complex
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


@njit(_QSMask(_QSMask, _QSIdx),
      locals={
          'lower': _QSMask,
          'higher': _QSMask
      },
      nogil=True,
      cache=True)
def _shifted(lower_mask: _QSMask, idx: _QSIdx) -> _QSMask:
    lower = idx & lower_mask
    higher = (idx & ~lower_mask) << 1
    return higher | lower


@njit(_QSMask[:](_QBIdx[:]), nogil=True, cache=True)
def _create_masks(indices: np.ndarray) -> np.ndarray:
    indices.sort()
    masks = np.empty(len(indices) + 1, dtype=_QSMask_dtype)
    for i, x in enumerate(indices):
        masks[i] = (1 << (x - i)) - 1
    masks[-1] = ~0
    for i in range(len(indices), 0, -1):
        masks[i] &= ~masks[i - 1]
    return masks


@njit(_QSMask(_QSMask[:], _QSIdx), nogil=True, cache=True)
def _mult_shifted(masks: np.ndarray, idx: _QSIdx) -> _QSMask:
    shifted = 0
    for i, x in enumerate(masks):
        shifted |= (idx & x) << i
    return shifted


@njit(locals={'lower_mask': _QSMask},
      nogil=True,
      parallel=True,
      fastmath=FASTMATH)
def _zgate(qubits: np.ndarray, n_qubits: _QSIdx, target: _QSIdx) -> None:
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        qubits[_shifted(lower_mask, i) + (1 << target)] *= -1


@njit(locals={'lower_mask': _QSMask},
      nogil=True,
      parallel=True,
      fastmath=FASTMATH)
def _xgate(qubits: np.ndarray, n_qubits: _QSIdx, target: _QSIdx) -> None:
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        qubits[i0] = qubits[i0 + (1 << target)]
        qubits[i0 + (1 << target)] = t


@njit(locals={'lower_mask': _QSMask},
      nogil=True,
      parallel=True,
      fastmath=FASTMATH)
def _ygate(qubits: np.ndarray, n_qubits: _QSIdx, target: _QSIdx) -> None:
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0] * 1j
        qubits[i0] = qubits[i0 + (1 << target)] * -1j
        qubits[i0 + (1 << target)] = t


@njit(locals={'lower_mask': _QSMask},
      nogil=True,
      parallel=True,
      fastmath=FASTMATH)
def _hgate(qubits: np.ndarray, n_qubits: _QSIdx, target: _QSIdx) -> None:
    sqrt2_inv = 0.7071067811865475
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] = (t + u) * sqrt2_inv
        qubits[i0 + (1 << target)] = (t - u) * sqrt2_inv


@njit(locals={'lower_mask': _QSMask},
      nogil=True,
      parallel=True,
      fastmath=FASTMATH)
def _diaggate(qubits: np.ndarray, n_qubits: _QSIdx, target: _QSIdx,
              factor: np.float64) -> None:
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i1 = _shifted(lower_mask, i) + (1 << target)
        qubits[i1] *= factor


@njit(locals={'lower_mask': _QSMask},
      nogil=True,
      parallel=True,
      fastmath=FASTMATH)
def _rzgate(qubits: np.ndarray, n_qubits: _QSIdx, target: _QSIdx,
            ang: np.float64) -> None:
    ang *= 0.5
    eit = cmath.exp(1.j * ang)
    eitstar = eit.conjugate()
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        qubits[i0] *= eitstar
        qubits[i0 + (1 << target)] *= eit


@njit(locals={'lower_mask': _QSMask},
      nogil=True,
      parallel=True,
      fastmath=FASTMATH)
def _rygate(qubits: np.ndarray, n_qubits: _QSIdx, target: _QSIdx,
            ang: np.float64) -> None:
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
      nogil=True,
      parallel=True,
      fastmath=FASTMATH)
def _rxgate(qubits: np.ndarray, n_qubits: _QSIdx, target: _QSIdx,
            ang: np.float64) -> None:
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
      nogil=True,
      parallel=True,
      fastmath=FASTMATH)
def _ugate(qubits: np.ndarray, n_qubits: _QSIdx, target: _QSIdx,
        theta: np.float64, phi: np.float64, lam: np.float64, gamma: np.float64) -> None:
    cos = math.cos(theta * 0.5)
    sin = math.sin(theta * 0.5)
    a = cos * cmath.exp(1j * gamma)
    b = -sin * cmath.exp(1j * (gamma + lam))
    c = sin * cmath.exp(1j * (gamma + phi))
    d = cos * cmath.exp(1j * (gamma + phi + lam))
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] = a * t + b * u
        qubits[i0 + (1 << target)] = c * t + d * u


@njit(locals={'lower_mask': _QSMask},
      nogil=True,
      parallel=True,
      fastmath=FASTMATH)
def _mat1gate(qubits: np.ndarray, n_qubits: _QSIdx, target: _QSIdx,
              mat: np.ndarray) -> None:
    lower_mask = (1 << _QSMask(target)) - 1
    a = mat[0, 0]
    b = mat[0, 1]
    c = mat[1, 0]
    d = mat[1, 1]
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        t = qubits[i0]
        u = qubits[i0 + (1 << target)]
        qubits[i0] = a * t + b * u
        qubits[i0 + (1 << target)] = c * t + d * u


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _czgate(qubits: np.ndarray, n_qubits: _QSIdx,
            controls_target: np.ndarray) -> None:
    #target = controls_target[-1]
    all1 = _QSMask(0)
    for b in controls_target:
        all1 |= _QSMask(1) << b
    n_loop = 1 << (_QSMask(n_qubits) - _QSMask(len(controls_target)))
    masks = _create_masks(controls_target)
    for i in prange(n_loop):
        i11 = _mult_shifted(masks, i) | all1
        qubits[i11] *= -1


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _cxgate(qubits: np.ndarray, n_qubits: _QSIdx,
            controls_target: np.ndarray) -> None:
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
def _crxgate(qubits: np.ndarray, n_qubits: _QSIdx, controls_target: np.ndarray,
             ang: np.float64) -> None:
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
def _crygate(qubits: np.ndarray, n_qubits: _QSIdx, controls_target: np.ndarray,
             ang: np.float64) -> None:
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
def _crzgate(qubits: np.ndarray, n_qubits: _QSIdx, controls_target: np.ndarray,
             ang: np.float64) -> None:
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
def _cphasegate(qubits: np.ndarray, n_qubits: _QSIdx,
                controls_target: np.ndarray, ang: np.float64) -> None:
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


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _cugate(qubits: np.ndarray, n_qubits: _QSIdx, controls_target: np.ndarray,
        theta: np.float64, phi: np.float64, lam: np.float64, gamma: np.float64) -> None:
    cos = math.cos(theta * 0.5)
    sin = math.sin(theta * 0.5)
    m11 = cos * cmath.exp(1j * gamma)
    m12 = -sin * cmath.exp(1j * (gamma + lam))
    m21 = sin * cmath.exp(1j * (gamma + phi))
    m22 = cos * cmath.exp(1j * (gamma + phi + lam))
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
        qubits[i10] = m11 * t + m12 * u
        qubits[i11] = m21 * t + m22 * u


@njit(locals={'lower_mask': _QSMask},
      nogil=True,
      parallel=True,
      fastmath=FASTMATH)
def _p0calc(qubits: np.ndarray, target: _QSIdx,
            n_qubits: _QSIdx) -> np.float64:
    p0 = 0.0
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        val = qubits[_shifted(lower_mask, i)]
        p0 += val.real * val.real + val.imag * val.imag
    return p0


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _reduce0(qubits: np.ndarray, target: _QSIdx, n_qubits: _QSIdx,
             p0: np.float64) -> None:
    sqrtp_inv = 1.0 / math.sqrt(p0)
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        qubits[i0] *= sqrtp_inv
        qubits[i0 + (1 << target)] = 0.0


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _reduce1(qubits: np.ndarray, target: _QSIdx, n_qubits: _QSIdx,
             p0: np.float64) -> None:
    sqrtp_inv = 1.0 / math.sqrt(1.0 - p0)
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        qubits[i0 + (1 << target)] *= sqrtp_inv
        qubits[i0] = 0.0


@njit(nogil=True, parallel=True, fastmath=FASTMATH)
def _reset1(qubits: np.ndarray, target: _QSIdx, n_qubits: _QSIdx,
            p0: np.float64) -> None:
    sqrtp_inv = 1.0 / math.sqrt(1.0 - p0)
    lower_mask = (1 << _QSMask(target)) - 1
    for i in prange(1 << (_QSMask(n_qubits) - 1)):
        i0 = _shifted(lower_mask, i)
        qubits[i0] = qubits[i0 + (1 << target)] * sqrtp_inv
        qubits[i0 + (1 << target)] = 0.0


class _NumbaBackendContext:
    """This class is internally used in NumbaBackend"""
    def __init__(self,
                 n_qubits: int,
                 save_cxt_cache: bool,
                 cache: Optional[np.ndarray],
                 cache_idx: int,
                 dtype=DEFAULT_DTYPE) -> None:
        self.n_qubits: int = n_qubits
        self.qubits: np.ndarray = np.zeros(2**n_qubits, dtype)
        self.save_cxt_cache: bool = save_cxt_cache
        self.shots_result: Counter = Counter()
        self.cregs: List[int] = [0] * self.n_qubits
        self.cache: Optional[np.ndarray] = cache
        self.cache_idx: int = cache_idx

    def prepare(self, initial: Optional[np.ndarray]) -> None:
        """Prepare to run next shot."""
        if self.cache is not None:
            np.copyto(self.qubits, self.cache)
        elif initial is not None:
            np.copyto(self.qubits, initial)
        else:
            self.qubits.fill(0.0)
            self.qubits[0] = 1.0
        self.cregs = [0] * self.n_qubits

    def store_shot(self) -> None:
        """Store current cregs to shots_result"""
        def to_str(cregs):
            return ''.join(str(b) for b in cregs)

        key = to_str(self.cregs)
        self.shots_result[key] = self.shots_result.get(key, 0) + 1


class NumbaBackend(Backend):
    """Simulator backend which uses numba."""
    __return_type: Dict[str, Callable[[_NumbaBackendContext], Any]] = {
        "statevector": lambda ctx: ctx.qubits,
        "shots": lambda ctx: ctx.shots_result,
        "statevector_and_shots": lambda ctx: (ctx.qubits, ctx.shots_result),
        "_inner_ctx": lambda ctx: ctx,
    }
    DEFAULT_SHOTS: int = 1024

    def __init__(self) -> None:
        self.cache = None
        self.cache_idx = -1

    def __clear_cache(self) -> None:
        self.cache = None
        self.cache_idx = -1

    def __clear_cache_if_invalid(self, n_qubits: int, dtype: type) -> None:
        if self.cache is None:
            self.__clear_cache()
            return
        if len(self.cache) != 2**n_qubits:
            self.__clear_cache()
            return
        if self.cache.dtype != dtype:
            self.__clear_cache()
            return

    def run(self,
            gates: List[Operation],
            n_qubits: int,
            shots: Optional[int] = None,
            returns: Optional[str] = None,
            initial: Optional[np.ndarray] = None,
            save_cache: bool = False,
            ignore_global: bool = False,
            dtype: type = DEFAULT_DTYPE,
            enable_ctx_cache: Optional[bool] = None,
            **kwargs) -> Any:
        def __parse_shots_returns(shots: Optional[int],
                                  returns: Optional[str]) -> Tuple[int, str]:
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
                warnings.warn(
                    "When `returns` = 'statevector', `shots` = 1 is enough.")
            return shots, returns

        shots, returns = __parse_shots_returns(shots, returns)

        if enable_ctx_cache is None:
            enable_ctx_cache = shots > 1
        elif enable_ctx_cache is False:
            self.__clear_cache()

        if kwargs:
            warnings.warn(f"Unknown arguments {kwargs}")

        if initial is not None:
            if not isinstance(initial, np.ndarray):
                raise ValueError(f"`initial` must be a np.ndarray, but {type(initial)}")
            if initial.shape != (2**n_qubits,):
                raise ValueError(f"`initial.shape` is not matched. Expected: {(2**n_qubits,)}, Actual: {initial.shape}")
            if initial.dtype != DEFAULT_DTYPE:
                initial = initial.astype(DEFAULT_DTYPE)
            if save_cache:
                warnings.warn("When initial is not None, saving cache is disabled.")
                save_cache = False
            self.__clear_cache()

        self.__clear_cache_if_invalid(n_qubits, dtype)

        ctx = _NumbaBackendContext(n_qubits, enable_ctx_cache, self.cache,
                                   self.cache_idx, dtype)

        def run_single_gate(gate: Operation) -> None:
            nonlocal ctx
            action = self._get_action(gate)
            if action is not None:
                ctx = action(gate, ctx)
            else:
                for g in gate.fallback(n_qubits):
                    run_single_gate(g)

        for _ in range(shots):
            ctx.prepare(initial)
            cache_idx = ctx.cache_idx
            for gate in gates[cache_idx + 1:]:
                run_single_gate(gate)
                if ctx.save_cxt_cache:
                    cache_idx += 1
            if ctx.cache_idx != cache_idx:
                ctx.cache_idx = cache_idx
                if ctx.save_cxt_cache:
                    ctx.cache = ctx.qubits.copy()
                if save_cache:
                    self.cache_idx = ctx.cache_idx
                    self.cache = ctx.cache
            if ctx.cregs:
                ctx.store_shot()

        if ignore_global:
            _ignore_global(ctx.qubits)
        return self.__return_type[returns](ctx)

    def make_cache(self, gates: List[Operation], n_qubits: int) -> None:
        self.run(gates, n_qubits)

    @staticmethod
    def gate_x(gate: XGate, ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of X gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _xgate(qubits, n_qubits, target)
        return ctx

    @staticmethod
    def gate_y(gate: YGate, ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of Y gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _ygate(qubits, n_qubits, target)
        return ctx

    @staticmethod
    def gate_z(gate: ZGate, ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of Z gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _zgate(qubits, n_qubits, target)
        return ctx

    @staticmethod
    def gate_h(gate: HGate, ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of H gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _hgate(qubits, n_qubits, target)
        return ctx

    @staticmethod
    def gate_t(gate: TGate, ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of T gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = cmath.exp(0.25j * math.pi)
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    @staticmethod
    def gate_tdg(gate: TDagGate,
                 ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of T† gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = cmath.exp(-0.25j * math.pi)
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    @staticmethod
    def gate_s(gate: SGate, ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of S gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = 1.j
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    @staticmethod
    def gate_sdg(gate: SDagGate,
                 ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of S† gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = -1.j
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    @staticmethod
    def gate_cu(gate: CUGate, ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of CU gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for control, target in gate.control_target_iter(n_qubits):
            _cugate(qubits, n_qubits, np.array([control, target], dtype=_QBIdx_dtype),
                gate.theta, gate.phi, gate.lam, gate.gamma)
        return ctx

    @staticmethod
    def gate_cx(gate: CXGate,
                ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of CX gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for control, target in gate.control_target_iter(n_qubits):
            _cxgate(qubits, n_qubits,
                    np.array([control, target], dtype=_QBIdx_dtype))
        return ctx

    @staticmethod
    def gate_cz(gate: CZGate,
                ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of CZ gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for control, target in gate.control_target_iter(n_qubits):
            _czgate(qubits, n_qubits,
                    np.ndarray([control, target], dtype=_QBIdx_dtype))
        return ctx

    @staticmethod
    def gate_rx(gate: RXGate,
                ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of RX gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            _rxgate(qubits, n_qubits, target, theta)
        return ctx

    @staticmethod
    def gate_ry(gate: RYGate,
                ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of RY gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            _rygate(qubits, n_qubits, target, theta)
        return ctx

    @staticmethod
    def gate_rz(gate: RZGate,
                ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of RZ gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            _rzgate(qubits, n_qubits, target, theta)
        return ctx

    @staticmethod
    def gate_phase(gate: PhaseGate,
                   ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of Phase gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        factor = cmath.exp(1.j * gate.theta)
        for target in gate.target_iter(n_qubits):
            _diaggate(qubits, n_qubits, target, factor)
        return ctx

    @staticmethod
    def gate_crx(gate: CRXGate,
                 ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of CRX gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for control, target in gate.control_target_iter(n_qubits):
            _crxgate(qubits, n_qubits,
                     np.array([control, target], dtype=_QBIdx_dtype), theta)
        return ctx

    @staticmethod
    def gate_cry(gate: CRYGate,
                 ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of CRY gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for control, target in gate.control_target_iter(n_qubits):
            _crygate(qubits, n_qubits,
                     np.array([control, target], dtype=_QBIdx_dtype), theta)
        return ctx

    @staticmethod
    def gate_crz(gate: CRZGate,
                 ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of CRZ gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for control, target in gate.control_target_iter(n_qubits):
            _crzgate(qubits, n_qubits,
                     np.array([control, target], dtype=_QBIdx_dtype), theta)
        return ctx

    @staticmethod
    def gate_cphase(gate: CPhaseGate,
                    ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of CPhase gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        theta = gate.theta
        for control, target in gate.control_target_iter(n_qubits):
            _cphasegate(qubits, n_qubits,
                        np.array([control, target], dtype=_QBIdx_dtype),
                        theta)
        return ctx

    @staticmethod
    def gate_ccz(gate: CCZGate,
                 ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of CCZ gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        _czgate(qubits, n_qubits, np.array(gate.targets, dtype=_QBIdx_dtype))
        return ctx

    @staticmethod
    def gate_ccx(gate: ToffoliGate,
                 ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of Toffoli gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        _cxgate(qubits, n_qubits, np.array(gate.targets, dtype=_QBIdx_dtype))
        return ctx

    @staticmethod
    def gate_u(gate: UGate,
                ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of U gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            _ugate(qubits, n_qubits, target, gate.theta, gate.phi, gate.lam, gate.gamma)
        return ctx

    @staticmethod
    def gate_mat1(gate: Mat1Gate,
                  ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of Mat1 gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        mat = gate.matrix()
        for target in gate.target_iter(n_qubits):
            _mat1gate(qubits, n_qubits, target, mat)
        return ctx

    @staticmethod
    def gate_measure(gate: Measurement,
                     ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of measurement operation."""
        if ctx.save_cxt_cache:
            ctx.cache = ctx.qubits.copy()
        ctx.save_cxt_cache = False
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

    @staticmethod
    def gate_reset(gate: Reset,
                   ctx: _NumbaBackendContext) -> _NumbaBackendContext:
        """Implementation of reset operation."""
        if ctx.save_cxt_cache:
            ctx.cache = ctx.qubits.copy()
        ctx.save_cxt_cache = False
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        for target in gate.target_iter(n_qubits):
            rand = random.random()
            p0 = _p0calc(qubits, target, n_qubits)
            if rand < p0:
                _reduce0(qubits, target, n_qubits, p0)
            else:
                _reset1(qubits, target, n_qubits, p0)
        return ctx


@njit(nogil=True, cache=True)
def _ignore_global(qubits: np.ndarray) -> np.ndarray:
    for q in qubits:
        if abs(q) > 0.0000001:
            ang = abs(q) / q
            qubits *= ang
            break
    return qubits
