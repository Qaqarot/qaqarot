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

from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import cmath
import random
import warnings

import numpy as np

from ..gate import *
from ..utils import ignore_global_phase
from .backendbase import Backend

DEFAULT_DTYPE = complex


class _NumPyBackendContext:
    """This class is internally used in NumPyBackend"""
    def __init__(self, n_qubits: int, cache: Optional[np.ndarray],
                 cache_idx: int) -> None:
        self.n_qubits = n_qubits
        self.qubits = np.zeros(2**n_qubits, dtype=DEFAULT_DTYPE)
        self.qubits_buf = np.zeros(2**n_qubits, dtype=DEFAULT_DTYPE)
        self.indices = np.arange(2**n_qubits, dtype=np.uint32)
        self.save_ctx_cache = True
        self.cache = cache
        self.cache_idx = cache_idx
        self.shots_result = Counter()
        self.cregs = [0] * self.n_qubits

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
        def to_str(cregs: List[int]) -> str:
            return ''.join(str(b) for b in cregs)

        key = to_str(self.cregs)
        self.shots_result[key] = self.shots_result.get(key, 0) + 1


class NumPyBackend(Backend):
    """Simulator backend which uses numpy. This backend is Blueqat's default backend."""
    __return_type: Dict[str, Callable[[_NumPyBackendContext], Any]] = {
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

    def run(
            self,
            gates: List[Operation],
            n_qubits,
            shots: Optional[int] = None,
            initial: Optional[np.ndarray] = None,
            returns: Optional[str] = None,
            ignore_global: bool = False,
            save_cache: bool = False,
            **kwargs) -> Any:
        def __parse_run_args(shots: Optional[int],
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

        shots, returns = __parse_run_args(shots, returns)
        if kwargs:
            warnings.warn(f"Unknown run arguments: {kwargs}")

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
        else:
            self.__clear_cache_if_invalid(n_qubits, DEFAULT_DTYPE)

        ctx = _NumPyBackendContext(n_qubits, self.cache, self.cache_idx)

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
            for gate in gates[ctx.cache_idx + 1:]:
                run_single_gate(gate)
                if ctx.save_ctx_cache:
                    ctx.cache = ctx.qubits.copy()
                    ctx.cache_idx += 1
                    if save_cache:
                        self.cache = ctx.cache
                        self.cache_idx = ctx.cache_idx
            if ctx.cregs:
                ctx.store_shot()

        if ignore_global:
            ignore_global_phase(ctx.qubits)
        return self.__return_type[returns](ctx)

    def make_cache(self, gates: List[Operation], n_qubits: int) -> None:
        self.run(gates, n_qubits, save_cache=True)

    @staticmethod
    def gate_x(gate: XGate, ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of X gate."""
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[t0] = qubits[t1]
            newq[t1] = qubits[t0]
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    @staticmethod
    def gate_y(gate: YGate, ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of Y gate."""
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[t0] = -1.0j * qubits[t1]
            newq[t1] = 1.0j * qubits[t0]
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    @staticmethod
    def gate_z(gate: ZGate, ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of Z gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= -1
        return ctx

    @staticmethod
    def gate_h(gate: HGate, ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of H gate."""
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[t0] = qubits[t0] + qubits[t1]
            newq[t1] = qubits[t0] - qubits[t1]
            newq *= 1.0 / math.sqrt(2)
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    @staticmethod
    def gate_rx(gate: RXGate,
                ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of RX gate."""
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        halftheta = gate.theta * 0.5
        a00 = a11 = math.cos(halftheta)
        a01 = a10 = -1j * math.sin(halftheta)
        for target in gate.target_iter(n_qubits):
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[t0] = a00 * qubits[t0] + a01 * qubits[t1]
            newq[t1] = a10 * qubits[t0] + a11 * qubits[t1]
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    @staticmethod
    def gate_ry(gate: RYGate,
                ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of RY gate."""
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        halftheta = gate.theta * 0.5
        a00 = a11 = math.cos(halftheta)
        a10 = math.sin(halftheta)
        a01 = -a10
        for target in gate.target_iter(n_qubits):
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[t0] = a00 * qubits[t0] + a01 * qubits[t1]
            newq[t1] = a10 * qubits[t0] + a11 * qubits[t1]
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    @staticmethod
    def gate_rz(gate: RZGate,
                ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of RZ gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        halftheta = gate.theta * 0.5
        a0 = complex(math.cos(halftheta), -math.sin(halftheta))
        a1 = complex(math.cos(halftheta), math.sin(halftheta))
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) == 0] *= a0
            qubits[(i & (1 << target)) != 0] *= a1
        return ctx

    @staticmethod
    def gate_phase(gate: PhaseGate,
                   ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of Phase gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        theta = gate.theta
        a = complex(math.cos(theta), math.sin(theta))
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= a
        return ctx

    @staticmethod
    def gate_t(gate: PhaseGate,
               ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of T gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices

        sqrt2_inv = 1 / math.sqrt(2)
        factor = complex(sqrt2_inv, sqrt2_inv)
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= factor
        return ctx

    @staticmethod
    def gate_s(gate: SGate, ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of S gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= 1.j
        return ctx

    @staticmethod
    def gate_cz(gate: CZGate,
                ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of CZ gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for control, target in gate.control_target_iter(n_qubits):
            qubits[((i & (1 << control)) != 0)
                   & ((i & (1 << target)) != 0)] *= -1
        return ctx

    @staticmethod
    def gate_cx(gate: CXGate,
                ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of CX gate."""
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for control, target in gate.control_target_iter(n_qubits):
            np.copyto(newq, qubits)
            c1 = (i & (1 << control)) != 0
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[c1 & t0] = qubits[c1 & t1]
            newq[c1 & t1] = qubits[c1 & t0]
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    @staticmethod
    def gate_cu(gate: CUGate,
                 ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of CU gate."""
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        theta = gate.theta
        phi = gate.phi
        lam = gate.lam
        gamma = gate.gamma
        globalphase = cmath.exp(1j * gamma)
        a00 = math.cos(theta * 0.5) * globalphase
        a11 = a00 * cmath.exp(1j * (phi + lam))
        a01 = a10 = math.sin(theta * 0.5) * globalphase
        a01 *= -cmath.exp(1j * lam)
        a10 *= cmath.exp(1j * phi)
        for control, target in gate.control_target_iter(n_qubits):
            np.copyto(newq, qubits)
            c1 = (i & (1 << control)) != 0
            c1t0 = ((i & (1 << target)) == 0) & c1
            c1t1 = ((i & (1 << target)) != 0) & c1
            newq[c1t0] = a00 * qubits[c1t0] + a01 * qubits[c1t1]
            newq[c1t1] = a10 * qubits[c1t0] + a11 * qubits[c1t1]
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    @staticmethod
    def gate_crx(gate: CRXGate,
                 ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of CRX gate."""
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        halftheta = gate.theta * 0.5
        a00 = a11 = math.cos(halftheta)
        a01 = a10 = -1j * math.sin(halftheta)
        for control, target in gate.control_target_iter(n_qubits):
            np.copyto(newq, qubits)
            c1 = (i & (1 << control)) != 0
            c1t0 = ((i & (1 << target)) == 0) & c1
            c1t1 = ((i & (1 << target)) != 0) & c1
            newq[c1t0] = a00 * qubits[c1t0] + a01 * qubits[c1t1]
            newq[c1t1] = a10 * qubits[c1t0] + a11 * qubits[c1t1]
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    @staticmethod
    def gate_cry(gate: CRYGate,
                 ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of CRY gate."""
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        halftheta = gate.theta * 0.5
        a00 = a11 = math.cos(halftheta)
        a10 = math.sin(halftheta)
        a01 = -a10
        for control, target in gate.control_target_iter(n_qubits):
            np.copyto(newq, qubits)
            c1 = (i & (1 << control)) != 0
            c1t0 = ((i & (1 << target)) == 0) & c1
            c1t1 = ((i & (1 << target)) != 0) & c1
            newq[c1t0] = a00 * qubits[c1t0] + a01 * qubits[c1t1]
            newq[c1t1] = a10 * qubits[c1t0] + a11 * qubits[c1t1]
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    @staticmethod
    def gate_crz(gate: CRZGate,
                 ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of CRZ gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        halftheta = gate.theta * 0.5
        a0 = complex(math.cos(halftheta), -math.sin(halftheta))
        a1 = complex(math.cos(halftheta), math.sin(halftheta))
        for control, target in gate.control_target_iter(n_qubits):
            c1t0 = ((i & (1 << control)) != 0) & ((i & (1 << target)) == 0)
            c1t1 = ((i & (1 << control)) != 0) & ((i & (1 << target)) != 0)
            qubits[c1t0] *= a0
            qubits[c1t1] *= a1
        return ctx

    @staticmethod
    def gate_cphase(gate: CPhaseGate,
                    ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of CPhase gate."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        theta = gate.theta
        a = complex(math.cos(theta), math.sin(theta))
        for control, target in gate.control_target_iter(n_qubits):
            c1t1 = ((i & (1 << control)) != 0) & ((i & (1 << target)) != 0)
            qubits[c1t1] *= a
        return ctx

    @staticmethod
    def gate_ccz(gate: CCZGate,
                 ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of CCZ gate."""
        c1, c2, t = gate.targets
        qubits = ctx.qubits
        i = ctx.indices
        indices = (i & (1 << c1)) != 0
        indices &= (i & (1 << c2)) != 0
        indices &= (i & (1 << t)) != 0
        qubits[indices] *= -1
        return ctx

    @staticmethod
    def gate_ccx(gate: ToffoliGate,
                 ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of Toffoli gate."""
        _, _, t = gate.targets
        ctx = NumPyBackend.gate_h(HGate(t), ctx)
        ctx = NumPyBackend.gate_ccz(CCZGate(gate.targets), ctx)
        ctx = NumPyBackend.gate_h(HGate(t), ctx)
        return ctx

    @staticmethod
    def gate_u(gate: UGate,
                ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of U gate."""
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        theta = gate.theta
        phi = gate.phi
        lam = gate.lam
        gamma = gate.gamma
        globalphase = cmath.exp(1j * gamma)
        a00 = math.cos(theta * 0.5) * globalphase
        a11 = a00 * cmath.exp(1j * (phi + lam))
        a01 = a10 = math.sin(theta * 0.5) * globalphase
        a01 *= -cmath.exp(1j * lam)
        a10 *= cmath.exp(1j * phi)
        for target in gate.target_iter(n_qubits):
            np.copyto(newq, qubits)
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[t0] = qubits[t0] * a00
            newq[t0] += qubits[t1] * a01
            newq[t1] = qubits[t0] * a10
            newq[t1] += qubits[t1] * a11
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    @staticmethod
    def gate_mat1(gate: Mat1Gate,
                  ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of U3 gate."""
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        mat = gate.mat
        for target in gate.target_iter(n_qubits):
            np.copyto(newq, qubits)
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[t0] = mat[0, 0] * qubits[t0]
            newq[t0] += mat[0, 1] * qubits[t1]
            newq[t1] = mat[1, 0] * qubits[t0]
            newq[t1] += mat[1, 1] * qubits[t1]
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    @staticmethod
    def gate_measure(gate: Measurement,
                     ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of measurement operation."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            p_zero = np.linalg.norm(qubits[(i & (1 << target)) == 0])**2
            rand = random.random()
            if rand < p_zero:
                qubits[(i & (1 << target)) != 0] = 0.0
                qubits /= math.sqrt(p_zero)
                ctx.cregs[target] = 0
            else:
                qubits[(i & (1 << target)) == 0] = 0.0
                qubits /= math.sqrt(1.0 - p_zero)
                ctx.cregs[target] = 1
        ctx.save_ctx_cache = False
        return ctx

    @staticmethod
    def gate_reset(gate: Reset,
                   ctx: _NumPyBackendContext) -> _NumPyBackendContext:
        """Implementation of measurement operation."""
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            p_zero = np.linalg.norm(qubits[(i & (1 << target)) == 0])**2
            rand = random.random()
            t1 = (i & (1 << target)) != 0
            if rand < p_zero:
                qubits[t1] = 0.0
                qubits /= math.sqrt(p_zero)
            else:
                qubits[(i & (1 << target)) == 0] = qubits[t1]
                qubits[t1] = 0.0
                qubits /= math.sqrt(1.0 - p_zero)
        ctx.save_ctx_cache = False
        return ctx
