from collections import Counter
import math
import random
import warnings
import numpy as np
from ..gate import *
from .backendbase import Backend

DEFAULT_DTYPE = np.complex128


class _NumPyBackendContext:
    """This class is internally used in NumPyBackend"""

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.qubits = np.zeros(2**n_qubits, dtype=DEFAULT_DTYPE)
        self.qubits_buf = np.zeros(2**n_qubits, dtype=DEFAULT_DTYPE)
        self.indices = np.arange(2**n_qubits, dtype=np.uint32)
        self.save_cache = True
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


class NumPyBackend(Backend):
    """Simulator backend which uses numpy. This backend is Blueqat's default backend."""
    __return_type = {
        "statevector": lambda ctx: _ignore_globals(ctx.qubits),
        "shots": lambda ctx: ctx.shots_result,
        "statevector_and_shots": lambda ctx: (_ignore_globals(ctx.qubits), ctx.shots_result),
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
        def __parse_run_args(shots=None, returns=None, **_kwargs):
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
            return shots, returns

        shots, returns = __parse_run_args(*args, **kwargs)

        self.__clear_cache_if_invalid(n_qubits, DEFAULT_DTYPE)
        ctx = _NumPyBackendContext(n_qubits)

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
            for gate in gates[self.cache_idx + 1:]:
                run_single_gate(gate)
                if ctx.save_cache:
                    self.cache = ctx.qubits.copy()
                    self.cache_idx += 1
            if ctx.cregs:
                ctx.store_shot()

        return self.__return_type[returns](ctx)

    def make_cache(self, gates, n_qubits):
        self.run(gates, n_qubits)

    def gate_x(self, gate, ctx):
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

    def gate_y(self, gate, ctx):
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

    def gate_z(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= -1
        return ctx

    def gate_h(self, gate, ctx):
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[t0] = qubits[t0] + qubits[t1]
            newq[t1] = qubits[t0] - qubits[t1]
            newq /= np.sqrt(2)
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    def gate_cz(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for control, target in gate.control_target_iter(n_qubits):
            qubits[((i & (1 << control)) != 0) & ((i & (1 << target)) != 0)] *= -1
        return ctx

    def gate_cx(self, gate, ctx):
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

    def gate_rx(self, gate, ctx):
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[t0] = np.cos(theta / 2) * qubits[t0] + -1.0j * np.sin(theta / 2) * qubits[t1]
            newq[t1] = -1.0j * np.sin(theta / 2) * qubits[t0] + np.cos(theta / 2) * qubits[t1]
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    def gate_ry(self, gate, ctx):
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[t0] = np.cos(theta / 2) * qubits[t0] + -np.sin(theta / 2) * qubits[t1]
            newq[t1] = np.sin(theta / 2) * qubits[t0] + np.cos(theta / 2) * qubits[t1]
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    def gate_rz(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= complex(math.cos(theta), math.sin(theta))
        return ctx

    def gate_t(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices

        sqrt2_inv = 1 / np.sqrt(2)
        factor = complex(sqrt2_inv, sqrt2_inv)
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= factor
        return ctx

    def gate_s(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= 1.j
        return ctx

    def gate_u1(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        lambd = gate.lambd
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= complex(math.cos(lambd), math.sin(lambd))
        return ctx

    def gate_u3(self, gate, ctx):
        qubits = ctx.qubits
        newq = ctx.qubits_buf
        n_qubits = ctx.n_qubits
        i = ctx.indices
        theta = gate.theta
        phi = gate.phi
        lambd = gate.lambd
        a00 = math.cos(theta / 2)
        a11 = a00 * complex(math.cos(phi + lambd), math.sin(phi + lambd))
        a01 = a10 = math.sin(theta / 2)
        a01 *= complex(math.cos(lambd), math.sin(lambd))
        a10 *= complex(math.cos(phi), math.sin(phi))
        for target in gate.target_iter(n_qubits):
            np.copyto(newq, qubits)
            t0 = (i & (1 << target)) == 0
            t1 = (i & (1 << target)) != 0
            newq[t0] = qubits[t0] * a00
            newq[t0] -= qubits[t1] * a01
            newq[t1] = qubits[t0] * a10
            newq[t1] += qubits[t1] * a11
            qubits, newq = newq, qubits
        ctx.qubits = qubits
        ctx.qubits_buf = newq
        return ctx

    def gate_measure(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            p_zero = np.linalg.norm(qubits[(i & (1 << target)) == 0]) ** 2
            rand = random.random()
            if rand < p_zero:
                qubits[(i & (1 << target)) != 0] = 0.0
                qubits /= np.sqrt(p_zero)
                ctx.cregs[target] = 0
            else:
                qubits[(i & (1 << target)) == 0] = 0.0
                qubits /= np.sqrt(1.0 - p_zero)
                ctx.cregs[target] = 1
        ctx.save_cache = False
        return ctx


def _ignore_globals(qubits):
    for q in qubits:
        if abs(q) > 0.0000001:
            ang = abs(q) / q
            qubits *= ang
            break
    return qubits
