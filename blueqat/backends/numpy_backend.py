import math
import random
import numpy as np
from ..gate import *
from .backendbase import Backend

DEFAULT_DTYPE = np.complex128

class _NumPyBackendContext:
    """This class is internally used in NumPyBackend"""
    def __init__(self, n_qubits, cache, cache_idx, save_cache=True):
        self.n_qubits = n_qubits
        if cache is not None:
            self.qubits = cache.copy()
        else:
            self.qubits = np.zeros(2**n_qubits, dtype=DEFAULT_DTYPE)
            self.qubits[0] = 1.0

        self.indices = np.arange(2**n_qubits, dtype=np.uint32)
        self.cregs = [0] * n_qubits
        self.save_cache = save_cache

class NumPyBackend(Backend):
    """Simulator backend which uses numpy. This backend is Blueqat's default backend."""

    def __init__(self):
        self.cache = None
        self.cache_idx = -1

        # run_history will be deprecated.
        self.run_history = []

    def __save_cache(self, ctx):
        if ctx.save_cache:
            self.cache = ctx.qubits.copy()
            self.cache_idx += 1

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

    def _preprocess_run(self, gates, _args, _kwargs):
        n_qubits = find_n_qubits(gates)
        self.__clear_cache_if_invalid(n_qubits, DEFAULT_DTYPE)
        ctx = _NumPyBackendContext(n_qubits, self.cache, self.cache_idx)
        return self._resolve_fallback(gates)[self.cache_idx + 1:], ctx

    def _postprocess_run(self, ctx):
        if ctx.n_qubits:
            self.run_history.append(tuple(ctx.cregs))
        return _ignore_globals(ctx.qubits)

    def run(self, gates, args, kwargs):
        return self._run(gates, args, kwargs)

    def gate_x(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            newq = np.zeros_like(qubits)
            newq[(i & (1 << target)) == 0] = qubits[(i & (1 << target)) != 0]
            newq[(i & (1 << target)) != 0] = qubits[(i & (1 << target)) == 0]
            qubits = newq
        ctx.qubits = qubits
        self.__save_cache(ctx)
        return ctx

    def gate_y(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            newq = np.zeros_like(qubits)
            newq[(i & (1 << target)) == 0] = -1.0j * qubits[(i & (1 << target)) != 0]
            newq[(i & (1 << target)) != 0] = 1.0j * qubits[(i & (1 << target)) == 0]
            qubits = newq
        ctx.qubits = qubits
        self.__save_cache(ctx)
        return ctx

    def gate_z(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= -1
        self.__save_cache(ctx)
        return ctx

    def gate_h(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            newq = np.zeros_like(qubits)
            newq[(i & (1 << target)) == 0] = qubits[(i & (1 << target)) == 0] + qubits[(i & (1 << target)) != 0]
            newq[(i & (1 << target)) != 0] = qubits[(i & (1 << target)) == 0] - qubits[(i & (1 << target)) != 0]
            newq /= np.sqrt(2)
            qubits = newq
        ctx.qubits = qubits
        self.__save_cache(ctx)
        return ctx

    def gate_cz(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for control, target in gate.control_target_iter(n_qubits):
            qubits[((i & (1 << control)) != 0) & ((i & (1 << target)) != 0)] *= -1
        self.__save_cache(ctx)
        return ctx

    def gate_cx(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for control, target in gate.control_target_iter(n_qubits):
            newq = qubits.copy()
            newq[((i & (1 << control)) != 0) & ((i & (1 << target)) != 0)] = qubits[((i & (1 << control)) != 0) & ((i & (1 << target)) == 0)]
            newq[((i & (1 << control)) != 0) & ((i & (1 << target)) == 0)] = qubits[((i & (1 << control)) != 0) & ((i & (1 << target)) != 0)]
            qubits = newq
        ctx.qubits = qubits
        self.__save_cache(ctx)
        return ctx

    def gate_rx(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            newq = np.zeros_like(qubits)
            newq[(i & (1 << target)) == 0] = (
                np.cos(theta / 2) * qubits[(i & (1 << target)) == 0] +
                -1.0j * np.sin(theta / 2) * qubits[(i & (1 << target)) != 0]
            )
            newq[(i & (1 << target)) != 0] = (
                -1.0j * np.sin(theta / 2) * qubits[(i & (1 << target)) == 0] +
                np.cos(theta / 2) * qubits[(i & (1 << target)) != 0]
            )
            qubits = newq
        ctx.qubits = qubits
        self.__save_cache(ctx)
        return ctx

    def gate_ry(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            newq = np.zeros_like(qubits)
            newq[(i & (1 << target)) == 0] = (
                np.cos(theta / 2) * qubits[(i & (1 << target)) == 0] +
                -np.sin(theta / 2) * qubits[(i & (1 << target)) != 0]
            )
            newq[(i & (1 << target)) != 0] = (
                np.sin(theta / 2) * qubits[(i & (1 << target)) == 0] +
                np.cos(theta / 2) * qubits[(i & (1 << target)) != 0]
            )
            qubits = newq
        ctx.qubits = qubits
        self.__save_cache(ctx)
        return ctx

    def gate_rz(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        theta = gate.theta
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= complex(math.cos(theta), math.sin(theta))
        self.__save_cache(ctx)
        return ctx

    def gate_t(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices

        sqrt2_inv = 1 / np.sqrt(2)
        factor = complex(sqrt2_inv, sqrt2_inv)
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= factor
        self.__save_cache(ctx)
        return ctx

    def gate_s(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            qubits[(i & (1 << target)) != 0] *= 1.j
        self.__save_cache(ctx)
        return qubits

    def gate_measure(self, gate, ctx):
        qubits = ctx.qubits
        n_qubits = ctx.n_qubits
        i = ctx.indices
        for target in gate.target_iter(n_qubits):
            p_zero = (qubits[(i & (1 << target)) == 0].T.conjugate() @ qubits[(i & (1 << target)) == 0]).real
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
