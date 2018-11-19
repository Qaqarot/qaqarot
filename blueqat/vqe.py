from collections import Counter, defaultdict
from functools import reduce
import itertools
import random
import warnings

import numpy as np
from scipy.optimize import minimize as scipy_minimizer
from .circuit import Circuit
from .utils import to_inttuple

class AnsatzBase:
    def __init__(self, hamiltonian, n_params):
        self.hamiltonian = hamiltonian
        self.n_params = n_params
        self.n_qubits = self.hamiltonian.max_n() + 1

    def get_circuit(self, params):
        """Make a circuit from parameters."""
        raise NotImplementedError

    def get_energy(self, circuit, sampler):
        """Calculate energy from circuit and sampler."""
        val = 0.0
        for meas in self.hamiltonian:
            c = circuit.copy()
            for op in meas.ops:
                if op.op == "X":
                    c.h[op.n]
                elif op.op == "Y":
                    c.rx(-np.pi / 2)[op.n]
            measured = sampler(c, meas.n_iter())
            for bits, prob in measured.items():
                if sum(bits) % 2:
                    val -= prob * meas.coeff
                else:
                    val += prob * meas.coeff
        return val.real

    def get_objective(self, sampler):
        """Get an objective function to be optimized."""
        def objective(params):
            circuit = self.get_circuit(params)
            circuit.make_cache()
            return self.get_energy(circuit, sampler)
        return objective

class QaoaAnsatz(AnsatzBase):
    def __init__(self, hamiltonian, step=1, init_circuit=None):
        super().__init__(hamiltonian, step * 2)
        self.hamiltonian = hamiltonian.to_expr().simplify()
        if not self.check_hamiltonian():
            raise ValueError("Hamiltonian terms are not commutable")

        self.step = step
        self.n_qubits = self.hamiltonian.max_n() + 1
        if init_circuit:
            self.init_circuit = init_circuit
            if init_circuit.n_qubits > self.n_qubits:
                self.n_qubits = init_circuit.n_qubits
        else:
            self.init_circuit = Circuit(self.n_qubits).h[:]
        self.init_circuit.make_cache()
        self.time_evolutions = [term.get_time_evolution() for term in self.hamiltonian]

    def check_hamiltonian(self):
        """Check hamiltonian is commutable. This condition is required for QaoaAnsatz"""
        return self.hamiltonian.is_all_terms_commutable()

    def get_circuit(self, params):
        c = self.init_circuit.copy()
        betas = params[:self.step]
        gammas = params[self.step:]
        for beta, gamma in zip(betas, gammas):
            beta *= np.pi
            gamma *= 2 * np.pi
            for evo in self.time_evolutions:
                evo(c, gamma)
            c.rx(beta)[:]
        return c

class VqeResult:
    def __init__(self, vqe=None, params=None, circuit=None):
        self.vqe = vqe
        self.params = params
        self.circuit = circuit
        self._probs = None

    def most_common(self, n=1):
        return tuple(sorted(self.get_probs().items(), key=lambda item: -item[1]))[:n]

    @property
    def probs(self):
        """Get probabilities. This property is obsoleted. Use get_probs()."""
        warnings.warn("VqeResult.probs is obsoleted. " +
                      "Use VqeResult.get_probs().", DeprecationWarning)
        return self.get_probs()

    def get_probs(self, sampler=None, rerun=None, store=True):
        """Get probabilities."""
        if rerun is None:
            rerun = sampler is not None
        if self._probs is not None and not rerun:
            return self._probs
        if sampler is None:
            sampler = self.vqe.sampler
        probs = sampler(self.circuit, range(self.circuit.n_qubits))
        if store:
            self._probs = probs
        return probs


class Vqe:
    def __init__(self, ansatz, minimizer=None, sampler=None):
        self.ansatz = ansatz
        self.minimizer = minimizer or get_scipy_minimizer(
            method="Powell",
            options={"ftol": 5.0e-2, "xtol": 5.0e-2, "maxiter": 1000}
        )
        self.sampler = sampler or non_sampling_sampler
        self._result = None

    def run(self, verbose=False):
        objective = self.ansatz.get_objective(self.sampler)
        if verbose:
            def verbose_objective(objective):
                def f(params):
                    val = objective(params)
                    print("params:", params, "val:", val)
                    return val
                return f
            objective = verbose_objective(objective)
        params = self.minimizer(objective, self.ansatz.n_params)
        c = self.ansatz.get_circuit(params)
        return VqeResult(self, params, c)

    @property
    def result(self):
        """Vqe.result is deprecated. Use `result = Vqe.run()`."""
        warnings.warn("Vqe.result is deprecated. Use `result = Vqe.run()`",
                      DeprecationWarning)
        return self._result if self._result is not None else VqeResult()

def get_scipy_minimizer(**kwargs):
    """Get minimizer which uses `scipy.optimize.minimize`"""
    def minimizer(objective, n_params):
        params = [random.random() for _ in range(n_params)]
        result = scipy_minimizer(objective, params, **kwargs)
        return result.x
    return minimizer

def expect(qubits, meas):
    "For the VQE simulation without sampling."
    result = {}
    i = np.arange(len(qubits))
    meas = tuple(meas)

    def to_mask(n):
        return reduce(lambda acc, im: acc | (n & (1 << im[0])) << (im[1] - im[0]), enumerate(meas), 0)

    mask = reduce(lambda acc, v: acc | (1 << v), meas, 0)

    cnt = defaultdict(float)
    for i, v in enumerate(qubits):
        cnt[i & mask] += (v * v.conjugate()).real
    #print("m:", meas)
    #print("cnt:", dict(cnt))

    #for i in range(2**len(meas)):
    #    print(f"to_mask({i}) = {to_mask(i)}")

    ret = {m: cnt[to_mask(i)] for i, m in enumerate(itertools.product((0, 1), repeat=len(meas)))}
    #print("ret:", ret)
    return ret

def non_sampling_sampler(circuit, meas):
    """Calculate the expectations without sampling."""
    meas = tuple(meas)
    if len(meas) == circuit.n_qubits and meas == tuple(range(circuit.n_qubits)):
        qubits = circuit.run(returns="statevector")
        probs = (qubits.conjugate() * qubits).real
        return {tuple(map(int, prod[::-1])): val \
                for prod, val in zip(itertools.product("01", repeat=circuit.n_qubits), probs) if val}
    return expect(circuit.run(returns="statevector"), meas)

def get_measurement_sampler(n_sample, run_options=None):
    """Returns a function which get the expectations by sampling the measured circuit"""
    if run_options is None:
        run_options = {}

    def sampling_by_measurement(circuit, meas):
        def reduce_bits(bits, meas):
            bits = [int(x) for x in bits[::-1]]
            return tuple(bits[m] for m in meas)

        meas = tuple(meas)
        circuit.measure[meas]
        counter = circuit.run(shots=n_sample, returns="shots", **run_options)
        counts = Counter({reduce_bits(bits, meas): val for bits, val in counter.items()})
        return {k: v / n_sample for k, v in counts.items()}

    return sampling_by_measurement

def get_state_vector_sampler(n_sample):
    """Returns a function which get the expectations by sampling the state vector"""
    def sampling_by_measurement(circuit, meas):
        val = 0.0
        e = expect(circuit.run(returns="statevector"), meas)
        bits, probs = zip(*e.items())
        dists = np.random.multinomial(n_sample, probs) / n_sample
        return dict(zip(tuple(bits), dists))
    return sampling_by_measurement

def get_qiskit_sampler(backend, **execute_kwargs):
    """Returns a function which get the expectation by sampling via Qiskit.

    This function requires `qiskit` module.
    """
    try:
        import qiskit
    except ImportError:
        raise ImportError("blueqat.vqe.get_qiskit_sampler() requires qiskit. Please install before call this function.")
    try:
        shots = execute_kwargs['shots']
    except KeyError:
        execute_kwargs['shots'] = shots = 1024

    def reduce_bits(bits, meas):
        bits = [int(x) for x in bits[::-1]]
        return tuple(bits[m] for m in meas)

    def sampling(circuit, meas):
        meas = tuple(meas)
        circuit.measure[meas]
        qasm = circuit.to_qasm()
        qk_circuit = qiskit.load_qasm_string(qasm)
        result = qiskit.execute(qk_circuit, backend, **execute_kwargs).result()
        counts = Counter({reduce_bits(bits, meas): val for bits, val in result.get_counts().items()})
        return {k: v / shots for k, v in counts.items()}

    return sampling
