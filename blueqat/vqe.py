from collections import Counter
import itertools
from math import pi

import numpy as np
from scipy.optimize import minimize as scipy_minimizer
from .circuit import Circuit

class QaoaAnsatz:
    def __init__(self, hamiltonian, step=1, init_circuit=None):
        self.hamiltonian = hamiltonian.simplify()
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
        self.init_circuit.run() # To make a cache.
        self.time_evolutions = [term.get_time_evolution() for term in self.hamiltonian]

    def check_hamiltonian(self):
        return self.hamiltonian.is_all_terms_commutable()

    def n_params(self):
        return self.step * 2

    def get_circuit(self, params):
        c = self.init_circuit.copy()
        betas = params[:len(params)/2]
        gammas = params[len(params)/2:]
        for beta, gamma in zip(betas, gammas):
            for evo in self.time_evolutions:
                evo(c, gamma)
            c.rx(beta)[:]
        return c

    def get_objective(self, sampler):
        def objective(params):
            c = self.get_circuit(params)
            return sampler(c, self.hamiltonian.terms)
        return objective

class VqeResult:
    def __init__(self, params=None, circuit=None, probs=None):
        self.params = params
        self.circuit = circuit
        self.probs = probs

class Vqe:
    def __init__(self, ansatz, minimizer=None, sampler=None):
        self.ansatz = ansatz
        self.minimizer = minimizer
        self.sampler = sampler
        self.result = VqeResult()

    def run(self):
        objective = self.ansatz.get_objective(self.sampler)
        params = self.minimizer(objective)
        c = self.ansatz.get_circuit(params)
        self.result.params = params
        self.result.circuit = c
        self.result.probs = self.sampler(c)
        return self.result

def get_scipy_minimizer(**kwargs):
    """Get minimizer which uses `scipy.optimize.minimize`"""
    def minimizer(objective, params):
        result = scipy_minimizer(objective, params, **kwargs)
        return result.x
    return minimizer

def expect(qubits, meas):
    "For the VQE simulation without sampling."
    d = {"": (qubits, 1.0)}
    result = {}
    i = np.arange(len(qubits))

    def get(bits):
        if bits in d:
            return d[bits]
        n = len(bits)
        m = meas[n - 1]
        qb, p = get(bits[:-1])
        p_zero = (qb[(i & (1 << m)) == 0].T.conjugate() @ qubits[(i & (1 << m)) == 0]).real

        qb_zero = qb.copy()
        qb_zero[(i & (1 << m)) != 0] = 0.0
        qb_zero /= np.sqrt(p_zero)
        d[bits[:-1] + "0"] = qb_zero, p * p_zero

        qb_one = qb.copy()
        qb_one[(i & (1 << m)) == 0] = 0.0
        qb_one /= np.sqrt(1.0 - p_zero)
        d[bits[:-1] + "1"] = qb_one, p * (1 - p_zero)
        return d[bits]

    for m in itertools.product("01", repeat=len(meas)):
        m = "".join(m)
        result[m] = get(m)[1]
    return result

def non_sampling_sampler(circuit, measures):
    """Calculate the expectations without sampling."""
    circuit.run()
    val = 0.0
    for meas in measures:
        c = circuit.copy(copy_cache=True, copy_history=False)
        for op in meas.ops:
            if op.op == "X":
                c.h[op.n]
            elif op.op == "Y":
                c.rx(-pi / 2)[op.n]
        e = expect(c.run(), meas)
        for bits, p in e.items():
            if bits.count("1") % 2:
                val -= p
            else:
                val += p
    return val

def get_measurement_sampler(n_sample):
    """Returns a function which get the expectations by sampling the measured circuit"""
    def sampling_by_measurement(circuit, measures):
        circuit.run()
        val = 0.0
        for meas in measures:
            c = circuit.copy(copy_cache=True, copy_history=False)
            for op in meas.ops:
                if op.op == "X":
                    c.h[op.n]
                elif op.op == "Y":
                    c.rx(-pi / 2)[op.n]
            c.measure[tuple(meas.ops.n_iter())]
            for _ in range(n_sample):
                c.run()
            counter = Counter(sum(e) % 2 for e in c.run_history)
            for bits, cnt in counter.items():
                if bits:
                    val -= cnt / n_sample
                else:
                    val += cnt / n_sample
        return val
    return sampling_by_measurement

def get_state_vector_sampler(n_sample):
    """Returns a function which get the expectations by sampling the state vector"""
    def sampling_by_measurement(circuit, measures):
        circuit.run(copy_cache=True, copy_history=False)
        val = 0.0
        for meas in measures:
            c = circuit.copy()
            for op in meas.ops:
                if op.op == "X":
                    c.h[op.n]
                elif op.op == "Y":
                    c.rx(-pi / 2)[op.n]
            e = expect(c.run(), meas)
            pvals = []
            bits = []
            for k, v in e.items():
                bits.append(k)
                pvals.append(v)
            vals = np.array([k.count("1") % 2 for k in bits], dtype=np.float64) * -2 + 1
            vals *= np.random.multinomial(n_sample, pvals)
            vals /= n_sample
            val += np.sum(vals)
        return val
    return sampling_by_measurement
