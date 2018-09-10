from collections import Counter
import itertools
import random

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
        betas = params[:self.step]
        gammas = params[self.step:]
        for beta, gamma in zip(betas, gammas):
            beta *= np.pi
            gamma *= np.pi
            for evo in self.time_evolutions:
                evo(c, beta)
            c.rx(gamma)[:]
        return c

    def get_objective(self, sampler):
        def objective(params):
            circuit = self.get_circuit(params)
            circuit.run()
            val = 0.0
            for meas in self.hamiltonian:
                c = circuit.copy(copy_cache=True, copy_history=False)
                for op in meas.ops:
                    if op.op == "X":
                        c.h[op.n]
                    elif op.op == "Y":
                        c.rx(-np.pi / 2)[op.n]
                measured = sampler(c, meas.n_iter())
                #print("sampler out:", measured)
                for bits, prob in measured.items():
                    if sum(bits) % 2:
                        val -= prob * meas.coeff
                    else:
                        val += prob * meas.coeff
            #print(params, val)
            return val
        return objective

class VqeResult:
    def __init__(self, params=None, circuit=None, probs=None):
        self.params = params
        self.circuit = circuit
        self.probs = probs

    def __repr__(self):
        return "VqeResult" + repr((self.params, self.circuit, self.probs))

    def most_common(self, n=1):
        return tuple(sorted(self.probs.items(), key=lambda item: -item[1]))[:n]

class Vqe:
    def __init__(self, ansatz, minimizer=None, sampler=None):
        self.ansatz = ansatz
        self.minimizer = minimizer or get_scipy_minimizer(
            method="Powell",
            options={"ftol": 5.0e-2, "xtol": 5.0e-2, "maxiter": 1000}
        )
        self.sampler = sampler or non_sampling_sampler
        self.result = VqeResult()

    def run(self):
        objective = self.ansatz.get_objective(self.sampler)
        params = self.minimizer(objective, self.ansatz.n_params())
        c = self.ansatz.get_circuit(params)
        self.result.params = params
        self.result.probs = self.sampler(c, range(self.ansatz.n_qubits))
        self.result.circuit = c
        return self.result

def get_scipy_minimizer(**kwargs):
    """Get minimizer which uses `scipy.optimize.minimize`"""
    def minimizer(objective, n_params):
        params = [random.random() for _ in range(n_params)]
        result = scipy_minimizer(objective, params, **kwargs)
        return result.x
    return minimizer

def expect(qubits, meas):
    "For the VQE simulation without sampling."
    d = {"": (qubits, 1.0)}
    result = {}
    i = np.arange(len(qubits))
    meas = tuple(meas)
    #print("from qubits:", (qubits.conjugate() * qubits).real)

    def get(bits):
        if bits in d:
            return d[bits]
        n = len(bits)
        m = meas[n - 1]
        qb, p = get(bits[:-1])
        if p == 0.0:
            d[bits[:-1] + "0"] = qb, 0.0
            d[bits[:-1] + "1"] = qb, 0.0
            return d[bits]

        p_zero = (qb[(i & (1 << m)) == 0].T.conjugate() @ qb[(i & (1 << m)) == 0]).real

        try:
            factor = 1.0 / np.sqrt(p_zero)
            if not np.isfinite(factor):
                raise ZeroDivisionError
            qb_zero = qb.copy()
            qb_zero[(i & (1 << m)) != 0] = 0.0
            qb_zero *= factor
            d[bits[:-1] + "0"] = qb_zero, p * p_zero
        except ZeroDivisionError:
            d[bits[:-1] + "0"] = qb, 0.0

        try:
            factor = 1.0 / np.sqrt(1.0 - p_zero)
            if not np.isfinite(factor):
                raise ZeroDivisionError
            qb_one = qb.copy()
            qb_one[(i & (1 << m)) == 0] = 0.0
            qb_one *= factor
            d[bits[:-1] + "1"] = qb_one, p * (1.0 - p_zero)
        except ZeroDivisionError:
            d[bits[:-1] + "1"] = qb, 0.0
        return d[bits]

    for m in itertools.product("01", repeat=len(meas)):
        m = "".join(m)
        v = get(m)
        if v[1]:
            result[m] = v[1]
    return {tuple(map(int, k)): v for k, v in result.items()}

def non_sampling_sampler(circuit, meas):
    """Calculate the expectations without sampling."""
    return expect(circuit.run(), meas)

def get_measurement_sampler(n_sample):
    """Returns a function which get the expectations by sampling the measured circuit"""
    def sampling_by_measurement(circuit, meas):
        circuit.measure[tuple(meas)]
        for _ in range(n_sample):
            circuit.run()
        counter = Counter(circuit.run_history)
        return {k: v / n_sample for k, v in counter.items()}
    return sampling_by_measurement

def get_state_vector_sampler(n_sample):
    """Returns a function which get the expectations by sampling the state vector"""
    def sampling_by_measurement(circuit, meas):
        val = 0.0
        e = expect(circuit.run(), meas)
        bits, probs = zip(*e.items())
        dists = np.random.multinomial(n_sample, probs) / n_sample
        return dict(zip(tuple(map(int, bits)), dists))
    return sampling_by_measurement
