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
        self.sparse = None

    def make_sparse(self, fmt='csc', make_method=None):
        """Make sparse matrix. This method may be changed in the future release."""
        if make_method:
            self.sparse = make_method(self.hamiltonian)
        else:
            self.sparse = self.hamiltonian.to_matrix(sparse=fmt)

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

    def get_energy_sparse(self, circuit):
        """Get energy using sparse matrix. This method may be changed in the future release."""
        return sparse_expectation(self.sparse, circuit.run())

    def get_objective(self, sampler=None):
        """Get an objective function to be optimized."""
        def objective(params):
            circuit = self.get_circuit(params)
            circuit.make_cache()
            return self.get_energy(circuit, sampler)

        def obj_expect(params):
            circuit = self.get_circuit(params)
            circuit.make_cache()
            return self.get_energy_sparse(circuit)

        if sampler is not None:
            return objective
        if self.sparse is None:
            self.make_sparse()
        return obj_expect


class QaoaAnsatz(AnsatzBase):
    """Ansatz for QAOA."""
    def __init__(self, hamiltonian, step=1, init_circuit=None, mixer=None):
        """
        Args:
            hamiltonian (Pauli expr): Hamiltonian for optimization.
            step (optional, int): The number of step for repeat the circuit.
                Generally, larger is precise and slow.
            init_circuit (optional, Circuit): Initial circuit for QAOA.
                By default, `Circuit().h[:]` is used.
                If mixer is specified, this argument is required.
            mixer (optional, list of Pauli expr): Mixer for Quantum Alternative Operator Ansatz.
                By default, mixer is None and normal Quantum Approximate Optimization Algorithm is used.
                This feature is experimental. It may be modified.
        """
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
            if mixer:
                raise ValueError(
                    'init_circuit is required when mixer is not default.')
            self.init_circuit = Circuit(self.n_qubits).h[:]
        self.mixer = mixer
        self.init_circuit.make_cache()
        self.time_evolutions = [
            term.get_time_evolution() for term in self.hamiltonian
        ]
        self.mixer_time_evolutions = [
            term.get_time_evolution() for term in self.mixer
        ] if mixer else []

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
            if self.mixer is None:
                c.rx(beta)[:]
            else:
                for evo in self.mixer_time_evolutions:
                    evo(c, beta)
        return c


class VqeResult:
    def __init__(self, vqe=None, params=None, circuit=None):
        self.vqe = vqe
        self.params = params
        self.circuit = circuit
        self._probs = None

    def most_common(self, n=1):
        return tuple(
            sorted(self.get_probs().items(), key=lambda item: -item[1]))[:n]

    @property
    def probs(self):
        """Get probabilities. This property is obsoleted. Use get_probs()."""
        warnings.warn(
            "VqeResult.probs is obsoleted. " + "Use VqeResult.get_probs().",
            DeprecationWarning)
        return self.get_probs()

    def get_probs(self, sampler=None, rerun=None, store=True):
        """Get probabilities."""
        if rerun is None:
            rerun = sampler is not None
        if self._probs is not None and not rerun:
            return self._probs
        if sampler is None:
            sampler = self.vqe.sampler

        if sampler is None:
            probs = expect(self.circuit.run(returns="statevector"),
                           range(self.circuit.n_qubits))
        else:
            probs = sampler(self.circuit, range(self.circuit.n_qubits))
        if store:
            self._probs = probs
        return probs


class Vqe:
    def __init__(self, ansatz, minimizer=None, sampler=None):
        self.ansatz = ansatz
        self.minimizer = minimizer or get_scipy_minimizer(method="Powell",
                                                          options={
                                                              "ftol": 5.0e-2,
                                                              "xtol": 5.0e-2,
                                                              "maxiter": 1000
                                                          })
        self.sampler = sampler

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
        return reduce(
            lambda acc, im: acc | (n & (1 << im[0])) << (im[1] - im[0]),
            enumerate(meas), 0)

    def to_key(k):
        return tuple(1 if k & (1 << i) else 0 for i in meas)

    mask = reduce(lambda acc, v: acc | (1 << v), meas, 0)

    cnt = defaultdict(float)
    for i, v in enumerate(qubits):
        p = v.real**2 + v.imag**2
        if p != 0.0:
            cnt[i & mask] += p
    return {to_key(k): v for k, v in cnt.items()}


def non_sampling_sampler(circuit, meas):
    """Calculate the expectations without sampling."""
    meas = tuple(meas)
    n_qubits = circuit.n_qubits
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
        counts = Counter(
            {reduce_bits(bits, meas): val
             for bits, val in counter.items()})
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
        raise ImportError(
            "blueqat.vqe.get_qiskit_sampler() requires qiskit. Please install before call this function."
        )
    try:
        shots = execute_kwargs['shots']
    except KeyError:
        execute_kwargs['shots'] = shots = 1024

    def reduce_bits(bits, meas):
        # In Qiskit 0.6.1, For example
        # Aer backend returns bit string and IBMQ backend returns hex string.
        # Sample code:
        """
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ
IBMQ.load_accounts()
q, c = QuantumRegister(4, 'q'), ClassicalRegister(4, 'c')
circ = QuantumCircuit(q, c)
circ.x(q[1])
for i in range(4):
    circ.measure(q[i], c[i])
print("Aer qasm_simulator_py")
print(execute(circ, Aer.get_backend('qasm_simulator_py')).result().get_counts())
print("IBMQ ibmq_qasm_simulator")
print(execute(circ, IBMQ.get_backend('ibmq_qasm_simulator')).result().get_counts())
        """
        # The result is,
        # Aer: {'0010': 1024}
        # IBMQ: {'0x2': 1024}
        # This is workaround for this IBM's specifications.
        if bits.startswith("0x"):
            bits = int(bits, base=16)
            bits = "0" * 100 + format(bits, "b")
        bits = [int(x) for x in bits[::-1]]
        return tuple(bits[m] for m in meas)

    def sampling(circuit, meas):
        meas = tuple(meas)
        if not meas:
            return {}
        circuit.measure[meas]
        result = circuit.run_with_ibmq(qiskit_backend=backend,
                                       returns="qiskit_result",
                                       **execute_kwargs)
        counts = Counter({
            reduce_bits(bits, meas): val
            for bits, val in result.get_counts().items()
        })
        return {k: v / shots for k, v in counts.items()}

    return sampling


def sparse_expectation(mat, vec):
    """Calculate expectation value <vec|mat|vec>.

    Args:
        mat (scipy sparse matrix): Sparse matrix
        vec (numpy array): Vector

    Returns:
        (Real part of) expectation value <vec|mat|vec>.
        Remarks: when mat is Hermitian, <vec|mat|vec> is real.
    """
    return np.vdot(vec, mat.dot(vec)).real
