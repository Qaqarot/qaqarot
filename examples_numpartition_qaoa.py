from collections import Counter
import itertools
import numpy as np
from scipy.optimize import minimize
from circuit import Circuit

def numpartition_qaoa(n_step, edges, minimizer=None, sampler=None, verbose=True):
    """Do the Number partition QAOA.

    :param n_step: The number of step of QAOA
    :param n_sample: The number of sampling time of each measurement in VQE.
                     If None, use calculated ideal value.
    :param edges: The edges list of the graph.
    :returns result of QAOA
    """
    ma = NumPartitionQaoaCalculator(n_step, edges, minimizer, sampler, verbose)
    return ma.result

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

def get_scipy_minimizer(**kwargs):
    return lambda fun, x0: minimize(fun, x0, **kwargs)

def calculate_from_state_vector(circuit, measures):
    """Calculate the expectations without sampling."""
    qubits = circuit.run()
    val = 0.0
    for meas in measures:
        e = expect(qubits, meas)
        for bits, p in e.items():
            if bits.count("1") % 2:
                val -= p
            else:
                val += p
    return val

def get_measurement_sampler(n_sample):
    """Returns a function which get the expectations by sampling the measured circuit"""
    def sampling_by_measurement(circuit, measures):
        val = 0.0
        for meas in measures:
            c = circuit.copy(copy_cache=True, copy_history=False)
            c.measure[tuple(meas)]
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
        qubits = circuit.run()
        val = 0.0
        for meas in measures:
            e = expect(qubits, meas)
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

class NumPartitionQaoaCalculator:
    def __init__(self, n_step, nums, minimizer=None, sampler=None, verbose=True):
        self.n_step = n_step
        self.nums = nums
        self.verbose = verbose

        self.n_query = 0
        self.query_history = []

        self.sampler = sampler or calculate_from_state_vector

        self.circuit = None

        n_qubits = len(nums)
        self.n_qubits = n_qubits

        obj_f = self.get_objective_func()
        params = np.random.rand(2 * n_step) * np.pi
        params[:n_step] *= 2

        if minimizer is None:
            minimizer = get_scipy_minimizer(
                method="Powell",
                options={"ftol": 2.0e-2, "xtol": 2.0e-2, "disp": self.verbose}
            )
        optimized = minimizer(obj_f, params)
        gammas = optimized.x[:len(params)//2]
        betas = optimized.x[len(params)//2:]
        if self.verbose:
            print("initial   params:", params)
            print("optimized params:", optimized.x)
        c = self.get_circuit(gammas, betas)
        qubits = c.run()
        #print("qubits:")
        #print(qubits)
        p = (qubits.conjugate() * qubits).real
        #print("probabilities:")
        #print(p)
        maxi = p.argmax()
        maxi_bitstring = ("{:0" + str(n_qubits) + "b}").format(maxi)[::-1]
        self.result = list(maxi_bitstring)
        if self.verbose:
            print("Most significant index:", maxi_bitstring)

    def get_circuit(self, gammas, betas):
        c = Circuit(self.n_qubits)
        c.h[:]
        for gamma, beta in zip(gammas, betas):
            for i,x1 in enumerate(self.nums):
                for j,x2 in enumerate(self.nums[i:]):
                    ang = beta * (x1 * x2)
                    if i != j:
                        ang += ang
                    c.cz(ang)[i, j]
            c.rx(-2.0 * beta)[:]
        return c

    def get_objective_func(self):
        def obj_f(params):
            self.n_query += 1
            sampler = self.sampler
            gammas = params[:len(params)//2]
            betas = params[len(params)//2:]
            val = 0.0
            c = self.get_circuit(gammas, betas)
            for i,x1 in enumerate(self.nums):
                for j,x2 in enumerate(self.nums):
                    factor = x1 * x2
                    if i != j:
                        factor += factor
                    val += factor * sampler(c, ((x1, x2),))
            val += sampler(c, self.edges)
            if self.verbose:
                print("params:", params)
                print("val:", val)
            self.query_history.append((params, val))
            return val
        return obj_f

if __name__ == "__main__":
    #result = numpartition_qaoa(2, [3,2,6,9,2,5,7,3,3,6,7,3,5,3,2,2,2,6,8,4,6,3,3,6,4,3,3,2,2,5,8,9])
    result = numpartition_qaoa(2, [3,2,6,9,2,5,7,3,3,6,7,3,5,3,2,2,2,6,8,4,6,3,3,6,4,3])
    print(result)
