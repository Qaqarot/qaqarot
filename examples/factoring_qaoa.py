import itertools
from collections import Counter, OrderedDict

import numpy as np
from scipy.optimize import minimize

from blueqat.pauli import ising_bit
from blueqat import Circuit

def factoring_qaoa(n_step, n, minimizer=None, sampler=None, verbose=True):
    """Do the Number partition QAOA.

    :param n_step: The number of step of QAOA
    :param n_sample: The number of sampling time of each measurement in VQE.
                     If None, use calculated ideal value.
    :param edges: The edges list of the graph.
    :returns result of QAOA
    """
    ma = FactoringQaoaCalculator(n_step, n, minimizer, sampler, verbose)
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
        if p_zero > 0.0000000001:
            qb_zero /= np.sqrt(p_zero)
            d[bits[:-1] + "0"] = qb_zero, p * p_zero
        else:
            d[bits[:-1] + "0"] = qb_zero, 0.0

        qb_one = qb.copy()
        qb_one[(i & (1 << m)) == 0] = 0.0
        if 1.0 - p_zero > 0.0000000001:
            qb_one /= np.sqrt(1.0 - p_zero)
            d[bits[:-1] + "1"] = qb_one, p * (1.0 - p_zero)
        else:
            d[bits[:-1] + "1"] = qb_one, 0.0
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
    #print(e, val)
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

class FactoringQaoaCalculator:
    def __init__(self, n_step, num, minimizer=None, sampler=None, verbose=True):
        self.n_step = n_step
        self.num = num
        self.verbose = verbose

        self.n_query = 0
        self.query_history = []

        self.sampler = sampler or calculate_from_state_vector

        self.circuit = None

        def get_nbit(n):
            m = 1
            while 2**m < n:
                m += 1
            return m
        self.n1_bits = get_nbit(int(num**0.5)) - 1
        self.n2_bits = get_nbit(int(num**0.5))
        self.n_qubits = self.n1_bits + self.n2_bits

        def mk_expr(start, n):
            expr = 1
            for i in range(start, n):
                expr = expr + 2**(i-start+1) * ising_bit(i)
            return expr
        self.qubo = (num - mk_expr(0, self.n1_bits) * mk_expr(self.n1_bits, self.n_qubits))**2
        self.h = self.qubo.simplify()

        #factor = 1 / max(abs(x) for x in self.h.values())
        #for k in self.h:
        #    self.h[k] *= factor


        obj_f = self.get_objective_func()
        params = np.random.rand(2 * n_step) * np.pi
        params[:n_step] *= 2

        if minimizer is None:
            minimizer = get_scipy_minimizer(
                method="Powell",
                options={"ftol": 5.0e-2, "xtol": 5.0e-2, "maxiter": 1000, "disp": self.verbose}
            )
        optimized = minimizer(obj_f, params)
        gammas = optimized.x[:len(params)//2]
        betas = optimized.x[len(params)//2:]
        if self.verbose:
            print("initial   params:", params)
            print("optimized params:", optimized.x)
        c = self.get_circuit(gammas, betas)
        qubits = c.run()
        p = list(enumerate((qubits.conjugate() * qubits).real.tolist()))
        p.sort(key=lambda a:-a[1])

        def to_n(bitstring):
            val = 1
            mul = 2
            for c in bitstring:
                val += mul * (c == '0')
                mul *= 2
            return val

        def to_result(n):
            bitstring = list(("{:0" + str(self.n_qubits) + "b}").format(n)[::-1])
            return to_n(bitstring[:self.n1_bits]), to_n(bitstring[self.n1_bits:])

        self.result = OrderedDict((to_result(a[0]), a[1]) for a in p)

    def get_circuit(self, gammas, betas):
        evolves = [term.get_time_evolution() for term in self.h.terms]
        c = Circuit(self.n_qubits)
        c.h[:]
        for gamma, beta in zip(gammas, betas):
            for evolve in evolves:
                evolve(c, gamma)
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
            for term in self.h.terms:
                val += term.coeff * sampler(c, [[op.n for op in term.ops]])
            if self.verbose:
                print("params:", params)
                print("val:", val)
            self.query_history.append((params, val))
            return val
        return obj_f

if __name__ == "__main__":
    num = 7*5
    calc = FactoringQaoaCalculator(1, num)
    result = calc.result
    print("n_qubits:", calc.n_qubits, "(%d patterns)" % 2**calc.n_qubits)
    for i, (ans, p) in enumerate(result.items()):
        is_ok = ans[0] * ans[1] == num
        print("{:2}: {:4} {} {:2} * {:2} (p = {:.4})".format(i+1, num, ("!=", "==")[is_ok], ans[0], ans[1], p))
        if is_ok:
            break
