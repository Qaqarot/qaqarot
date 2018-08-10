from collections import Counter
import itertools
import numpy as np
from scipy.optimize import minimize
from circuit import Circuit

def maxcut_qaoa(n_step, n_sample, edges):
    """Do the Maxcut QAOA.

    :param n_step: The number of step of QAOA
    :param n_sample: The number of sampling time of each measurement in VQE.
                     If None, use calculated ideal value.
    :param edges: The edges list of the graph.

    Calculation is done in constructor.
    To get the result, use `result` property.
    """
    ma = MaxcutQaoaCalculator(n_step, n_sample, edges)
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

class MaxcutQaoaCalculator:
    def __init__(self, n_step, n_sample, edges):
        self.n_step = n_step
        self.n_sample = n_sample
        self.edges = edges

        n_qubits = -1
        for i, j in edges:
            n_qubits = max(n_qubits, i, j)
        n_qubits += 1
        self.n_qubits = n_qubits

        obj_f = self.get_objective_func()
        params = np.random.rand(2 * n_step) * np.pi
        params[:n_step] *= 2

        #optimized = minimize(obj_f, params, options={"disp": True})
        optimized = minimize(obj_f, params, method="Powell", options={"ftol": 2.0e-2, "xtol": 2.0e-2, "disp": True})
        gammas = optimized.x[:len(params)//2]
        betas = optimized.x[len(params)//2:]
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
        print("Most significant index:", maxi_bitstring)

    def get_circuit(self, gammas, betas):
        c = Circuit(self.n_qubits)
        c.h[:]
        for gamma, beta in zip(gammas, betas):
            for i, j in self.edges:
                c.cx[i, j].rz(gamma)[j].cx[i, j]
            c.rx(-2.0 * beta)[:]
        return c

    def get_objective_func(self):
        def obj_f(params):
            n_sample = self.n_sample
            gammas = params[:len(params)//2]
            betas = params[len(params)//2:]
            val = 0.0
            for i, j in self.edges:
                c = self.get_circuit(gammas, betas)
                a = c.run()
                if n_sample:
                    c.measure[i, j]
                    for _ in range(n_sample):
                        c.run()
                    counter = Counter(sum(e) % 2 for e in c.run_history)
                    for bits, cnt in counter.items():
                        if bits:
                            val -= cnt / n_sample
                        else:
                            val += cnt / n_sample
                else:
                    e = expect(a, [i, j])
                    for bits, p in e.items():
                        if bits.count("1") % 2:
                            val -= p
                        else:
                            val += p
            print("params:", params)
            print("val:", val)
            return val
        return obj_f

if __name__ == "__main__":
    result = maxcut_qaoa(2, None, [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2), (4, 0), (4, 3)])
    print("""
       {4}
      / \\
     {0}---{3}
     | x |
     {1}---{2}
""".format(*result))
