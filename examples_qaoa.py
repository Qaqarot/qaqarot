from collections import Counter
import numpy as np
from scipy.optimize import minimize
from circuit import Circuit

class MaxcutQaoa:
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
        optimized = minimize(obj_f, params, options={"disp": True})
        gammas = optimized.x[:len(params)//2]
        betas = optimized.x[len(params)//2:]
        print("initial   params:", params)
        print("optimized params:", optimized.x)
        c = self.get_circuit(gammas, betas)
        qubits = c.run()
        print("qubits:")
        print(qubits)
        p = (qubits.conjugate() * qubits).real
        print("probabilities:")
        print(p)
        maxi = p.argmax()
        maxi_bitstring = ("{:0" + str(n_qubits) + "b}").format(maxi)[::-1]
        self.result = list(maxi_bitstring)
        print("Most significant index:", maxi_bitstring)

    def get_circuit(self, gammas, betas):
        c = Circuit(self.n_qubits)
        c.h[:]
        for gamma, beta in zip(gammas, betas):
            for i,j in self.edges:
                c.cx[i, j].rz(gamma)[j].cx[i, j]
            c.rx(beta)[:]
        return c

    def get_objective_func(self):
        def obj_f(params):
            n_sample = self.n_sample
            gammas = params[:len(params)//2]
            betas = params[len(params)//2:]
            val = 0.0
            for i, j in self.edges:
                c = self.get_circuit(gammas, betas)
                c.measure[i, j]
                for _ in range(n_sample):
                    c.run()
                counter = Counter(sum(e) % 2 for e in c.run_history)
                for bits, cnt in counter.items():
                    if bits:
                        val -= cnt / n_sample
                    else:
                        val += cnt / n_sample
            print("val:", val)
            return val
        return obj_f

if __name__ == "__main__":
    ma = MaxcutQaoa(2, 250, [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2), (4, 0), (4, 3)])
    print("""
       {4}
      / \\
     {0}---{3}
     | x |
     {1}---{2}""".format(*ma.result))
