from collections import Counter
import itertools
import numpy as np
from scipy.optimize import minimize
from blueqat import Circuit, pauli, vqe

def numpartition_qaoa(n_step, nums, minimizer=None, sampler=None):
    """Do the Number partition QAOA.

    :param n_step: The number of step of QAOA
    :param nums: The edges list of the graph.
    :returns Vqe object
    """
    hamiltonian = pauli.Expr.zero()
    for i, x in enumerate(nums):
        hamiltonian += pauli.Z[i] * x
    hamiltonian = (hamiltonian ** 2).simplify()

    return vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, n_step), minimizer, sampler)

if __name__ == "__main__":
    minimizer = vqe.get_scipy_minimizer(
        method="Powell",
        options={"disp": True}
    )
    nums = [3,2,6,9,2,5,7,3,3,6,7,3]
    vqe = numpartition_qaoa(2, nums, minimizer=minimizer)
    vqe.run(verbose=True)
    print("Num partition:", nums)
    best = vqe.result.most_common()[0]
    print("Probability:", best[1])
    result = "".join(map(str, best[0]))
    group0 = [a for a, b in zip(nums, result) if b == '0']
    group1 = [a for a, b in zip(nums, result) if b == '1']
    print("Group 0:", sum(group0), group0)
    print("Group 1:", sum(group1), group1)
