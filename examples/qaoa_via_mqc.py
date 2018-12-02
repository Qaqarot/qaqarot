from blueqat import pauli, vqe, BlueqatGlobalSetting

def maxcut_qaoa(n_step, edges, minimizer=None, sampler=None, verbose=True):
    """Setup QAOA.

    :param n_step: The number of step of QAOA
    :param n_sample: The number of sampling time of each measurement in VQE.
                     If None, use calculated ideal value.
    :param edges: The edges list of the graph.
    :returns Vqe object
    """
    sampler = sampler or vqe.non_sampling_sampler
    minimizer = minimizer or vqe.get_scipy_minimizer(
        method="Powell",
        options={"ftol": 5.0e-2, "xtol": 5.0e-2, "maxiter": 1000, "disp": True}
    )
    hamiltonian = pauli.I() * 0

    for i, j in edges:
        hamiltonian += pauli.Z(i) * pauli.Z(j)

    return vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, n_step), minimizer, sampler)

if __name__ == "__main__":
    print("Input token:")
    token = input().strip()
    BlueqatGlobalSetting.set_default_backend("mqc")
    #sampler = vqe.get_measurement_sampler(1024)
    sampler = vqe.get_measurement_sampler(1024, {"token": token})

    runner = maxcut_qaoa(2, [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2), (4, 0), (4, 3)], sampler=sampler)
    result = runner.run(verbose=True)
    print("""
       {4}
      / \\
     {0}---{3}
     | x |
     {1}---{2}
""".format(*result.most_common()[0][0]))
