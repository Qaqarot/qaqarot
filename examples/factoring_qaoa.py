from blueqat import Circuit, pauli, vqe
from blueqat.pauli import qubo_bit as q

def factoring_qaoa(n_step, num, minimizer=None, sampler=None, verbose=True):
    """Do the Number partition QAOA.

    :param num: The number to be factoring.
    :param n_step: The number of step of QAOA
    :param edges: The edges list of the graph.
    :returns result of QAOA
    """
    def get_nbit(n):
        m = 1
        while 2**m < n:
            m += 1
        return m

    n1_bits = get_nbit(int(num**0.5)) - 1
    n2_bits = get_nbit(int(num**0.5))

    def mk_expr(offset, n):
        expr = pauli.Expr.from_number(1)
        for i in range(n):
            expr = expr + 2**(i + 1) * q(i + offset)
        return expr

    def bitseparator(bits):
        assert len(bits) == n1_bits + n2_bits
        p = 1
        m = 1
        for b in bits[:n1_bits]:
            if b:
                p += 2**m
            m += 1
        q = 1
        m = 1
        for b in bits[n1_bits:]:
            if b:
                q += 2**m
            m += 1
        return p, q

    hamiltonian = (num - mk_expr(0, n1_bits) * mk_expr(n1_bits, n2_bits))**2
    return vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, n_step), minimizer, sampler), bitseparator

if __name__ == "__main__":
    num = 11*3
    vqe, separator = factoring_qaoa(2, num)
    result = vqe.run(verbose=True)
    for i, (bits, prob) in enumerate(result.most_common(100)):
        p, q = separator(bits)
        is_ok = p * q == num
        print("{:2}: {:4} {} {:2} * {:2} (p = {:.4})".format(i+1, num, ("!=", "==")[is_ok], p, q, prob))
        if is_ok:
            break
