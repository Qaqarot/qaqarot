from circuit import Circuit
from pauli import *

def to_pauli_expr(qubit_operator):
    def convert_ops(qo_ops, coeff):
        return Term.from_ops_iter((pauli_from_char(c, n) for n, c in qo_ops), coeff)

    return Expr.from_terms_iter(convert_ops(ops, coeff) for ops, coeff in qubit_operator.terms.items())
