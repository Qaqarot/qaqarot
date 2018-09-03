from openfermion.ops import QubitOperator

from circuit import Circuit
from pauli import *

def to_pauli_expr(qubit_operator):
    def convert_ops(qo_ops, coeff):
        return Term.from_ops_iter((pauli_from_char(c, n) for n, c in qo_ops), coeff)

    return Expr.from_terms_iter(convert_ops(ops, coeff) for ops, coeff in qubit_operator.terms.items())

def from_pauli_term(term):
    term = term.to_term()
    def ops_to_str(bq_ops):
        s_ops = []
        for op in bq_ops:
            s_ops.append(op.op + str(op.n))
        return " ".join(s_ops)
    return QubitOperator(ops_to_str(term.ops), term.coeff)

def from_pauli_expr(expr):
    terms = expr.to_expr().terms
    if not terms:
        return QubitOperator("I", 0)
    qo = from_pauli_term(terms[0])
    for term in terms[1:]:
        qo += from_pauli_term(term)
    return qo
