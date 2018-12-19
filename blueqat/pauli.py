"""The module for calculate Pauli matrices."""

from collections import defaultdict, namedtuple
from functools import reduce
from itertools import combinations, product
from numbers import Number, Integral
from math import pi

import numpy as np

_PauliTuple = namedtuple("_PauliTuple", "n")
half_pi = pi / 2

def pauli_from_char(ch, n=0):
    """Make Pauli matrix from an character.

    Args:
        ch (str): "X" or "Y" or "Z" or "I".
        n (int, optional): Make Pauli matrix as n-th qubits.

    Returns:
        If ch is "X" => X, "Y" => Y, "Z" => Z, "I" => I

    Raises:
        ValueError: When ch is not "X", "Y", "Z" nor "I".
    """
    ch = ch.upper()
    if ch == "I":
        return I
    if ch == "X":
        return X(n)
    if ch == "Y":
        return Y(n)
    if ch == "Z":
        return Z(n)
    raise ValueError("ch shall be X, Y, Z or I")

def term_from_chars(chars):
    """Make Pauli's Term from chars which is written by "X", "Y", "Z" or "I".
    e.g. "XZIY" => X(0) * Z(1) * Y(3)

    Args:
        chars (str): Written in "X", "Y", "Z" or "I".

    Returns:
        Term: A `Term` object.

    Raises:
        ValueError: When chars conteins the character which is "X", "Y", "Z" nor "I".
    """
    return Term.from_chars(chars)

def to_term(pauli):
    """Convert to Term from Pauli operator (X, Y, Z, I).

    Args:
        pauli (X, Y, Z or I): A Pauli operator

    Returns:
        Term: A `Term` object.
    """
    return pauli.to_term()

def to_expr(term):
    """Convert to Expr from Term or Pauli operator (X, Y, Z, I).

    Args:
        term: (Term, X, Y, Z or I): A Term or Pauli operator.

    Returns:
        Expr: An `Expr` object.
    """
    return term.to_expr()

def commutator(expr1, expr2):
    """Returns [expr1, expr2] = expr1 * expr2 - expr2 * expr1.

    Args:
        expr1 (Expr, Term or Pauli operator): Pauli's expression.
        expr2 (Expr, Term or Pauli operator): Pauli's expression.

    Returns:
        Expr: expr1 * expr2 - expr2 * expr1.
    """
    expr1 = expr1.to_expr().simplify()
    expr2 = expr2.to_expr().simplify()
    return (expr1 * expr2 - expr2 * expr1).simplify()

def is_commutable(expr1, expr2, eps=0.00000001):
    """Test whether expr1 and expr2 are commutable.

    Args:
        expr1 (Expr, Term or Pauli operator): Pauli's expression.
        expr2 (Expr, Term or Pauli operator): Pauli's expression.
        eps (float, optional): Machine epsilon.
            If |[expr1, expr2]| < eps, consider it is commutable.

    Returns:
        bool: if expr1 and expr2 are commutable, returns True, otherwise False.
    """
    return sum((x * x.conjugate()).real for x in commutator(expr1, expr2).coeffs()) < eps

# To avoid pylint error
def _n(pauli):
    return pauli.n

def _GetItem(self_, n):
    return type(self_)(n)

class _PauliImpl:
    @property
    def op(self):
        """Return operator type (X, Y, Z, I)"""
        return self.__class__.__name__

    @property
    def is_identity(self):
        """If `self` is I, returns True, otherwise False."""
        return self.op == "I"

    def __hash__(self):
        return hash((self.op, _n(self)))

    def __eq__(self, other):
        if isinstance(other, _PauliImpl):
            if self.is_identity:
                return other.is_identity
            return _n(self) == _n(other) and self.op == other.op
        if isinstance(other, Term):
            return self.to_term() == other
        if isinstance(other, Expr):
            return self.to_expr() == other
        return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __mul__(self, other):
        if isinstance(other, Number):
            return Term.from_pauli(self, other)
        if not isinstance(other, _PauliImpl):
            return NotImplemented
        if self.is_identity:
            return other.to_term()
        if other.is_identity:
            return self.to_term()
        if _n(self) == _n(other) and self.op == other.op:
            return I.to_term()
        return Term.from_paulipair(self, other)

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Term.from_pauli(self, other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            if other:
                return Term.from_pauli(self, 1.0 / other)
            raise ZeroDivisionError
        return NotImplemented

    def __add__(self, other):
        return self.to_expr() + other

    def __radd__(self, other):
        return other + self.to_expr()

    def __sub__(self, other):
        return self.to_expr() - other

    def __rsub__(self, other):
        return other - self.to_expr()

    def __neg__(self):
        return Term.from_pauli(self, -1.0)

    def __repr__(self):
        if self.is_identity:
            return "I"
        return self.op + "[" + str(_n(self)) + "]"

    def to_term(self):
        """Convert to Pauli Term"""
        return Term.from_pauli(self)

    def to_expr(self):
        """Convert to Pauli Expr"""
        return self.to_term().to_expr()

    _matrix = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }

    @property
    def matrix(self):
        """Matrix reprentation of this operator."""
        return self._matrix[self.op].copy()

    def to_matrix(self, n_qubits=-1):
        """Convert to the matrix."""
        if self.is_identity:
            if n_qubits == -1:
                return self.matrix
            else:
                return reduce(np.kron, [I.matrix for _ in range(n_qubits)])
        if n_qubits == -1:
            n_qubits = _n(self) + 1
        if _n(self) == 0:
            mat = self.matrix
        else:
            mat = reduce(np.kron, [I.matrix for _ in range(_n(self))])
            mat = np.kron(mat, self.matrix)
        if n_qubits > _n(self) + 1:
            mat = reduce(np.kron, [I.matrix for _ in range(n_qubits - _n(self) - 1)], mat)
        return mat

class X(_PauliImpl, _PauliTuple):
    """Pauli's X operator"""

class Y(_PauliImpl, _PauliTuple):
    """Pauli's Y operator"""

class Z(_PauliImpl, _PauliTuple):
    """Pauli's Z operator"""

class _PauliCtor:
    def __init__(self, ty):
        self.ty = ty

    def __call__(self, n):
        return self.ty(n)

    def __getitem__(self, n):
        return self.ty(n)

X = _PauliCtor(X)
Y = _PauliCtor(Y)
Z = _PauliCtor(Z)

class I(_PauliImpl, namedtuple("_I", "")):
    """Identity operator"""
    def __call__(self):
        return self

I = I()
_TermTuple = namedtuple("_TermTuple", "ops coeff")

class Term(_TermTuple):
    """Multiplication of Pauli matrices with coefficient.
    Note that this class is immutable.

    Multiplied Pauli matrices are very important for quantum computation
    because it is an unitary matrix (without coefficient) and also
    it can be consider the time evolution of the term (with real coefficient)
    without Suzuki-Trotter expansion.
    """
    @staticmethod
    def from_paulipair(pauli1, pauli2):
        """Make new Term from two Pauli operator."""
        return Term(Term.join_ops((pauli1,), (pauli2,)), 1.0)

    @staticmethod
    def from_pauli(pauli, coeff=1.0):
        """Make new Term from an Pauli operator"""
        if pauli.is_identity or coeff == 0:
            return Term((), coeff)
        return Term((pauli,), coeff)

    @staticmethod
    def from_ops_iter(ops, coeff):
        """For internal use."""
        return Term(tuple(ops), coeff)

    @staticmethod
    def from_chars(chars):
        """Make Pauli's Term from chars which is written by "X", "Y", "Z" or "I".
        e.g. "XZIY" => X(0) * Z(1) * Y(3)

        Args:
            chars (str): Written in "X", "Y", "Z" or "I".

        Returns:
            Term: A `Term` object.

        Raises:
            ValueError: When chars conteins the character which is "X", "Y", "Z" nor "I".
        """
        paulis = [pauli_from_char(c, n) for n, c in enumerate(chars) if c != "I"]
        if not paulis:
            return 1.0 * I
        if len(paulis) == 1:
            return 1.0 * paulis[0]
        return reduce(lambda a, b: a * b, paulis)

    @staticmethod
    def join_ops(ops1, ops2):
        """For internal use."""
        i = len(ops1) - 1
        j = 0
        while i >= 0 and j < len(ops2):
            if ops1[i] == ops2[j]:
                i -= 1
                j += 1
            else:
                break
        return ops1[:i + 1] + ops2[j:]

    @property
    def is_identity(self):
        """If `self` is I, returns True, otherwise False."""
        return not self.ops

    def __mul__(self, other):
        if isinstance(other, Number):
            return Term(self.ops, self.coeff * other)
        if isinstance(other, Term):
            ops = Term.join_ops(self.ops, other.ops)
            coeff = self.coeff * other.coeff
            return Term(ops, coeff)
        if isinstance(other, _PauliImpl):
            if other.is_identity:
                return self
            return Term(Term.join_ops(self.ops, (other,)), self.coeff)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Term(self.ops, self.coeff * other)
        if isinstance(other, _PauliImpl):
            if other.is_identity:
                return self
            return Term(Term.join_ops((other,), self.ops), self.coeff)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other:
                return Term(self.ops, self.coeff / other)
            raise ZeroDivisionError
        return NotImplemented

    def __pow__(self, n):
        if isinstance(n, Integral):
            if n < 0:
                raise ValueError("`pauli_term ** n` or `pow(pauli_term, n)`: " +
                                 "n shall not be negative value.")
            if n == 0:
                return Term.from_pauli(I)
            return Term(self.ops * n, self.coeff ** n)
        return NotImplemented

    def __add__(self, other):
        return Expr.from_term(self) + other

    def __radd__(self, other):
        return other + Expr.from_term(self)

    def __sub__(self, other):
        return Expr.from_term(self) - other

    def __rsub__(self, other):
        return other - Expr.from_term(self)

    def __neg__(self):
        return Term(self.ops, -self.coeff)

    def __repr__(self):
        if self.coeff == 0:
            return "0*I"
        if self.coeff == -1.0:
            s_coeff = "-"
        else:
            s_coeff = str(self.coeff) + "*"
        if self.ops == ():
            s_ops = "I"
        else:
            s_ops = "*".join(op.op + "[" + repr(op.n) + "]" for op in self.ops)
        return s_coeff + s_ops

    def __eq__(self, other):
        if isinstance(other, _PauliImpl):
            other = other.to_term()
        return _TermTuple.__eq__(self, other) or \
               _TermTuple.__eq__(self.simplify(), other.simplify())

    def __ne__(self, other):
        return not self == other

    def to_term(self):
        """Do nothing. This method is prepared to avoid TypeError."""
        return self

    def to_expr(self):
        """Convert to Expr."""
        return Expr.from_term(self)

    def commutator(self, other):
        """Returns commutator."""
        return commutator(self, other)

    def is_commutable_with(self, other):
        """Test whether `self` is commutable with `other`."""
        return is_commutable(self, other)

    def simplify(self):
        """Simplify the Term."""
        def mul(op1, op2):
            if op1 == "I":
                return 1.0, op2
            if op2 == "I":
                return 1.0, op1
            if op1 == op2:
                return 1.0, "I"
            if op1 == "X":
                return (-1j, "Z") if op2 == "Y" else (1j, "Y")
            if op1 == "Y":
                return (-1j, "X") if op2 == "Z" else (1j, "Z")
            if op1 == "Z":
                return (-1j, "Y") if op2 == "X" else (1j, "X")

        before = defaultdict(list)
        for op in self.ops:
            if op.op == "I":
                continue
            before[op.n].append(op.op)
        new_coeff = self.coeff
        new_ops = []
        for n in sorted(before.keys()):
            ops = before[n]
            assert ops
            k = 1.0
            op = ops[0]
            for _op in ops[1:]:
                _k, op = mul(op, _op)
                k *= _k
            new_coeff *= k
            if new_coeff.imag == 0:
                # cast to float
                new_coeff = new_coeff.real
            if op != "I":
                new_ops.append(pauli_from_char(op, n))
        return Term(tuple(new_ops), new_coeff)

    def n_iter(self):
        """Returns an iterator which yields indices for each Pauli matrices in the Term."""
        return (op.n for op in self.ops)

    def max_n(self):
        """Returns the maximum index of Pauli matrices in the Term."""
        return max(self.n_iter())

    def append_to_circuit(self, circuit, simplify=True):
        """Append Pauli gates to `Circuit`."""
        if simplify:
            term = self.simplify()
        else:
            term = self
        for op in term.ops[::-1]:
            gate = op.op.lower()
            if gate != "i":
                getattr(circuit, gate)[op.n]

    def get_time_evolution(self):
        """Get the function to append the time evolution of this term.

        Returns:
            function(circuit: Circuit, t: float):
                Add gates for time evolution to `circuit` with time `t`
        """
        term = self.simplify()
        coeff = term.coeff
        if coeff.imag:
            raise ValueError("Not a real coefficient.")
        ops = term.ops
        def append_to_circuit(circuit, t):
            if not ops:
                return
            for op in ops:
                n = op.n
                if op.op == "X":
                    circuit.h[n]
                elif op.op == "Y":
                    circuit.rx(-half_pi)[n]
            for i in range(1, len(ops)):
                circuit.cx[ops[i-1].n, ops[i].n]
            circuit.rz(-2 * coeff * t)[ops[-1].n]
            for i in range(len(ops)-1, 0, -1):
                circuit.cx[ops[i-1].n, ops[i].n]
            for op in ops:
                n = op.n
                if op.op == "X":
                    circuit.h[n]
                elif op.op == "Y":
                    circuit.rx(half_pi)[n]
        return append_to_circuit

    def to_matrix(self, n_qubits=-1):
        """Convert to the matrix."""
        if n_qubits == -1:
            n_qubits = self.max_n() + 1
        mat = I.to_matrix(n_qubits)
        for op in self.ops:
            if op.is_identity:
                continue
            mat = mat @ op.to_matrix(n_qubits)
        return mat * self.coeff


_ExprTuple = namedtuple("_ExprTuple", "terms")
class Expr(_ExprTuple):
    @staticmethod
    def from_number(num):
        """Make new Expr from a number"""
        if num:
            return Expr.from_term(Term((), num))
        else:
            return Expr.zero()

    @staticmethod
    def from_term(term):
        """Make new Expr from a Term"""
        if term.coeff:
            return Expr((term,))
        else:
            return Expr.zero()

    @staticmethod
    def from_terms_iter(terms):
        """For internal use."""
        return Expr(tuple(term for term in terms if term.coeff))

    def terms_to_dict(self):
        """For internal use."""
        return {term[0]: term[1] for term in self.terms if term.coeff}

    @staticmethod
    def from_terms_dict(terms_dict):
        """For internal use."""
        return Expr(tuple(Term(k, v) for k, v in terms_dict.items() if v))

    @staticmethod
    def zero():
        """Returns 0 as Term"""
        return Expr(())

    @property
    def is_identity(self):
        """If `self` is I, returns True, otherwise False."""
        if not self.terms:
            return True
        return len(self.terms) == 1 and not self.terms[0].ops and self.terms[0].coeff == 1.0

    def __eq__(self, other):
        if isinstance(other, (_PauliImpl, Term)):
            other = other.to_expr()
        if isinstance(other, Expr):
            return self.terms == other.terms or self.simplify().terms == other.simplify().terms
        return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __add__(self, other):
        if isinstance(other, Number):
            other = Expr.from_number(other)
        elif isinstance(other, Term):
            other = Expr.from_term(other)
        if isinstance(other, Expr):
            terms = self.terms_to_dict()
            for op, coeff in other.terms:
                if op in terms:
                    terms[op] += coeff
                    if terms[op] == 0:
                        del terms[op]
                else:
                    terms[op] = coeff
            return Expr.from_terms_dict(terms)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Number):
            other = Expr.from_number(other)
        elif isinstance(other, Term):
            other = Expr.from_term(other)
        if isinstance(other, Expr):
            terms = self.terms_to_dict()
            for op, coeff in other.terms:
                if op in terms:
                    terms[op] -= coeff
                    if terms[op] == 0:
                        del terms[op]
                else:
                    terms[op] = -coeff
            return Expr.from_terms_dict(terms)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Number):
            return Expr.from_number(other) + self
        if isinstance(other, Term):
            return Expr.from_term(other) + self
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Number):
            return Expr.from_number(other) - self
        if isinstance(other, Term):
            return Expr.from_term(other) - self
        return NotImplemented

    def __neg__(self):
        return Expr(tuple((op, -coeff) for op, coeff in self.terms))

    def __mul__(self, other):
        if isinstance(other, Number):
            if other == 0:
                return Expr.from_number(0.0)
            return Expr.from_terms_iter(Term(op, coeff * other) for op, coeff in self.terms)
        if isinstance(other, _PauliImpl):
            other = other.to_term()
        if isinstance(other, Term):
            return Expr(tuple(term * other for term in self.terms))
        if isinstance(other, Expr):
            terms = defaultdict(float)
            for t1, t2 in product(self.terms, other.terms):
                term = t1 * t2
                terms[term.ops] += term.coeff
            return Expr.from_terms_dict(terms)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            if other == 0:
                return Expr.from_number(0.0)
            return Expr.from_terms_iter(Term(op, coeff * other) for op, coeff in self.terms)
        if isinstance(other, _PauliImpl):
            other = other.to_term()
        if isinstance(other, Term):
            return Expr(tuple(other * term for term in self.terms))
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            if other:
                return Expr(tuple(term / other for term in self.terms))
            raise ZeroDivisionError
        return NotImplemented

    def __pow__(self, n):
        if isinstance(n, Integral):
            if n < 0:
                raise ValueError("`pauli_expr ** n` or `pow(pauli_expr, n)`: " +
                                 "n shall not be negative value.")
            if n == 0:
                return Expr.from_number(1.0)
            val = self
            for _ in range(n - 1):
                val *= self
            return val
        return NotImplemented

    def __iter__(self):
        return iter(self.terms)

    def __repr__(self):
        if not self.terms:
            return "0*I+0"
        s_terms = [repr(self.terms[0])]
        for term in self.terms[1:]:
            s = repr(term)
            if s[0] == "+":
                s_terms.append("+")
                s_terms.append(s[1:])
            elif s[0] == "-":
                s_terms.append("-")
                s_terms.append(s[1:])
            else:
                s_terms.append("+")
                s_terms.append(s)
        return " ".join(s_terms)

    def to_expr(self):
        """Do nothing. This method is prepared to avoid TypeError."""
        return self

    def max_n(self):
        """Returns the maximum index of Pauli matrices in the Term."""
        return max(term.max_n() for term in self.terms if term.ops)

    def coeffs(self):
        """Generator which yields a coefficent for each Term."""
        for term in self.terms:
            yield term.coeff

    def commutator(self, other):
        """Returns commutator."""
        return commutator(self, other)

    def is_commutable_with(self, other):
        """Test whether `self` is commutable with `other`."""
        return is_commutable(self, other)

    def is_all_terms_commutable(self):
        """Test whether all terms are commutable. This function may very slow."""
        return all(is_commutable(a, b) for a, b in combinations(self.terms, 2))

    def simplify(self):
        """Simplify the Expr."""
        d = defaultdict(float)
        for term in self.terms:
            term = term.simplify()
            d[term.ops] += term.coeff
        return Expr.from_terms_iter(
            Term.from_ops_iter(k, d[k]) for k in sorted(d, key=repr) if d[k])

    def to_matrix(self, n_qubits=-1):
        """Convert to the matrix."""
        if n_qubits == -1:
            n_qubits = self.max_n() + 1
        return sum(term.to_matrix(n_qubits) for term in self.terms)

def qubo_bit(n):
    """Represent QUBO's bit to Pauli operator of Ising model.

    Args:
        n (int): n-th bit in QUBO

    Returns:
        Expr: Pauli expression of QUBO bit.
    """
    return 0.5 - 0.5*Z[n]
