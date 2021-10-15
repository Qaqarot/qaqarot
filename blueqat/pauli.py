# Copyright 2019 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The module for calculate Pauli matrices."""

from bisect import bisect_left
from collections import defaultdict, namedtuple
from functools import reduce
from itertools import combinations, groupby, product
from numbers import Number, Integral
from math import pi
from typing import Sequence

import numpy as np
import scipy.sparse

_PauliTuple = namedtuple("_PauliTuple", "n")
half_pi = pi / 2

_sparse_types = {
    'bsr': scipy.sparse.bsr_matrix,
    'coo': scipy.sparse.coo_matrix,
    'csc': scipy.sparse.csc_matrix,
    'csr': scipy.sparse.csr_matrix,
    'dia': scipy.sparse.dia_matrix,
    'dok': scipy.sparse.dok_matrix,
    'lil': scipy.sparse.lil_matrix,
}

_matrix = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex)
}

_mul_map = {
    ('X', 'X'): (1.0, 'I'),
    ('X', 'Y'): (1j, 'Z'),
    ('X', 'Z'): (-1j, 'Y'),
    ('Y', 'X'): (-1j, 'Z'),
    ('Y', 'Y'): (1.0, 'I'),
    ('Y', 'Z'): (1j, 'X'),
    ('Z', 'X'): (1j, 'Y'),
    ('Z', 'Y'): (-1j, 'X'),
    ('Z', 'Z'): (1.0, 'I'),
}

_sparse_matrix = {
    ty: {ch: fn(mat, dtype=complex)
         for ch, mat in _matrix.items()}
    for ty, fn in _sparse_types.items()
}


def _kron_1d(a, b):
    """This function is for internal use.
    Returns a⊗b for 1d array.
    """
    nb = b.size
    d = np.repeat(a, nb).reshape(-1, nb)
    d *= b
    return d.reshape(-1)


def _kron_1d_rec(krons, cumsum, lo, hi):
    """This function is for internal use.
    Equivalent with reduce(_kron_1d, krons[lo:hi]), but faster.
    """
    if hi - lo == 1:
        return krons[lo]
    if hi - lo == 2:
        return _kron_1d(krons[lo], krons[lo + 1])
    mid = bisect_left(cumsum, (cumsum[lo] + cumsum[hi - 1]) // 2, lo, hi)
    if mid == lo:
        return _kron_1d(krons[lo], _kron_1d_rec(krons, cumsum, lo + 1, hi))
    return _kron_1d(_kron_1d_rec(krons, cumsum, lo, mid),
                    _kron_1d_rec(krons, cumsum, mid, hi))


def _term_to_dataarray(term, n_qubits, rowmajor):
    """This function is for internal use.
    Make data of sparse Kronecker product matrix."""
    y_mat = np.array([-1j, 1j]) if rowmajor else np.array([1j, -1j])
    paulis = ['I'] * n_qubits
    data_list = []
    for op in term.ops:
        paulis[op.n] = op.op
    for g, l in groupby(paulis):
        n = len(tuple(l))
        if g == 'Y':
            data_list += [y_mat.copy() for _ in range(n)]
        elif g == 'Z':
            data_list += [np.array([1, -1], dtype=complex) for _ in range(n)]
        else:
            data_list.append(np.repeat(np.array([1], dtype=complex), 2**n))
    t = min(data_list, key=len)
    t *= term.coeff
    data_list.reverse()
    cumsum = np.array([k.size for k in data_list]).cumsum()
    return _kron_1d_rec(data_list, cumsum, 0, len(cumsum))


def _term_to_indices(term, dim, dtype, rowcol):
    """This function is for internal use.
    Make indices for sparse Kronecker product matrix."""
    xor_bits = sum(1 << op.n for op in term.ops if op.op in 'XY')
    if rowcol:
        col = np.arange(dim, dtype=dtype)
        row = col ^ xor_bits
        return row, col
    return np.arange(dim, dtype=dtype) ^ xor_bits


def pauli_from_char(ch, n=0):
    """Make Pauli matrix from an character.

    Args:
        | ch (str): "X" or "Y" or "Z" or "I".
        | n (int, optional): Make Pauli matrix as n-th qubits.

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
    e.g. "XZIY" => X(3) * Z(2) * Y(0)

    Args:
        chars (str): Written in "X", "Y", "Z" or "I".

    Returns:
        Term: A `Term` object.

    Raises:
        ValueError: When chars conteins the character which is "X", "Y", "Z" nor "I".
    """
    return Term.from_chars(reversed(chars))


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
        | expr1 (Expr, Term or Pauli operator): Pauli's expression.
        | expr2 (Expr, Term or Pauli operator): Pauli's expression.

    Returns:
        Expr: expr1 * expr2 - expr2 * expr1.
    """
    expr1 = expr1.to_expr().simplify()
    expr2 = expr2.to_expr().simplify()
    return (expr1 * expr2 - expr2 * expr1).simplify()


def is_commutable(expr1, expr2, eps=0.00000001):
    """Test whether expr1 and expr2 are commutable.

    Args:
        | expr1 (Expr, Term or Pauli operator): Pauli's expression.
        | expr2 (Expr, Term or Pauli operator): Pauli's expression.
        | eps (float, optional): Machine epsilon.
            If | \[expr1, expr2 \]| < eps, consider it is commutable.

    Returns:
        bool: if expr1 and expr2 are commutable, returns True, otherwise False.
    """
    return sum((x * x.conjugate()).real
               for x in commutator(expr1, expr2).coeffs()) < eps


# To avoid pylint error
def _n(pauli):
    return pauli.n


def _GetItem(self_, n):
    return type(self_)(n)


class _PauliImpl:
    @property
    def op(self):
        """Return operator type (X, Y, Z, I)"""
        return self.__class__.__name__[1]

    @property
    def is_identity(self):
        """If `self` is I, returns True, otherwise False."""
        return self.op == "I"

    @property
    def n_qubits(self):
        """Returns `self.n + 1` if self is not I. otherwise 0."""
        return 0 if self.is_identity else _n(self) + 1

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

    @property
    def matrix(self):
        """Matrix reprentation of this operator."""
        return _matrix[self.op].copy()

    def to_matrix(self, n_qubits=-1, *, sparse=None):
        """Convert to the matrix."""
        return self.to_term().to_matrix(n_qubits, sparse=sparse)


class _X(_PauliImpl, _PauliTuple):
    """Pauli's X operator"""


class _Y(_PauliImpl, _PauliTuple):
    """Pauli's Y operator"""


class _Z(_PauliImpl, _PauliTuple):
    """Pauli's Z operator"""


class _PauliCtor:
    def __init__(self, ty):
        self.ty = ty

    def __call__(self, n):
        return self.ty(n)

    def __getitem__(self, n):
        return self.ty(n)

    @property
    def matrix(self):
        """Matrix reprentation of this operator."""
        return _matrix[self.ty.__name__[-1]].copy()


X = _PauliCtor(_X)
Y = _PauliCtor(_Y)
Z = _PauliCtor(_Z)


class _I(_PauliImpl, namedtuple("_I", "")):
    """Identity operator"""
    def __call__(self):
        return self

    @property
    def matrix(self):
        """Matrix reprentation of this operator."""
        return _matrix['I'].copy()


I = _I()
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
        return Term(Term.join_ops((pauli1, ), (pauli2, )), 1.0)

    @staticmethod
    def from_pauli(pauli, coeff=1.0):
        """Make new Term from an Pauli operator"""
        if pauli.is_identity or coeff == 0:
            return Term((), coeff)
        return Term((pauli, ), coeff)

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
        paulis = [
            pauli_from_char(c, n) for n, c in enumerate(chars) if c != "I"
        ]
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
            return Term(Term.join_ops(self.ops, (other, )), self.coeff)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Term(self.ops, self.coeff * other)
        if isinstance(other, _PauliImpl):
            if other.is_identity:
                return self
            return Term(Term.join_ops((other, ), self.ops), self.coeff)
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
                raise ValueError(
                    "`pauli_term ** n` or `pow(pauli_term, n)`: " +
                    "n shall not be negative value.")
            if n == 0:
                return Term.from_pauli(I)
            return Term(self.ops * n, self.coeff**n)
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
            return _mul_map[op1, op2]

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
        """Returns the maximum index of Pauli matrices in the Term.
        If there's no Pauli matrices, returns -1.
        """
        try:
            return max(self.n_iter())
        except ValueError:
            return -1

    @property
    def n_qubits(self):
        """Returns the number of qubits of the term.
        If the term is constant with identity matrix, n_qubits is 0."""
        return self.max_n() + 1

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
                circuit.cx[ops[i - 1].n, ops[i].n]
            circuit.rz(-2 * coeff * t)[ops[-1].n]
            for i in range(len(ops) - 1, 0, -1):
                circuit.cx[ops[i - 1].n, ops[i].n]
            for op in ops:
                n = op.n
                if op.op == "X":
                    circuit.h[n]
                elif op.op == "Y":
                    circuit.rx(half_pi)[n]

        return append_to_circuit

    def to_matrix(self, n_qubits=-1, *, sparse=None):
        """Convert to the matrix."""
        if not (sparse is None or sparse in _sparse_types):
            raise ValueError(f'Unknown sparse format {sparse}.')
        if n_qubits == -1:
            n_qubits = self.n_qubits
        if n_qubits == 0:
            m = np.array([[self.coeff]])
            if sparse is None:
                return m
            return _sparse_types[sparse](m)
        dim = 2**n_qubits
        term = self.simplify()
        data = _term_to_dataarray(term, n_qubits, sparse == 'csr')
        dtype_idx = np.int32 if n_qubits < 31 else np.int64
        if sparse == 'csc':
            indices = _term_to_indices(term, dim, dtype_idx, False)
            return scipy.sparse.csc_matrix(
                (data, indices, np.arange(dim + 1, dtype=dtype_idx)),
                shape=(dim, dim))
        if sparse == 'csr':
            indices = _term_to_indices(term, dim, dtype_idx, False)
            return scipy.sparse.csr_matrix(
                (data, indices, np.arange(dim + 1, dtype=dtype_idx)),
                shape=(dim, dim))
        row, col = _term_to_indices(term, dim, dtype_idx, True)
        m = scipy.sparse.coo_matrix((data, (row, col)), shape=(dim, dim))
        if sparse is None:
            return m.toarray()
        return _sparse_types[sparse](m)


_ExprTuple = namedtuple("_ExprTuple", "terms")


class Expr(_ExprTuple):
    @staticmethod
    def from_number(num):
        """Make new Expr from a number"""
        if num:
            return Expr.from_term(Term((), num))
        return Expr.zero()

    @staticmethod
    def from_term(term):
        """Make new Expr from a Term"""
        if term.coeff:
            return Expr((term, ))
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
        return len(
            self.terms
        ) == 1 and not self.terms[0].ops and self.terms[0].coeff == 1.0

    def __eq__(self, other):
        if isinstance(other, (_PauliImpl, Term)):
            other = other.to_expr()
        if isinstance(other, Expr):
            return self.terms == other.terms or self.simplify(
            ).terms == other.simplify().terms
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
        return Expr(tuple(Term(op, -coeff) for op, coeff in self.terms))

    def __mul__(self, other):
        if isinstance(other, Number):
            if other == 0:
                return Expr.from_number(0.0)
            return Expr.from_terms_iter(
                Term(op, coeff * other) for op, coeff in self.terms)
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
            return Expr.from_terms_iter(
                Term(op, coeff * other) for op, coeff in self.terms)
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
                raise ValueError(
                    "`pauli_expr ** n` or `pow(pauli_expr, n)`: " +
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

    def __getnewargs__(self):
        return (self.terms, )

    def to_expr(self):
        """Do nothing. This method is prepared to avoid TypeError."""
        return self

    def max_n(self):
        """Returns the maximum index of Pauli matrices in the Expr.
        If Expr is empty or only constant and identity matrix, returns -1.
        """
        try:
            return max(term.max_n() for term in self.terms if term.ops)
        except ValueError:
            return -1

    @property
    def n_qubits(self):
        """Returns the number of qubits of the Term.

        If Expr is empty or only constant and identity matrix, returns 0.
        """
        return self.max_n() + 1

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

    def to_matrix(self, n_qubits=-1, *, sparse=None):
        """Convert to the matrix."""
        if not (sparse is None or sparse in _sparse_types):
            raise ValueError(f'Unknown sparse format {sparse}.')
        if n_qubits == -1:
            n_qubits = self.n_qubits
        if n_qubits == 0:
            m = np.array([[sum(term.coeff for term in self.terms)]])
            if sparse is None:
                return m
            return _sparse_types[sparse](m)
        expr = self.simplify()
        grpkey = lambda pau: sum(1 << op.n for op in pau.ops if op.op in 'XY')
        dim = 2**n_qubits
        is_csr = sparse == 'csr'
        gr_terms = [
            list(g)
            for _, g in groupby(sorted(expr.terms, key=grpkey), key=grpkey)
        ]
        n_groups = len(gr_terms)
        n_vals = n_groups * dim
        dtype_idx = np.int32 if n_qubits < 31 else np.int64
        vals = np.empty(n_vals, dtype=complex)
        inds = np.empty(n_vals, dtype=dtype_idx)
        for i_grp, grp in enumerate(gr_terms):
            val_acc = _term_to_dataarray(grp[0], n_qubits, is_csr)
            inds[i_grp::n_groups] = _term_to_indices(grp[0], dim, dtype_idx,
                                                     False)
            for term in grp[1:]:
                val_acc += _term_to_dataarray(term, n_qubits, is_csr)
            vals[i_grp::n_groups] = val_acc
        if not is_csr:
            m = scipy.sparse.csc_matrix(
                (vals, inds, np.arange(0, n_vals + 1, n_groups)),
                shape=(dim, dim))
        else:
            m = scipy.sparse.csr_matrix(
                (vals, inds, np.arange(0, n_vals + 1, n_groups)),
                shape=(dim, dim))
        m.eliminate_zeros()
        if sparse is None:
            return m.toarray()
        return _sparse_types[sparse](m)


def qubo_bit(n):
    """Represent QUBO's bit to Pauli operator of Ising model.

    Args:
        n (int): n-th bit in QUBO

    Returns:
        Expr: Pauli expression of QUBO bit.
    """
    return 0.5 - 0.5 * Z[n]


def from_qubo(qubo: Sequence[Sequence[float]]) -> Expr:
    """
    Convert to pauli operators of universal gate model.
    """
    h = 0.0
    assert all(len(q) == len(qubo) for q in qubo)
    for i in range(len(qubo)):
        h += qubo_bit(i) * qubo[i][i]
        for j in range(i + 1, len(qubo)):
            h += qubo_bit(i) * qubo_bit(j) * (qubo[i][j] + qubo[j][i])
    return h
