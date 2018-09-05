from collections import Counter, defaultdict, namedtuple
from functools import reduce
from itertools import product
from numbers import Number, Integral

_PauliTuple = namedtuple("_PauliTuple", "n")

def pauli_from_char(ch, n=0):
    """"X" => X, "Y" => Y, "Z" => Z, "I" => I"""
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
    """"XZIY" => X(0) * Z(1) * Y(3)"""
    return Term.from_chars(chars)

def to_term(pauli):
    return pauli.to_term()

def to_expr(term):
    return term.to_expr()

def commutation(expr1, expr2):
    expr1 = expr1.to_expr().simplify()
    expr2 = expr2.to_expr().simplify()
    return expr1 * expr2 - expr2 * expr1

def is_commutable(expr1, expr2, eps=0.00000001):
    return sum(x * x for x in commutation(expr1, expr2).coeffs()) < eps

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

class X(_PauliImpl, _PauliTuple):
    """Pauli's X operator"""
    pass

class Y(_PauliImpl, _PauliTuple):
    """Pauli's Y operator"""
    pass

class Z(_PauliImpl, _PauliTuple):
    """Pauli's Z operator"""
    pass

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
    @staticmethod
    def from_paulipair(pauli1, pauli2):
        return Term(Term.join_ops((pauli1,), (pauli2,)), 1.0)

    @staticmethod
    def from_pauli(pauli, coeff=1.0):
        """From X, Y, Z, I to Term"""
        if pauli.is_identity or coeff == 0:
            return Term((), coeff)
        return Term((pauli,), coeff)

    @staticmethod
    def from_ops_iter(ops, coeff):
        return Term(tuple(ops), coeff)

    @staticmethod
    def from_chars(chars):
        """"XZIY" => X(0) * Z(1) * Y(3)"""
        paulis = [pauli_from_char(c, n) for n, c in enumerate(chars) if c != "I"]
        if not paulis:
            return 1.0 * I
        if len(paulis) == 1:
            return 1.0 * paulis[0]
        return reduce(lambda a, b: a * b, paulis)

    @staticmethod
    def join_ops(ops1, ops2):
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
                raise ValueError("`pauli_term ** n` or `pow(pauli_term, n)`: n shall not be negative value.")
            if n == 0:
                return Term.from_pauli(I)
            return Term(self.ops * n, self.coeff ** n)
        return NotImplemented

    def __add__(self, other):
        return Expr.from_term(self) + other

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        return Expr.from_term(self) - other

    def __rsub__(self, other):
        return other - self

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

    def to_term(self):
        return self

    def __eq__(self, other):
        if isinstance(other, _PauliImpl):
            other = other.to_term()
        return _TermTuple.__eq__(self, other) or _TermTuple.__eq__(self.simplify(), other.simplify())

    def __ne__(self, other):
        return not self == other

    def to_expr(self):
        return Expr.from_term(self)

    def simplify(self):
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

    def append_to_circuit(self, circuit, simplify=True):
        if simplify:
            term = self.simplify()
        else:
            term = self
        for op in term.ops[::-1]:
            gate = op.op.lower()
            if gate != "i":
                getattr(circuit, gate)[op.n]

_ExprTuple = namedtuple("_ExprTuple", "terms")
class Expr(_ExprTuple):
    @staticmethod
    def from_number(num):
        return Expr.from_term(Term((), num))

    @staticmethod
    def from_term(term):
        return Expr((term,))

    @staticmethod
    def from_terms_iter(terms):
        return Expr(tuple(terms))

    def terms_to_dict(self):
        return {term[0]: term[1] for term in self.terms}

    @staticmethod
    def from_terms_dict(terms_dict):
        return Expr(tuple(Term(k, v) for k, v in terms_dict.items()))

    @property
    def is_identity(self):
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
            return Expr(tuple((op, coeff * other) for op, coeff in self.terms))
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
            return Expr(tuple((op, coeff * other) for op, coeff in self.terms))
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
                raise ValueError("`pauli_expr ** n` or `pow(pauli_expr, n)`: n shall not be negative value.")
            if n == 0:
                return Expr.from_number(1.0)
            val = self
            for _ in range(n - 1):
                val *= self
            return val
        return NotImplemented

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
        return self

    def coeffs(self):
        for term in self.terms:
            yield term.coeff

    def commutation(self, other):
        return commutation(self, other)

    def is_commutable_with(self, other):
        return is_commutable(self, other)

    def simplify(self):
        d = defaultdict(float)
        for term in self.terms:
            term = term.simplify()
            d[term.ops] += term.coeff
        return Expr.from_terms_iter((k, d[k]) for k in sorted(d, key=repr))

def ising_bit(n):
    return Expr((Term((Z(n),), 0.5), Term((), 0.5)))
