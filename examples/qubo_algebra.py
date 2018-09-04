from collections import Counter

class Term:
    def __init__(self, bit_tuple, coeff=1):
        if coeff == 0:
            raise ValueError("coeff shall not be zero")
        if not bit_tuple:
            raise ValueError("bit_tuple shall not be empty")
        self.bits = bit_tuple
        self.coeff = coeff

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Term(self.bits, self.coeff * other)
        if isinstance(other, Term):
            return Term(self.bits + other.bits, self.coeff * other.coeff)
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other:
                return Term(self.bits, self.coeff / other)
            raise ZeroDivisionError
        return NotImplemented

    def __pow__(self, n):
        if isinstance(n, int) and n >= 2:
            return Term(self.bits * n, self.coeff ** n)
        return NotImplemented

    def __add__(self, other):
        return Expr((self,)) + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return Expr((self,)) - other

    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        return Term(self.bits, -self.coeff)

    def __repr__(self):
        return "Term(" + repr(self.bits) + ", " + repr(self.coeff) + ")"

class Expr:
    def __init__(self, terms, constant=0):
        self.terms = terms
        self.constant = constant

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Expr(self.terms, self.constant + other)
        if isinstance(other, Term):
            return Expr(self.terms + (other,), self.constant)
        if isinstance(other, Expr):
            return Expr(self.terms + other.terms, self.constant + other.constant)
        return NotImplemented

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Expr(tuple(Term(t.bits, t.coeff * other) for t in self.terms), self.constant * other)
        if isinstance(other, Term):
            if self.constant:
                return Expr(tuple(t * other for t in self.terms) + (other * self.constant,))
            else:
                return Expr(tuple(t * other for t in self.terms))
        if isinstance(other, Expr):
            total = self.constant * other
            for t in self.terms:
                total += t * other
            return total
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other:
                return Expr(tuple(Term(t.bits, t.coeff / other) for t in self.terms), self.constant / other)
            raise ZeroDivisionError
        return NotImplemented

    def __pow__(self, n):
        if isinstance(n, int) and n >= 2:
            a = self
            for _ in range(n-1):
                a = a * self
            return a
        return NotImplemented

    def __neg__(self):
        return Expr(tuple(-t for t in self.terms), -self.constant)

    def __repr__(self):
        return "Expr(" + repr(self.terms) + ", " + repr(self.constant) + ")"

def bit(n):
    return Expr((Term((n,), 0.5),), 0.5)

def extract(expr):
    def extract_term(term):
        return tuple(sorted(k for k,v in Counter(term.bits).items() if v % 2))

    d = {}
    for term in expr.terms:
        bits = extract_term(term)
        if bits:
            d[bits] = d.get(bits, 0) + term.coeff
    return d

