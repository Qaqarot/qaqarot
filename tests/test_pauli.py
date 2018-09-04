from blueqat import Circuit
from blueqat.pauli import *

def test_equality_identity_matrix():
    assert I == I()
    assert I == pauli_from_char("I")

def test_equality_identity_matrix2():
    assert I is I()
    assert I is pauli_from_char("I")

def test_is_identity():
    assert I.is_identity
    assert not X(0).is_identity
    assert I.to_term().is_identity
    assert not X(0).to_term().is_identity
    assert I.to_expr().is_identity
    assert not X(0).to_expr().is_identity

def test_equality():
    assert X(0) == X(0)
    assert X(0).to_term() == X(0).to_term()
    assert X(0).to_expr() == X(0).to_expr()

def test_equality_diff_type():
    assert X(0) == X(0)
    assert X(0) == X(0).to_term()
    assert X(0).to_term() == X(0)
    assert X(0) == X(0).to_expr()
    assert X(0).to_expr() == X(0)

def test_non_equality():
    assert X(0) != Y(0)
    assert X(0) != X(1)
    assert X(0) * X(1) != X(0)
    assert 1.2*Z(0) != Z(0)

def test_equality_calced_term():
    assert 2 * X(0) * Y(1) * X(2) == (0.5j * Y(1) * X(2)) * (-4j * X(0))
    assert Y(2) * X(2) == 1j * Z(2)
    assert 1j * Y(2) == X(2) * Z(2) * -Z(2) * -I * X(3) * I * Z(2) * Z(0) * Z(0) * X(3)

def test_nonequality_calced_term():
    assert -X(0) != X(0)
    assert -X(0) * X(0) != X(0) * X(0)
    assert I != 0 * I
    assert X(1) * Z(1) != X(0) * Z(1)

def test_equality_calced_expr():
    assert X(0) + X(1) == X(1) + X(0)
    assert X(0) + X(0) == 2 * X(0)
    assert X(1) + Y(2) == 1j * Z(2) * X(2) + X(1)
    assert (X(0) + Y(1)) * (Z(2) - 2*Y(1)) == Z(2) * X(0) + Y(1) * Z(2) - 2 * X(0) * Y(1) - 2
