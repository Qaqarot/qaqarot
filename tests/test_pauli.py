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


def test_notation():
    assert X(0) * Z[2] + Y(1) * Z[2] == term_from_chars("XIZ") + term_from_chars("IYZ")


def test_all_terms_commutable():
    (X[0] * X[1] + Y[1] * X[2]).is_all_terms_commutable()
    assert not (X[0] * X[1] + Y[1] * X[2]).is_all_terms_commutable()
    assert (X[0] + Z[1]).is_all_terms_commutable()
    assert (X[0] * Z[0] + Y[0]).is_all_terms_commutable()
    assert (X[0] * Z[1] + Z[0] * X[1]).is_all_terms_commutable()


def test_div():
    assert X[0] / 0.5 == X[0] * 2
    assert (X[0] + Y[0]) / 0.5 == (X[0] + Y[0]) * 2


def test_rmul():
    assert 2*X[0] == X[0]*2
    assert (X[0] + Y[0]) * 3 == 3 * (X[0] + Y[0])
    assert (X[0] + Y[0] - X[0]) * Z[0] == Y[0] * (X[0] + Z[0] - X[0])


def test_radd():
    assert X[0]*Z[0] + X[0] == X[0] + X[0]*Z[0]
    assert X[0]*Z[0] + 123 == 123 + X[0]*Z[0]
    assert X[0] + 123 == 123 + X[0]


def test_simplify1():
    assert (Z[0] + 0*X[1]).simplify() == Z[0].to_expr()


def test_pickle():
    import pickle
    h = X[0] * Z[0] * -1.23 + 4.56 + Z[2] * Z[3] * Z[4] * X[2] - 4 + X[1] * Y[2] + X[2] * Z[2]
    assert h == pickle.loads(pickle.dumps(h))
    assert Expr(()) == pickle.loads(pickle.dumps(Expr(())))
    assert X[0].to_expr() == pickle.loads(pickle.dumps(X[0].to_expr()))
    assert I.to_expr() == pickle.loads(pickle.dumps(I.to_expr()))

def test_expr_neg():
    assert -(X[0] + 2 * Y[0]) == X[0] * -1 - Y[0] * 2
