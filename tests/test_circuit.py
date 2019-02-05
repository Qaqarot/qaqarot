import pytest
import numpy as np
import pytest
import numpy as np
from collections import Counter
from functools import reduce
from sympy import eye, zeros, symbols, simplify, sin, cos, exp, pi, I, Matrix
from sympy.physics.quantum import gate, TensorProduct
from blueqat import Circuit, BlueqatGlobalSetting


EPS = 1e-16


def vec_distsq(a, b):
    diff = a - b
    return diff.T.conjugate() @ diff


def is_vec_same(a, b, eps=EPS):
    return vec_distsq(a, b) < eps


def test_hgate1():
    assert is_vec_same(Circuit().h[1].h[0].run(), np.array([0.5, 0.5, 0.5, 0.5]))


def test_hgate2():
    assert is_vec_same(Circuit().x[0].h[0].run(), np.array([1 / np.sqrt(2), -1 / np.sqrt(2)]))


def test_hgate3():
    assert is_vec_same(Circuit().h[:2].run(), Circuit().h[0].h[1].run())


def test_pauli1():
    assert is_vec_same(Circuit().x[0].y[0].run(), Circuit().z[0].run())


def test_pauli2():
    assert is_vec_same(Circuit().y[0].z[0].run(), Circuit().x[0].run())


def test_pauli3():
    assert is_vec_same(Circuit().z[0].x[0].run(), Circuit().y[0].run())


def test_cx1():
    assert is_vec_same(
        Circuit().h[0].h[1].cx[1, 0].h[0].h[1].run(),
        Circuit().cx[0, 1].run()
    )


def test_cx2():
    assert is_vec_same(
        Circuit().x[2].cx[:4:2, 1:4:2].run(),
        Circuit().x[2:4].run()
    )


def test_rz1():
    assert is_vec_same(Circuit().h[0].rz(np.pi)[0].run(), Circuit().x[0].h[0].run())


def test_rz2():
    assert is_vec_same(
        Circuit().h[0].rz(np.pi / 3)[0].h[1].rz(np.pi / 3)[1].run(),
        Circuit().h[0, 1].rz(np.pi / 3)[:].run()
    )


def test_tgate():
    assert is_vec_same(Circuit().t[0].run(), Circuit().rz(np.pi / 4)[0].run())


def test_sgate():
    assert is_vec_same(Circuit().s[0].run(), Circuit().rz(np.pi / 2)[0].run())


def test_tdg_gate():
    assert is_vec_same(Circuit().s[1].tdg[1].tdg[1].run(), Circuit().i[1].run())


def test_sdg_gate():
    assert is_vec_same(Circuit().s[1].sdg[1].run(), Circuit().i[1].run())


@pytest.mark.parametrize('bin', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_toffoli_gate(bin):
    c = Circuit()
    if bin[0]:
        c.x[0]
    if bin[1]:
        c.x[1]
    c.ccx[0, 1, 2].m[2]
    expected_meas = "001" if bin[0] and bin[1] else "000"
    assert c.run(shots=1) == Counter([expected_meas])


def test_u3_gate():
    assert is_vec_same(Circuit().u3(1.23, 4.56, -5.43)[1].run(), Circuit().rz(-5.43)[1].ry(1.23)[1].rz(4.56)[1].run())


def test_u2_gate():
    assert is_vec_same(Circuit().u2(-1.23, 4.56)[1].run(), Circuit().u3(np.pi / 2, -1.23, 4.56)[1].run())


def test_u1_gate():
    assert is_vec_same(Circuit().u1(-1.23)[1].run(), Circuit().u3(0, 0, -1.23)[1].run())


def test_rotation1():
    assert is_vec_same(
        Circuit().ry(-np.pi / 2)[0].rz(np.pi / 6)[0].ry(np.pi / 2)[0].run(),
        Circuit().rx(np.pi / 6)[0].run()
    )


def test_measurement1():
    c = Circuit().m[0]
    cnt = c.run(shots=10000)
    assert cnt.most_common(1) == [("0", 10000)]


def test_measurement2():
    c = Circuit().x[0].m[0]
    cnt = c.run(shots=10000)
    assert cnt.most_common(1) == [("1", 10000)]


def test_measurement3():
    # 75% |0> + 25% |1>
    c = Circuit().rx(np.pi / 3)[0].m[0]
    n = 10000
    cnt = c.run(shots=n)
    most_common = cnt.most_common(1)[0]
    assert most_common[0] == "0"
    # variance of binomial distribution (n -> ∞) is np(1-p)
    # therefore, 2σ = 2 * sqrt(np(1-p))
    two_sigma = 2 * np.sqrt(n * 0.75 * 0.25)
    assert abs(most_common[1] - 0.75 * n) < two_sigma


def test_measurement_multiqubit1():
    c = Circuit().x[0].m[1]
    cnt = c.run(shots=10000)
    # 0-th qubit is also 0 because it is not measured.
    assert cnt.most_common(1) == [("00", 10000)]


def test_measurement_multiqubit2():
    c = Circuit().x[0].m[1::-1]
    cnt = c.run(shots=10000)
    assert cnt.most_common(1) == [("10", 10000)]


def test_measurement_entangled_state():
    # 1/sqrt(2) (|0> + |1>)
    c = Circuit().h[0].cx[0, 1]
    for _ in range(10000):
        cnt = c.run(shots=1)
        result = cnt.most_common()
        assert result == [("00", 1)] or result == [("11", 1)]


def test_measurement_hadamard1():
    n = 10000
    c = Circuit().h[0].m[0]
    cnt = c.run(shots=n)
    a, b = cnt.most_common(2)
    assert a[1] + b[1] == n
    # variance of binomial distribution (n -> ∞) is np(1-p)
    # therefore, 2σ = 2 * sqrt(np(1-p))
    two_sigma = 2 * np.sqrt(n * 0.5 * 0.5)
    assert abs(a[1] - n / 2) < two_sigma


def test_measurement_after_qubits1():
    for _ in range(50):
        c = Circuit().h[0].m[0]
        a, cnt = c.run(shots=1, returns="statevector_and_shots")
        if cnt.most_common(1)[0] == ('0', 1):
            assert is_vec_same(a, np.array([1, 0]))
        else:
            assert is_vec_same(a, np.array([0, 1]))


def test_caching_then_expand():
    c = Circuit().h[0]
    c.run()
    qubits = c.i[1].run()
    assert is_vec_same(qubits, Circuit().h[0].i[1].run())


def test_copy_empty():
    c = Circuit()
    c.run()
    # copy_history: deprecated.
    cc = c.copy(copy_backends=True)
    assert c.ops == cc.ops and c.ops is not cc.ops
    assert c._backends['numpy'].cache is None and cc._backends['numpy'].cache is None
    assert c._backends['numpy'].cache_idx == cc._backends['numpy'].cache_idx == -1


def test_cache_then_append():
    c = Circuit()
    c.x[0]
    c.run()
    c.h[0]
    c.run()
    assert is_vec_same(c.run(), Circuit().x[0].h[0].run())


def test_concat_circuit1():
    c1 = Circuit()
    c1.h[0]
    c1.run()
    c2 = Circuit()
    c2.h[1]
    c2.run()
    c1 += c2
    assert is_vec_same(c1.run(), Circuit().h[0].h[1].run())


def test_concat_circuit2():
    c1 = Circuit()
    c1.h[1]
    c1.run()
    c2 = Circuit()
    c2.h[0]
    c2.run()
    c1 += c2
    assert is_vec_same(c1.run(), Circuit().h[1].h[0].run())


def test_concat_circuit3():
    c1 = Circuit()
    c1.x[0]
    c2 = Circuit()
    c2.h[0]
    c1 += c2
    assert is_vec_same(c1.run(), Circuit().x[0].h[0].run())
    c1 = Circuit()
    c1.h[0]
    c2 = Circuit()
    c2.x[0]
    c1 += c2
    assert is_vec_same(c1.run(), Circuit().h[0].x[0].run())


def test_concat_circuit4():
    c1 = Circuit()
    c1.x[0]
    c2 = Circuit()
    c2.h[0]
    c = c1 + c2
    c.run()
    assert is_vec_same(c.run(), Circuit().x[0].h[0].run())
    assert is_vec_same(c1.run(), Circuit().x[0].run())
    assert is_vec_same(c2.run(), Circuit().h[0].run())


def test_switch_backend1():
    c = Circuit().x[0].h[0]
    assert np.array_equal(c.run(), c.run(backend="numpy"))

    BlueqatGlobalSetting.set_default_backend("qasm_output")
    assert c.run() == c.to_qasm()

    # Different instance of QasmOutputBackend is used.
    # Lhs is owned by Circuit, rhs is passed as argument. But in this case same result.
    from blueqat.backends.qasm_output_backend import QasmOutputBackend
    assert c.run(output_prologue=False) == c.run(False, backend=QasmOutputBackend())

    BlueqatGlobalSetting.set_default_backend("numpy")
    assert c.run(shots=5) == c.run_with_numpy(shots=5)


def test_sympy_backend_for_one_qubit_gate():
    E = eye(2)
    X = gate.X(0).get_target_matrix()
    Y = gate.Y(0).get_target_matrix()
    Z = gate.Z(0).get_target_matrix()
    H = gate.H(0).get_target_matrix()
    T = gate.T(0).get_target_matrix()
    S = gate.S(0).get_target_matrix()

    x, y, z = symbols('x, y, z')
    RX = Matrix([[cos(x / 2), -I * sin(x / 2)], [-I * sin(x / 2), cos(x / 2)]])
    RY = Matrix([[cos(y / 2), -sin(y / 2)], [sin(y / 2), cos(y / 2)]])
    RZ = Matrix([[exp(-I * z / 2), 0], [0, exp(I * z / 2)]])

    actual_1 = Circuit().x[0, 1].y[1].z[2].run(backend="sympy_unitary")
    expected_1 = reduce(TensorProduct, [Z, Y * X, X])
    assert actual_1 == expected_1

    actual_2 = Circuit().y[0].z[3].run(backend="sympy_unitary")
    expected_2 = reduce(TensorProduct, [Z, E, E, Y])
    assert actual_2 == expected_2

    actual_3 = Circuit().x[0].z[3].h[:].t[1].s[2].run(backend="sympy_unitary")
    expected_3 = reduce(TensorProduct, [H * Z, S * H, T * H, H * X])
    assert actual_3 == expected_3

    actual_4 = Circuit().rx(-pi / 2)[0].rz(pi / 2)[1].ry(pi)[2].run(backend="sympy_unitary")
    expected_4 = reduce(TensorProduct, [RY, RZ, RX]).subs([[x, -pi / 2], [y, pi], [z, pi / 2]])
    assert actual_4 == expected_4


def test_sympy_backend_for_two_qubit_gate():
    E = eye(2)
    UPPER = Matrix([[1, 0], [0, 0]])
    LOWER = Matrix([[0, 0], [0, 1]])
    X = gate.X(0).get_target_matrix()
    Z = gate.Z(0).get_target_matrix()
    H = gate.H(0).get_target_matrix()
    H_3 = reduce(TensorProduct, [H, E, H])
    H_4 = reduce(TensorProduct, [H, E, E, H])
    CX_3 = reduce(TensorProduct, [E, E, UPPER]) + reduce(TensorProduct, [X, E, LOWER])
    CZ_3 = reduce(TensorProduct, [E, E, UPPER]) + reduce(TensorProduct, [Z, E, LOWER])
    CX_4 = reduce(TensorProduct, [E, E, E, UPPER]) + reduce(TensorProduct, [X, E, E, LOWER])
    CZ_4 = reduce(TensorProduct, [E, E, E, UPPER]) + reduce(TensorProduct, [Z, E, E, LOWER])

    actual_1 = Circuit().cx[0, 3].run(backend="sympy_unitary")
    assert actual_1 == CX_4

    actual_2 = Circuit().cx[1, 3].x[4].run(backend="sympy_unitary")
    expected_2 = reduce(TensorProduct, [X, CX_3, E])
    assert actual_2 == expected_2

    actual_3 = Circuit().cz[0, 3].run(backend="sympy_unitary")
    assert actual_3 == CZ_4

    actual_4 = Circuit().cz[1, 3].x[4].run(backend="sympy_unitary")
    expected_4 = reduce(TensorProduct, [X, CZ_3, E])
    assert actual_4 == expected_4

    actual_5 = Circuit().cx[3, 0].run(backend="sympy_unitary")
    assert actual_5 == H_4 * CX_4 * H_4

    actual_6 = Circuit().cx[3, 1].x[4].run(backend="sympy_unitary")
    assert actual_6 == reduce(TensorProduct, [X, H_3 * CX_3 * H_3, E])

    actual_7 = Circuit().cz[3, 0].run(backend="sympy_unitary")
    assert actual_7 == CZ_4

    actual_8 = Circuit().cz[3, 1].x[4].run(backend="sympy_unitary")
    assert actual_8 == reduce(TensorProduct, [X, CZ_3, E])


def test_sympy_cx_cz():
    assert Circuit().cx[1, 2].run(backend="sympy_unitary") == Circuit().h[2].cz[2, 1].h[2].run(backend="sympy_unitary")
    assert Circuit().cx[2, 1].run(backend="sympy_unitary") == Circuit().h[1].cz[2, 1].h[1].run(backend="sympy_unitary")


def test_u1():
    lambd = symbols("lambd")

    actual_1 = Circuit().u1(lambd)[0].run(backend="sympy_unitary")
    expected_1 = Circuit().rz(lambd)[0].run_with_sympy_unitary()
    assert simplify(actual_1 - expected_1) == zeros(2)


def test_u1_realvalue():
    lambd = pi / 11

    actual_1 = Circuit().u1(lambd)[0].run(backend="sympy_unitary")
    assert actual_1[0, 0] != 0
    expected_1 = Circuit().rz(lambd)[0].run_with_sympy_unitary()
    assert expected_1[0, 0] != 0
    assert actual_1 == expected_1

    actual_2 = Circuit().u1(lambd.evalf())[0].run_with_numpy()
    expected_2 = np.array(expected_1.col(0)).astype(complex).reshape(-1)
    # ignore global phase
    for i in range(len(actual_2)):
        if actual_2[i] != 0:
            actual_2 *= expected_2[i] / actual_2[i]
            break
    assert is_vec_same(actual_2, expected_2)

    actual_3 = Circuit().x[0].u1(lambd.evalf())[0].run_with_numpy()
    expected_3 = np.array(expected_1.col(1)).astype(complex).reshape(-1)
    # ignore global phase
    for i in range(len(actual_3)):
        if actual_3[i] != 0:
            actual_3 *= expected_3[i] / actual_3[i]
            break
    assert is_vec_same(actual_3, expected_3)


def test_u2():
    phi, lambd = symbols("phi lambd")

    actual_1 = Circuit().u2(phi, lambd)[0].run(backend="sympy_unitary")
    expected_1 = Circuit().rz(lambd)[0].ry(pi / 2)[0].rz(phi)[0].run_with_sympy_unitary()
    assert simplify(actual_1 - expected_1) == zeros(2)


def test_u3():
    theta, phi, lambd = symbols("theta phi lambd")

    actual_1 = Circuit().u3(theta, phi, lambd)[0].run(backend="sympy_unitary")
    assert actual_1[0, 0] != 0
    expected_1 = Circuit().rz(lambd)[0].ry(theta)[0].rz(phi)[0].run_with_sympy_unitary()
    assert expected_1[0, 0] != 0
    assert simplify(actual_1 - expected_1) == zeros(2)


def test_u3_realvalue():
    theta = pi * 7 / 11
    phi = pi * 5 / 13
    lambd = pi * 8 / 17

    actual_1 = Circuit().u3(theta, phi, lambd)[0].run(backend="sympy_unitary")
    assert actual_1[0, 0] != 0
    expected_1 = Circuit().rz(lambd)[0].ry(theta)[0].rz(phi)[0].run_with_sympy_unitary()
    assert expected_1[0, 0] != 0
    assert actual_1 == expected_1

    actual_2 = Circuit().u3(theta.evalf(), phi.evalf(), lambd.evalf())[0].run_with_numpy()
    expected_2 = np.array(expected_1.col(0)).astype(complex).reshape(-1)
    # ignore global phase
    actual_2 *= expected_2[0] / actual_2[0]
    assert is_vec_same(actual_2, expected_2)

    actual_3 = Circuit().x[0].u3(theta.evalf(), phi.evalf(), lambd.evalf())[0].run_with_numpy()
    expected_3 = np.array(expected_1.col(1)).astype(complex).reshape(-1)
    # ignore global phase
    actual_3 *= expected_3[0] / actual_3[0]
    assert is_vec_same(actual_3, expected_3)


def test_cu1():
    E = eye(2)
    UPPER = Matrix([[1, 0], [0, 0]])
    LOWER = Matrix([[0, 0], [0, 1]])
    lambd = symbols("lambd")
    U = Circuit().rz(lambd)[0].run_with_sympy_unitary()
    U /= U[0, 0]

    actual_1 = Circuit().cu1(lambd)[0, 1].run(backend="sympy_unitary")
    actual_1 /= actual_1[0, 0]
    expected_1 = reduce(TensorProduct, [UPPER, E]) + reduce(TensorProduct, [LOWER, U])
    assert simplify(actual_1 - expected_1) == zeros(4)


def test_cu1_realvalue():
    E = eye(2)
    UPPER = Matrix([[1, 0], [0, 0]])
    LOWER = Matrix([[0, 0], [0, 1]])
    lambd = pi * 8 / 17
    U = Circuit().rz(lambd)[0].run_with_sympy_unitary()
    U /= U[0, 0]

    actual_1 = Circuit().cu1(lambd)[0, 1].run(backend="sympy_unitary")
    actual_1 /= actual_1[0, 0] # Ignore global phase
    expected_1 = reduce(TensorProduct, [E, UPPER]) + reduce(TensorProduct, [U, LOWER])
    assert actual_1 == expected_1

    for i in range(4):
        c = Circuit()
        if i % 2 == 1:
            c.x[0]
        if (i // 2) % 2 == 1:
            c.x[1]
        actual_2i = c.cu1(lambd.evalf())[0, 1].run_with_numpy()
        expected_2i = np.array(expected_1.col(i)).astype(complex).reshape(-1)
        assert 0.99999 < np.abs(np.dot(actual_2i.conj(), expected_2i)) < 1.00001


def test_cu3():
    E = eye(2)
    UPPER = Matrix([[1, 0], [0, 0]])
    LOWER = Matrix([[0, 0], [0, 1]])
    theta, phi, lambd = symbols("theta phi lambd")
    U = Circuit().rz(lambd)[0].ry(theta)[0].rz(phi)[0].run_with_sympy_unitary()

    actual_1 = Circuit().cu3(theta, phi, lambd)[0, 1].run(backend="sympy_unitary")
    expected_1 = reduce(TensorProduct, [E, UPPER]) + reduce(TensorProduct, [U, LOWER])
    print("actual")
    print(simplify(actual_1))
    print("expected")
    print(simplify(expected_1))
    print("diff")
    print(simplify(actual_1 - expected_1))
    assert simplify(actual_1 - expected_1) == zeros(4)


def test_cu3_realvalue():
    E = eye(2)
    UPPER = Matrix([[1, 0], [0, 0]])
    LOWER = Matrix([[0, 0], [0, 1]])
    theta = pi * 7 / 11
    phi = pi * 5 / 13
    lambd = pi * 8 / 17
    U = Circuit().u3(theta, phi, lambd)[0].run_with_sympy_unitary()
    expected_1 = reduce(TensorProduct, [E, UPPER]) + reduce(TensorProduct, [U, LOWER])
    print(expected_1)

    for i in range(4):
        c = Circuit()
        if i % 2 == 1:
            c.x[0]
        if (i // 2) % 2 == 1:
            c.x[1]
        actual_i = c.cu3(theta.evalf(), phi.evalf(), lambd.evalf())[0, 1].run_with_numpy()
        actual_i = np.array(actual_i).astype(complex).reshape(-1)
        expected_i = np.array(expected_1.col(i)).astype(complex).reshape(-1)
        assert 0.99999 < np.abs(np.dot(actual_i.conj(), expected_i)) < 1.00001
