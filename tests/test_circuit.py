from blueqat import Circuit, BlueqatGlobalSetting
import numpy as np
import sys
from collections import Counter

EPS = 1e-16

def vec_distsq(a, b):
    diff = a - b
    return diff.T.conjugate() @ diff

def is_vec_same(a, b, eps=EPS):
    return vec_distsq(a, b) < eps

def test_hgate1():
    assert is_vec_same(Circuit().h[1].h[0].run(), np.array([0.5, 0.5, 0.5, 0.5]))

def test_hgate2():
    assert is_vec_same(Circuit().x[0].h[0].run(), np.array([1/np.sqrt(2), -1/np.sqrt(2)]))

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
        Circuit().h[0].h[1].cx[1,0].h[0].h[1].run(),
        Circuit().cx[0,1].run()
    )

def test_cx2():
    assert is_vec_same(
        Circuit().x[2].cx[:4:2,1:4:2].run(),
        Circuit().x[2:4].run()
    )

def test_rz1():
    assert is_vec_same(Circuit().h[0].rz(np.pi)[0].run(), Circuit().x[0].h[0].run())

def test_rz2():
    assert is_vec_same(
        Circuit().h[0].rz(np.pi / 3)[0].h[1].rz(np.pi / 3)[1].run(),
        Circuit().h[0,1].rz(np.pi / 3)[:].run()
    )

def test_tgate():
    assert is_vec_same(Circuit().t[0].run(), Circuit().rz(np.pi / 4)[0].run())

def test_sgate():
    assert is_vec_same(Circuit().s[0].run(), Circuit().rz(np.pi / 2)[0].run())

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
    assert abs(a[1] - n/2) < two_sigma

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
    cc = c.copy(copy_cache=True, copy_history=True)
    assert c.ops == cc.ops and c.ops is not cc.ops
    assert c.cache is None and cc.cache is None
    assert c.cache_idx == cc.cache_idx == -1
    # run_history: deprecated.
    assert c.run_history == cc.run_history
    assert c.run_history is not cc.run_history

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
    assert np.array_equal(c.run(), c.run_with_backend("run_with_numpy"))

    BlueqatGlobalSetting.set_default_backend("to_qasm")
    assert c.run() == c.to_qasm()

    # Different instance of QasmOutputBackend is used.
    # Lhs is owned by Circuit, rhs is passed as argument. But in this case same result.
    from blueqat.backends.qasm_output_backend import QasmOutputBackend
    assert c.run(output_prologue=False) == c.run_with_backend(QasmOutputBackend(), False)
    assert c.run(False) == c.run_with_backend(QasmOutputBackend(), output_prologue=False)

    BlueqatGlobalSetting.set_default_backend("run_with_numpy")
    assert c.run(5) == c.run_with_numpy(shots=5)
