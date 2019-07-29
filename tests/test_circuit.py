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

from collections import Counter
from functools import reduce

import pytest
import numpy as np

from blueqat import Circuit, BlueqatGlobalSetting
from blueqat.utils import ignore_global_phase


EPS = 1e-16


def vec_distsq(a, b):
    diff = a - b
    return diff.T.conjugate() @ diff


def is_vec_same(a, b, ignore_global='', eps=EPS):
    if 'a' in ignore_global:
        ignore_global_phase(a)
    if 'b' in ignore_global:
        ignore_global_phase(b)
    return vec_distsq(a, b) < eps


def test_hgate1(backend):
    assert is_vec_same(Circuit().h[1].h[0].run(backend=backend), np.array([0.5, 0.5, 0.5, 0.5]))


def test_hgate2(backend):
    assert is_vec_same(Circuit().x[0].h[0].run(backend=backend),
                       np.array([1 / np.sqrt(2), -1 / np.sqrt(2)]))


def test_hgate3(backend):
    assert is_vec_same(Circuit().h[:2].run(backend=backend),
                       Circuit().h[0].h[1].run(backend=backend))


def test_pauli1(backend):
    assert is_vec_same(Circuit().x[0].y[0].run(backend=backend),
                       Circuit().z[0].run(backend=backend),
                       ignore_global='ab')


def test_pauli2(backend):
    assert is_vec_same(Circuit().y[0].z[0].run(backend=backend),
                       Circuit().x[0].run(backend=backend),
                       ignore_global='ab')


def test_pauli3(backend):
    assert is_vec_same(Circuit().z[0].x[0].run(backend=backend),
                       Circuit().y[0].run(backend=backend),
                       ignore_global='ab')


def test_cx1(backend):
    assert is_vec_same(
        Circuit().h[0].h[1].cx[1, 0].h[0].h[1].run(backend=backend),
        Circuit().cx[0, 1].run(backend=backend)
    )


def test_cx2(backend):
    assert is_vec_same(
        Circuit().x[2].cx[:4:2, 1:4:2].run(backend=backend),
        Circuit().x[2:4].run(backend=backend)
    )


def test_rz1(backend):
    assert is_vec_same(Circuit().h[0].rz(np.pi)[0].run(backend=backend),
                       Circuit().x[0].h[0].run(backend=backend))


def test_rz2(backend):
    assert is_vec_same(
        Circuit().h[0].rz(np.pi / 3)[0].h[1].rz(np.pi / 3)[1].run(backend=backend),
        Circuit().h[0, 1].rz(np.pi / 3)[:].run(backend=backend)
    )


def test_tgate(backend):
    assert is_vec_same(Circuit().t[0].run(backend=backend),
                       Circuit().rz(np.pi / 4)[0].run(backend=backend),
                       ignore_global='ab')


def test_sgate(backend):
    assert is_vec_same(Circuit().s[0].run(backend=backend),
                       Circuit().rz(np.pi / 2)[0].run(backend=backend),
                       ignore_global='ab')


def test_tdg_gate(backend):
    assert is_vec_same(Circuit().s[1].tdg[1].tdg[1].run(backend=backend),
                       Circuit().i[1].run(backend=backend),
                       ignore_global='ab')


def test_sdg_gate(backend):
    assert is_vec_same(Circuit().s[1].sdg[1].run(backend=backend),
                       Circuit().i[1].run(backend=backend),
                       ignore_global='ab')


@pytest.mark.parametrize('bin', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_toffoli_gate(bin, backend):
    c = Circuit()
    if bin[0]:
        c.x[0]
    if bin[1]:
        c.x[1]
    c.ccx[0, 1, 2].m[2]
    expected_meas = "001" if bin[0] and bin[1] else "000"
    assert c.run(backend=backend, shots=1) == Counter([expected_meas])


def test_u3_gate(backend):
    assert is_vec_same(Circuit().u3(1.23, 4.56, -5.43)[1].run(backend=backend),
                       Circuit().rz(-5.43)[1].ry(1.23)[1].rz(4.56)[1].run(backend=backend))


def test_u2_gate(backend):
    assert is_vec_same(Circuit().u2(-1.23, 4.56)[1].run(backend=backend),
                       Circuit().u3(np.pi / 2, -1.23, 4.56)[1].run(backend=backend))


def test_u1_gate(backend):
    assert is_vec_same(Circuit().u1(-1.23)[1].run(backend=backend),
                       Circuit().u3(0, 0, -1.23)[1].run(backend=backend))


def test_rotation1(backend):
    assert is_vec_same(
        Circuit().ry(-np.pi / 2)[0].rz(np.pi / 6)[0].ry(np.pi / 2)[0].run(backend=backend),
        Circuit().rx(np.pi / 6)[0].run(backend=backend),
        ignore_global='ab'
    )


def test_measurement1(backend):
    c = Circuit().m[0]
    cnt = c.run(backend=backend, shots=10000)
    assert cnt.most_common(1) == [("0", 10000)]


def test_measurement2(backend):
    c = Circuit().x[0].m[0]
    cnt = c.run(backend=backend, shots=10000)
    assert cnt.most_common(1) == [("1", 10000)]


def test_measurement3(backend):
    # 75% |0> + 25% |1>
    c = Circuit().rx(np.pi / 3)[0].m[0]
    n = 10000
    cnt = c.run(backend=backend, shots=n)
    most_common = cnt.most_common(1)[0]
    assert most_common[0] == "0"
    # variance of binomial distribution (n -> ∞) is np(1-p)
    # therefore, 3σ = 3 * sqrt(np(1-p))
    three_sigma = 3 * np.sqrt(n * 0.75 * 0.25)
    assert abs(most_common[1] - 0.75 * n) < three_sigma


def test_measurement_multiqubit1(backend):
    c = Circuit().x[0].m[1]
    cnt = c.run(backend=backend, shots=10000)
    # 0-th qubit is also 0 because it is not measured.
    assert cnt.most_common(1) == [("00", 10000)]


def test_measurement_multiqubit2(backend):
    c = Circuit().x[0].m[1::-1]
    cnt = c.run(backend=backend, shots=10000)
    assert cnt.most_common(1) == [("10", 10000)]


def test_measurement_entangled_state(backend):
    # 1/sqrt(2) (|0> + |1>)
    c = Circuit().h[0].cx[0, 1]
    for _ in range(10000):
        cnt = c.run(backend=backend, shots=1)
        result = cnt.most_common()
        assert result == [("00", 1)] or result == [("11", 1)]


def test_measurement_hadamard1(backend):
    n = 10000
    c = Circuit().h[0].m[0]
    cnt = c.run(backend=backend, shots=n)
    a, b = cnt.most_common(2)
    assert a[1] + b[1] == n
    # variance of binomial distribution (n -> ∞) is np(1-p)
    # therefore, 3σ = 3 * sqrt(np(1-p))
    three_sigma = 3 * np.sqrt(n * 0.5 * 0.5)
    assert abs(a[1] - n / 2) < three_sigma


def test_measurement_after_qubits1(backend):
    for _ in range(50):
        c = Circuit().h[0].m[0]
        a, cnt = c.run(backend=backend, shots=1, returns="statevector_and_shots")
        if cnt.most_common(1)[0] == ('0', 1):
            assert is_vec_same(a, np.array([1, 0]))
        else:
            assert is_vec_same(a, np.array([0, 1]))


def test_caching_then_expand(backend):
    c = Circuit().h[0]
    c.run(backend=backend)
    qubits = c.i[1].run(backend=backend)
    assert is_vec_same(qubits, Circuit().h[0].i[1].run())


def test_copy_empty_numpy():
    c = Circuit()
    c.run(backend='numpy')
    # copy_history: deprecated.
    cc = c.copy(copy_backends=True)
    assert c.ops == cc.ops and c.ops is not cc.ops
    assert c._backends['numpy'].cache is None and cc._backends['numpy'].cache is None
    assert c._backends['numpy'].cache_idx == cc._backends['numpy'].cache_idx == -1


def test_copy_empty_numba():
    c = Circuit()
    c.run(backend='numba')
    # copy_history: deprecated.
    cc = c.copy(copy_backends=True)
    assert c.ops == cc.ops and c.ops is not cc.ops
    assert c._backends['numba'].cache is None and cc._backends['numba'].cache is None
    assert c._backends['numba'].cache_idx == cc._backends['numba'].cache_idx == -1


def test_cache_then_append(backend):
    c = Circuit()
    c.x[0]
    c.run()
    c.h[0]
    c.run()
    assert is_vec_same(c.run(backend=backend), Circuit().x[0].h[0].run(backend=backend))


def test_concat_circuit1(backend):
    c1 = Circuit()
    c1.h[0]
    c1.run()
    c2 = Circuit()
    c2.h[1]
    c2.run()
    c1 += c2
    assert is_vec_same(c1.run(backend=backend), Circuit().h[0].h[1].run(backend=backend))


def test_concat_circuit2(backend):
    c1 = Circuit()
    c1.h[1]
    c1.run()
    c2 = Circuit()
    c2.h[0]
    c2.run()
    c1 += c2
    assert is_vec_same(c1.run(backend=backend), Circuit().h[1].h[0].run(backend=backend))


def test_concat_circuit3(backend):
    c1 = Circuit()
    c1.x[0]
    c2 = Circuit()
    c2.h[0]
    c1 += c2
    assert is_vec_same(c1.run(backend=backend), Circuit().x[0].h[0].run(backend=backend))
    c1 = Circuit()
    c1.h[0]
    c2 = Circuit()
    c2.x[0]
    c1 += c2
    assert is_vec_same(c1.run(backend=backend), Circuit().h[0].x[0].run(backend=backend))


def test_concat_circuit4(backend):
    c1 = Circuit()
    c1.x[0]
    c2 = Circuit()
    c2.h[0]
    c = c1 + c2
    c.run()
    assert is_vec_same(c.run(backend=backend), Circuit().x[0].h[0].run(backend=backend))
    assert is_vec_same(c1.run(backend=backend), Circuit().x[0].run(backend=backend))
    assert is_vec_same(c2.run(backend=backend), Circuit().h[0].run(backend=backend))


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


def test_macro():
    def macro(c, i):
        return c.h[i]
    BlueqatGlobalSetting.register_macro('foo', macro)
    try:
        assert is_vec_same(Circuit().foo(1).run(), Circuit().h[1].run())
    finally:
        BlueqatGlobalSetting.unregister_macro('foo')
