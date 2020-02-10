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

import sympy
import pytest

from blueqat import Circuit

p1, p2, p3, p4, p5 = sympy.symbols('p1 p2 p3 p4 p5')

@pytest.mark.parametrize('circuit', {
    Circuit().i[0],
    Circuit().x[0],
    Circuit().y[0],
    Circuit().z[0],
    Circuit().h[0],
    Circuit().t[0],
    Circuit().tdg[0],
    Circuit().s[0],
    Circuit().sdg[0],
    Circuit().cz[1, 0],
    Circuit().cy[1, 0],
    Circuit().cx[1, 0],
    Circuit().ch[0, 1],
    Circuit().rx(p1)[1],
    Circuit().ry(p1)[1],
    Circuit().rz(p1)[1],
    Circuit().r(p1)[2],
    Circuit().crx(p1)[2, 0],
    Circuit().cry(p1)[2, 0],
    Circuit().crz(p1)[2, 0],
    Circuit().cr(p1)[2, 0],
    Circuit().u1(p1)[2],
    Circuit().u2(p1, p2)[2],
    Circuit().u3(p1, p2, p3)[2],
    Circuit().cu1(p1)[1, 0],
    Circuit().cu2(p1, p2)[1, 0],
    Circuit().cu3(p1, p2, p3)[1, 0],
    Circuit().swap[2, 0],
    Circuit().ccx[2, 0, 1],
})
def test_dagger_unitary(circuit):
    circuit += circuit.dagger()
    u = sympy.simplify(sympy.trigsimp(circuit.to_unitary()))
    s1, s2 = u.shape
    assert s1 == s2
    assert u == sympy.eye(s1)


def test_dagger_unitary2():
    c = Circuit().h[0].s[0]
    c += c.dagger()
    u = sympy.simplify(c.to_unitary())
    assert u.shape == (2, 2)
    assert u == sympy.eye(2)
