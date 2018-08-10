from circuit import Circuit

QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.23) q[2];
x q[2];
y q[2];
cz q[2],q[1];
z q[1];
ry(4.56) q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];"""

def test_qasm1():
    qasm = Circuit().h[0,1].cx[0,1].rz(1.23)[2].x[2].y[2].cz[2,1].z[1].ry(4.56)[0].m[:].to_qasm()
    assert QASM == qasm
