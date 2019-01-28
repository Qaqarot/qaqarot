from blueqat import Circuit
from qiskit import Aer

def test_ibmq_backend():
    c = Circuit().x[0].h[1].h[1].m[:]
    assert c.run(shots=1234) == c.run_with_ibmq(Aer.get_backend("qasm_simulator"), shots=1234)
