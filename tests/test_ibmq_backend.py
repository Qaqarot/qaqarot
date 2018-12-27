from blueqat import Circuit

def test_ibmq_backend():
    c = Circuit().x[0].h[1].h[1].m[:]
    assert c.run(shots=1234) == c.run_with_ibmq(ibmq_backend="qasm_simulator_py", shots=1234)
