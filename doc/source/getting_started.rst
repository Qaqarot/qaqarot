===============
Getting Started
===============

Prerequisites
=============

- Python3
- numpy
- scipy

Install
=======

.. code-block:: bash

    $ git clone https://github.com/Blueqat/Blueqat
    $ cd blueqat
    $ pip3 install -e .

or

.. code-block:: bash

    $ pip3 install blueqat


Basics
======

Circuit
-------

.. code-block:: python

    from blueqat import Circuit
    import math

    # number of qubit is not specified
    c = Circuit()

    #if you want to specify the number of qubit explicitly
    c = Circuit(3) # 3 qubits

Method Chain
------------

.. code-block:: python

    # write as a single chain
    Circuit().h[0].x[0].z[0]

    # in separate lines
    c = Circuit().h[0]
    c.x[0].z[0]

Index slicing
-------------

.. code-block:: python

    Circuit().z[1:3] # Z-gate on 1,2
    Circuit().x[:3] # X-gate on (0, 1, 2)
    Circuit().h[:] # H-gate on all qubits
    Circuit().x[1, 2] # another way to spcefify 1 qubit gate on qubit 1 and 2.

Rotation Gate
-------------

.. code-block:: python

    Circuit().rz(math.pi / 4)[0]

Measurement
-----------

.. code-block:: python

    Circuit().m[0]

Single shot run
---------------

.. code-block:: python

    Circuit().h[0].cx[0,1].run()

Run(shots=n)
------------

.. code-block:: python

    c = Circuit().h[0].cx[0,1].m[:]
    c.run(shots=100) # => Counter({'00': 48, '11': 52}) (result may vary due to randomness)

Hamiltonian
-----------

.. code-block:: python

    from blueqat.pauli import *

    hamiltonian1 = (1.23 * Z[0] + 4.56 * X[1] * Z[2]) ** 2
    hamiltonian2 = (2.46 * Y[0] + 5.55 * Z[1] * X[2] * X[1]) ** 2
    hamiltonian = hamiltonian1 + hamiltonian2
    print(hamiltonian)

simplify the hamiltonian

.. code-block:: python

    hamiltonian = hamiltonian.simplify()
    print(hamiltonian)

VQE
---

.. code-block:: python

    import numpy as np
    import scipy.optimize as optimize
    from blueqat.pauli import X, Y, Z, I

    # hamiltonian
    hamiltonian = 1.23 * I - 4.56 * X(0) + 2.45 * Y(0) + 2.34 * Z(0)
    hamiltonian = hamiltonian.to_expr()

    def f(params):
        params = params
        return Circuit().rx(params[0])[0].rz(params[1])[0].run(hamiltonian = hamiltonian)

    initial_guess = np.array([np.random.rand()*np.pi*2 for _ in range(2)])
    optimal_params = optimize.minimize(f, initial_guess, method="Powell", options={"ftol": 5.0e-2, "xtol": 5.0e-2, "maxiter": 1000})

    print(f'Estimated energy by VQE = {f(optimal_params.x)}', )


Blueqat to Qiskit
-----------------

.. code-block:: python

    import qiskit
    c = Circuit().h[0].cx[0, 1].m[:]
    c.run_with_ibmq(shots=100)

Blueqat to QASM
---------------

.. code-block:: python

    c = Circuit().h[0].cx[0, 1]
    print(c.to_qasm())
    
    # OPENQASM 2.0;
    # include "qelib1.inc";
    # qreg q[2];
    # creg c[2];
    # h q[0];
    # cx q[0],q[1];

Examples
========

2-qubit Grover
--------------

.. code-block:: python

    from blueqat import Circuit
    c = Circuit().h[:2].cz[0,1].h[:].x[:].cz[0,1].x[:].h[:].m[:]
    c.run(shots=1) # => Counter({'11': 1})

Maxcut QAOA
-----------

.. code-block:: python

    from blueqat.pauli import Z
    from blueqat.utils import qaoa

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2), (4, 0), (4, 3)]
    hamiltonian = sum(Z[e[0]]*Z[e[1]] for e in edges)
    step = 1

    result = qaoa(hamiltonian, step)
    b = result.circuit.run(shots=10)
    sample = b.most_common(1)[0][0]

    print("sample:"+ str(sample))
    print(
    """  {4}
     / \\
    {0}---{3}
    | x |
    {1}---{2}""".format(*b.most_common()[0][0]))