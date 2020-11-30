.. image:: https://raw.githubusercontent.com/Blueqat/Blueqat/master/blueqat_logo_blue.png
    :target: https://github.com/Blueqat/Blueqat
    :alt: Blueqat
    :scale: 30 %

=======
blueqat
=======

A quantum computing SDK

Version
=======
.. image:: https://badge.fury.io/py/blueqat.svg
    :target: https://badge.fury.io/py/blueqat

Build info
==========
.. image:: https://circleci.com/gh/Blueqat/Blueqat.svg?style=svg
    :target: https://circleci.com/gh/Blueqat/Blueqat

Tutorial
========
https://github.com/Blueqat/Blueqat-tutorials

Install
=======

.. code-block:: sh

    git clone https://github.com/Blueqat/Blueqat
    cd Blueqat
    pip3 install -e .

or

.. code-block:: sh

    pip3 install blueqat

Circuit
=======

.. code-block:: python

    from blueqat import Circuit
    import math

    #number of qubit is not specified
    c = Circuit()

    #if you want to specified the number of qubit
    c = Circuit(3) #3qubits

Method Chain
============

.. code-block:: python

    # write as chain
    Circuit().h[0].x[0].z[0]

    # write in separately
    c = Circuit().h[0]
    c.x[0].z[0]

Slice
=====

.. code-block:: python

    Circuit().z[1:3] # Zgate on 1,2
    Circuit().x[:3] # Xgate on (0, 1, 2)
    Circuit().h[:] # Hgate on all qubits
    Circuit().x[1, 2] # 1qubit gate with comma

Rotation Gate
=============

.. code-block:: python

    Circuit().rz(math.pi / 4)[0]

Measurement
===========

.. code-block:: python

    Circuit().m[0]

Run()
=====

.. code-block:: python

    Circuit().h[0].cx[0,1].run()

Run(shots=n)
============

.. code-block:: python

    c = Circuit().h[0].cx[0,1].m[:]
    c.run(shots=100) # => Counter({'00': 48, '11': 52}) (random value.)

Hamiltonian
===========

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
===

.. code-block:: python

    from blueqat import vqe
    from blueqat.pauli import qubo_bit as q

    hamiltonian = -3*q(0)-3*q(1)-3*q(2)-3*q(3)-3*q(4)+2*q(0)*q(1)+2*q(0)*q(2)+2*q(0)*q(3)+2*q(0)*q(4)+2*q(1)*q(2)+2*q(1)*q(3)+2*q(1)*q(4)+2*q(2)*q(3)+2*q(2)*q(4)+2*q(3)*q(4)
    step = 2

    result = vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, step)).run()
    print(result.most_common(12))

If you want to create an ising model hamiltonian use Z(x) instead of q(x) in the equation

.. code-block:: python

    hamiltonian = Z(0)-3*Z(1)+2*Z(0)*Z(1)+2*Z(0)*Z(2)

Blueqat to Qiskit
=================

.. code-block:: python

    qiskit.register(APItoken)
    sampler = blueqat.vqe.get_qiskit_sampler(backend="backend name")
    result = blueqat.vqe.Vqe(QaoaAnsatz(...), sampler=sampler).run(verbose=True)

Blueqat to QASM
===============

.. code-block:: python

    Circuit.to_qasm()
    
    #OPENQASM 2.0;
    #include "qelib1.inc";
    #qreg q[1];
    #creg c[1];
    #h q[0];

Example
=======

2-qubit Grover
--------------

.. code-block:: python

    from blueqat import Circuit
    c = Circuit().h[:2].cz[0,1].h[:].x[:].cz[0,1].x[:].h[:].m[:]
    print(c.run(shots=1))

Maxcut QAOA
-----------

.. code-block:: python

    from blueqat import vqe, pauli
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2), (4, 0), (4, 3)]
    ansatz = vqe.QaoaAnsatz(sum([pauli.Z(i) * pauli.Z(j) for i, j in edges]), 1)
    result = vqe.Vqe(ansatz).run()
    print(
    """   {4}
      / \\
     {0}---{3}
     | x |
     {1}---{2}""".format(*result.most_common()[0][0]))

Document
========

https://blueqat.readthedocs.io/en/latest/

Author
======
Takumi Kato (blueqat), Yuichiro Minato (blueqat), Yuma Murata (D Slit Technologies), Satoshi Takezawa (TerraSky)

Disclaimer
==========
Copyright 2020 The Blueqat Developers.
