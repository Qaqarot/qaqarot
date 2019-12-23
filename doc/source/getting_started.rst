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

    $ git clone https://github.com/mdrft/blueqat
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

    #number of qubit is not specified
    c = Circuit()

    #if you want to specified the number of qubit
    c = Circuit(3) #3qubits

Method Chain
------------

.. code-block:: python

    # write as chain
    Circuit().h[0].x[0].z[0]

    # write in separately
    c = Circuit().h[0]
    c.x[0].z[0]

Slice
-----

.. code-block:: python

    Circuit().z[1:3] # Zgate on 1,2
    Circuit().x[:3] # Xgate on (0, 1, 2)
    Circuit().h[:] # Hgate on all qubits
    Circuit().x[1, 2] # 1qubit gate with comma

Rotation Gate
-------------

.. code-block:: python

    Circuit().rz(math.pi / 4)[0]

Measurement
-----------

.. code-block:: python

    Circuit().m[0]

Run()
-----

.. code-block:: python

    Circuit().h[0].cx[0,1].run()

Run(shots=n)
------------

.. code-block:: python

    c = Circuit().h[0].cx[0,1].m[:]
    c.run(shots=100) # => Counter({'00': 48, '11': 52}) (random value.)

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
-----------------

.. code-block:: python

    qiskit.register(APItoken)
    sampler = blueqat.vqe.get_qiskit_sampler(backend="backend name")
    result = blueqat.vqe.Vqe(QaoaAnsatz(...), sampler=sampler).run(verbose=True)

Blueqat to QASM
---------------

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
    c.run()
    print(c.last_result()) # => (1, 1)

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

Optimization
-------------------------

.. code-block:: python

    import blueqat.opt as wq
    c = wq.opt().add([[1,1],[1,1]]).add("(q0+q1)^2")

    #qaoa
    print(c.qaoa().most_common(5))
    #=>(((0, 0), 0.7639901896866), ((1, 0), 0.10321404014639714), ((0, 1), 0.10321404014639707), ((1, 1), 0.029581730020605202))

    #annealing
    print(c.run())
    [0, 0]

    
SA Annealing
-----------------

.. code-block:: python

    import blueqat.opt as wq
    a = wq.opt()
    a.qubo = wq.sel(3,1) #creating QUBO matrix
    result = a.sa(shots=100,sampler="fast")
    wq.counter(result)
    
    Counter({'010': 29, '100': 34, '001': 37})

SA Parameters
-----------------

Some parameters for simualtion is adjustable

.. code-block:: python

    #for sa
    a.Ts  = 10    #default 5
    a.R   = 0.99  #default 0.95
    a.ite = 10000 #default 1000

SA Energy Function
------------------

Energy function of the calculation is stored in attribute E as an array.

.. code-block:: python

    print(a.E[-1]) #=>[0.0]

    #if you want to check the time evolution
    a.plot()

SA Sampling
-----------------

Sampling and counter function with number of shots.

.. code-block:: python

    result = a.sa(shots=100,sampler="fast")

    print(result)

    [[0, 1, 0],
     [0, 0, 1],
     [0, 1, 0],
     [0, 0, 1],
     [0, 1, 0],
 
     counter(result) # => Counter({'001': 37, '010': 25, '100': 38})

Connection to D-Wave cloud
-----------------------------

Direct connection to D-Wave machine with apitoken
https://github.com/dwavesystems/dwave-cloud-client is required

.. code-block:: python

    from blueqat.opt import Opt
    a = Opt()
    a.dwavetoken = "your token here"
    a.qubo = [[0,0,0,0,-4],[0,2,0,0,-4],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,4]] 
    a.dw()

    # => [1,1,-1,1,1,0,0,0,0,0,0]

QUBO Functions
-----------------

sel(N,K,array)
Automatically create QUBO which select K qubits from N qubits

.. code-block:: python

    print(wq.sel(5,2))
    #=>
    [[-3  2  2  2  2]
     [ 0 -3  2  2  2]
     [ 0  0 -3  2  2]
     [ 0  0  0 -3  2]
     [ 0  0  0  0 -3]]
     
if you set array on the 3rd params, the result likely to choose the nth qubit in the array

.. code-block:: python

    print(wq.sel(5,2,[0,2]))
    #=>
    [[-3.5  2.   2.   2.   2. ]
     [ 0.  -3.   2.   2.   2. ]
     [ 0.   0.  -3.5  2.   2. ]
     [ 0.   0.   0.  -3.   2. ]
     [ 0.   0.   0.   0.  -3. ]]

net(arr,N)
Automatically create QUBO which has value 1 for all connectivity defined by array of edges and graph size N

.. code-block:: python

    print(wq.net([[0,1],[1,2]],4))
    #=>
    [[0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]

this create 4*4 QUBO and put value 1 on connection between 0th and 1st qubit, 1st and 2nd qubit

zeros(N) Create QUBO with all element value as 0

.. code-block:: python

    print(wq.zeros(3))
    #=>
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]

diag(list) Create QUBO with diag from list

.. code-block:: python

    print(wq.diag([1,2,1]))
    #=>
    [[1 0 0]
     [0 2 0]
     [0 0 1]]
     
rands(N) Create QUBO with random number

.. code-block:: python

    print(wq.rands(2))
    #=>
    [[0.89903411 0.68839641]
     [0.         0.28554602]]
