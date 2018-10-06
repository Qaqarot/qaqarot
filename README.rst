=======
blueqat
=======

A quantum gate simulator

Install
=======
::

    git clone https://github.com/mdrft/blueqat
    cd blueqat
    pip3 install -e .

or ::

    pip3 install blueqat

Circuit
=======

::

    from blueqat import Circuit
    import math

    #number of qubit is not specified
    c = Circuit()

    #if you want to specified the number of qubit
    c = Circuit(3) #3qubits

::

Method Chain
=======

::
    # write as chain
    Circuit().h[0].x[0].z[0]

    # write in separately
    c = Circuit().h[0]
    c.x[0].z[0]
::

Slice
=======

::
    Circuit().z[1:3] # Zgate on 1,2
    Circuit().x[:3] # Xgate on (0, 1, 2)
    Circuit().h[:] # Hgate on all qubits
    Circuit().x[1, 2] # 1qubit gate with comma
::

Rotation Gate
=======

::
    Circuit().rz(math.pi / 4)[0]
::

Measurement
=======

::
    Circuit().m[0]
::

Run()
=======

::
    Circuit().h[0].cx[0,1].run()
::

last_result() Method
=======

::
    c = Circuit().h[0].cx[0,1].m[0]
    c.run() # array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j])
    c.last_result() # (1, 0)
::

Example
=======

2-qubit Grover
--------------
::

    from blueqat import Circuit
    c = Circuit().h[:2].cz[0,1].h[:].x[:].cz[0,1].x[:].h[:].m[:]
    c.run()
    print(c.last_result()) # => (1, 1)

Maxcut QAOA
-----------
::

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

Tutorial
======
日本語

https://github.com/mdrft/Blueqat/tree/master/tutorial_ja

Author
======
Takumi Kato (MDR),Yuichiro Minato(MDR)

Disclaimer
==========
Copyright 2018 The Blueqat Developers.
