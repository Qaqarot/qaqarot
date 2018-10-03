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
<a href="https://github.com/mdrft/Blueqat/tree/master/tutorial_ja">https://github.com/mdrft/Blueqat/tree/master/tutorial_ja</a>

Author
======
Takumi Kato (MDR)
Yuichiro Minato (MDR)

Disclaimer
==========
Copyright 2018 The Blueqat Developers.
