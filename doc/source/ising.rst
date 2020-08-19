Ising/QUBO problem
==================

Blueqat "wq" module is a module for solbing Ising and Quadratic Unconstrained Binary Optimization (QUBO) problems. 
It includes local solver for Simulated Annealing (SA) and Simulated Quantum Annealing (SQA).
You can also submit problems to D-Wave cloud machine using wq module.

What is Ising model
-------------------

Real Quantum Annealing (QA) machines are built upon physical model called Ising model, which can be computationally
simulated on our laptops with algorithms called Simulated Annealing (SA) or Simulated Quantum Annealing (SQA).
1-dimensional Ising model is a 1D array of quantum bits (qubits), each of them has a ‘spin’ of +1 (up) or -1 (down).
2-dimensional Ising model is similar, it consists of a plainer lattice and has more adjacent qubits than 1D.
Although the complex physics may be overwhelming to you, wq module let you easily calculate the model without knowing much about them.


Combinatorial Optimization problem and SA
-----------------------------------------

Simulated Annealing (SA) can be used to solve combinatorial optimization problems of some forms, and Ising model is one of them.
Metropolis sampling based Monte-Carlo methods are used in such procedures.


Hamiltonian
-----------

To solve Ising model with SA, we have to set Jij/hi which describes how strong a pair of qubits are coupled, and how strong a qubit is biased, respectively.


Simulated Annealing and Simulated Quantum Annealing
---------------------------------------------------

We also have algorithm called Simulated Quantum Annealing(SQA) to solve Ising problems, in which quantum effects are taken into account.
The effect of quantum superposition is approximated by the parallel execution of different world-line,
by which we can effectively simulate wilder quantum nature. Technically, path-integral
methods based on Suzuki-Trotter matrix decomposition are used in the algorithm.


QUBO
----

Ising model problems are represented by Quadratic Unconstrained Binary Optimization (QUBO) problems.
Although variables in combinatorial optimization problems are of {0, 1}, quantum spins above are represented by {-1, 1},
so we have to transform their representation. wq module can automatically handle them, so you do not have to know about {-1, 1} things.


Checking and Verifying solutions
--------------------------------

We can calculate how good (or bad) a solution is by calculating ‘Energy’ of the solution, which can be done by a wq module one-liner.
Repeatedly solving Ising model, comparing that energy may let you know which is the best, or better answer to the problem.
If the problem is of NP, then checking whether the constraints are fulfilled can also be calculated in polynomial time.