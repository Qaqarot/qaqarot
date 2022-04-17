![Logo](https://raw.githubusercontent.com/Blueqat/Blueqat/master/blueqat_logo_blue.png)

# blueqat
A Quantum Computing SDK

### Version
[![Version](https://badge.fury.io/py/blueqat.svg)](https://badge.fury.io/py/blueqat)

### Build info
[![Build](https://circleci.com/gh/Blueqat/Blueqat.svg?style=svg)](https://circleci.com/gh/Blueqat/Blueqat)

### Tutorial
https://github.com/Blueqat/Blueqat-tutorials

### Install
```
git clone https://github.com/Blueqat/Blueqat
cd Blueqat
pip3 install -e .
```

or

```
pip3 install blueqat
```

### Circuit
```python
from blueqat import Circuit
import math

#number of qubit is not specified
c = Circuit()

#if you want to specified the number of qubit
c = Circuit(3) #3qubits
```

### Method Chain
```python
# write as chain
Circuit().h[0].x[0].z[0]

# write in separately
c = Circuit().h[0]
c.x[0].z[0]
```

### Slice
```python
Circuit().z[1:3] # Zgate on 1,2
Circuit().x[:3] # Xgate on (0, 1, 2)
Circuit().h[:] # Hgate on all qubits
Circuit().x[1, 2] # 1qubit gate with comma
```

### Rotation Gate
```python
Circuit().rz(math.pi / 4)[0]
```

### Measurement
```python
Circuit().m[0]
```

### Run() to get state vector
```python
Circuit().h[0].cx[0,1].run()
# => array([0.70710678+0.j, 0.+0.j, 0.+0.j, 0.70710678+0.j])
```

### Run(shots=n)
```python
c = Circuit().h[0].cx[0,1].m[:]
c.run(shots=100)
# => Counter({'00': 48, '11': 52})
```

### State Vector Initialization
```python
Circuit(2).m[:].run(shots=100, initial=np.array([0, 1, 1, 0])/np.sqrt(2))
# => Counter({'10': 51, '01': 49})
```

### Sympy Unitary
```python
Circuit().h[0].cx[0, 1].run(backend="sympy_unitary")
```

### Blueqat to QASM
```python
Circuit().h[0].to_qasm()
    
#OPENQASM 2.0;
#include "qelib1.inc";
#qreg q[1];
#creg c[1];
#h q[0];
```

### Hamiltonian
```python
from blueqat.pauli import *

hamiltonian1 = (1.23 * Z[0] + 4.56 * X[1] * Z[2]) ** 2
hamiltonian2 = (2.46 * Y[0] + 5.55 * Z[1] * X[2] * X[1]) ** 2
hamiltonian = hamiltonian1 + hamiltonian2
print(hamiltonian)
    
# => 7.5645*I + 5.6088*Z[0]*X[1]*Z[2] + 5.6088*X[1]*Z[2]*Z[0] + 20.793599999999998*X[1]*Z[2]*X[1]*Z[2] + 13.652999999999999*Y[0]*Z[1]*X[2]*X[1] + 13.652999999999999*Z[1]*X[2]*X[1]*Y[0] + 30.8025*Z[1]*X[2]*X[1]*Z[1]*X[2]*X[1]
```

### Simplify the Hamiltonian
```python
hamiltonian = hamiltonian.simplify()
print(hamiltonian)

#=>-2.4444000000000017*I + 27.305999999999997j*Y[0]*Y[1]*X[2] + 11.2176*Z[0]*X[1]*Z[2]
```

### QUBO Hamiltonian
```python
from blueqat.pauli import qubo_bit as q

hamiltonian = -3*q(0)-3*q(1)-3*q(2)-3*q(3)-3*q(4)+2*q(0)*q(1)+2*q(0)*q(2)+2*q(0)*q(3)+2*q(0)*q(4)
print(hamiltonian)
    
# => -5.5*I + 1.0*Z[1] + 1.0*Z[2] + 1.0*Z[3] + 1.0*Z[4] + 0.5*Z[0]*Z[1] + 0.5*Z[0]*Z[2] + 0.5*Z[0]*Z[3] - 0.5*Z[0] + 0.5*Z[0]*Z[4]
```

### VQE
```python
from blueqat import Circuit
from blueqat.pauli import X, Y, Z, I
from blueqat.vqe import AnsatzBase, Vqe

class OneQubitAnsatz(AnsatzBase):
    def __init__(self, hamiltonian):
        super().__init__(hamiltonian.to_expr(), 2)
        self.step = 1

    def get_circuit(self, params):
        a, b = params
        return Circuit().rx(a)[0].rz(b)[0]

# hamiltonian
h = 1.23 * I - 4.56 * X(0) + 2.45 * Y(0) + 2.34 * Z(0)

result = Vqe(OneQubitAnsatz(h)).run()
print(runner.ansatz.get_energy_sparse(result.circuit))

# => -4.450804074762511
```

### Time Evolution
```python
hamiltonian = [1.0*Z(0), 1.0*X[0]]
a = [term.get_time_evolution() for term in hamiltonian]

time_evolution = Circuit().h[0]
for evo in a:
    evo(time_evolution, np.random.rand())
    
print(time_evolution)

# => Circuit(1).h[0].rz(-1.4543063361067243)[0].h[0].rz(-1.8400416676737137)[0].h[0]
```

### QAOA
```python
from blueqat import vqe
from blueqat.pauli import *
from blueqat.pauli import qubo_bit as q
    
hamiltonian = q(0)-3*q(1)+2*q(0)*q(1)
#hamiltonian = -0.5*I - Z[0] + 1.0*Z[1] + 0.5*Z[0]*Z[1]
step = 2

result = vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, step)).run()
print(result.most_common(4))
    
# => (((0, 1), 0.9874053861648978), ((1, 0), 0.00967786055983366), ((0, 0), 0.0014583766376339746), ((1, 1), 0.0014583766376339703))
```

### QAOA Mixer
```python
hamiltonian = q(0)-3*q(1)+2*q(0)*q(1)
step = 2
init = Circuit().h[0].cx[0,1].x[1]
mixer = (X[0]*X[1] + Y[0]*Y[1])*0.5

result = vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, step, init, mixer)).run()
print(result.most_common(4))
    
# => (((0, 1), 0.9999886003516928), ((1, 0), 1.1399648305716677e-05), ((0, 0), 1.5176327961771419e-31), ((1, 1), 4.006785342235446e-32))
```

### Select Scipy Minimizer
```python
minimizer = vqe.get_scipy_minimizer(method="COBYLA")
result = vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, step), minimizer=minimizer).run()
```

### Circuit Drawing Backend
```python
from blueqat import vqe
from blueqat.pauli import *
from blueqat.pauli import qubo_bit as q

#hamiltonian = q(0)-3*q(1)+2*q(0)*q(1)+3*q(2)*q(3)+q(4)*q(7)
hamiltonian = Z[0]-3*Z[1]+2*Z[0]*Z[1]+3*Z[2]*Z[3]+Z[4]
step = 8

result = vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, step)).run()
result.circuit.run(backend='draw')
```

![draw](https://raw.githubusercontent.com/Blueqat/Blueqat/master/draw.png)

### Cloud System Connection (API Key is required)
```python
from bqcloud import register_api
api = register_api("Your API Key")

from bqcloud import load_api
api = load_api()

from blueqat import Circuit
from bqcloud import Device

task = api.execute(Circuit().h[0].cx[0, 1], Device.IonQDevice, 10)
#task = api.execute(Circuit().h[0].cx[0, 1], Device.AspenM1, 10)

# Wait 10 sec. If complete, result is returned, otherwise, None is returned.
result = task.wait(timeout=10)

if result:
    print(result.shots())
else:
    print("timeout")
```

### Example1: GHZ
```python
from blueqat import Circuit
Circuit().h[0].cx[0,1].cx[1,2].m[:].run(shots=100)

# => Counter({'000': 48, '111': 52})
```

### Document
https://blueqat.readthedocs.io/en/latest/

### Contributors
[Contributors](https://github.com/Blueqat/Blueqat/graphs/contributors)

### Disclaimer
Copyright 2022 The Blueqat Developers.
