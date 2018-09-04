# A quantum gate simulator.

## Example: 2-qubit Grover

```
from circuit import Circuit
c = Circuit().h[:2].cz[0,1].h[:].x[:].cz[0,1].x[:].h[:].m[:]
c.run()
print(c.last_result()) # => (1, 1)
```

## Example: Maxcut QAOA

```
from examples_qaoa import *
result = maxcut_qaoa(2, [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2), (4, 0), (4, 3)])
print("""
       {4}
      / \\
     {0}---{3}
     | x |
     {1}---{2}
""".format(*result))
```
