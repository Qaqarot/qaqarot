# A quantum gate simulator.

## Example: 2-qubit Grover

```
from circuit import Circuit
c = Circuit().h[:2].cz[0,1].h[:].x[:].cz[0,1].x[:].h[:].m[:]
c.run()
print(c.last_result()) # => (1, 1)
```
