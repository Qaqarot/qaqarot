# A quantum gate simulator.

## Install
```bash
git clone https://github.com/mdrft/blueqat
cd blueqat
pip3 install -e .
```

## Example: 2-qubit Grover

```python
from blueqat import Circuit
c = Circuit().h[:2].cz[0,1].h[:].x[:].cz[0,1].x[:].h[:].m[:]
c.run()
print(c.last_result()) # => (1, 1)
```

## Example: Maxcut QAOA

```bash
python examples/examples_maxcut_qaoa.py
```
