# A quantum gate simulator.

## Install

```bash
git clone https://github.com/mdrft/blueqat
cd blueqat
pip3 install -e .
```

or

```bash
pip3 install blueqat
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
python examples/maxcut_qaoa.py
```

## Author

## Disclaimer

Copyright 2018 The Blueqat Developers.

