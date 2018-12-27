from collections import Counter
import pytest
from blueqat.utils import to_inttuple

@pytest.mark.parametrize('arg, expect', [
    ("01011", (0, 1, 0, 1, 1)),
    ({"00011": 2, "10100": 3}, {(0, 0, 0, 1, 1): 2, (1, 0, 1, 0, 0): 3}),
    (Counter({"00011": 2, "10100": 3}), Counter({(0, 0, 0, 1, 1): 2, (1, 0, 1, 0, 0): 3}))
])
def test_to_inttuple(arg, expect):
    assert to_inttuple(arg) == expect
