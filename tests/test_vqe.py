import pytest
from blueqat import Circuit
from blueqat.vqe import expect


@pytest.mark.parametrize('qubits, meas, expected', [
    (
        Circuit(4).h[:].run(), (1, 2),
        {
            (0, 0): 0.25,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.25,
        }
    ),
    (
        Circuit(4).h[1:3].run(), (1, 2),
        {
            (0, 0): 0.25,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.25,
        }
    ),
])
def test_expect(qubits, meas, expected):

    def assert_sampling(actual, expected, eps=0.0000001):
        assert list(actual.keys()) == list(expected.keys())
        for k in expected:
            assert abs(actual[k] - expected[k]) < eps

    ex = expect(qubits, meas)
    assert_sampling(ex, expected)
