import types
import pytest
from blueqat import (
    Circuit,
)
from blueqat.vqe import (
    VqeResult,
    non_sampling_sampler,
    expect,
)


class TestVqeResult(object):

    @pytest.mark.parametrize('probs, expected', [
        (
            {
                (0, 0): 0.185990056957774,
                (0, 1): 0.31400994304222624,
                (1, 0): 0.3140099430422262,
                (1, 1): 0.18599005695777357
            },
            (((0, 1), 0.31400994304222624),)
        ),
    ])
    def test_most_common(self, probs, expected):
        result = VqeResult()
        result._probs = probs
        assert result.most_common() == expected


@pytest.mark.parametrize('circuit, meas, expected', [
    (
        Circuit(4),
        (1, 2),
        {(0, 0): 1.0},
    ),
    (
        Circuit(2),
        (0, 1),
        {(0, 0): 1.0},
    ),
])
def test_non_sampling_sampler(circuit, meas, expected):
    assert isinstance(non_sampling_sampler, types.FunctionType)
    assert non_sampling_sampler(circuit, meas) == expected


@pytest.mark.parametrize('qubits, meas, expected', [
    (
        Circuit(4).h[:].run(),
        (1, 2),
        {
            (0, 0): 0.25,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.25,
        }
    ),
    (
        Circuit(4).h[1:3].run(),
        (1, 2),
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
        assert list(sorted(actual.keys())) == list(sorted(expected.keys()))
        for k in expected:
            assert abs(actual[k] - expected[k]) < eps

    assert isinstance(expect, types.FunctionType)
    assert_sampling(expect(qubits, meas), expected)
