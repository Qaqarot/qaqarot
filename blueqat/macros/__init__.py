"""This module is experiment features. We may delete or change provided functions.

The module provides builtin macros."""

import math
from typing import Any, Sequence, Optional

from blueqat import Circuit
from blueqat.decorators import circuitmacro
from blueqat.utils import gen_gray_controls


@circuitmacro
def draw(c: Circuit, **kwargs) -> Any:
    """Draw the circuit. This function requires Qiskit."""
    return c.run(backend='ibmq', returns="draw", **kwargs)


@circuitmacro
def margolus(c: Circuit, c0: int, c1: int) -> Circuit:
    """Simplified Toffoli gate implementation by Margolus.

    This gate is also as know as relative Toffoli gate.
    It's almost as same as Toffoli gate, but only relative phases are different.
    Refer https://arxiv.org/abs/quant-ph/0312225v1 for details.
    (Remarks: It's also described in Nielsen & Chuang, Exercise 4.26.)"""
    c.ry(math.pi * 0.25)[t].cx[c1, t].ry(math.pi * 0.25)[t].cx[c0, t]
    c.ry(math.pi * -0.25)[t].cx[c1, t].ry(math.pi * -0.25)[t]
    return c


@circuitmacro
def c3z(c: Circuit, c0: int, c1: int, c2: int, t: int) -> Circuit:
    return c.mcz_gray([c0, c1, c2], t)


@circuitmacro
def c4z(c: Circuit, c0: int, c1: int, c2: int, c3: int, t: int) -> Circuit:
    return c.mcz_gray([c0, c1, c2, c3], t)


@circuitmacro
def c3x(c: Circuit, c0: int, c1: int, c2: int, t: int) -> Circuit:
    return c.h[t].c3z(c0, c1, c2, t).h[t]


@circuitmacro
def c4x(c: Circuit, c0: int, c1: int, c2: int, c3: int, t: int) -> Circuit:
    return c.h[t].c4z(c0, c1, c2, c3, t).h[t]


@circuitmacro
def mcx_gray(c: Circuit, ctrl: Sequence[int], target: int) -> Circuit:
    return c.h[target].mcz_gray(ctrl, target).h[target]


@circuitmacro
def mcx_with_ancilla(c: Circuit,
                     ctrl: Sequence[int],
                     target: int,
                     ancilla: Optional[int] = None) -> Circuit:
    if ancilla is None:
        ancilla = c.n_qubits
    pass


@circuitmacro
def mcz_gray(c: Circuit, ctrl: Sequence[int], target: int) -> Circuit:
    n_ctrl = len(ctrl)
    if n_ctrl == 0:
        return c.z[target]
    angles = [math.pi / 2**(n_ctrl - 1), -math.pi / 2**(n_ctrl - 1)]
    for c0, c1, parity in gen_gray_controls(n_ctrl):
        if c0 >= 0:
            c.cx[ctrl[c0], ctrl[c1]]
        c.cr(angles[parity])[ctrl[c1], target]
    return c


@circuitmacro
def mcz_with_ancilla(c: Circuit,
                     ctrl: Sequence[int],
                     target: int,
                     ancilla: Optional[int] = None) -> Circuit:
    return c.h[target].mcx_with_ancilla(ctrl, target, ancilla).h[target]


@circuitmacro
def mcrx_gray(c: Circuit, theta: float, ctrl: Sequence[int],
              target: int) -> Circuit:
    n_ctrl = len(ctrl)
    if n_ctrl == 0:
        return c.rx(theta)[target]
    angles = [theta / 2**(n_ctrl - 1), -theta / 2**(n_ctrl - 1)]
    for c0, c1, parity in gen_gray_controls(n_ctrl):
        if c0 >= 0:
            c.cx[ctrl[c0], ctrl[c1]]
        c.crx(angles[parity])[ctrl[c1], target]
    return c


@circuitmacro
def mcry_gray(c: Circuit, theta: float, ctrl: Sequence[int],
              target: int) -> Circuit:
    n_ctrl = len(ctrl)
    if n_ctrl == 0:
        return c.ry(theta)[target]
    angles = [theta / 2**(n_ctrl - 1), -theta / 2**(n_ctrl - 1)]
    for c0, c1, parity in gen_gray_controls(n_ctrl):
        if c0 >= 0:
            c.cx[ctrl[c0], ctrl[c1]]
        c.cry(angles[parity])[ctrl[c1], target]
    return c


@circuitmacro
def mcrz_gray(c: Circuit, theta: float, ctrl: Sequence[int],
              target: int) -> Circuit:
    n_ctrl = len(ctrl)
    if n_ctrl == 0:
        return c.rz(theta)[target]
    angles = [theta / 2**(n_ctrl - 1), -theta / 2**(n_ctrl - 1)]
    for c0, c1, parity in gen_gray_controls(n_ctrl):
        if c0 >= 0:
            c.cx[ctrl[c0], ctrl[c1]]
        c.crz(angles[parity])[ctrl[c1], target]
    return c


@circuitmacro
def mcu_gray(c: Circuit, theta: float, phi: float, lam: float, gamma: float,
             ctrl: Sequence[int], target: int) -> Circuit:
    pass
