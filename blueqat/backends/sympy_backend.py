# Copyright 2019 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warnings
from functools import reduce
from .backendbase import Backend


def lazy_import():
    global eye, symbols, sin, cos, exp, sqrt, pi, I, Matrix, sympy_gate, TensorProduct, sympy
    from sympy import eye, symbols, sin, cos, exp, sqrt, pi, I, Matrix
    from sympy.physics.quantum import TensorProduct
    import sympy


def _angle_simplify(ang):
    if isinstance(ang, float):
        nsimp = sympy.nsimplify(ang / np.pi)
        numer, denom = nsimp.as_numer_denom()
        if denom < 1e12:
            return pi * numer / denom
    return ang


class _SympyBackendContext:
    def __init__(self, n_qubits, ignore_global):
        self.n_qubits = n_qubits
        self.matrix_of_circuit = eye(2**n_qubits)
        self.ignore_global = ignore_global


class SympyBackend(Backend):
    def __init__(self):
        try:
            lazy_import()
        except ImportError:
            raise ImportError(
                'sympy_unitary requires sympy. Please install before call this option.'
            ) from None
        theta, phi, lam, gamma = symbols('theta phi lam gamma')
        self.theta = theta
        self.phi = phi
        self.lam = lam
        self.gamma = gamma
        self.SYMPY_GATE = {
            '_C0':
            Matrix([[1, 0], [0, 0]]),
            '_C1':
            Matrix([[0, 0], [0, 1]]),
            'H':
            Matrix([[1, 1], [1, -1]]) / sqrt(2),
            'PHASE':
            Matrix([[1, 0], [0, exp(I * theta)]]),
            'RX':
            Matrix([[cos(theta / 2), -I * sin(theta / 2)],
                    [-I * sin(theta / 2), cos(theta / 2)]]),
            'RY':
            Matrix([[cos(theta / 2), -sin(theta / 2)],
                    [sin(theta / 2), cos(theta / 2)]]),
            'RZ':
            Matrix([[cos(theta / 2) - I * sin(theta / 2), 0],
                    [0, cos(theta / 2) + I * sin(theta / 2)]]),
            'SX':
            Matrix([[1 + I, 1 - I], [1 - I, 1 + I]]) / 2,
            'SXDG':
            Matrix([[1 - I, 1 + I], [1 + I, 1 - I]]) / 2,
            'U':
            Matrix(
                [[cos(theta / 2), -(cos(lam) + I * sin(lam)) * sin(theta / 2)],
                 [(cos(phi) + I * sin(phi)) * sin(theta / 2),
                  (cos(phi + lam) + I * sin(phi + lam)) * cos(theta / 2)]]) *
            (cos(gamma) + I * sin(gamma)),
            'X':
            Matrix([[0, 1], [1, 0]]),
            'Y':
            Matrix([[0, -I], [I, 0]]),
            'Z':
            Matrix([[1, 0], [0, -1]]),
        }

    def _create_matrix_of_one_qubit_gate(self, n_qubits, targets,
                                         matrix_of_gate):
        targets = [idx for idx in targets]
        gates = []
        for idx in range(n_qubits):
            if idx in targets:
                gates.append(matrix_of_gate)
            else:
                gates.append(eye(2))
        return reduce(TensorProduct, reversed(gates))

    def _create_matrix_of_one_qubit_gate_circuit(self, gate, ctx,
                                                 matrix_of_gate):
        n_qubits = ctx.n_qubits
        m = self._create_matrix_of_one_qubit_gate(n_qubits,
                                                  gate.target_iter(n_qubits),
                                                  matrix_of_gate)
        ctx.matrix_of_circuit = m * ctx.matrix_of_circuit
        return ctx

    def _one_qubit_gate_noargs(self, gate, ctx):
        matrix_of_gate = self.SYMPY_GATE[gate.uppername]
        return self._create_matrix_of_one_qubit_gate_circuit(
            gate, ctx, matrix_of_gate)

    def _one_qubit_gate_args_theta(self, gate, ctx):
        theta = _angle_simplify(gate.theta)
        matrix_of_gate = self.SYMPY_GATE[gate.uppername].subs(
            self.theta, theta)
        return self._create_matrix_of_one_qubit_gate_circuit(
            gate, ctx, matrix_of_gate)

    gate_x = _one_qubit_gate_noargs
    gate_y = _one_qubit_gate_noargs
    gate_z = _one_qubit_gate_noargs
    gate_h = _one_qubit_gate_noargs
    gate_sx = _one_qubit_gate_noargs
    gate_sxdg = _one_qubit_gate_noargs
    gate_rx = _one_qubit_gate_args_theta
    gate_ry = _one_qubit_gate_args_theta
    gate_rz = _one_qubit_gate_args_theta
    gate_phase = _one_qubit_gate_args_theta

    def gate_u(self, gate, ctx):
        theta, phi, lam, gamma = map(_angle_simplify, gate.params)
        matrix_of_gate = self.SYMPY_GATE['U'].subs(
            [(self.lam, lam), (self.phi, phi), (self.theta, theta),
             (self.gamma, gamma)],
            simultaneous=True)
        return self._create_matrix_of_one_qubit_gate_circuit(
            gate, ctx, matrix_of_gate)

    def _create_matrix_of_two_qubit_gate(self,
                                         n_qubits,
                                         gate,
                                         control,
                                         target,
                                         subs=None):
        matrix_of_target_gate = self.SYMPY_GATE[gate.uppername[1:]]
        if subs:
            matrix_of_target_gate = matrix_of_target_gate.subs(
                subs, simultaneous=True)
        c0 = self._create_matrix_of_one_qubit_gate(n_qubits, [control],
                                                   self.SYMPY_GATE['_C0'])
        c1 = self._create_matrix_of_one_qubit_gate(n_qubits, [control],
                                                   self.SYMPY_GATE['_C1'])
        tgt = self._create_matrix_of_one_qubit_gate(n_qubits, [target],
                                                    matrix_of_target_gate)
        return c0 + tgt * c1

    def _two_qubit_gate_noargs(self, gate, ctx):
        n_qubits = ctx.n_qubits
        for control, target in gate.control_target_iter(n_qubits):
            m = self._create_matrix_of_two_qubit_gate(n_qubits, gate, control,
                                                      target)
            ctx.matrix_of_circuit = m * ctx.matrix_of_circuit
        return ctx

    gate_cx = _two_qubit_gate_noargs
    gate_cz = _two_qubit_gate_noargs

    def gate_cu(self, gate, ctx):
        theta, phi, lam, gamma = map(_angle_simplify, gate.params)
        n_qubits = ctx.n_qubits
        subs = [(self.lam, lam), (self.phi, phi), (self.theta, theta),
                (self.gamma, gamma)]
        for control, target in gate.control_target_iter(n_qubits):
            m = self._create_matrix_of_two_qubit_gate(n_qubits, gate, control,
                                                      target, subs)
            ctx.matrix_of_circuit = m * ctx.matrix_of_circuit
        return ctx

    def _two_qubit_gate_args_theta(self, gate, ctx):
        theta = _angle_simplify(gate.theta)
        matrix_of_gate = self.SYMPY_GATE[gate.uppername].subs(
            self.theta, theta)
        for control, target in gate.control_target_iter(ctx.n_qubits):
            control_gate_of_matrix = self._create_matrix_of_two_qubit_gate(
                gate, ctx, control, target)
            ctx.matrix_of_circuit = control_gate_of_matrix * ctx.matrix_of_circuit
        return ctx

    def _warn_unavailable_operation(self, gate, ctx):
        warnings.warn(
            f'`{gate.uppername}` operation is ignored because unavailable in this backend.'
        )
        return ctx

    gate_measure = _warn_unavailable_operation
    gate_reset = _warn_unavailable_operation

    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        kwargs.setdefault('ignore_global', False)
        return gates, _SympyBackendContext(n_qubits, kwargs['ignore_global'])

    def _postprocess_run(self, ctx):
        if ctx.ignore_global:
            mat = ctx.matrix_of_circuit
            for i in range(2**ctx.n_qubits):
                if mat[i, i] != 0:
                    return mat * (abs(mat[i, i]) / mat[i, i])
        return ctx.matrix_of_circuit
