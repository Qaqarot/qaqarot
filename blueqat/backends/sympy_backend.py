import numpy as np
from functools import reduce
from .backendbase import Backend


def lazy_import():
    global eye, symbols, sin, cos, exp, sqrt, pi, I, Matrix, sympy_gate, TensorProduct, sympy
    from sympy import eye, symbols, sin, cos, exp, sqrt, pi, I, Matrix
    from sympy.physics.quantum import gate as sympy_gate, TensorProduct
    import sympy


def _angle_simplify(ang):
    if isinstance(ang, float):
        nsimp = sympy.nsimplify(ang / np.pi)
        numer, denom = nsimp.as_numer_denom()
        if denom < 1e12:
            return pi * numer / denom
    return ang


class _SympyBackendContext:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.matrix_of_circuit = eye(2 ** n_qubits)


class SympyBackend(Backend):

    def __init__(self):
        try:
            lazy_import()
        except ImportError:
            raise ImportError('sympy_unitary requires sympy. Please install before call this option.')
        theta, phi, lambd = symbols('theta phi lambd')
        self.theta = theta
        self.phi = phi
        self.lambd = lambd
        self.SYMPY_GATE = {
            'X': sympy_gate.X(0).get_target_matrix(),
            'Y': sympy_gate.Y(0).get_target_matrix(),
            'Z': sympy_gate.Z(0).get_target_matrix(),
            'H': sympy_gate.H(0).get_target_matrix(),
            'T': sympy_gate.T(0).get_target_matrix(),
            'S': sympy_gate.S(0).get_target_matrix(),
            'RX': Matrix([[cos(theta / 2), -I * sin(theta / 2)], [-I * sin(theta / 2), cos(theta / 2)]]),
            'RY': Matrix([[cos(theta / 2), -sin(theta / 2)], [sin(theta / 2), cos(theta / 2)]]),
            'RZ': Matrix([[exp(-I * theta / 2), 0], [0, exp(I * theta / 2)]]),
            'TARGET_CX': sympy_gate.X(0).get_target_matrix(),
            'TARGET_CZ': sympy_gate.Z(0).get_target_matrix(),
            'U1': Matrix([[exp(-I * lambd / 2), 0], [0, exp(I * lambd / 2)]]),
            'U2': Matrix([
                [exp(-I * (phi + lambd) / 2) / sqrt(2), -exp(-I * (phi - lambd) / 2) / sqrt(2)],
                [exp(I * (phi - lambd) / 2) / sqrt(2), exp(I * (phi + lambd) / 2) / sqrt(2)]]),
            'U3': Matrix([
                [exp(-I * (phi + lambd) / 2) * cos(theta / 2), -exp(-I * (phi - lambd) / 2) * sin(theta / 2)],
                [exp(I * (phi - lambd) / 2) * sin(theta / 2), exp(I * (phi + lambd) / 2) * cos(theta / 2)]]),
        }

    def _create_matrix_of_one_qubit_gate_circuit(self, gate, ctx, matrix_of_gate):
        targets = [idx for idx in gate.target_iter(ctx.n_qubits)]
        gates = []
        for idx in range(ctx.n_qubits):
            if idx in targets:
                gates.append(matrix_of_gate)
            else:
                gates.append(eye(2))
        ctx.matrix_of_circuit = reduce(TensorProduct, reversed(gates)) * ctx.matrix_of_circuit
        return ctx

    def _one_qubit_gate_noargs(self, gate, ctx):
        matrix_of_gate = self.SYMPY_GATE[gate.uppername]
        return self._create_matrix_of_one_qubit_gate_circuit(gate, ctx, matrix_of_gate)

    def _one_qubit_gate_args_theta(self, gate, ctx):
        theta = _angle_simplify(gate.theta)
        matrix_of_gate = self.SYMPY_GATE[gate.uppername].subs(self.theta, theta)
        return self._create_matrix_of_one_qubit_gate_circuit(gate, ctx, matrix_of_gate)

    gate_x = _one_qubit_gate_noargs
    gate_y = _one_qubit_gate_noargs
    gate_z = _one_qubit_gate_noargs
    gate_h = _one_qubit_gate_noargs
    gate_t = _one_qubit_gate_noargs
    gate_s = _one_qubit_gate_noargs
    gate_rx = _one_qubit_gate_args_theta
    gate_ry = _one_qubit_gate_args_theta
    gate_rz = _one_qubit_gate_args_theta

    def _one_qubit_gate_ugate(self, gate, ctx):
        if len(gate.params) == 3:
            phi = _angle_simplify(gate.phi)
            theta = _angle_simplify(gate.theta)
        elif len(gate.params) == 2:
            phi = _angle_simplify(gate.phi)
            theta = pi / 2
        else:
            phi = theta = 0
        lambd = _angle_simplify(gate.lambd)
        matrix_of_gate = self.SYMPY_GATE[gate.uppername].subs([
            (self.lambd, lambd),
            (self.phi, phi),
            (self.theta, theta)], simultaneous=True)
        return self._create_matrix_of_one_qubit_gate_circuit(gate, ctx, matrix_of_gate)

    gate_u1 = _one_qubit_gate_ugate
    gate_u2 = _one_qubit_gate_ugate
    gate_u3 = _one_qubit_gate_ugate

    def _create_control_gate_of_matrix(self, type_of_gate, control, target):
        unit_of_upper_triangular_matrix = Matrix([[1, 0], [0, 0]])
        unit_of_lower_triangular_matrix = Matrix([[0, 0], [0, 1]])

        control_gates = [eye(2), unit_of_upper_triangular_matrix]
        target_gates = [self.SYMPY_GATE['TARGET_%s' % type_of_gate], unit_of_lower_triangular_matrix]
        number_between_of_qubits = abs(target - control) - 1
        if number_between_of_qubits != 0:
            between_unit_gate = eye(2 ** number_between_of_qubits)
            control_gates.insert(1, between_unit_gate)
            target_gates.insert(1, between_unit_gate)

        control_gate_of_matrix = reduce(TensorProduct, control_gates) + reduce(TensorProduct, target_gates)
        if control < target or type_of_gate == 'CZ':
            return control_gate_of_matrix
        else:
            gates = [self.SYMPY_GATE['H'], self.SYMPY_GATE['H']]
            if number_between_of_qubits != 0:
                gates.insert(1, eye(2 ** number_between_of_qubits))
            transformation_of_matrix = reduce(TensorProduct, gates)
            return transformation_of_matrix * control_gate_of_matrix * transformation_of_matrix

    def _embedded_control_gate(self, n_qubits, upper_qubit, lower_qubit, control_gate):
        gates = [control_gate]
        if upper_qubit > 0:
            gates.append(eye(2**upper_qubit))

        number_of_lower_qubits = n_qubits - lower_qubit - 1
        if number_of_lower_qubits > 0:
            gates.insert(0, eye(2**number_of_lower_qubits))

        return reduce(TensorProduct, gates)

    def _two_qubit_gate_noargs(self, gate, ctx):
        for control, target in gate.control_target_iter(ctx.n_qubits):
            control_gate_of_matrix = self._create_control_gate_of_matrix(gate.uppername, control, target)
            if ctx.n_qubits == abs(target - control) + 1:
                ctx.matrix_of_circuit = control_gate_of_matrix * ctx.matrix_of_circuit
            elif control > target:
                ctx.matrix_of_circuit = self._embedded_control_gate(ctx.n_qubits, target, control, control_gate_of_matrix) * ctx.matrix_of_circuit
            else:
                ctx.matrix_of_circuit = self._embedded_control_gate(ctx.n_qubits, control, target, control_gate_of_matrix) * ctx.matrix_of_circuit
        return ctx

    gate_cx = _two_qubit_gate_noargs
    gate_cz = _two_qubit_gate_noargs

    def gate_measure(self, gate, ctx):
        return ctx

    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        return gates, _SympyBackendContext(n_qubits)

    def _postprocess_run(self, ctx):
        return ctx.matrix_of_circuit
