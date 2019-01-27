import numpy as np
from functools import reduce
from .backendbase import Backend


is_import = True
try:
    from sympy import eye, symbols, sin, cos, exp, I, Matrix
    from sympy.physics.quantum import gate as sympy_gate, TensorProduct
except ImportError:
    is_import = False


class _SympyBackendContext:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.matrix_of_circuit = eye(2 ** n_qubits)


class SympyBackend(Backend):

    def __init__(self):
        if not is_import:
            raise ImportError('sympy_unitary requires sympy. Please install before call this option.')
        theta = symbols('theta')
        self.theta = theta
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
            'TAEGET_CX': sympy_gate.X(0).get_target_matrix(),
            'TAEGET_CZ': sympy_gate.Z(0).get_target_matrix(),
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
        matrix_of_gate = self.SYMPY_GATE[gate.uppername].subs(self.theta, gate.theta)
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

    def _two_qubit_gate_noargs(self, gate, ctx):
        for control, target in gate.control_target_iter(ctx.n_qubits):
            control_gate_of_matrix = self._create_control_gate_of_matrix(gate.uppername, control, target)
            if ctx.n_qubits == abs(target - control) + 1:
                ctx.matrix_of_circuit = control_gate_of_matrix * ctx.matrix_of_circuit
            else:
                matrix = control_gate_of_matrix
                number_of_upper_qubits = control
                if number_of_upper_qubits > 0:
                    matrix = TensorProduct(matrix, eye(2 ** number_of_upper_qubits))
                number_of_lower_qubits = ctx.n_qubits - target - 1
                if number_of_lower_qubits > 0:
                    matrix = TensorProduct(eye(2 ** number_of_lower_qubits), control_gate_of_matrix)
                ctx.matrix_of_circuit = matrix * ctx.matrix_of_circuit
        return ctx

    def _create_control_gate_of_matrix(self, type_of_gate, control, target):
        number_between_of_qubits = abs(target - control) - 1
        unit_of_upper_triangular_matrix = Matrix([[1, 0], [0, 0]])
        control_of_matrix = eye(2 ** (number_between_of_qubits + 1))
        unit_of_lower_triangular_matrix = Matrix([[0, 0], [0, 1]])
        target_of_matrix = self.SYMPY_GATE['TAEGET_%s' % type_of_gate]

        if number_between_of_qubits != 0:
            target_of_matrix = TensorProduct(eye(2 ** number_between_of_qubits), target_of_matrix)

        control_gate_of_matrix = TensorProduct(unit_of_upper_triangular_matrix, control_of_matrix) + TensorProduct(unit_of_lower_triangular_matrix, target_of_matrix)
        if control < target or type_of_gate == 'CZ':
            return control_gate_of_matrix
        else:
            transformation_of_matrix = self.SYMPY_GATE['H']
            if number_between_of_qubits != 0:
                transformation_of_matrix = TensorProduct(eye(2 ** number_between_of_qubits), transformation_of_matrix)
            transformation_of_matrix = TensorProduct(transformation_of_matrix, transformation_of_matrix)
            return transformation_of_matrix * control_gate_of_matrix * transformation_of_matrix

    gate_cx = _two_qubit_gate_noargs
    gate_cz = _two_qubit_gate_noargs

    def gate_measure(self, gate, ctx):
        return ctx

    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        return gates, _SympyBackendContext(n_qubits)

    def _postprocess_run(self, ctx):
        return ctx.matrix_of_circuit
