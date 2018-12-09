import numpy as np
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
            'X': sympy_gate.XGate(1).get_target_matrix(),
            'Y': sympy_gate.YGate(1).get_target_matrix(),
            'Z': sympy_gate.ZGate(1).get_target_matrix(),
            'H': sympy_gate.HadamardGate(1).get_target_matrix(),
            'T': sympy_gate.TGate(1).get_target_matrix(),
            'S': sympy_gate.PhaseGate(1).get_target_matrix(),
            'RX': Matrix([[cos(theta / 2), -I * sin(theta / 2)], [-I * sin(theta / 2), cos(theta / 2)]]),
            'RY': Matrix([[cos(theta / 2), -sin(theta / 2)], [sin(theta / 2), cos(theta / 2)]]),
            'RZ': Matrix([[exp(-I * theta / 2), 0], [0, exp(I * theta / 2)]]),
        }

    def _one_qubit_gate_noargs(self, gate, ctx):
        matrix = eye(2)
        targets = [i for i in gate.target_iter(ctx.n_qubits)]
        for idx in range(ctx.n_qubits):
            if idx in targets:
                matrix_of_gate = self.SYMPY_GATE[gate.uppername]
                matrix = matrix_of_gate if idx == 0 else TensorProduct(matrix_of_gate, matrix)
            elif not idx == 0:
                matrix = TensorProduct(eye(2), matrix)
        ctx.matrix_of_circuit = matrix * ctx.matrix_of_circuit
        return ctx

    def _one_qubit_gate_args_theta(self, gate, ctx):
        matrix = eye(2)
        targets = [i for i in gate.target_iter(ctx.n_qubits)]
        for idx in range(ctx.n_qubits):
            if idx in targets:
                matrix_of_gate = self.SYMPY_GATE[gate.uppername].subs(self.theta, gate.theta)
                matrix = matrix_of_gate if idx == 0 else TensorProduct(matrix_of_gate, matrix)
            elif not idx == 0:
                matrix = TensorProduct(eye(2), matrix)
        ctx.matrix_of_circuit = matrix * ctx.matrix_of_circuit
        return ctx

    gate_x = _one_qubit_gate_noargs
    gate_y = _one_qubit_gate_noargs
    gate_z = _one_qubit_gate_noargs
    gate_h = _one_qubit_gate_noargs
    gate_t = _one_qubit_gate_noargs
    gate_s = _one_qubit_gate_noargs
    gate_rx = _one_qubit_gate_args_theta
    gate_ry = _one_qubit_gate_args_theta
    gate_rz = _one_qubit_gate_args_theta

    def gate_cz(self, gate, ctx):
        print('gate_cz')
        for control, target in gate.control_target_iter(ctx[1]):
            print(control, target)
        return ctx

    def gate_cx(self, gate, ctx):
        print('gate_cx')
        for control, target in gate.control_target_iter(ctx[1]):
            print(control, target)
        return ctx

    def gate_measure(self, gate, ctx):
        return ctx

    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        return gates, _SympyBackendContext(n_qubits)

    def _postprocess_run(self, ctx):
        return ctx.matrix_of_circuit
