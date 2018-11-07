import json
import math
import random
import numpy as np
from ..gate import *
from .backendbase import Backend

class MQCBackend(Backend):
    """Backend for MDR Quantum Cloud."""
    def _preprocess_run(self, gates, args, kwargs):
        # TODO: Token and shots!
        n_qubits = find_n_qubits(gates)
        return gates, ([], n_qubits)

    def _postprocess_run(self, ctx):
        # TODO: POST to cloud. Not JSON dump!
        return json.dumps(ctx[0])

    def run(self, gates, args, kwargs):
        return self._run(gates, args, kwargs)

    def _one_qubit_gate_noargs(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append({gate.uppername: [idx]})
        return ctx

    gate_x = _one_qubit_gate_noargs
    gate_y = _one_qubit_gate_noargs
    gate_z = _one_qubit_gate_noargs
    gate_h = _one_qubit_gate_noargs
    gate_t = _one_qubit_gate_noargs
    gate_s = _one_qubit_gate_noargs

    def _two_qubit_gate_noargs(self, gate, ctx):
        for control, target in gate.control_target_iter(ctx[1]):
            ctx[0].append({gate.uppername: [control, target]})
        return ctx

    gate_cz = _two_qubit_gate_noargs
    gate_cx = _two_qubit_gate_noargs

    def _one_qubit_gate_args_theta(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append({gate.uppername: [gate.theta, idx]})
        return ctx

    gate_rx = _one_qubit_gate_args_theta
    gate_ry = _one_qubit_gate_args_theta
    gate_rz = _one_qubit_gate_args_theta

    def gate_measure(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append({"m": [idx]})
        return ctx
