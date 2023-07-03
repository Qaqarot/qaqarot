from .backendbase import Backend
from ..circuit import Circuit
from ..gate import *

import numpy as np
import math


class Quimb(Backend):
    
    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        import quimb.tensor as qtn
        
        circ = qtn.Circuit(n_qubits)
        regs = list(range(n_qubits))
        
        if "shots" in kwargs:
            n_shots = kwargs["shots"]
        else:
            n_shots = 1
        if "amplitude" in kwargs:
            amp = kwargs["amplitude"]
        else:
            amp = None
        if "hamiltonian" in kwargs:
            hami = kwargs["hamiltonian"]
        else:
            hami = None
        return gates, (circ, regs, n_qubits, n_shots, amp, hami)

    def _postprocess_run(self, ctx):
        import quimb as qu
        import collections
        import re
        
        if ctx[5]:
            hami = ctx[5].simplify()
            c = 0

            # because term not included into the array
            if len(re.split('[+,-]', str(hami))) == 1:
                kron_hami = qu.pauli(hami[0][0].op)
                qu_list = [hami[0][0][0]]
                for i in range(1, len(hami[0])):
                    kron_hami = qu.kron(kron_hami, qu.pauli(hami[0][i].op))
                    qu_list.append(hami[0][i][0])
                
                c += hami[1] * ctx[0].local_expectation(kron_hami, tuple(qu_list)).real
            
            else:
                for item in hami[0]:
                    if item[0]:
                        op = item[0][0].op
                        qubit = item[0][0][0]

                    else:
                        op = "I"
                        qubit = 0
                    kron_hami = qu.pauli(op)
                    qu_list = [qubit]
                
                    for i in range(1, len(item[0])):
                        if item[0]:
                            op = item[0][i].op
                            qubit = item[0][i][0]
                        else:
                            op = "I"
                            qubit = 0
                        kron_hami = qu.kron(kron_hami, qu.pauli(op))
                        qu_list.append(qubit)

                    c += item[1] * ctx[0].local_expectation(kron_hami, tuple(qu_list)).real
                
        # amplitude
        elif ctx[4]:
            c = ctx[0].amplitude(ctx[4])
            
        # sampling or state vector
        else:
            if ctx[3] is not None:
                if ctx[3] < 0:
                    c = np.squeeze(ctx[0].to_dense(reverse=True))
                else:
                    c = collections.Counter(ctx[0].sample(ctx[3]))
            else:
                c = collections.Counter(ctx[0].sample(1))

        return c

    def _one_qubit_gate_noargs(self, gate, ctx):
        for idx in gate.target_iter(ctx[2]):
            ctx[0].apply_gate(gate.lowername, ctx[1][idx])
        return ctx

    def _one_qubit_gate_args_theta(self, gate, ctx):
        if gate.lowername == 'u':
            for idx in gate.target_iter(ctx[2]):
                ctx[0].apply_gate('U3', gate.theta, gate.phi, gate.lam, ctx[1][idx])
        elif gate.lowername == 'phase':
            for idx in gate.target_iter(ctx[2]):
                ctx[0].apply_gate('U1', gate.theta, ctx[1][idx])
        else:
            for idx in gate.target_iter(ctx[2]):
                ctx[0].apply_gate(gate.lowername, gate.theta, ctx[1][idx])
        return ctx
    
    def _two_qubit_gate_noargs(self, gate, ctx):
        for control, target in gate.control_target_iter(ctx[2]):
            ctx[0].apply_gate(gate.lowername, ctx[1][control], ctx[1][target])
        return ctx

    def _two_qubit_gate_args_theta(self, gate, ctx):
        if gate.lowername == 'rxx':
            for control, target in gate.control_target_iter(ctx[2]):
                ctx[0].apply_gate('H', ctx[1][control])
                ctx[0].apply_gate('H', ctx[1][target])
                ctx[0].apply_gate('RZZ', gate.theta, ctx[1][control], ctx[1][target])
                ctx[0].apply_gate('H', ctx[1][control])
                ctx[0].apply_gate('H', ctx[1][target])
        elif gate.lowername == 'ryy':
            for control, target in gate.control_target_iter(ctx[2]):
                ctx[0].apply_gate('RX', -np.pi/4, ctx[1][control])
                ctx[0].apply_gate('RX', -np.pi/4, ctx[1][target])
                ctx[0].apply_gate('RZZ', gate.theta, ctx[1][control], ctx[1][target])
                ctx[0].apply_gate('RX', np.pi/4, ctx[1][control])
                ctx[0].apply_gate('RX', np.pi/4, ctx[1][target])
        elif gate.lowername == 'cphase':
            for control, target in gate.control_target_iter(ctx[2]):
                ctx[0].apply_gate('CU1', gate.theta, ctx[1][control], ctx[1][target])
        elif gate.lowername == 'crx':
            for control, target in gate.control_target_iter(ctx[2]):
                cu_params = (gate.theta, -math.pi / 2.0, math.pi / 2.0)
                ctx[0].apply_gate('CU3', *cu_params, ctx[1][control], ctx[1][target])
        else:
            for control, target in gate.control_target_iter(ctx[2]):
                ctx[0].apply_gate(gate.lowername, gate.theta, ctx[1][control], ctx[1][target])
        return ctx

    # def _three_qubit_gate_noargs(self, gate, ctx):
    #     return ctx

    def gate_measure(self, gate, ctx):
        return ctx

    gate_x = gate_y = gate_z = gate_h = gate_t = gate_s = _one_qubit_gate_noargs
    gate_rx = gate_ry = gate_rz = gate_phase = gate_u = _one_qubit_gate_args_theta
    gate_cx = gate_cy = gate_cz = _two_qubit_gate_noargs
    gate_rxx = gate_ryy = gate_rzz = gate_cphase = _two_qubit_gate_args_theta
    #gate_crx = gate_cry = gate_crz = _two_qubit_gate_args_theta
    gate_crx = _two_qubit_gate_args_theta
    # gate_ccx = gate_cswap = _three_qubit_gate_noargs
    gate_reset = _one_qubit_gate_noargs