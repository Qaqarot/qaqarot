from collections import Counter
from .backendbase import Backend
import blueqat.gate as bqgate
import numpy as np

class QgateBackend(Backend) :

    def __init__(self) :
        import sys
        this = sys.modules[__name__]
        if not hasattr(this, 'qgate') :
            import qgate
            this.qgate = qgate
            this.model = qgate.model
            this.gtype = qgate.model.gate_type

            QgateBackend.gatetypes = {
                'i' : (gtype.ID, 0, 1),
                'x' : (gtype.X, 0, 1), 'y' : (gtype.Y, 0, 1), 'z' : (gtype.Z, 0, 1),
                'h' : (gtype.H, 0, 1), 't' : (gtype.T, 0, 1), 's' : (gtype.S, 0, 1),
                # adjoint
                'tdg' : (gtype.T, 0, 1), 'sdg' : (gtype.S, 0, 1),
                # 1-control-bit gate
                'cz' : (gtype.Z, 1, 1), 'cx' : (gtype.X, 1, 1), 'cnot' : (gtype.X, 1, 1),
                # rotation, phase gate
                'rx' : (gtype.RX, 0, 1), 'ry' : (gtype.RY, 0, 1), 'rz' : (gtype.RZ, 0, 1),
                'phase' : (gtype.U1, 0, 1),
                # controlled rotation gate
                'crx' : (gtype.RX, 1, 1), 'cry' : (gtype.RY, 1, 1), 'crz' : (gtype.RZ, 1, 1),
                'cphase' : (gtype.U1, 1, 1),
                # U gate
                'u1' : (gtype.RZ, 0, 1), 'u2' : (gtype.U2, 0, 1), 'u3' : (gtype.U, 0, 1),
                # gtype of U1 gate is RZ instead of U1, for compatibility with OpenQASM.
                # controlled U gate
                'cu1' : (gtype.U1, 1, 1), 'cu2' : (gtype.U2, 1, 1), 'cu3' : (gtype.U, 1, 1),
                # swap and multi-controlled-bit gate
                'swap' : (gtype.SWAP, 0, 2), 'ccx' : (gtype.X, 2, 1),
            }

    def run(self, gates, n_qubits, *args, **kwargs):
        self.n_qubits = n_qubits
        # returns
        r = kwargs.get('returns', '')
        sampling = kwargs.get('sampling', 'qgate')
        circuit_prep = kwargs.get('circuit_prep', qgate.prefs.dynamic)
        simprefs = {'circuit_prep': circuit_prep}
        shots = kwargs.get('shots', 0)
        if shots != 0 and r == '' :
            r = 'shots'
        elif r == '' :
            r = 'statevector'

        # create simulator instance
        runtime = kwargs.get('runtime', 'cpu')
        # runtime = kwargs.get('runtime', 'py')
        if runtime == 'cuda' :
            sim = qgate.simulator.cuda(**simprefs)
        elif runtime == 'cpu' :
            sim = qgate.simulator.cpu(**simprefs)
        elif runtime == 'py' :
            sim = qgate.simulator.py(**simprefs)
        else :
            raise RuntimeError("Unknown runtime, {}.  Accetable values are 'cpu', 'cuda' and 'py'.".format(runtime))

        # creating registers and references.
        self.qregs = [model.Qreg() for _ in range(n_qubits)]
        self.refs = [model.Reference() for _ in range(n_qubits)]
        sim.qubits.set_ordering(self.qregs)

        ops = self.convert_operators(gates)
        # qgate.dump(ops)
        if r == 'statevector' :
            return self.get_state_vector(sim, ops)
        elif r == 'shots' :
            return Counter(self.sample(sim, ops, shots, sampling))
        elif r == 'statevector_and_shots' :
            if 1 < shots :
                raise RuntimeError('shots must be 1 when returns="statevector_and_shots"')
            vec, shots = self.get_state_vector_and_sample(sim, ops)
            return vec, Counter(shots)
        else :
            raise RuntimeError('Unkown returns token, {}.'.format(r))

    def convert_operators(self, gates) :
        ops = list()
        for op in gates :
            typeinfo = QgateBackend.gatetypes.get(op.lowername, None)
            if typeinfo is not None :
                if typeinfo[2] == 1 :
                    ops += self.convert_one_qubit_gate(op, typeinfo)
                else :
                    ops += self.convert_multi_qubit_gate(op, typeinfo)
            elif op.lowername in ('measure', 'm') :
                # "measure": gate.Measurement, "m": gate.Measurement,
                ops += self.convert_measure(op)
            else :
                assert False
        return ops

    def convert_one_qubit_gate(self, gate, typeinfo) :
        glist = list()

        if typeinfo[1] == 0 :
            # no control bit
            assert isinstance(gate, bqgate.OneQubitGate)
            one_qubit_gates = self.create_gates(typeinfo[0], gate)
            # "tdg": gate.TDagGate, "sdg": gate.SDagGate,
            if gate.lowername[1:] == 'dg' :
                for qg in one_qubit_gates :
                    qg.set_adjoint(True)
            glist += one_qubit_gates
        else :
            if isinstance(gate, bqgate.TwoQubitGate) :
                # controlled gate
                for ctrlidx, qregidx in gate.control_target_iter(self.n_qubits) :
                    gt = typeinfo[0](*gate.params)
                    g = model.Gate(gt)
                    ctrl = self.qregs[ctrlidx]
                    qreg = self.qregs[qregidx]
                    g.set_ctrllist(ctrl)
                    g.set_qreg(qreg)
                    glist.append(g)
            elif isinstance(gate, bqgate.ToffoliGate) :
                gt = typeinfo[0](*gate.params)
                g = model.Gate(gt)
                c0, c1, t = gate.targets
                ctrls = (self.qregs[c0], self.qregs[c1])
                qreg = self.qregs[t]
                g.set_ctrllist(ctrls)
                g.set_qreg(qreg)
                glist.append(g)
            else :
                raise RuntimeError('Unknown gate, {}.'.format(repr(gate)))

        return glist

    def convert_multi_qubit_gate(self, gate, typeinfo) :
        glist = list()
        g = model.MultiQubitGate(typeinfo[0](*gate.params))
        t0, t1 = gate.targets
        qregs = (self.qregs[t0], self.qregs[t1])
        g.set_qreglist(qregs)
        glist.append(g)
        return glist

    def convert_measure(self, measure) :
        mlist = list()
        for qregidx in measure.target_iter(self.n_qubits) :
            m = model.Measure(self.refs[qregidx], self.qregs[qregidx])
            mlist.append(m)
        return mlist

    def create_gates(self, gate_type_factory, gate) :
        glist = list()
        for qregidx in gate.target_iter(self.n_qubits) :
            gt = gate_type_factory(*gate.params)
            g = model.Gate(gt)
            qreg = self.qregs[qregidx]
            g.set_qreg(qreg)
            glist.append(g)
        return glist

    def create_global_phase_gates(self, phase, gate) :
        glist = list()
        for qregidx in gate.target_iter(self.n_qubits) :
            gt = gtype.ExpiI(phase)
            g = model.Gate(gt)
            qreg = self.qregs[qregidx]
            g.set_qreg(qreg)
            glist.append(g)
        return glist

    def get_state_vector(self, sim, ops) :
        sim.run(ops)
        return sim.qubits.states[:]

    def sample(self, sim, ops, shots, sampling_method) :
        sampling_qregs = list()
        ops_no_measure = list()

        for op in ops :
            if isinstance(op, model.Measure) :
                sampling_qregs.append(op.qreg)
            else :
                ops_no_measure.append(op)

        sim.run(ops_no_measure)
        # print(sim.qubits.states[:])

        qregs_no_dup = set(sampling_qregs)
        sampling_qregs = list(qregs_no_dup)
        sampling_qregs.sort(key = self.qregs.index)

        if sampling_method == 'blueqat' :
            sampling_pool = sim.qubits.create_sampling_pool(sampling_qregs,
                                                            BlueqatCompatibleSamplingPool)
            # print(sampling_pool.prob)
        else :
            sampling_pool = sim.qubits.create_sampling_pool(sampling_qregs)
        obs = sampling_pool.sample(shots)

        # FIXME: faster conversion for bit representation
        hist = obs.histgram()
        bit_format = '0{}b'.format(len(self.qregs))
        sampling_pos = [self.qregs.index(qreg) for qreg in sampling_qregs]
        max_bit_pos = len(self.qregs) - 1
        strkey_hist = dict()
        for v, c in hist.items() :
            permuted = 0
            for idx, pos in enumerate(sampling_pos) :
                if (v & (1 << idx)) != 0 :
                    permuted |= 1 << (max_bit_pos - pos)
            strbits = format(permuted, bit_format)
            strkey_hist[strbits] = c
        return strkey_hist

    def get_state_vector_and_sample(self, sim, ops) :
        sim.run(ops)
        # state vector
        state_vector = sim.qubits.states[:]
        # shot
        reversed_refs = list(reversed(self.refs))
        obs = sim.obs(reversed_refs)
        bit_format = '0{}b'.format(len(self.refs))
        strkey_hist = dict()
        strbits = format(obs.int, bit_format)
        strkey_hist[strbits] = 1 # observed 1 time.
        return state_vector, strkey_hist

# provided for compatiblity and tests
class BlueqatCompatibleSamplingPool :
    def __init__(self, prob, empty_lanes, qreg_ordering) :
        #prob *= (1. / np.sum(prob))
        self.prob = prob

        self.qreg_ordering = qreg_ordering
        self.empty_lanes = empty_lanes
        self.mask = 0
        for idx in empty_lanes :
            self.mask |= 1 << idx

    def sample(self, n_samples) :
        obs = np.empty([n_samples], np.int)
        import random
        for idx in range(n_samples) :
            redprob = self.prob
            v = 0
            p_all = 1
            for bitpos in range(len(self.qreg_ordering)) :
                rnum = random.random()
                p_zero = np.sum(redprob[::2])
                if rnum < (p_zero / p_all) :
                    redprob = redprob[::2]
                    p_all = p_zero
                else :
                    redprob = redprob[1::2]
                    p_all = p_all - p_zero
                    v |= (1 << bitpos)

            obs[idx] = v

        if self.mask != 0 :
            self.shift_for_empty_lanes(obs)
        obslist = qgate.simulator.observation.ObservationList(self.qreg_ordering, obs, self.mask)
        return obslist

    def shift_for_empty_lanes(self, obs) :
        # create control bit mask
        bits = [1 << lane for lane in self.empty_lanes]
        n_shifts = len(bits) + 1

        mask = bits[0] - 1
        masks = [ mask ]
        for idx in range(len(bits) - 1) :
            mask = (bits[idx + 1] - 1) & (~(bits[idx] * 2 - 1))
            masks.append(mask)
        mask = ~(bits[-1] * 2 - 1)
        masks.append(mask)

        for idx in range(len(obs)) :
            v = obs[idx]
            v_shifted = 0
            for shift in range(n_shifts) :
                v_shifted |= (v << shift) & masks[shift]
            obs[idx] = v_shifted
