from .backendbase import Backend
from ..circuit import Circuit
from ..gate import *

import numpy as np

class cuTN(Backend):

  def _preprocess_run(self, gates, n_qubits, args, kwargs):
    import opt_einsum as oe

    arr_tensor = []
    arr_arm = []
    arr_state = []
    n_symbols = 0

    #initial state vec
    psi = np.array([1,0], dtype='complex128')

    #arm, state and tensor
    for i in range(n_qubits):
      arr_arm.append(oe.get_symbol(n_symbols))
      arr_state.append(oe.get_symbol(n_symbols))
      arr_tensor.append(psi)
      n_symbols += 1

    #number of shots for samples
    if 'shots' in kwargs:
      n_shots = kwargs['shots']
    else:
      n_shots = 1

    #for expectation value of hamiltonian
    if 'hamiltonian' in kwargs:
      hami = kwargs['hamiltonian']
    else:
      hami = None

    return gates, (arr_arm, arr_tensor, arr_state, [n_symbols], n_qubits, n_shots, hami)

  def _postprocess_run(self, ctx):
    import cuquantum
    import cupy as cp

    # Set the pathfinder options
    options = cuquantum.OptimizerOptions()
    options.slicing.disable_slicing = 0
    options.samples = 100

    #execution
    stream = cp.cuda.Stream()
    D_d, info = cuquantum.contract(','.join(ctx[0]), *ctx[1], optimize=options, stream=stream, return_info=True)
    stream.synchronize()

    #state vec out of memory
    #D_d.reshape(2**ctx[4])

    print('beta : only H, X, RX, RY, RZ, CX, RZZ are available')

    return f'{info[1].opt_cost/1e9} GFLOPS', D_d.reshape(2**ctx[4])

  def _one_qubit_gate_noargs(self, gate, ctx):
    import opt_einsum as oe

    #fixed rotation
    H = np.array([[1,1],[1,-1]], dtype='complex128')/np.sqrt(2)
    X = np.array([[0,1],[1,0]], dtype='complex128')

    #ctx[4] is n_qubits
    for idx in gate.target_iter(ctx[4]):

      #01.arm
      ctx[0].append(ctx[2][idx] + oe.get_symbol(ctx[3][0]))

      #02.tensor
      ctx[1].append(locals()[gate.uppername])

      #03.state
      ctx[2][idx] = oe.get_symbol(ctx[3][0])

      #04.n_symbols
      ctx[3][0] = ctx[3][0] + 1

    return ctx

  def _one_qubit_gate_args_theta(self, gate, ctx):
    import opt_einsum as oe

    #ctx[4] is n_qubits
    for idx in gate.target_iter(ctx[4]):

      #arbitrary rotation
      RX = np.array([[np.cos(gate.theta/2),-1j*np.sin(gate.theta/2)],[-1j*np.sin(gate.theta/2),np.cos(gate.theta/2)]], dtype='complex128')
      RY = np.array([[np.cos(gate.theta/2),-1*np.sin(gate.theta/2)],[np.sin(gate.theta/2),np.cos(gate.theta/2)]], dtype='complex128')
      RZ = np.array([[np.exp(-1j*gate.theta/2),0],[0,np.exp(1j*gate.theta/2)]], dtype='complex128')

      #01.arm
      ctx[0].append(ctx[2][idx] + oe.get_symbol(ctx[3][0]))

      #02.tensor
      ctx[1].append(locals()[gate.uppername])

      #03.state
      ctx[2][idx] = oe.get_symbol(ctx[3][0])

      #04.n_symbols
      ctx[3][0] = ctx[3][0] + 1

    return ctx

  def _two_qubit_gate_noargs(self, gate, ctx):
    import opt_einsum as oe

    #fixed lotation
    CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype='complex128').reshape(2,2,2,2)

    #ctx[4] is n_qubits
    for control, target in gate.control_target_iter(ctx[4]):

      #01.arm
      ctx[0].append(ctx[2][control] + ctx[2][target] + oe.get_symbol(ctx[3][0]) + oe.get_symbol(ctx[3][0]+1))

      #02.tensor
      ctx[1].append(locals()[gate.uppername])

      #03.state
      ctx[2][control] = oe.get_symbol(ctx[3][0])
      ctx[2][target] = oe.get_symbol(ctx[3][0]+1)

      #04.n_symbols
      ctx[3][0] = ctx[3][0] + 2

    return ctx

  def _two_qubit_gate_args_theta(self, gate, ctx):
    import opt_einsum as oe

    #ctx[4] is n_qubits
    for control, target in gate.control_target_iter(ctx[4]):

      #arbitrary lotation
      RZZ = np.array([[np.exp(-1j*gate.theta/2),0,0,0],[0,np.exp(1j*gate.theta/2),0,0],[0,0,np.exp(1j*gate.theta/2),0],[0,0,0,np.exp(-1j*gate.theta/2)]], dtype='complex128').reshape(2,2,2,2)

      #01.arm
      ctx[0].append(ctx[2][control] + ctx[2][target] + oe.get_symbol(ctx[3][0]) + oe.get_symbol(ctx[3][0]+1))

      #02.tensor
      ctx[1].append(locals()[gate.uppername])

      #03.state
      ctx[2][control] = oe.get_symbol(ctx[3][0])
      ctx[2][target] = oe.get_symbol(ctx[3][0]+1)

      #04.n_symbols
      ctx[3][0] = ctx[3][0] + 2

    return ctx

  def _three_qubit_gate_noargs(self, gate, ctx):
    return ctx

  def gate_measure(self, gate, ctx):
    return ctx

  gate_x = gate_y = gate_z = gate_h = gate_t = gate_s = _one_qubit_gate_noargs
  gate_rx = gate_ry = gate_rz = gate_phase = _one_qubit_gate_args_theta
  gate_cx = gate_cy = gate_cz = _two_qubit_gate_noargs
  gate_rxx = gate_ryy = gate_rzz = _two_qubit_gate_args_theta
  gate_ccx = gate_cswap = _three_qubit_gate_noargs