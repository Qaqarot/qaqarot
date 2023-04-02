from .backendbase import Backend
from ..circuit import Circuit
from ..gate import *

import numpy as np

class cuSV(Backend):
  def _preprocess_run(self, gates, n_qubits, args, kwargs):
    import cupy as cp

    # initialize by |00...00>
    h_sv = np.asarray(np.full(2**n_qubits,0.0+0.0j), dtype=np.complex64)
    h_sv[0] = 1.0+0.0j
    d_sv = cp.asarray(h_sv)
    # returns gates, ctx. In this case, ctx = (d_sv, n_qubits)
    return gates, (d_sv, n_qubits)

  def _postprocess_run(self, ctx):
    return ctx[0]

  def _one_qubit_gate_noargs(self, gate, ctx):
    import cupy as cp
    import cuquantum
    from cuquantum import custatevec as cusv
    
    for idx in gate.target_iter(ctx[1]):

      #if you want a new gate please write down here.
      if gate.lowername == 'x':
        matrix = cp.array([
          [0, 1],
          [1, 0]
        ], dtype=np.complex64)
      elif gate.lowername == 'y':
        matrix = cp.array([
          [0, -1j],
          [1j, 0]
        ], dtype=np.complex64)
      elif gate.lowername == 'z':
        matrix = cp.array([
          [1, 0],
          [0, -1]
        ], dtype=np.complex64)
      elif gate.lowername == 'h':
        matrix = cp.array(np.array([
          [1, 1],
          [1, -1]
        ]) / np.sqrt(2), dtype=np.complex64)
      elif gate.lowername == 't':
        matrix = cp.array([
          [1, 0],
          [0, np.exp(np.pi/4*1j)]
        ], dtype=np.complex64)
      elif gate.lowername == 's':
        matrix = cp.array([
          [1, 0],
          [0, 1j]
        ], dtype=np.complex64)
      else:
        matrix = cp.asarray([1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0+0.0j], dtype=np.complex64)

      nIndexBits = ctx[1]
      nSvSize  = (1 << nIndexBits)
      nTargets  = 1
      nControls = 0
      adjoint  = 0

      targets  = np.asarray([idx], dtype=np.int32)
      controls  = np.asarray([], dtype=np.int32)

      if isinstance(matrix, cp.ndarray):
        matrix_ptr = matrix.data.ptr
      elif isinstance(matrix, np.ndarray):
        matrix_ptr = matrix.ctypes.data
      else:
        raise ValueError

      # cuStateVec handle initialization
      handle = cusv.create()
      workspaceSize = cusv.apply_matrix_get_workspace_size(
        handle, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, nTargets, nControls, cuquantum.ComputeType.COMPUTE_32F)

      # check the size of external workspace
      if workspaceSize > 0:
        workspace = cp.cuda.memory.alloc(workspaceSize)
        workspace_ptr = workspace.ptr
      else:
        workspace_ptr = 0

      # apply gate
      cusv.apply_matrix(
        handle, ctx[0].data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
        cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)

      # destroy handle
      cusv.destroy(handle)

    return ctx

  gate_x = _one_qubit_gate_noargs
  gate_y = _one_qubit_gate_noargs
  gate_z = _one_qubit_gate_noargs
  gate_h = _one_qubit_gate_noargs
  gate_t = _one_qubit_gate_noargs
  gate_s = _one_qubit_gate_noargs

  def _one_qubit_gate_args_theta(self, gate, ctx):
    import cupy as cp
    import cuquantum
    from cuquantum import custatevec as cusv
    
    for idx in gate.target_iter(ctx[1]):

      #if you want a new gate please write down here.
      if gate.lowername == 'rx':
        matrix = cp.array([
          [np.cos(gate.theta/2), -np.sin(gate.theta/2)*1j],
          [-np.sin(gate.theta/2)*1j, np.cos(gate.theta/2)]
        ], dtype=np.complex64)
      elif gate.lowername == 'ry':
        matrix = cp.array([
          [np.cos(gate.theta/2) -np.sin(gate.theta/2)],
          [np.sin(gate.theta/2), np.cos(gate.theta/2)]
        ], dtype=np.complex64)
      elif gate.lowername == 'rz':
        matrix = cp.array([
          [np.exp(-gate.theta/2*1j), 0],
          [0, np.exp(gate.theta/2*1j)]
        ], dtype=np.complex64)
      elif gate.lowername == 'p':
        matrix = cp.array([
          [1, 0],
          [0, np.exp(gate.theta*1j)]
        ], dtype=np.complex64)
      elif gate.lowername == 'u':
        matrix = cp.array([
          [np.cos(gate.theta/2), -np.exp(gate.lam*1j)*np.sin(gate.theta/2)],
          [np.exp(gate.phi*1j)*np.sin(gate.theta/2), np.exp((gate.phi+gate.lam)*1j)*np.cos(gate.theta/2)]
        ], dtype=np.complex64)
      else:
        matrix = cp.asarray([1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0+0.0j], dtype=np.complex64)

      nIndexBits = ctx[1]
      nSvSize  = (1 << nIndexBits)
      nTargets  = 1
      nControls = 0
      adjoint  = 0

      targets  = np.asarray([idx], dtype=np.int32)
      controls  = np.asarray([], dtype=np.int32)

      if isinstance(matrix, cp.ndarray):
        matrix_ptr = matrix.data.ptr
      elif isinstance(matrix, np.ndarray):
        matrix_ptr = matrix.ctypes.data
      else:
        raise ValueError

      # cuStateVec handle initialization
      handle = cusv.create()
      workspaceSize = cusv.apply_matrix_get_workspace_size(
        handle, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, nTargets, nControls, cuquantum.ComputeType.COMPUTE_32F)

      # check the size of external workspace
      if workspaceSize > 0:
        workspace = cp.cuda.memory.alloc(workspaceSize)
        workspace_ptr = workspace.ptr
      else:
        workspace_ptr = 0

      # apply gate
      cusv.apply_matrix(
        handle, ctx[0].data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
        cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)

      # destroy handle
      cusv.destroy(handle)

    return ctx

  gate_rx = _one_qubit_gate_args_theta
  gate_ry = _one_qubit_gate_args_theta
  gate_rz = _one_qubit_gate_args_theta
  gate_p = gate_phase = _one_qubit_gate_args_theta
  gate_u = _one_qubit_gate_args_theta

  def _two_qubit_gate_noargs(self, gate, ctx):
    import cupy as cp
    import cuquantum
    from cuquantum import custatevec as cusv
    
    for control, target in gate.control_target_iter(ctx[1]):

      if gate.lowername == 'cx':
        matrix = cp.asarray([
          [0, 1],
          [1, 0]
        ], dtype=np.complex64)
      elif gate.lowername == 'cy':
        matrix = cp.array([
          [0, -1j],
          [1j, 0]
        ], dtype=np.complex64)
      elif gate.lowername == 'cz':
        matrix = cp.array([
          [1, 0],
          [0, -1]
        ], dtype=np.complex64)
      else:
        matrix = cp.asarray([1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0+0.0j], dtype=np.complex64)

      nIndexBits = ctx[1]
      nSvSize  = (1 << nIndexBits)
      nTargets  = 1
      nControls = 1
      adjoint  = 0

      targets  = np.asarray([target], dtype=np.int32)
      controls  = np.asarray([control], dtype=np.int32)

      if isinstance(matrix, cp.ndarray):
        matrix_ptr = matrix.data.ptr
      elif isinstance(matrix, np.ndarray):
        matrix_ptr = matrix.ctypes.data
      else:
        raise ValueError

      # cuStateVec handle initialization
      handle = cusv.create()
      workspaceSize = cusv.apply_matrix_get_workspace_size(
        handle, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, nTargets, nControls, cuquantum.ComputeType.COMPUTE_32F)

      # check the size of external workspace
      if workspaceSize > 0:
        workspace = cp.cuda.memory.alloc(workspaceSize)
        workspace_ptr = workspace.ptr
      else:
        workspace_ptr = 0

      # apply gate
      cusv.apply_matrix(
        handle, ctx[0].data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
        cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)

      # destroy handle
      cusv.destroy(handle)
    return ctx

  gate_cx = gate_cy = gate_cz = _two_qubit_gate_noargs

  # https://docs.nvidia.com/cuda/cuquantum/custatevec/getting_started.html#code-example

  def _three_qubit_gate_noargs(self, gate, ctx):
    import cupy as cp
    import cuquantum
    from cuquantum import custatevec as cusv
    
    c1, c2, target = gate.targets

    #if you want a new gate please write down here.
    if gate.lowername == 'ccx':
      matrix = cp.array([
        [0, 1],
        [1, 0]
      ], dtype=np.complex64)
    else:
      matrix = cp.asarray([1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0+0.0j], dtype=np.complex64)

    nIndexBits = ctx[1]
    nSvSize  = (1 << nIndexBits)
    nTargets  = 1
    nControls = 2
    adjoint  = 0

    targets  = np.asarray([target], dtype=np.int32)
    controls  = np.asarray([c1, c2], dtype=np.int32)

    if isinstance(matrix, cp.ndarray):
      matrix_ptr = matrix.data.ptr
    elif isinstance(matrix, np.ndarray):
      matrix_ptr = matrix.ctypes.data
    else:
      raise ValueError

    # cuStateVec handle initialization
    handle = cusv.create()
    workspaceSize = cusv.apply_matrix_get_workspace_size(
      handle, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
      cusv.MatrixLayout.ROW, adjoint, nTargets, nControls, cuquantum.ComputeType.COMPUTE_32F)

    # check the size of external workspace
    if workspaceSize > 0:
      workspace = cp.cuda.memory.alloc(workspaceSize)
      workspace_ptr = workspace.ptr
    else:
      workspace_ptr = 0

    # apply gate
    cusv.apply_matrix(
      handle, ctx[0].data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
      cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
      cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)

    # destroy handle
    cusv.destroy(handle)

    return ctx

  gate_ccx = _three_qubit_gate_noargs

  # https://github.com/NVIDIA/cuQuantum/tree/main/python/samples/custatevec

  def _cswap(self, gate, ctx):
    import cupy as cp
    import cuquantum
    from cuquantum import custatevec as cusv
    control, t1, t2 = gate.targets

    #if you want a new gate please write down here.
    if gate.lowername != 'cswap':
      raise NotImplementedError()

    nIndexBits = ctx[1]
    nSvSize  = (1 << nIndexBits)
    nBitSwaps = 1
    bitSwaps = [(t1, t2)]
    maskLen = 1
    maskBitString = [1]
    maskOrdering = [control]

    # cuStateVec handle initialization
    handle = cusv.create()
    cusv.swap_index_bits(
      handle, ctx[0].data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits,
      bitSwaps, nBitSwaps,
      maskBitString, maskOrdering, maskLen)

    # destroy handle
    cusv.destroy(handle)

    return ctx

  gate_cswap = _cswap

  def gate_measure(self, gate, ctx):
    import cupy as cp
    import cuquantum
    from cuquantum import custatevec as cusv
    return ctx

  gate_reset = _one_qubit_gate_noargs