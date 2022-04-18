
"""
`gateOps` module implements calculation for quantum gate operations.
This module may be redefined as a backend option in later versions.
"""


import numpy as np
from .bosonicLadder import *
from .WignerFunc import reduceState, partialTrace
from scipy.special import eval_hermite as hermite
from scipy.linalg import expm

def Displacement(state, mode, alpha, modeNum, cutoff):
    state = _singleGate_preProcess(state, mode)
    state = exp_annihiration(state, -np.conj(alpha), cutoff = cutoff)
    state = exp_creation(state, alpha, cutoff = cutoff)
    state = _singleGate_postProcess(state, mode, modeNum)
    state = state * np.exp(-np.abs(alpha)**2 / 2)
    return state

def Beamsplitter(state, mode1, mode2, theta, modeNum, cutoff):
    if modeNum < 2:
        raise ValueError("The gate requires more than one mode.")
    state = _twoModeGate_preProcess(state, mode1, mode2)
    state = exp_BS(state, -theta, cutoff = cutoff)
    state = _twoModeGate_postProcess(state, mode1, mode2, modeNum)
    return state

def Squeeze(state, mode, r, phi, modeNum, cutoff):
    G = np.exp(2 * 1j * phi) * np.tanh(r)
    state = _singleGate_preProcess(state, mode)
    state = exp_annihiration(state, np.conj(G) / 2, order = 2, cutoff = cutoff)
    state = exp_photonNum(state, -np.log(np.cosh(r)), cutoff = cutoff)
    state = exp_creation(state, -G / 2, order = 2, cutoff = cutoff)
    state = _singleGate_postProcess(state, mode, modeNum)
    state = state / np.sqrt(np.cosh(r))
    return state

# def Squeeze(state, mode, r, phi, modeNum, cutoff):
#     state = _singleGate_preProcess(state, mode)
#     state = exp_aa_minus_AA(state, r * np.exp(2j * phi) / 2, cutoff = cutoff)
#     state = _singleGate_postProcess(state, mode, modeNum)
#     return state

def KerrEffect(state, mode, chi, modeNum, cutoff):
    state = _singleGate_preProcess(state, mode)
    state = exp_AAaa(state, 1j * chi / 2, cutoff = cutoff)
    state = _singleGate_postProcess(state, mode, modeNum)
    return state

def HamiltonianEvo(state, mode, expr, gamma, modeNum, cutoff):
    dim = state.shape[0]
    str_aA = expand_xp_to_aA(expr, dim, evaluate = False, hbar = 1)
    str_exp_aA = exp_str_aA(str_aA, -1j * gamma, N = 5)
    mat = str_to_aA_mat(str_exp_aA, dim, hbar = 1)
    state = _singleGate_preProcess(state, mode)
    mat_ = expm(-1j * gamma * mat)
    state = np.dot(mat_, state.T)
    state = _singleGate_postProcess(state.T, mode, modeNum)
    return state

def photonMeasurement(state, mode, post_select):
    reducedDensity = reduceState(state, mode)
    probs = np.real(np.diag(reducedDensity))
    probs = probs / np.sum(probs)
    if post_select is None:
        res = np.random.choice(probs.shape[0], 1, p = probs)
    else:
        res = post_select
    prob = probs[res]
    
    state_ = np.swapaxes(state, mode, 0)
    ind = np.ones((state.shape[-1]), bool)
    ind[res] = False
    state_[ind] = 0
    state_ = np.swapaxes(state_, mode, 0)
    state_ = state_ / np.sqrt(prob)
    return res, state_

def homodyneMeasurement(state, mode, theta, post_select):
    if post_select is None:
        res, psi = homodyneFock(state, mode, theta, ite = 1, hbar = 1)
    else:
        res = post_select   
    state_ = _afterHomodyne(state, mode, res, psi, theta)
    return res, state_

def homodyneFock(state, mode, theta, ite = 1, hbar = 1):
    dim = state.shape[0]
    rho = reduceState(state, mode)
    lim = np.sqrt(dim * 2) + 3 # heuristic
    q = np.arange(-lim, lim, lim / 500)
    q_ = q / np.sqrt(hbar)
    psi = _positionWaveFunc(dim, q_, theta) # psi_{n}(q, theta)
    probs = np.zeros(q.shape) + 0j
    for i in range(dim):
        for j in range(dim):
            tmp = rho[i, j] * psi[i, :] * np.conj(psi[j, :])# * np.exp(1j * theta * (j - i))
            probs += tmp
    probs = np.abs(np.real(probs))
    probs_sum = np.abs(np.sum(probs))
    if 1 - probs_sum < 0.01:
        probs = probs / probs_sum
    else:
        print("sum of probability", probs_sum)
        raise ValueError('probabilities do not sum to 1')
    res = np.random.choice(probs.shape[0], ite, p = probs)
    return q[res], psi[:, res]

def _afterHomodyne(state, mode, q, psi, theta):
    dim = state.shape[-1]
    modeNum = state.ndim
    #psi = _positionWaveFunc(dim, q, theta)
    state = _singleGate_preProcess(state, mode)
    state = state * psi.T
    state[:, 0] = np.sum(state, axis = -1)
    state[:, 1:] = 0
    state = _singleGate_postProcess(state, mode, modeNum)
    state = state / np.sqrt(np.sum(np.abs(state) ** 2))
    return state

def _positionWaveFunc(n, q, theta, hbar = 1):
    #lim = np.sqrt(n * 2) + 3 # heuristic
    n_ = np.arange(n)[:, np.newaxis]
    H = hermite(n_, q)
    A = (np.pi * hbar) ** 0.25 / np.sqrt(2 ** n_ * fact(n))
    psi = A * H * np.exp(-q ** 2 / 2 / hbar) * np.exp(-1j * n_ * theta)
    C = np.sum(np.abs(psi) ** 2, axis=1) + 0j # sum of squeres
    psi = psi / np.sqrt(C)[:, np.newaxis] # normalized wave functions
    return psi

def _singleGate_preProcess(fockState, mode):
    cutoff = fockState.shape[-1] - 1
    fockState = np.swapaxes(fockState, mode, fockState.ndim - 1)
    return fockState.reshape(-1, cutoff + 1)

def _singleGate_postProcess(fockState, mode, modeNum):
    cutoff = fockState.shape[-1] - 1
    fockState = fockState.reshape([cutoff + 1] * modeNum)
    return np.swapaxes(fockState, mode, modeNum - 1)

def _twoModeGate_preProcess(fockState, mode1, mode2):
    cutoff = fockState.shape[-1] - 1
    modeNum = fockState.ndim
    fockState = np.swapaxes(fockState, mode2, modeNum - 1)
    fockState = np.swapaxes(fockState, mode1, modeNum - 2)
    return fockState.reshape(-1, (cutoff + 1) ** 2)

def _twoModeGate_postProcess(fockState, mode1, mode2, modeNum):
    dim = np.int(np.sqrt(fockState.shape[-1]))
    fockState = fockState.reshape([dim] * modeNum)
    fockState = np.swapaxes(fockState, mode1, modeNum - 2)
    fockState = np.swapaxes(fockState, mode2, modeNum - 1)
    return fockState
